"""A22-voxcpm (E12): offset-aware replacements for voxcpm.training.data loaders.

Why (documented local edit, reports/voxcpm_local_edits.patch): the official
``load_audio_text_datasets`` casts the manifest's ``audio`` column to
``datasets.Audio`` and can only decode WHOLE files, while our training units are
defined as [offset_start, offset_end) slices of the project's source flacs
(data/train/{long,short}/utt2offset; prereg E12: audio comes from the original
sources, read-only, no re-cut copies on the crisis SSD). These replacements keep
the manifest's ``audio`` column as a plain path string plus offset columns and do
soundfile seek-read + resample-to-16k-mono in ``__getitem__``:

  * decode: soundfile.read(start=round(off0*sr), stop=round(off1*sr)) -- reads only
    the slice, mono-mixes channels by mean;
  * resample: librosa.resample(res_type default 'soxr_hq') -- the same library the
    upstream inference path uses for prompt audio (librosa.load in
    src/voxcpm/model/voxcpm2.py:417), so train-side and inference-side audio go
    through the same resampler family;
  * everything downstream (HFVoxCPMDataset sample dict contract, collate_fn,
    AudioFeatureProcessingPacker) is reused from upstream unchanged.

The long arm has offset_start == 0 and offset_end == duration for every row
(verified: 0 non-zero offsets in data/train/long/utt2offset), so for it this path
is simply "decode the whole file"; the short arm slices windows out of the same
parents. RAM per worker peaks at one decoded slice (a 15-min 44.1 kHz stereo flac
is ~270 MB as float64 -> read as float32, ~137 MB) -- checked before training per
the E12 prereg discipline.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from voxcpm.training.data import (  # noqa: F401  (re-exported for the train script)
    DEFAULT_AUDIO_COLUMN,
    DEFAULT_ID_COLUMN,
    DEFAULT_TEXT_COLUMN,
    HFVoxCPMDataset,
)

REQUIRED_COLUMNS = ('text', 'audio', 'duration', 'offset_start', 'offset_end')


def load_audio_text_datasets_offsets(
    train_manifest: str,
    val_manifest: str = "",
    sample_rate: int = 16_000,
) -> Tuple[Dataset, Optional[Dataset], int]:
    """Load JSONL manifests WITHOUT casting audio -- paths + offsets stay as-is."""
    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest
    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    def prepare(ds: Dataset) -> Dataset:
        missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
        if missing:
            raise ValueError('manifest lacks required columns {}'.format(missing))
        if DEFAULT_ID_COLUMN not in ds.column_names:
            ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
        return ds

    train_ds = prepare(dataset_dict["train"])
    val_ds = prepare(dataset_dict["validation"]) if "validation" in dataset_dict else None
    return train_ds, val_ds, sample_rate


def _read_slice_16k_mono(path: str, off0: float, off1: float, target_sr: int) -> np.ndarray:
    import soundfile as sf
    import librosa
    info = sf.info(path)
    sr = info.samplerate
    start = int(round(float(off0) * sr))
    stop = int(round(float(off1) * sr))
    stop = min(stop, info.frames)
    wave, sr = sf.read(path, start=start, stop=stop, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)                      # mono mix (datasets.Audio does the same)
    if sr != target_sr:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(wave, dtype=np.float32)


class OffsetVoxCPMDataset(TorchDataset):
    """Same sample contract as voxcpm.training.data.HFVoxCPMDataset, but the audio
    is decoded from (path, offset_start, offset_end) at 16 kHz mono on access."""

    def __init__(self, dataset: Dataset, sample_rate: int = 16_000):
        self.dataset = dataset
        self.sample_rate = int(sample_rate)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        wave = _read_slice_16k_mono(item['audio'], item['offset_start'], item['offset_end'],
                                    self.sample_rate)
        return {
            "text_ids": item["text_ids"],
            "audio_array": wave,
            "audio_sampling_rate": self.sample_rate,
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }


def build_dataloader_offsets(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    sample_rate: int = 16_000,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    torch_dataset = OffsetVoxCPMDataset(hf_dataset, sample_rate=sample_rate)
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=HFVoxCPMDataset.collate_fn,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------------- #
# Precomputed-latent path (long arm). The AudioVAE encoder's first conv holds
# T_samples x 128 channels fp32: a 568-s unit is a single 4.33 GiB allocation, which
# OOMs INSIDE the training process next to the 2B model's fp32 weights+grads+AdamW
# (P2 probe, packed 7800/8138). Mirroring the Q-series "prepared" pipeline, the long
# arm's latents are encoded OFFLINE (scripts/voxcpm_p3_encode_latents.py: identical
# read/resample/pad/encode math as the packer -- soundfile slice -> librosa 16k mono
# -> right-pad to patch_len 2560 -> AudioVAE.encode fp32 mean) and stored as
# .npy [T_latent, 64] fp32; training then feeds latents straight into the packer,
# whose encode step becomes an identity. The short arm (windows <= 40 s, packed <= 542)
# keeps the upstream on-GPU encode path.
# ---------------------------------------------------------------------------------- #
LATENT_PAD = -100.0


class LatentVoxCPMDataset(TorchDataset):
    """Sample contract mirrors HFVoxCPMDataset, but audio comes as precomputed VAE
    latents loaded from the manifest's ``latents`` npy path."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        lat = np.load(item['latents'])          # [T_latent, 64] float32
        return {
            "text_ids": item["text_ids"],
            "latent": np.ascontiguousarray(lat, dtype=np.float32),
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }

    @classmethod
    def collate_fn(cls, batch):
        text_tensors = [torch.tensor(s["text_ids"], dtype=torch.int32) for s in batch]
        lat_tensors = [torch.tensor(s["latent"], dtype=torch.float32) for s in batch]
        max_t = max(t.shape[0] for t in text_tensors)
        max_l = max(x.shape[0] for x in lat_tensors)
        text_padded = torch.stack([
            torch.nn.functional.pad(t, (0, max_t - t.shape[0]), value=-100) for t in text_tensors])
        lat_padded = torch.stack([
            torch.nn.functional.pad(x, (0, 0, 0, max_l - x.shape[0]), value=LATENT_PAD)
            for x in lat_tensors])
        return {
            "text_tokens": text_padded,
            "audio_tokens": lat_padded,      # latents, consumed by LatentPacker
            "task_ids": torch.ones(text_padded.size(0), dtype=torch.int32),
            "dataset_ids": torch.tensor([s["dataset_id"] for s in batch], dtype=torch.int32),
            "is_prompts": [bool(s.get("is_prompt", False)) for s in batch],
        }


def make_latent_packer(config, audio_vae):
    """AudioFeatureProcessingPacker whose encode step is an identity over latents.

    unpad_audio_tokens: a padded latent row is all LATENT_PAD; the base class looks for
    the first -100 position, which on a [T, 64] tensor returns the first padded ROW
    index -- correct here. encode_audio returns the latents as [1, T, D] and never
    touches the VAE (which can stay off-GPU for the whole training).
    """
    from voxcpm.training.packers import AudioFeatureProcessingPacker

    class LatentPacker(AudioFeatureProcessingPacker):
        def encode_audio(self, latent: torch.Tensor):
            return latent.unsqueeze(0)       # [1, T_latent, D]; extract_audio_feats then
                                             # pads T to a patch multiple and rearranges

    return LatentPacker(
        dataset_cnt=1,
        max_len=config.max_length,
        patch_size=config.patch_size,
        feat_dim=config.feat_dim,
        audio_vae=audio_vae,
    )


class LatentBatchProcessor:
    """Drop-in for voxcpm.training.BatchProcessor over precomputed latents."""

    def __init__(self, *, config, audio_vae, dataset_cnt: int, device):
        self.device = device
        self.packer = make_latent_packer(config, audio_vae)
        self.packer.dataset_cnt = max(dataset_cnt, 1)

    def __call__(self, batch):
        return self.packer(
            audio_tokens=batch["audio_tokens"].to(self.device),
            text_tokens=batch["text_tokens"].to(self.device),
            task_ids=batch["task_ids"].to(self.device),
            dataset_ids=batch["dataset_ids"].to(self.device),
            is_prompts=batch["is_prompts"],
        )


def build_dataloader_latents(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    torch_dataset = LatentVoxCPMDataset(hf_dataset)
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=LatentVoxCPMDataset.collate_fn,
        drop_last=drop_last,
    )
