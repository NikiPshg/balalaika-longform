"""Shared F5 loading, unpadded data, and reproducibility helpers.

The only architecture adaptation is extension of the nonpersistent sinusoidal
text-position buffer. The extension is identical for all experimental arms.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.utils.checkpoint

SAMPLE_RATE = 24000
HOP_LENGTH = 256
MEL_CHANNELS = 100
DEFAULT_UPSTREAM = os.environ.get('F5_ROOT', 'third_party/F5-TTS')
CFM_KWARGS = dict(sigma=0.0, audio_drop_prob=0.3, cond_drop_prob=0.2,
                  frac_lengths_mask=[0.7, 1.0])
EMA_PACKAGE_VERSION = "0.8.3"
# Exact defaults used by the pinned upstream Trainer's
# EMA(model, include_online_model=False, **{}). Freeze them explicitly.
EMA_KWARGS = dict(beta=0.9999, update_after_step=100, update_every=10,
                  inv_gamma=1.0, power=2 / 3, min_value=0.0,
                  include_online_model=False, allow_different_devices=False,
                  use_foreach=False, update_model_with_ema_every=None,
                  update_model_with_ema_beta=0.0, move_ema_to_online_device=False,
                  coerce_dtype=False, lazy_init_ema=False)


def create_ema(model):
    """Match pinned upstream EMA from the initial weights, before any updates."""
    from ema_pytorch import EMA
    installed = version("ema-pytorch")
    if installed != EMA_PACKAGE_VERSION:
        raise RuntimeError(f"Expected ema-pytorch {EMA_PACKAGE_VERSION}, found {installed}")
    return EMA(model, **EMA_KWARGS).to(next(model.parameters()).device)


def ema_to_model_state(ema_state: dict) -> dict:
    """Convert upstream EMA state to strict CFM weights for an inference export.

The full resumable EMA state retains `step`, `initted`, and `ema_model.*` keys.
Only this exported model-weight dictionary strips the prefix/counter fields.
"""
    state = {}
    for name, value in ema_state.items():
        if name in {"initted", "step"}:
            continue
        if not name.startswith("ema_model."):
            raise ValueError(f"Unexpected EMA checkpoint key: {name}")
        state[name.removeprefix("ema_model.")] = value
    if not state:
        raise ValueError("Empty EMA model state")
    return state


def checkpoint_ema_history(path: str | Path) -> dict:
    """Read source history for provenance without importing it into a new SFT EMA."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            return dict(step=int(handle.get_tensor("step")) if "step" in keys else None,
                        initted=bool(handle.get_tensor("initted")) if "initted" in keys else None)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema_model_state_dict", {})
    return dict(step=int(state["step"]) if "step" in state else None,
                initted=bool(state["initted"]) if "initted" in state else None)


def add_upstream(path: str | Path = DEFAULT_UPSTREAM) -> None:
    source = str(Path(path).resolve() / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_cuda() -> torch.device:
    """Require exactly one authorized physical GPU, mapped to logical cuda:0."""
    # Use the first device exposed by the caller; no lab-specific GPU assignment.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # Never silently fall back to an O(sequence_length**2) attention tensor.
    torch.backends.cuda.enable_math_sdp(False)
    return torch.device("cuda:0")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def load_model(checkpoint: str | Path, vocab: str | Path, *,
               upstream: str | Path = DEFAULT_UPSTREAM,
               max_duration_seconds: float = 930.0,
               checkpoint_activations: bool = True,
               device: str | torch.device = "cpu"):
    """Load every Russian EMA weight strictly, or a locally trained state dict."""
    add_upstream(upstream)
    from f5_tts.model import CFM, DiT
    from f5_tts.model.modules import precompute_freqs_cis
    from f5_tts.model.utils import get_tokenizer

    vocab_map, vocab_size = get_tokenizer(str(vocab), "custom")
    backbone = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                   conv_layers=4, text_num_embeds=vocab_size, mel_dim=MEL_CHANNELS,
                   text_mask_padding=True, attn_backend="torch",
                   attn_mask_enabled=False,
                   checkpoint_activations=checkpoint_activations)
    model = CFM(transformer=backbone, vocab_char_map=vocab_map,
                **CFM_KWARGS,
                mel_spec_kwargs=dict(n_fft=1024, hop_length=HOP_LENGTH,
                                     win_length=1024, n_mel_channels=MEL_CHANNELS,
                                     target_sample_rate=SAMPLE_RATE,
                                     mel_spec_type="vocos"),
                odeint_kwargs={"method": "euler"})
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file
        raw = load_file(str(checkpoint), device="cpu")
        state = ema_to_model_state(raw)
    else:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Full training checkpoints keep online weights separately for resume.
        # Synthesis and training initialization from a full checkpoint use EMA,
        # matching the public F5 inference convention.
        state = (ema_to_model_state(raw["ema_model_state_dict"])
                 if "ema_model_state_dict" in raw else raw["model_state_dict"])
    model.load_state_dict(state, strict=True)
    del raw, state
    positions = max(8192, math.ceil(max_duration_seconds * SAMPLE_RATE / HOP_LENGTH) + 1)
    embedding = model.transformer.text_embed
    embedding.freqs_cis = precompute_freqs_cis(512, positions)
    embedding.precompute_max_pos = positions
    # All optimizer/master parameters remain float32; autocast is used externally.
    model = model.float().to(device)
    return model


def read_manifest(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        for field in ("utt", "text", "audio", "duration", "parent"):
            if field not in row:
                raise ValueError(f"Manifest row missing {field}: {row.get('utt')}")
        if row["utt"] in seen:
            raise ValueError(f"Duplicate utterance: {row['utt']}")
        seen.add(row["utt"])
        if not row["text"].strip() or not math.isfinite(row["duration"]) or row["duration"] <= 0:
            raise ValueError(f"Invalid text/duration: {row['utt']}")
    return rows


def parent_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["parent"], []).append(row)
    return {parent: sorted(group, key=lambda row: (row.get("offset_start", 0), row["utt"]))
            for parent, group in sorted(groups.items())}


def load_audio(row: dict) -> torch.Tensor:
    """Read actual source offsets; resampling does not change the source ledger."""
    import soundfile as sf
    import torchaudio

    with sf.SoundFile(row["audio"]) as stream:
        rate = stream.samplerate
        start = round(float(row.get("offset_start", 0)) * rate)
        end_value = row.get("offset_end")
        end = round(float(end_value) * rate) if end_value is not None else len(stream)
        # Corpus offsets are rounded to centiseconds. A requested end at physical
        # EOF may exceed it by <= half of that precision; accept only that bound.
        overrun = max(0, end - len(stream))
        if overrun and overrun <= round(0.005 * rate):
            end = len(stream)
        if not 0 <= start < end <= len(stream):
            raise ValueError(f"Invalid audio offsets for {row.get('utt')}: {start}:{end}/{len(stream)}")
        row["_source_samples"] = end - start
        row["_source_sample_rate"] = rate
        row["_actual_audio_seconds"] = (end - start) / rate
        row["_metadata_end_overrun_samples"] = overrun
        stream.seek(start)
        audio = stream.read(end - start, dtype="float32", always_2d=True).mean(axis=1)
    waveform = torch.from_numpy(audio)
    if rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)
    row["_resampled_samples"] = waveform.numel()
    if not torch.isfinite(waveform).all() or waveform.numel() < 1024:
        raise ValueError(f"Invalid audio for {row.get('utt')}")
    return waveform


def mel_for_row(model, row: dict, device: str | torch.device) -> torch.Tensor:
    """Return unpadded [1, frames, 100] float32 mel; cache format is [frames, 100]."""
    if row.get("mel_cache"):
        # Keep exact source-sample exposure even when only a cached mel is loaded.
        import soundfile as sf
        info = sf.info(row["audio"])
        start = round(float(row.get("offset_start", 0)) * info.samplerate)
        requested_end = round(float(row["offset_end"]) * info.samplerate) if row.get("offset_end") is not None else info.frames
        overrun = max(0, requested_end - info.frames)
        if overrun > round(0.005 * info.samplerate):
            raise ValueError(f"Cached mel source has invalid audio offsets: {row['utt']}")
        end = min(requested_end, info.frames)
        if not 0 <= start < end:
            raise ValueError(f"Cached mel source has invalid audio offsets: {row['utt']}")
        row.update(_source_samples=end-start, _source_sample_rate=info.samplerate,
                   _actual_audio_seconds=(end-start)/info.samplerate,
                   _metadata_end_overrun_samples=overrun,
                   _resampled_samples=math.ceil((end-start)*SAMPLE_RATE/info.samplerate))
        path = Path(row["mel_cache"])
        if path.suffix == ".npy":
            mel = torch.from_numpy(np.load(path, allow_pickle=False))
        else:
            mel = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(mel, dict):
            mel = mel["mel"]
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        mel = mel.float().to(device)
    else:
        waveform = load_audio(row).to(device)
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            mel = model.mel_spec(waveform.unsqueeze(0)).transpose(1, 2).contiguous()
    if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[-1] != MEL_CHANNELS:
        raise ValueError(f"Expected [1, frames, 100] mel for {row['utt']}, got {tuple(mel.shape)}")
    expected_frames = row["_resampled_samples"] // HOP_LENGTH + 1
    if mel.shape[1] != expected_frames:
        raise ValueError(f"Mel/audio length mismatch for {row['utt']}: {mel.shape[1]} != {expected_frames}")
    if not torch.isfinite(mel).all():
        raise ValueError(f"Nonfinite mel for {row['utt']}")
    return mel


def capture_rng() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def estimated_duration_frames(prompt_samples: int, reference_text: str,
                              target_text: str, speed: float = 1.0) -> int:
    """Upstream UTF-8 length estimator, without reference target audio duration."""
    if speed <= 0 or not reference_text.strip():
        raise ValueError("Positive speed and nonempty reference transcript are required")
    prompt_frames = prompt_samples // HOP_LENGTH
    local_speed = 0.3 if len(target_text.encode("utf-8")) < 10 else speed
    return prompt_frames + int(prompt_frames / len(reference_text.encode("utf-8"))
                               * len(target_text.encode("utf-8")) / local_speed)
