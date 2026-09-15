# coding=utf-8
"""Streaming SOVA/Balalaika filtering and full-utterance feature encoding.

The source dataset is never materialized locally. ``datasets`` always runs in
streaming mode and only accepted WAV bytes are decoded, stressed and encoded.
No word timestamps, speaker IDs, or ref_text/target_text split are used.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_finetuning.full_utterance_utils import (  # noqa: E402
    GIGAAM_E2E_FIELD,
    apply_silero_stress,
    asr_consensus,
    build_assistant_text,
    deterministic_prefix_frames,
    dotenv_value,
    select_training_transcript,
    str2bool,
)


TARGET_SR = 24_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full-WAV/main-Talker LoRA rows from a streaming HF WebDataset."
    )
    parser.add_argument("--dataset_name", default="lab260/sova_balalaika")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument(
        "--dataset_revision",
        default="be89f9bbc9908afd34b28e05441bbb9f655c0183",
    )
    parser.add_argument("--streaming", type=str2bool, default=True)
    parser.add_argument("--hf_token_env", default="HF_TOKEN")
    parser.add_argument("--env_file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--model_path", default="bitmanagerai/Qwen3TTS-RL-2")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--stats_json", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--validation_samples", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--shuffle_buffer", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--min_duration_sec", type=float, default=5.0)
    parser.add_argument("--max_duration_sec", type=float, default=20.0)
    parser.add_argument("--min_prefix_sec", type=float, default=3.0)
    parser.add_argument("--max_prefix_sec", type=float, default=8.0)
    parser.add_argument("--min_continuation_sec", type=float, default=2.0)
    parser.add_argument("--codec_fps", type=float, default=12.0)
    parser.add_argument("--min_distill_mos", type=float, default=3.5)
    parser.add_argument("--max_music_prob", type=float, default=0.30)
    parser.add_argument("--max_asr_wer", type=float, default=0.25)
    parser.add_argument("--max_e2e_rover_wer", type=float, default=0.0)
    parser.add_argument("--require_single_speaker", type=str2bool, default=True)
    parser.add_argument("--language", default="Russian")
    parser.add_argument("--min_words", type=int, default=4)
    parser.add_argument("--max_text_chars", type=int, default=800)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[prepare-full] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def torch_dtype(torch: Any, name: str) -> Any:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def tokenize(processor: Any, text: str) -> list[int]:
    encoded = processor(text=text, return_tensors="pt", padding=False)["input_ids"]
    if encoded.ndim == 2:
        encoded = encoded[0]
    return encoded.cpu().tolist()


def decode_wav_bytes(wav_bytes: bytes, librosa: Any, soundfile: Any) -> tuple[np.ndarray, float]:
    wav, sample_rate = soundfile.read(
        io.BytesIO(wav_bytes),
        dtype="float32",
        always_2d=True,
    )
    wav = np.asarray(wav, dtype=np.float32).mean(axis=1)
    if int(sample_rate) != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=int(sample_rate), target_sr=TARGET_SR).astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32, copy=False)
    return wav, float(len(wav) / TARGET_SR)


def extract_speaker_embeddings_batch(model: Any, wavs: Sequence[np.ndarray], torch: Any) -> list[Any]:
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    max_len = max(len(wav) for wav in wavs)
    padded = np.zeros((len(wavs), max_len), dtype=np.float32)
    for index, wav in enumerate(wavs):
        padded[index, : len(wav)] = wav
    mels = mel_spectrogram(
        torch.from_numpy(padded),
        n_fft=1024,
        num_mels=128,
        sampling_rate=TARGET_SR,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    embeddings = model.speaker_encoder(mels.to(model.device, dtype=model.dtype))
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.shape[0] != len(wavs):
        raise RuntimeError(f"speaker embedding batch mismatch: {embeddings.shape[0]} != {len(wavs)}")
    return [embeddings[index] for index in range(embeddings.shape[0])]


def streaming_dataset(args: argparse.Namespace, token: str | None) -> Iterable[dict[str, Any]]:
    from datasets import Value, load_dataset

    if not args.streaming:
        raise ValueError("This pipeline intentionally requires --streaming true")
    load_kwargs = {
        "path": args.dataset_name,
        "split": args.dataset_split,
        "revision": args.dataset_revision,
        "streaming": True,
        "token": token,
    }
    dataset = load_dataset(**load_kwargs)
    if "wav" not in dataset.features:
        raise ValueError(f"Dataset {args.dataset_name} has no 'wav' feature: {dataset.features}")

    # datasets>=5 routes Audio through TorchCodec even when the source already
    # contains bytes. Override the feature as binary to keep true WebDataset
    # streaming and decode accepted rows ourselves with soundfile.
    dataset = dataset.cast_column("wav", Value("binary"))
    if args.shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return dataset


def candidate_from_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    accentor: Any,
    librosa: Any,
    soundfile: Any,
    stats: Counter[str],
) -> dict[str, Any] | None:
    stats["seen"] += 1
    metadata = row.get("json") or {}
    duration = float(metadata.get("total_duration") or 0.0)
    if not args.min_duration_sec <= duration <= args.max_duration_sec:
        stats["skip_duration"] += 1
        return None
    if args.require_single_speaker and not bool(metadata.get("is_single_speaker")):
        stats["skip_multispeaker"] += 1
        return None
    if float(metadata.get("DistillMOS") or 0.0) < args.min_distill_mos:
        stats["skip_mos"] += 1
        return None
    music_prob = metadata.get("music_prob")
    if float(1.0 if music_prob is None else music_prob) > args.max_music_prob:
        stats["skip_music"] += 1
        return None
    consensus_ok, consensus_scores = asr_consensus(metadata, max_wer=args.max_asr_wer)
    if not consensus_ok:
        stats["skip_asr_consensus"] += 1
        return None

    source_text, transcript_source, e2e_rover_wer = select_training_transcript(
        metadata, max_wer=args.max_e2e_rover_wer
    )
    if not source_text:
        stats["skip_missing_transcript"] += 1
        return None
    if not args.min_words <= len(source_text.split()) or len(source_text) > args.max_text_chars:
        stats["skip_text_length"] += 1
        return None
    try:
        stressed_text = apply_silero_stress(accentor, source_text)
    except Exception as exc:
        stats["skip_stress_error"] += 1
        log(f"Silero Stress failed for key={row.get('__key__')}: {exc}")
        return None
    wav_bytes = row.get("wav")
    if not isinstance(wav_bytes, bytes) or not wav_bytes:
        stats["skip_missing_wav"] += 1
        return None
    try:
        wav, decoded_duration = decode_wav_bytes(wav_bytes, librosa, soundfile)
    except Exception as exc:
        stats["skip_audio_decode"] += 1
        log(f"audio decode failed for key={row.get('__key__')}: {exc}")
        return None
    if not args.min_duration_sec <= decoded_duration <= args.max_duration_sec:
        stats["skip_decoded_duration"] += 1
        return None

    source_url = str(row.get("__url__") or "")
    source_key = str(row.get("__key__") or "")
    stable_key = f"{source_url}#{source_key}"
    if transcript_source == GIGAAM_E2E_FIELD:
        stats["transcript_gigaam_e2e"] += 1
    else:
        stats["transcript_punct_fallback"] += 1
    return {
        "key": stable_key,
        "source_key": source_key,
        "source_url": source_url,
        "wav": wav,
        "duration": decoded_duration,
        "text": stressed_text,
        "source_text": source_text,
        "transcript_source": transcript_source,
        "gigaam_e2e_rover_wer": e2e_rover_wer,
        "metadata_accent_text": metadata.get("accent.txt"),
        "asr_consensus_wer": consensus_scores,
        "quality": {
            "distill_mos": float(metadata.get("DistillMOS") or 0.0),
            "music_prob": float(metadata.get("music_prob") or 0.0),
            "silence_percent": float(metadata.get("silence_percent") or 0.0),
            "score_bonafide": float(metadata.get("score_bonafide") or 0.0),
        },
    }


def prepared_rows(
    batch: list[dict[str, Any]],
    *,
    qwen3tts: Any,
    args: argparse.Namespace,
    torch: Any,
    serialize: bool = True,
    output_device: Any | None = None,
) -> list[dict[str, Any]]:
    wavs = [item["wav"] for item in batch]
    # Use no_grad rather than inference_mode: the online training path keeps
    # these tensors on GPU and later mixes the frozen speaker embedding into a
    # trainable graph. The tokenizer itself still uses inference_mode, so its
    # codec IDs are cloned below to turn them back into regular tensors.
    with torch.no_grad():
        codes_batch = qwen3tts.model.speech_tokenizer.encode(wavs, sr=TARGET_SR).audio_codes
        speaker_embeddings = extract_speaker_embeddings_batch(qwen3tts.model, wavs, torch)
    if not len(codes_batch) == len(speaker_embeddings) == len(batch):
        raise RuntimeError(
            f"encoder batch mismatch: rows={len(batch)}, codes={len(codes_batch)}, "
            f"speakers={len(speaker_embeddings)}"
        )

    rows: list[dict[str, Any]] = []
    min_prefix_frames = round(args.min_prefix_sec * args.codec_fps)
    max_prefix_frames = round(args.max_prefix_sec * args.codec_fps)
    min_continuation_frames = round(args.min_continuation_sec * args.codec_fps)
    for item, codes, speaker_embedding in zip(batch, codes_batch, speaker_embeddings, strict=True):
        tensor_device = "cpu" if serialize or output_device is None else output_device
        # Qwen3TTSTokenizer.encode() returns inference tensors. A same-device
        # ``to(dtype=...)`` can be a no-op, and trainable embeddings cannot save
        # inference indices for backward. clone() outside inference_mode makes
        # the IDs safe while keeping codec extraction fully frozen.
        codes = codes.detach().to(tensor_device, dtype=torch.long).clone()
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        if codes.ndim != 2 or int(codes.shape[1]) != 16:
            raise ValueError(f"full_codes must be [T,16], got {tuple(codes.shape)} for {item['key']}")
        prefix_frames = deterministic_prefix_frames(
            key=item["key"],
            total_frames=int(codes.shape[0]),
            seed=args.seed,
            min_prefix_frames=min_prefix_frames,
            max_prefix_frames=max_prefix_frames,
            min_continuation_frames=min_continuation_frames,
        )
        text_ids = tokenize(qwen3tts.processor, build_assistant_text(item["text"]))
        text_ids_tensor = torch.tensor(
            text_ids, dtype=torch.long, device=tensor_device
        )
        speaker_embedding = speaker_embedding.detach().to(
            tensor_device, dtype=torch.float32
        )
        rows.append(
            {
                "full_codes": codes.tolist() if serialize else codes,
                "speaker_embedding": (
                    speaker_embedding.cpu().tolist()
                    if serialize
                    else speaker_embedding
                ),
                "text_ids": text_ids if serialize else text_ids_tensor,
                "text": item["text"],
                "source_text": item["source_text"],
                "transcript_source": item["transcript_source"],
                "gigaam_e2e_rover_wer": item["gigaam_e2e_rover_wer"],
                "metadata_accent_text": item["metadata_accent_text"],
                "prefix_frames": prefix_frames,
                "continuation_frames": int(codes.shape[0]) - prefix_frames,
                "duration": item["duration"],
                "language": args.language,
                "dataset": args.dataset_name,
                "dataset_revision": args.dataset_revision,
                "source_key": item["source_key"],
                "source_url": item["source_url"],
                "asr_consensus_wer": item["asr_consensus_wer"],
                "quality": item["quality"],
            }
        )
    return rows


def write_batch(
    rows: Sequence[dict[str, Any]],
    *,
    train_file: Any,
    val_file: Any,
    validation_samples: int,
    stats: Counter[str],
) -> None:
    for row in rows:
        if stats["val_written"] < validation_samples:
            output = val_file
            stats["val_written"] += 1
        else:
            output = train_file
            stats["train_written"] += 1
        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stats["accepted"] += 1


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= args.validation_samples:
        raise ValueError("max_samples must be larger than validation_samples")
    token = os.environ.get(args.hf_token_env) or dotenv_value(args.env_file, args.hf_token_env)

    log("importing runtime dependencies")
    import librosa
    import soundfile
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from silero_stress import load_accentor

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    accentor = load_accentor()
    log("Silero Stress ready on CPU")
    log(f"loading Qwen3-TTS tokenizer/model from {args.model_path}")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype(torch, args.dtype),
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        token=token,
    )
    qwen3tts.model.eval()
    log("model ready; opening streaming dataset")
    dataset = streaming_dataset(args, token)

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_json) if args.stats_json else train_path.with_suffix(".stats.json")
    stats = Counter()
    batch: list[dict[str, Any]] = []
    started = time.monotonic()

    with train_path.open("w", encoding="utf-8") as train_file, val_path.open("w", encoding="utf-8") as val_file:
        for source_row in dataset:
            candidate = candidate_from_row(source_row, args, accentor, librosa, soundfile, stats)
            if candidate is not None:
                batch.append(candidate)
            if len(batch) >= args.batch_size:
                rows = prepared_rows(batch, qwen3tts=qwen3tts, args=args, torch=torch)
                remaining = None if args.max_samples is None else args.max_samples - stats["accepted"]
                if remaining is not None:
                    rows = rows[:remaining]
                write_batch(
                    rows,
                    train_file=train_file,
                    val_file=val_file,
                    validation_samples=args.validation_samples,
                    stats=stats,
                )
                batch.clear()
                train_file.flush()
                val_file.flush()
                if stats["accepted"] % max(1, args.log_every) < args.batch_size:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    log(
                        f"accepted={stats['accepted']} train={stats['train_written']} "
                        f"val={stats['val_written']} seen={stats['seen']} "
                        f"rate={stats['accepted'] / elapsed:.2f} rows/s"
                    )
                if args.max_samples is not None and stats["accepted"] >= args.max_samples:
                    break
        if batch and (args.max_samples is None or stats["accepted"] < args.max_samples):
            remaining = None if args.max_samples is None else args.max_samples - stats["accepted"]
            rows = prepared_rows(batch, qwen3tts=qwen3tts, args=args, torch=torch)
            if remaining is not None:
                rows = rows[:remaining]
            write_batch(
                rows,
                train_file=train_file,
                val_file=val_file,
                validation_samples=args.validation_samples,
                stats=stats,
            )

    stats_payload = {
        **dict(stats),
        "elapsed_sec": time.monotonic() - started,
        "args": {key: value for key, value in vars(args).items() if key != "hf_token"},
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if stats["val_written"] < args.validation_samples:
        raise RuntimeError(
            f"Only {stats['val_written']} validation rows were prepared; requested {args.validation_samples}"
        )
    if stats["train_written"] == 0:
        raise RuntimeError("No training rows were prepared")
    log(
        f"done: train={stats['train_written']} val={stats['val_written']} "
        f"stats={stats_path}"
    )


if __name__ == "__main__":
    main()
