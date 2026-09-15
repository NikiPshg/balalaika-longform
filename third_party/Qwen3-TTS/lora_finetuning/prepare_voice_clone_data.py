# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TARGET_SR = 24000
PRIMARY_BOUNDARY = ".!?"
FALLBACK_BOUNDARY = ",;:"
TOKEN_RE = re.compile(r"[\wЁёА-Яа-я]+|[^\w\s]", re.UNICODE)
WORD_TOKEN_RE = re.compile(r"[\wЁёА-Яа-я]", re.UNICODE)
pd = None
librosa = None
sf = None
torch = None
Qwen3TTSModel = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Qwen3-TTS voice-clone JSONL from aligned utterances.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--input_format",
        type=str,
        choices=("auto", "aligned", "rusynth_packaged"),
        default="auto",
        help="Input layout. auto detects RuSynth packaged manifests or the original aligned layout.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="RuSynth packaged parquet manifest path. If omitted, all manifests in data_dir/manifests are processed.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional dataset label to include in prepared rows.",
    )
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--validation_jsonl", type=str, default=None)
    parser.add_argument("--validation_samples", type=int, default=0)
    parser.add_argument(
        "--validation_strategy",
        type=str,
        choices=("duration", "last5_random5"),
        default="duration",
        help="Validation selection. last5_random5 is intended for RuSynth packaged manifests.",
    )
    parser.add_argument("--validation_min_target_sec", type=float, default=0.0)
    parser.add_argument("--validation_target_sec", type=float, default=10.0)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--audio_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scan_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scan_chunk_size", type=int, default=5000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_ref_sec", type=float, default=3.0)
    parser.add_argument("--min_target_sec", type=float, default=2.0)
    parser.add_argument("--language", type=str, default="Auto")
    parser.add_argument(
        "--demonstration_dir",
        type=str,
        default=None,
        help="Optional folder for prepared artifacts, original audios, and decoded audio-code wavs.",
    )
    parser.add_argument(
        "--demo_samples",
        type=int,
        default=10,
        help="Number of prepared rows to copy/decode into demonstration_dir.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def import_runtime_deps() -> None:
    global librosa, sf, torch, Qwen3TTSModel
    try:
        import librosa as librosa_mod
        import soundfile as sf_mod
        import torch as torch_mod
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel as model_cls
    except ImportError as exc:
        raise SystemExit(
            "This script requires the Qwen3-TTS runtime audio/model dependencies. "
            "Install the project requirements before preparing data. "
            f"Missing import: {exc.name}"
        ) from exc
    librosa = librosa_mod
    sf = sf_mod
    torch = torch_mod
    Qwen3TTSModel = model_cls


def import_parquet_deps() -> None:
    global pd
    try:
        import pandas as pd_mod
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This script requires pandas and pyarrow to read parquet files. "
            "Install them with: pip install pandas pyarrow"
        ) from exc
    pd = pd_mod


def pick_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    existing = set(columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def normalize_relative_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def normalize_training_text(value: Any) -> str:
    return str(value).replace("+", "").replace("ё", "е").replace("Ё", "Е").strip()


def words_and_boundaries_from_punct(text: str) -> Tuple[List[str], List[Tuple[str, int, int]], List[str], List[int]]:
    words: List[str] = []
    boundaries: List[Tuple[str, int, int]] = []
    tokens = TOKEN_RE.findall(str(text).strip())
    word_token_indices: List[int] = []
    for token_index, token in enumerate(tokens):
        if WORD_TOKEN_RE.search(token):
            words.append(token)
            word_token_indices.append(token_index)
        elif token in PRIMARY_BOUNDARY + FALLBACK_BOUNDARY and words:
            boundaries.append((token, len(words) - 1, token_index))
    return words, boundaries, tokens, word_token_indices


def render_tokens(tokens: Sequence[str]) -> str:
    out = ""
    no_space_before = set(PRIMARY_BOUNDARY + FALLBACK_BOUNDARY + ")]}»")
    no_space_after = set("([{«")
    for token in tokens:
        if not out:
            out = token
        elif token in no_space_before or out[-1] in no_space_after:
            out += token
        else:
            out += " " + token
    return out.strip()


def timestamp_to_pair(item: Any) -> Optional[Tuple[float, float]]:
    if isinstance(item, dict):
        start = item.get("start", item.get("start_time", item.get("begin", item.get("s"))))
        end = item.get("end", item.get("end_time", item.get("finish", item.get("e"))))
    elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
        start, end = item[0], item[1]
    else:
        return None
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None
    if end_f <= start_f:
        return None
    return start_f, end_f


def parse_timestamps(value: Any) -> Optional[List[Tuple[float, float]]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        return None

    pairs = [timestamp_to_pair(item) for item in value]
    if any(pair is None for pair in pairs):
        return None
    return pairs


def row_timestamps(row: Any, ts_col: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    if ts_col is not None:
        return parse_timestamps(row[ts_col])

    starts_col = pick_column(row.index, ("start", "starts", "start_time", "word_start", "word_starts"))
    ends_col = pick_column(row.index, ("end", "ends", "end_time", "word_end", "word_ends"))
    if starts_col and ends_col:
        starts = row[starts_col]
        ends = row[ends_col]
        if isinstance(starts, np.ndarray):
            starts = starts.tolist()
        if isinstance(ends, np.ndarray):
            ends = ends.tolist()
        if isinstance(starts, (list, tuple)) and isinstance(ends, (list, tuple)) and len(starts) == len(ends):
            try:
                pairs = [(float(start), float(end)) for start, end in zip(starts, ends)]
            except (TypeError, ValueError):
                return None
            return pairs if all(end > start for start, end in pairs) else None
    return None


def choose_boundary(
    boundaries: Sequence[Tuple[str, int, int]],
    timestamps: Sequence[Tuple[float, float]],
    min_ref_sec: float,
    min_target_sec: float,
    target_end_sec: Optional[float] = None,
) -> Optional[int]:
    target_end_sec = timestamps[-1][1] if target_end_sec is None else float(target_end_sec)

    def eligible(indices: Iterable[int]) -> List[int]:
        out = []
        for idx in indices:
            if idx + 1 >= len(timestamps):
                continue
            ref_dur = timestamps[idx][1]
            target_dur = target_end_sec - timestamps[idx][1]
            if ref_dur >= min_ref_sec and target_dur >= min_target_sec:
                out.append(idx)
        return out

    primary = eligible(idx for mark, idx, _token_idx in boundaries if mark in PRIMARY_BOUNDARY)
    if primary:
        return primary[0]

    fallback = eligible(idx for mark, idx, _token_idx in boundaries if mark in FALLBACK_BOUNDARY)
    if fallback:
        return min(fallback, key=lambda i: abs(timestamps[i][1] - min_ref_sec))
    return None


def load_crop_resample(path: str, start_sec: float, end_sec: float) -> Tuple[np.ndarray, float]:
    info = sf.info(path)
    sr = int(info.samplerate)
    audio_duration = float(info.frames) / float(sr)
    start_sec = max(0.0, min(float(start_sec), audio_duration))
    end_sec = max(start_sec, min(float(end_sec), audio_duration))
    start_frame = int(round(start_sec * sr))
    stop_frame = int(round(end_sec * sr))
    frames = max(0, stop_frame - start_frame)
    wav, native_sr = sf.read(path, start=start_frame, frames=frames, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if int(native_sr) != TARGET_SR:
        wav = librosa.resample(y=wav, orig_sr=int(native_sr), target_sr=TARGET_SR).astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    return wav, audio_duration


def load_full_resample(path: str) -> Tuple[np.ndarray, float]:
    info = sf.info(path)
    duration = float(info.frames) / float(info.samplerate)
    return load_crop_resample(path, 0.0, duration)


def prepare_audio_item(item: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, float]:
    if item.get("input_format") == "rusynth_packaged":
        target_wav, audio_duration = load_full_resample(item["audio_path"])
        ref_wav, _ = load_full_resample(item["ref_audio_path"])
    else:
        target_wav, audio_duration = load_crop_resample(item["audio_path"], item["split_start"], item["split_end"])
        ref_wav, _ = load_crop_resample(item["audio_path"], item["ref_split_start"], item["ref_split_end"])
    if len(target_wav) == 0 or len(ref_wav) == 0:
        raise ValueError("empty crop")
    return item, target_wav, ref_wav, audio_duration


def extract_speaker_embeddings_batch(model: Any, ref_wavs: Sequence[np.ndarray]) -> List[Any]:
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    max_len = max(len(wav) for wav in ref_wavs)
    padded = np.zeros((len(ref_wavs), max_len), dtype=np.float32)
    for idx, wav in enumerate(ref_wavs):
        padded[idx, :len(wav)] = wav

    wav_tensor = torch.from_numpy(padded)
    mels = mel_spectrogram(
        wav_tensor,
        n_fft=1024,
        num_mels=128,
        sampling_rate=TARGET_SR,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    speaker_embeddings = model.speaker_encoder(mels.to(model.device).to(model.dtype))
    if speaker_embeddings.dim() == 1:
        speaker_embeddings = speaker_embeddings.unsqueeze(0)
    if speaker_embeddings.shape[0] != len(ref_wavs):
        raise ValueError(
            f"speaker embedding batch mismatch: got {speaker_embeddings.shape[0]}, expected {len(ref_wavs)}"
        )
    return [speaker_embeddings[idx] for idx in range(speaker_embeddings.shape[0])]


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_ref_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def tokenize(processor: Any, text: str) -> List[int]:
    inputs = processor(text=text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    return input_ids[0].cpu().tolist()


def chunks(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def iter_chunks(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def tensor_to_list(tensor: Any) -> List[Any]:
    return tensor.detach().cpu().tolist()


def tensor_from_codes(value: Any) -> Any:
    return torch.as_tensor(value, dtype=torch.long)


def copy_demo_audio(src: str, dest_dir: Path, name: str) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(src).suffix
    dest = dest_dir / f"{name}{suffix}"
    shutil.copy2(src, dest)
    return str(dest)


def write_demo_decoded_audio(model: Any, codes: Any, dest: Path) -> None:
    wavs, sr = model.speech_tokenizer.decode([{"audio_codes": tensor_from_codes(codes).to(model.device)}])
    wav = wavs[0]
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    sf.write(str(dest), wav, int(sr))


def item_duration_stats(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    if not items:
        return {
            "count": 0.0,
            "hours": 0.0,
            "mean_sec": 0.0,
            "min_sec": 0.0,
            "max_sec": 0.0,
        }
    values = [float(item[key]) for item in items]
    return {
        "count": float(len(values)),
        "hours": sum(values) / 3600.0,
        "mean_sec": sum(values) / float(len(values)),
        "min_sec": min(values),
        "max_sec": max(values),
    }


def store_split_stats(stats: Dict[str, float], prefix: str, items: Sequence[Dict[str, Any]]) -> None:
    target = item_duration_stats(items, "duration")
    ref = item_duration_stats(items, "ref_duration")
    stats[f"{prefix}_examples"] = target["count"]
    stats[f"{prefix}_target_hours"] = target["hours"]
    stats[f"{prefix}_target_mean_sec"] = target["mean_sec"]
    stats[f"{prefix}_target_min_sec"] = target["min_sec"]
    stats[f"{prefix}_target_max_sec"] = target["max_sec"]
    stats[f"{prefix}_ref_hours"] = ref["hours"]
    stats[f"{prefix}_ref_mean_sec"] = ref["mean_sec"]
    stats[f"{prefix}_ref_min_sec"] = ref["min_sec"]
    stats[f"{prefix}_ref_max_sec"] = ref["max_sec"]


def scan_candidate_chunk(payload: Tuple[List[Tuple[Any, ...]], Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows, cfg = payload
    stats = {
        "rows_scanned": 0.0,
        "usable": 0.0,
        "skipped": 0.0,
        "source_hours": 0.0,
        "prepared_target_hours": 0.0,
        "prepared_ref_hours": 0.0,
        "unprepared_hours": 0.0,
    }
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        stats["rows_scanned"] += 1
        relative_path = normalize_relative_path(row[0])
        punct_text = str(row[1]).strip()
        timestamps = parse_timestamps(row[2])
        if timestamps is None:
            stats["skipped"] += 1
            continue

        words, boundaries, tokens, word_token_indices = words_and_boundaries_from_punct(punct_text)
        if not words or len(words) != len(timestamps):
            stats["skipped"] += 1
            continue

        stats["usable"] += 1
        try:
            source_dur = max(float(row[3]), timestamps[-1][1])
        except (TypeError, ValueError):
            source_dur = timestamps[-1][1]
        stats["source_hours"] += source_dur / 3600.0

        boundary = choose_boundary(
            boundaries,
            timestamps,
            cfg["min_ref_sec"],
            cfg["min_target_sec"],
            target_end_sec=source_dur,
        )
        if boundary is None:
            stats["skipped"] += 1
            stats["unprepared_hours"] += source_dur / 3600.0
            continue

        ref_start = 0.0
        ref_end = timestamps[boundary][1]
        target_start = ref_end
        target_end = source_dur
        ref_dur = ref_end - ref_start
        target_dur = target_end - target_start
        if ref_dur < cfg["min_ref_sec"] or target_dur < cfg["min_target_sec"]:
            stats["skipped"] += 1
            stats["unprepared_hours"] += source_dur / 3600.0
            continue

        audio_path = os.path.join(cfg["data_dir"], relative_path)
        if not os.path.exists(audio_path):
            stats["skipped"] += 1
            stats["unprepared_hours"] += source_dur / 3600.0
            continue

        boundary_token_index = word_token_indices[boundary]
        for _mark, word_index, token_index in boundaries:
            if word_index == boundary:
                boundary_token_index = token_index
                break
        ref_text = render_tokens(tokens[:boundary_token_index + 1])
        target_text = render_tokens(tokens[boundary_token_index + 1:])

        candidates.append({
            "relative_path": relative_path,
            "audio_path": audio_path,
            "text": target_text,
            "ref_text": ref_text,
            "split_start": float(target_start),
            "split_end": float(target_end),
            "ref_split_start": float(ref_start),
            "ref_split_end": float(ref_end),
            "duration": float(target_dur),
            "ref_duration": float(ref_dur),
        })

        stats["prepared_target_hours"] += target_dur / 3600.0
        stats["prepared_ref_hours"] += ref_dur / 3600.0
        stats["unprepared_hours"] += max(0.0, source_dur - target_dur - ref_dur) / 3600.0
    return candidates, stats


def read_inputs(data_dir: str) -> Tuple[Any, int]:
    import_parquet_deps()
    records_path = os.path.join(data_dir, "records.parquet")
    timestamps_path = os.path.join(data_dir, "word_timestamps.parquet")
    records = pd.read_parquet(records_path)
    timestamps = pd.read_parquet(timestamps_path)
    rows_read = len(records)
    if "relative_path" not in records.columns or "relative_path" not in timestamps.columns:
        raise SystemExit("Both records.parquet and word_timestamps.parquet must contain a relative_path column.")
    records["relative_path"] = records["relative_path"].map(normalize_relative_path)
    timestamps["relative_path"] = timestamps["relative_path"].map(normalize_relative_path)
    joined = records.merge(timestamps, on="relative_path", how="inner", suffixes=("", "_ts"))
    return joined, rows_read


def detect_input_format(args: argparse.Namespace) -> str:
    if args.input_format != "auto":
        return args.input_format
    if os.path.isdir(os.path.join(args.data_dir, "manifests")):
        manifests = list(Path(args.data_dir, "manifests").glob("*.parquet"))
        if manifests:
            return "rusynth_packaged"
    return "aligned"


def packaged_manifest_paths(args: argparse.Namespace) -> List[Path]:
    if args.manifest_path:
        return [Path(args.manifest_path)]
    manifest_dir = Path(args.data_dir) / "manifests"
    paths = sorted(manifest_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"No parquet manifests found in {manifest_dir}")
    return paths


def packaged_manifest_label(path: Path) -> str:
    name = path.name
    if ".best." in name:
        return "best"
    if ".filtered." in name:
        return "filtered"
    return path.stem


def make_packaged_candidates(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    import_parquet_deps()
    global sf
    if sf is None:
        try:
            import soundfile as sf_mod
        except ImportError as exc:
            raise SystemExit(
                "This script requires soundfile to inspect RuSynth packaged audio durations. "
                "Install it with: pip install soundfile"
            ) from exc
        sf = sf_mod
    root = Path(args.data_dir)
    paths = packaged_manifest_paths(args)
    stats = {
        "rows_read": 0.0,
        "rows_scanned": 0.0,
        "usable": 0.0,
        "skipped": 0.0,
        "source_hours": 0.0,
        "prepared_target_hours": 0.0,
        "prepared_ref_hours": 0.0,
        "unprepared_hours": 0.0,
    }
    candidates: List[Dict[str, Any]] = []
    required = {
        "output_wav_path",
        "generated_text",
        "reference_path",
        "reference_text",
        "reference_duration",
        "generation_status",
    }

    for manifest_path in paths:
        df = pd.read_parquet(manifest_path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise SystemExit(f"{manifest_path} is missing required columns: {', '.join(missing)}")

        dataset_label = args.dataset_name or packaged_manifest_label(manifest_path)
        stats["rows_read"] += float(len(df))
        ok_mask = df["generation_status"].astype(str).str.lower().eq("ok")
        stats["skipped"] += float((~ok_mask).sum())

        for original_index, row in df.loc[ok_mask].iterrows():
            stats["rows_scanned"] += 1
            relative_path = normalize_relative_path(row["output_wav_path"])
            ref_relative_path = normalize_relative_path(row["reference_path"])
            audio_path = root / relative_path
            ref_audio_path = root / ref_relative_path
            if not audio_path.exists() or not ref_audio_path.exists():
                stats["skipped"] += 1
                continue

            try:
                target_info = sf.info(str(audio_path))
                target_duration = float(target_info.frames) / float(target_info.samplerate)
                ref_duration = float(row["reference_duration"])
            except Exception:
                stats["skipped"] += 1
                continue
            if target_duration < args.min_target_sec or ref_duration < args.min_ref_sec:
                stats["skipped"] += 1
                continue

            item = {
                "input_format": "rusynth_packaged",
                "dataset": dataset_label,
                "manifest_path": str(manifest_path),
                "manifest_row_index": int(original_index),
                "target_row_id": int(row["target_row_id"]) if "target_row_id" in row else None,
                "hypothesis_id": int(row["hypothesis_id"]) if "hypothesis_id" in row else None,
                "relative_path": relative_path,
                "audio_path": str(audio_path),
                "ref_relative_path": ref_relative_path,
                "ref_audio_path": str(ref_audio_path),
                "text": normalize_training_text(row["generated_text"]),
                "ref_text": normalize_training_text(row["reference_text"]),
                "split_start": 0.0,
                "split_end": float(target_duration),
                "ref_split_start": 0.0,
                "ref_split_end": float(ref_duration),
                "duration": float(target_duration),
                "ref_duration": float(ref_duration),
                "wer": float(row["wer"]) if "wer" in row else None,
                "cer": float(row["cer"]) if "cer" in row else None,
                "MOS": float(row["MOS"]) if "MOS" in row else None,
                "redimnet2_cos_sim": float(row["redimnet2_cos_sim"]) if "redimnet2_cos_sim" in row else None,
            }
            candidates.append(item)
            stats["usable"] += 1
            stats["source_hours"] += target_duration / 3600.0
            stats["prepared_target_hours"] += target_duration / 3600.0
            stats["prepared_ref_hours"] += ref_duration / 3600.0

            if args.max_samples is not None and len(candidates) >= args.max_samples:
                break
        if args.max_samples is not None and len(candidates) >= args.max_samples:
            break

    store_split_stats(stats, "candidate", candidates)
    return candidates, stats


def make_candidates(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if detect_input_format(args) == "rusynth_packaged":
        return make_packaged_candidates(args)

    joined, rows_read = read_inputs(args.data_dir)
    punct_col = pick_column(joined.columns, ("punct", "punctuated_text", "text", "sentence", "utterance"))
    status_col = pick_column(joined.columns, ("alignment_status", "alignment_status_ts"))
    ts_col = pick_column(joined.columns, ("word_timestamps", "timestamps", "words", "word_times", "segments"))
    duration_col = pick_column(joined.columns, ("duration_sec", "duration", "audio_duration", "audio_duration_sec"))
    if punct_col is None:
        raise SystemExit("Could not find a punctuated text column. Expected one of: punct, punctuated_text, text, sentence, utterance.")
    if status_col is None:
        raise SystemExit("Could not find alignment_status column.")

    stats = {
        "rows_read": float(rows_read),
        "rows_scanned": 0.0,
        "usable": 0.0,
        "skipped": float(max(0, rows_read - len(joined))),
        "source_hours": 0.0,
        "prepared_target_hours": 0.0,
        "prepared_ref_hours": 0.0,
        "unprepared_hours": 0.0,
    }
    candidates: List[Dict[str, Any]] = []

    status_values = joined[status_col].astype(str).str.lower()
    ok_mask = status_values.eq("ok")
    stats["skipped"] += float((~ok_mask).sum())
    if ts_col is None:
        raise SystemExit("Could not find word timestamp column.")
    scan_df = joined.loc[ok_mask, ["relative_path", punct_col, ts_col]].copy()
    scan_df["_duration_for_prep"] = joined.loc[ok_mask, duration_col] if duration_col is not None else np.nan

    cfg = {
        "data_dir": args.data_dir,
        "min_ref_sec": args.min_ref_sec,
        "min_target_sec": args.min_target_sec,
    }
    row_chunks = iter_chunks(scan_df.itertuples(index=False, name=None), max(1, args.scan_chunk_size))
    workers = max(1, min(args.scan_workers, os.cpu_count() or 1))

    try:
        from tqdm import tqdm
        total_chunks = (len(scan_df) + max(1, args.scan_chunk_size) - 1) // max(1, args.scan_chunk_size)
        progress = tqdm(total=len(scan_df), desc=f"Scanning rows ({workers} workers)")
    except ImportError:
        progress = None
        total_chunks = None

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            payloads = ((chunk, cfg) for chunk in row_chunks)
            for chunk_candidates, chunk_stats in executor.map(scan_candidate_chunk, payloads, chunksize=1):
                candidates.extend(chunk_candidates)
                for key, value in chunk_stats.items():
                    stats[key] += value
                if progress is not None:
                    progress.update(int(chunk_stats["rows_scanned"]))
                if args.max_samples is not None and len(candidates) >= args.max_samples:
                    break
    else:
        for chunk in row_chunks:
            chunk_candidates, chunk_stats = scan_candidate_chunk((chunk, cfg))
            candidates.extend(chunk_candidates)
            for key, value in chunk_stats.items():
                stats[key] += value
            if progress is not None:
                progress.update(int(chunk_stats["rows_scanned"]))
            if args.max_samples is not None and len(candidates) >= args.max_samples:
                break

    if progress is not None:
        progress.close()
    if args.max_samples is not None and len(candidates) > args.max_samples:
        candidates = candidates[:args.max_samples]
    store_split_stats(stats, "candidate", candidates)

    return candidates, stats


def encode_and_write(args: argparse.Namespace, candidates: List[Dict[str, Any]], stats: Dict[str, float]) -> int:
    val_indices: Set[int] = set()
    if args.validation_jsonl and args.validation_samples > 0:
        if args.validation_strategy == "last5_random5":
            eligible = [
                idx for idx, item in enumerate(candidates)
                if float(item["duration"]) >= float(args.validation_min_target_sec)
            ]
            tail_count = min(5, len(eligible))
            tail_indices = eligible[-tail_count:]
            remaining = [idx for idx in eligible if idx not in set(tail_indices)]
            rng = random.Random(args.split_seed)
            random_count = min(5, max(0, args.validation_samples - len(tail_indices)), len(remaining))
            random_indices = rng.sample(remaining, random_count) if random_count else []
            val_indices = set(tail_indices + random_indices)
        else:
            eligible = [
                idx for idx, item in enumerate(candidates)
                if float(item["duration"]) >= float(args.validation_min_target_sec)
            ]
            eligible.sort(
                key=lambda idx: (
                    abs(float(candidates[idx]["duration"]) - float(args.validation_target_sec)),
                    idx,
                )
            )
            val_count = min(args.validation_samples, len(eligible))
            val_indices = set(eligible[:val_count])
        if len(val_indices) < args.validation_samples:
            print(
                f"Warning: requested {args.validation_samples} validation samples, "
                f"but selected {len(val_indices)} with strategy {args.validation_strategy}.",
                file=sys.stderr,
            )
    train_candidates = [item for idx, item in enumerate(candidates) if idx not in val_indices]
    val_candidates = [item for idx, item in enumerate(candidates) if idx in val_indices]
    store_split_stats(stats, "train_candidate", train_candidates)
    store_split_stats(stats, "val_candidate", val_candidates)
    if val_candidates:
        val_durations = [float(item["duration"]) for item in val_candidates]
        print(
            "Validation split: "
            f"{len(val_candidates)} samples, "
            f"target_duration_min={min(val_durations):.2f}s, "
            f"target_duration_max={max(val_durations):.2f}s, "
            f"target_duration_mean={sum(val_durations) / len(val_durations):.2f}s"
        )

    with torch.inference_mode():
        qwen3tts = Qwen3TTSModel.from_pretrained(
            args.base_model_path,
            torch_dtype=torch_dtype(args.dtype),
            device_map=args.device,
            attn_implementation=args.attn_implementation,
        )
        qwen3tts.model.eval()
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        if args.validation_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(args.validation_jsonl)), exist_ok=True)
        demo_dir = Path(args.demonstration_dir) if args.demonstration_dir else None
        demo_rows: List[Dict[str, Any]] = []
        if demo_dir:
            (demo_dir / "original_audio").mkdir(parents=True, exist_ok=True)
            (demo_dir / "decoded_audio_codes").mkdir(parents=True, exist_ok=True)
            (demo_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        def write_rows(output_jsonl: str, rows: List[Dict[str, Any]], label: str) -> int:
            written = 0
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    chunks(rows, max(1, args.batch_size)),
                    total=(len(rows) + max(1, args.batch_size) - 1) // max(1, args.batch_size),
                    desc=f"Encoding {label}",
                )
            except ImportError:
                iterator = chunks(rows, max(1, args.batch_size))

            with open(output_jsonl, "w", encoding="utf-8") as out_f:
                for batch in iterator:
                    worker_count = min(max(1, args.audio_workers), len(batch))
                    if worker_count > 1:
                        with ThreadPoolExecutor(max_workers=worker_count) as executor:
                            futures = [(item, executor.submit(prepare_audio_item, item)) for item in batch]
                            prepared = []
                            for item, future in futures:
                                try:
                                    prepared.append(future.result())
                                except Exception as exc:
                                    print(f"Skipping {item['relative_path']}: audio load/crop failed: {exc}", file=sys.stderr)
                                    stats["skipped"] += 1
                    else:
                        prepared = []
                        for item in batch:
                            try:
                                prepared.append(prepare_audio_item(item))
                            except Exception as exc:
                                print(f"Skipping {item['relative_path']}: audio load/crop failed: {exc}", file=sys.stderr)
                                stats["skipped"] += 1

                    if not prepared:
                        continue

                    target_codes = qwen3tts.model.speech_tokenizer.encode([x[1] for x in prepared], sr=TARGET_SR).audio_codes
                    ref_codes = qwen3tts.model.speech_tokenizer.encode([x[2] for x in prepared], sr=TARGET_SR).audio_codes
                    spk_embeddings = extract_speaker_embeddings_batch(qwen3tts.model, [x[2] for x in prepared])
                    if not (len(target_codes) == len(ref_codes) == len(spk_embeddings) == len(prepared)):
                        raise RuntimeError(
                            "Batch encoder mismatch: "
                            f"prepared={len(prepared)} target_codes={len(target_codes)} "
                            f"ref_codes={len(ref_codes)} spk_embeddings={len(spk_embeddings)}"
                        )

                    for (item, _target_wav, _ref_wav, _audio_duration), code, ref_code, spk in zip(prepared, target_codes, ref_codes, spk_embeddings):
                        try:
                            row = {
                                "audio_codes": tensor_to_list(code),
                                "ref_audio_codes": tensor_to_list(ref_code),
                                "ref_spk_embedding": tensor_to_list(spk),
                                "text_ids": tokenize(qwen3tts.processor, build_assistant_text(item["text"])),
                                "ref_ids": tokenize(qwen3tts.processor, build_ref_text(item["ref_text"])),
                                "text": item["text"],
                                "ref_text": item["ref_text"],
                                "language": args.language,
                                "relative_path": item["relative_path"],
                                "split_start": item["split_start"],
                                "split_end": item["split_end"],
                                "ref_split_start": item["ref_split_start"],
                                "ref_split_end": item["ref_split_end"],
                                "duration": item["duration"],
                                "ref_duration": item["ref_duration"],
                            }
                            for optional_key in (
                                "dataset",
                                "manifest_path",
                                "manifest_row_index",
                                "target_row_id",
                                "hypothesis_id",
                                "ref_relative_path",
                                "wer",
                                "cer",
                                "MOS",
                                "redimnet2_cos_sim",
                            ):
                                if optional_key in item:
                                    row[optional_key] = item[optional_key]
                        except Exception as exc:
                            print(f"Skipping {item['relative_path']}: model encoding failed: {exc}", file=sys.stderr)
                            stats["skipped"] += 1
                            continue
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                        if demo_dir and len(demo_rows) < max(0, args.demo_samples):
                            demo_index = len(demo_rows)
                            demo_name = f"{label}_{demo_index:03d}"
                            demo_item = copy.deepcopy(row)
                            demo_item["original_target_audio"] = copy_demo_audio(
                                item["audio_path"], demo_dir / "original_audio", f"{demo_name}_target"
                            )
                            if item.get("ref_audio_path"):
                                demo_item["original_reference_audio"] = copy_demo_audio(
                                    item["ref_audio_path"], demo_dir / "original_audio", f"{demo_name}_reference"
                                )
                            target_decoded = demo_dir / "decoded_audio_codes" / f"{demo_name}_target_decoded.wav"
                            ref_decoded = demo_dir / "decoded_audio_codes" / f"{demo_name}_reference_decoded.wav"
                            write_demo_decoded_audio(qwen3tts.model, row["audio_codes"], target_decoded)
                            write_demo_decoded_audio(qwen3tts.model, row["ref_audio_codes"], ref_decoded)
                            demo_item["decoded_target_audio"] = str(target_decoded)
                            demo_item["decoded_reference_audio"] = str(ref_decoded)
                            demo_rows.append(demo_item)
            return written

        val_written = 0
        if args.validation_jsonl:
            val_written = write_rows(args.validation_jsonl, val_candidates, "validation")
            stats["val_written"] = float(val_written)
            print(f"  validation written: {val_written}")
        written = write_rows(args.output_jsonl, train_candidates, "train")
        stats["train_written"] = float(written)
        if demo_dir:
            shutil.copy2(args.output_jsonl, demo_dir / "artifacts" / Path(args.output_jsonl).name)
            if args.validation_jsonl and os.path.exists(args.validation_jsonl):
                shutil.copy2(args.validation_jsonl, demo_dir / "artifacts" / Path(args.validation_jsonl).name)
            with open(demo_dir / "artifacts" / "demo_manifest.jsonl", "w", encoding="utf-8") as out_f:
                for row in demo_rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["demo_examples"] = float(len(demo_rows))

    return written


def print_summary(stats: Dict[str, float], written: int) -> None:
    print("Summary")
    print(f"  rows read: {int(stats['rows_read'])}")
    print(f"  rows scanned with ok alignment: {int(stats['rows_scanned'])}")
    print(f"  text/timestamps usable: {int(stats['usable'])}")
    print(f"  skipped: {int(stats['skipped'])}")
    print(f"  candidate examples: {int(stats.get('candidate_examples', 0))}")
    print(f"  train candidates: {int(stats.get('train_candidate_examples', 0))}")
    print(f"  validation candidates: {int(stats.get('val_candidate_examples', 0))}")
    print(f"  train written: {int(stats.get('train_written', written))}")
    print(f"  validation written: {int(stats.get('val_written', 0))}")
    print(f"  source hours: {stats['source_hours']:.4f}")
    print(f"  candidate target hours: {stats.get('candidate_target_hours', 0.0):.4f}")
    print(f"  candidate ref hours: {stats.get('candidate_ref_hours', 0.0):.4f}")
    print(f"  train target hours: {stats.get('train_candidate_target_hours', 0.0):.4f}")
    print(f"  train ref hours: {stats.get('train_candidate_ref_hours', 0.0):.4f}")
    print(f"  validation target hours: {stats.get('val_candidate_target_hours', 0.0):.4f}")
    print(f"  validation ref hours: {stats.get('val_candidate_ref_hours', 0.0):.4f}")
    print(f"  target sec mean/min/max: {stats.get('candidate_target_mean_sec', 0.0):.2f}/{stats.get('candidate_target_min_sec', 0.0):.2f}/{stats.get('candidate_target_max_sec', 0.0):.2f}")
    print(f"  ref sec mean/min/max: {stats.get('candidate_ref_mean_sec', 0.0):.2f}/{stats.get('candidate_ref_min_sec', 0.0):.2f}/{stats.get('candidate_ref_max_sec', 0.0):.2f}")
    print(f"  truly unprepared source hours: {stats['unprepared_hours']:.4f}")


def main() -> None:
    args = parse_args()
    candidates, stats = make_candidates(args)
    import_runtime_deps()
    written = encode_and_write(args, candidates, stats)
    print_summary(stats, written)


if __name__ == "__main__":
    main()
