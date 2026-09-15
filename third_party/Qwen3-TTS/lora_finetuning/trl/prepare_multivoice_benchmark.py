#!/usr/bin/env python3
"""Materialize deterministic Qwen voice-clone references from proprietary-v2.

The source corpus has no populated speaker identifier.  To avoid claiming an
identity guarantee the output calls these ``reference_id`` values and selects
one clean utterance from each of 20 widely separated source shards.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_finetuning.trl.data_stream import without_feature_casting


DEFAULT_REPO = "bitmanagerai/balalaika_proprietary_v2"
DEFAULT_REVISION = "91a5696ce6125dad5e771f99fec848c060d4f622"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-references", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=519)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--min-distillmos", type=float, default=4.0)
    parser.add_argument("--max-music-prob", type=float, default=0.20)
    parser.add_argument("--max-asr-wer", type=float, default=0.25)
    parser.add_argument("--scan-rows", type=int, default=500)
    return parser.parse_args()


def _words(value: object) -> list[str]:
    return re.findall(r"[0-9a-zа-яё]+", str(value or "").lower())


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, lhs in enumerate(left, 1):
        current = [i]
        for j, rhs in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (lhs != rhs),
                )
            )
        previous = current
    return previous[-1]


def _wer(left: object, right: object) -> float:
    lhs, rhs = _words(left), _words(right)
    return _edit_distance(lhs, rhs) / max(len(lhs), 1)


def _audio_bytes(value: object) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    return None


def _spaced_shards(count: int, total: int) -> list[int]:
    if count <= 0 or count > total:
        raise ValueError("num-references must satisfy 1 <= count <= num-shards")
    if count == 1:
        return [0]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def _candidate(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    raw = _audio_bytes(row.get("mp3"))
    metadata = row.get("json")
    if raw is None or not isinstance(metadata, dict):
        return None
    try:
        info = sf.info(io.BytesIO(raw))
        duration = float(info.frames / info.samplerate)
        mos = float(metadata.get("DistillMOS"))
        music = float(metadata.get("music_prob"))
    except (TypeError, ValueError, RuntimeError):
        return None
    if not args.min_duration <= duration <= args.max_duration:
        return None
    if mos < args.min_distillmos or music > args.max_music_prob:
        return None
    reference_text = str(metadata.get("accent.txt") or "").strip()
    plain_text = str(metadata.get("punct.txt") or metadata.get("rover.txt") or "").strip()
    ctc = metadata.get("giga_ctc.txt")
    rnnt = metadata.get("giga_rnnt.txt")
    rover = metadata.get("rover.txt")
    if not reference_text or not plain_text or len(_words(plain_text)) < 4:
        return None
    agreement_wer = max(_wer(rover, ctc), _wer(rover, rnnt))
    if agreement_wer > args.max_asr_wer:
        return None
    return {
        "raw": raw,
        "duration_sec": duration,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "distillmos": mos,
        "music_prob": music,
        "asr_wer_max": agreement_wer,
        "reference_text": reference_text,
        "reference_text_plain": plain_text,
        "source_record_id": str(row.get("__key__") or ""),
        "source_relative_path": str(metadata.get("source_relative_path") or ""),
        "source_url": str(row.get("__url__") or ""),
    }


def select_from_shard(shard: int, args: argparse.Namespace) -> dict[str, Any]:
    filename = f"train/shard_{shard:06d}.tar"
    uri = f"hf://datasets/{args.repo}@{args.revision}/{filename}"
    stream = load_dataset(
        "webdataset",
        data_files={"train": uri},
        split="train",
        streaming=True,
        token=True,
    )
    stream = without_feature_casting(stream, ("json", "mp3", "__key__", "__url__"))
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(stream):
        if index >= args.scan_rows:
            break
        item = _candidate(row, args)
        if item is not None:
            candidates.append(item)
    if not candidates:
        raise RuntimeError(f"No qualifying reference in {filename} after {args.scan_rows} rows")
    # Prefer perceptual quality, then lower music probability and closer ASR agreement.
    candidates.sort(
        key=lambda item: (
            -item["distillmos"],
            item["music_prob"],
            item["asr_wer_max"],
            item["source_record_id"],
        )
    )
    return candidates[0]


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    for voice_index, shard in enumerate(
        _spaced_shards(args.num_references, args.num_shards)
    ):
        item = select_from_shard(shard, args)
        reference_id = f"corpus-{voice_index:02d}"
        audio_path = output / f"{reference_id}.mp3"
        audio_path.write_bytes(item.pop("raw"))
        item.update(
            {
                "reference_id": reference_id,
                "shard_index": shard,
                "audio_path": str(audio_path),
                "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            }
        )
        selected.append(item)
        print(
            f"[{voice_index + 1}/{args.num_references}] {reference_id} "
            f"shard={shard:06d} duration={item['duration_sec']:.2f}s "
            f"MOS={item['distillmos']:.3f}",
            flush=True,
        )
    source_ids = [item["source_record_id"] for item in selected]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Selected references do not have unique source record IDs")
    manifest = {
        "schema_version": 1,
        "dataset": args.repo,
        "dataset_revision": args.revision,
        "selection": {
            "method": "one-best-from-20-evenly-spaced-shards",
            "speaker_identity_guaranteed": False,
            "num_references": len(selected),
            "duration_sec": [args.min_duration, args.max_duration],
            "min_distillmos": args.min_distillmos,
            "max_music_prob": args.max_music_prob,
            "max_asr_wer": args.max_asr_wer,
            "scan_rows_per_shard": args.scan_rows,
        },
        "references": selected,
    }
    manifest_path = output / "references.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
