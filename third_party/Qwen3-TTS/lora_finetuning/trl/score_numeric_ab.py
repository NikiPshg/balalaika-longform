#!/usr/bin/env python3
"""Transcribe and score a paired numeric Base/SFT listening probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf
import soxr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grpo_dpo_finetuning.metrics import aggregate_metrics, score_text_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_root", type=Path)
    parser.add_argument("--asr-model", default="gigaam-v3-rnnt")
    return parser.parse_args()


def load_waveform(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16_000:
        waveform = soxr.resample(waveform, sample_rate, 16_000)
    value = np.asarray(waveform, dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"Invalid waveform: {path}")
    return value


def main() -> None:
    args = parse_args()
    selected = json.loads((args.probe_root / "selected_rows.json").read_text())
    selected_rows = selected["rows"]
    if len(selected_rows) != 10:
        raise ValueError("Expected exactly ten selected rows")

    pending: list[dict[str, Any]] = []
    waveforms: list[np.ndarray] = []
    for model_name in ("base", "sft"):
        manifest_path = args.probe_root / model_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        records = sorted(manifest["records"], key=lambda row: row["text_index"])
        if len(records) != len(selected_rows):
            raise ValueError(f"{model_name} manifest has {len(records)} records")
        for selected_row, generated_row in zip(selected_rows, records, strict=True):
            if selected_row["text"] != generated_row["source_text"]:
                raise ValueError(f"Text mismatch for {model_name} row {generated_row['text_index']}")
            wav_path = Path(generated_row["output_wav"])
            waveforms.append(load_waveform(wav_path))
            pending.append(
                {
                    "model": model_name,
                    "text_index": generated_row["text_index"],
                    "category": selected_row["category"],
                    "source_index": selected_row.get(
                        "probe_index", selected_row.get("csv_index")
                    ),
                    "reference": selected_row["text"],
                    "num_words": selected_row["num_words"],
                    "wav": str(wav_path.resolve()),
                    "duration_sec": generated_row["duration_sec"],
                    "model_text": generated_row["model_text"],
                }
            )

    from onnx_asr import load_model

    recognizer = load_model(args.asr_model, providers=["CPUExecutionProvider"])
    hypotheses = recognizer.recognize(waveforms, sample_rate=16_000)
    if len(hypotheses) != len(pending):
        raise ValueError("ASR returned the wrong number of hypotheses")

    scored = []
    for record, hypothesis in zip(pending, hypotheses, strict=True):
        hypothesis_text = str(hypothesis).strip()
        metrics = score_text_pair(
            record["reference"],
            hypothesis_text,
            record["num_words"],
        )
        scored.append(
            {
                **record,
                "hypothesis": hypothesis_text,
                "metrics": metrics.as_dict(),
            }
        )
        print(
            f"{record['model']} {record['text_index']:02d} {record['category']}: "
            f"WER={metrics.utterance_wer:.3f}, number_WER={metrics.number_wer:.3f}, "
            f"number_exact={metrics.number_exact}",
            flush=True,
        )

    report = {
        "dataset_repo": selected["dataset_repo"],
        "dataset_revision": selected["dataset_revision"],
        "dataset_file": selected["dataset_file"],
        "asr_model": args.asr_model,
        "asr_providers": ["CPUExecutionProvider"],
        "aggregates": aggregate_metrics(scored, group_keys="model"),
        "records": scored,
    }
    output_path = args.probe_root / "asr_report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
