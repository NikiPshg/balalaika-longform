#!/usr/bin/env python
"""Secondary ASR for the disagreement audit only (PLAN.md §9.4, §14 A4).

The primary and *only* metric ASR is GigaAM-v3 through onnx-asr with VAD
(`src/eval/asr_gigaam.py`).  This module exists for one narrow job: transcribing
a subset of outputs with a **second, non-GigaAM** model so that items where the
two ASRs disagree can be pulled out for the 5-10 % blind manual audit.

Rules baked in here, from PLAN.md §0.1 / §9.4:

* Whisper is forbidden anywhere in the project -- `whisper-*` is rejected by name.
* The second ASR must not be from the GigaAM family (it would share the primary's
  errors), so every ``gigaam-*`` id is rejected too.
* It never produces a published metric.  Its output feeds `disagreement_rows()`,
  which ranks items by hypothesis-vs-hypothesis WER; the audit then listens to
  the audio.  Nothing here changes WER, CER, coverage or any §17 table cell.
* Both voters of the dataset ROVER that onnx-asr ships (``alphacep/vosk-model-ru``
  and ``t-tech/t-one``) are allowed: for *disagreement* the shared-reference bias
  does not matter, since the comparison is hypothesis vs hypothesis on audio the
  ROVER never saw.

Usage::

    CUDA_VISIBLE_DEVICES=1 .venv-eval/bin/python src/eval/asr_secondary.py \\
        --per-item results/E1_native/per_item.jsonl \\
        --output results/E1_native/disagreement.jsonl \\
        --model alphacep/vosk-model-small-ru --top-frac 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.asr_gigaam import GigaAMTranscriber, TranscriptionResult  # noqa: E402
from eval.metrics import content_metrics  # noqa: E402

# Frozen allow-list: onnx-asr ids that are Russian, in the library catalogue, and
# outside the GigaAM family (loader.AsrNames, onnx-asr 0.12.0).
SECONDARY_MODELS = (
    "alphacep/vosk-model-small-ru",
    "alphacep/vosk-model-ru",
    "t-tech/t-one",
)
DEFAULT_SECONDARY = "alphacep/vosk-model-small-ru"


class SecondaryAsrError(ValueError):
    """The requested secondary ASR is not allowed by PLAN.md §0.1/§9.4."""


def check_secondary_model(model_id: str) -> str:
    """Reject Whisper and the GigaAM family; accept only the frozen allow-list."""
    mid = str(model_id)
    low = mid.lower()
    if "whisper" in low:
        raise SecondaryAsrError(
            f"{mid!r}: Whisper is forbidden in this project (PLAN.md §0.1)"
        )
    if "gigaam" in low:
        raise SecondaryAsrError(
            f"{mid!r}: the secondary ASR must be independent of the primary, and the "
            "whole GigaAM family is the primary (PLAN.md §9.4)"
        )
    if mid not in SECONDARY_MODELS:
        raise SecondaryAsrError(
            f"{mid!r} is not in the frozen secondary allow-list {SECONDARY_MODELS}"
        )
    return mid


class SecondaryTranscriber:
    """A :class:`GigaAMTranscriber` restricted to the allowed secondary models.

    Same VAD contract, same GPU guard, same in-repo HF cache -- only the weights
    differ, so a disagreement cannot be an artefact of a different chunking.
    """

    def __init__(self, model_id: str = DEFAULT_SECONDARY, device: str = "cuda", **kw: Any) -> None:
        self.model_id = check_secondary_model(model_id)
        self._tr = GigaAMTranscriber(model_id=self.model_id, use_vad=True, device=device, **kw)

    def transcribe(self, path: str | Path) -> TranscriptionResult:
        return self._tr.transcribe(path)

    def describe(self) -> dict:
        d = self._tr.describe()
        d["role"] = "secondary_disagreement_audit_only"
        return d


def disagreement_rows(
    per_item: Sequence[dict],
    secondary_texts: dict[str, str],
    variant: str = "lenient",
) -> list[dict]:
    """Rank items by primary-vs-secondary hypothesis WER (highest first).

    ``per_item`` rows are the evaluator's own output (they carry ``run_id``,
    ``asr_text`` and ``wer``); ``secondary_texts`` maps ``run_id`` to the second
    ASR's transcript.  The returned rows are an *audit worklist*, never a metric.
    """
    out: list[dict] = []
    for row in per_item:
        rid = row.get("run_id")
        if rid not in secondary_texts:
            continue
        primary = str(row.get("asr_text") or "")
        secondary = str(secondary_texts[rid] or "")
        # primary as the "reference": the number is a distance between two
        # hypotheses, symmetric enough for ranking and reported as such
        cm = content_metrics(primary, secondary, variant=variant)
        out.append({
            "run_id": rid,
            "text_id": row.get("text_id"),
            "checkpoint": row.get("checkpoint"),
            "bucket": row.get("bucket"),
            "status": row.get("status"),            # final §3.4 status (evaluator)
            "gen_status": row.get("gen_status"),    # provisional label (generator)
            "output_path": row.get("output_path"),
            "primary_wer_vs_reference": row.get("wer"),
            "primary_text": primary,
            "secondary_text": secondary,
            "disagreement_wer": cm["wer"],
            "disagreement_cer": cm["cer"],
            "n_primary_words": cm["n_ref_words"],
            "n_secondary_words": cm["n_hyp_words"],
        })
    out.sort(key=lambda r: (r["disagreement_wer"] is None, -(r["disagreement_wer"] or 0.0)))
    return out


def select_audit_subset(rows: Sequence[dict], top_frac: float = 0.10,
                        min_items: int = 1) -> list[dict]:
    """Top ``top_frac`` of the ranked rows (PLAN.md §9.4: 5-10 % blind audit)."""
    if not rows:
        return []
    n = max(min_items, int(round(len(rows) * float(top_frac))))
    return [dict(r, audit_selected=True) for r in rows[:n]]


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Secondary ASR disagreement audit (not a metric)")
    ap.add_argument("--per-item", required=True, help="per_item.jsonl from scripts/run_evaluation.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_SECONDARY, choices=list(SECONDARY_MODELS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--variant", default="lenient", choices=["strict", "lenient"])
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=None, help="transcribe at most N outputs")
    args = ap.parse_args(argv)

    per_item = read_jsonl(args.per_item)
    todo = [r for r in per_item if r.get("output_exists") and r.get("output_path")]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("[secondary] no existing outputs in the per-item file", file=sys.stderr)
        return 1

    tr = SecondaryTranscriber(model_id=args.model, device=args.device)
    print(json.dumps(tr.describe(), ensure_ascii=False), file=sys.stderr)

    texts: dict[str, str] = {}
    for row in todo:
        res = tr.transcribe(str(row["output_path"]))
        texts[row["run_id"]] = res.text
        print(f"[secondary] {row['run_id']}: {len(res.text.split())} words "
              f"({res.audio_duration_sec:.1f}s)", file=sys.stderr)

    rows = disagreement_rows(per_item, texts, variant=args.variant)
    selected = {r["run_id"] for r in select_audit_subset(rows, args.top_frac)}
    for r in rows:
        r["audit_selected"] = r["run_id"] in selected
        r["secondary_model_id"] = args.model
    write_jsonl(args.output, rows)
    print(f"[secondary] {len(rows)} compared, {len(selected)} flagged for the blind audit "
          f"-> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
