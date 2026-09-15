"""M4 analysis of the Short-SFT (E2) vs Long-SFT (E3) pilot (PLAN.md §2, §9, §12, §17).

Every number quoted in ``reports/pilot_sft_results.md`` is produced by this script
from immutable inputs and nothing else:

* ``data/benchmark/pilot.jsonl``                                   (A2, canonical §7.5 fields)
* ``results/v31_base/{E1,E0}/per_item.jsonl``                      (A4 evaluator)
* ``results/v31_sft/{E3_epoch_3_step_3001,E3_epoch_0_whole,
   E2_epoch_3_step_3001,E2_epoch_0_step_200}/per_item.jsonl``      (A4 evaluator)
* ``outputs/v31_{base,sft}/*/runs.jsonl``                          (A3/A7 run manifests, §10)
* ``exp/{long,short}/pilot/{train_stats.jsonl,run_info.json,
   budget_verify_step200.json}``                                   (A6 training, read-only)
* ``logs/v31_sft/train_{long,short}.log``                          (CV-loss lines)
* ``results/v31_stats/paired_deltas.json``                         (A8 cluster bootstrap, Table 4)
* ``data/train/{long,short}/stats.json``,
  ``data/train/long/token_manifest.jsonl``                         (A6 arms, for RQ6)

Cross-checks (all hard failures, the script refuses to print on disagreement):

1. the word alignment behind ``wer``/``hits``/``substitutions``/``deletions``/
   ``insertions``/``source_coverage``/``end_coverage``/``end_coverage_robust`` is
   re-derived from ``text_ref`` + ``asr_text`` for all 360 rows and compared with
   the evaluator's stored values;
2. every Table 3 / Table 5 cell this script computes is compared with the
   independent aggregate in ``results/v31_sft/main_table/v31_sft.csv``.

Usage::

    .venv-eval/bin/python src/stats/sft_analysis.py                 # tables + json + figures
    .venv-eval/bin/python src/stats/sft_analysis.py --no-figures
    .venv-eval/bin/python src/stats/sft_analysis.py --out-json PATH

No GPU, no model, no network.  Writes only under ``results/v31_sft_analysis/`` and
``reports/figures/v31_sft/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.alignment import align_words  # noqa: E402
from src.eval.metrics import last_aligned_run  # noqa: E402
from src.eval.normalize import words as norm_words  # noqa: E402

BUCKETS = ("B0", "B1", "B2", "B3", "B4")
STATUSES = (
    "complete",
    "degraded",
    "early_eos",
    "loop_cap",
    "hard_input_limit",
    "context_limit",
    "oom",
    "timeout",
    "empty_or_invalid_audio",
    "infrastructure_error",
)

# label -> (per_item dir, runs.jsonl dir, human-readable name)
ARMS: dict[str, tuple[str, str, str]] = {
    "E1_native": (
        "results/v31_base/E1",
        "outputs/v31_base/E1_native",
        "E1 Base-Native",
    ),
    "E0_official": (
        "results/v31_base/E0",
        "outputs/v31_base/E0_official",
        "E0 Base official-split (production control)",
    ),
    "E2_epoch_0_step_200": (
        "results/v31_sft/E2_epoch_0_step_200",
        "outputs/v31_sft/E2_epoch_0_step_200",
        "E2 Short-SFT step 200 (by dev loss)",
    ),
    "E2_epoch_3_step_3001": (
        "results/v31_sft/E2_epoch_3_step_3001",
        "outputs/v31_sft/E2_epoch_3_step_3001",
        "E2 Short-SFT step 3001 (selected)",
    ),
    "E3_epoch_0_whole": (
        "results/v31_sft/E3_epoch_0_whole",
        "outputs/v31_sft/E3_epoch_0_whole",
        "E3 Long-SFT step 772 (by dev loss)",
    ),
    "E3_epoch_3_step_3001": (
        "results/v31_sft/E3_epoch_3_step_3001",
        "outputs/v31_sft/E3_epoch_3_step_3001",
        "E3 Long-SFT step 3001 (selected)",
    ),
    # E4 Curriculum-SFT (PLAN §5 E4, *secondary* product recipe): pre-registered in
    # reports/decisions.md 2026-08-29, initialisation amended 2026-08-30 (starts from the
    # SELECTED E2 checkpoint, not the by-loss one).  Not budget-matched to E2/E3 -- 6000
    # optimizer steps in total against 3000 -- so it is published as its own row and never
    # replaces the E3-vs-E2 main comparison.
    "E4_epoch_0_whole": (
        "results/v31_curr/E4_epoch_0_whole",
        "outputs/v31_curr/E4_epoch_0_whole",
        "E4 Curriculum-SFT S3 step 773 (selected)",
    ),
    "E4_epoch_2_step_2001": (
        "results/v31_curr/E4_epoch_2_step_2001",
        "outputs/v31_curr/E4_epoch_2_step_2001",
        "E4 Curriculum-SFT S3 step 2001 (last)",
    ),
}
# print order for the tables
ORDER = (
    "E1_native",
    "E0_official",
    "E2_epoch_0_step_200",
    "E2_epoch_3_step_3001",
    "E3_epoch_0_whole",
    "E3_epoch_3_step_3001",
    "E4_epoch_0_whole",
    "E4_epoch_2_step_2001",
)
SELECTED = ("E1_native", "E2_epoch_3_step_3001", "E3_epoch_3_step_3001")

NGRAM_RANGE = (3, 8)


# ---------------------------------------------------------------- small utils


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _median(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return st.median(vals) if vals else None


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "n": len(vals),
        "min": min(vals),
        "median": st.median(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}"


def _print_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    print()


# ------------------------------------------------------------------ loading


def load_inputs() -> dict[str, Any]:
    bench = {r["text_id"]: r for r in _load_jsonl(REPO / "data/benchmark/pilot.jsonl")}
    per_item: dict[str, list[dict]] = {}
    runs: dict[str, dict[str, dict]] = {}
    for label, (res_dir, out_dir, _name) in ARMS.items():
        rows = _load_jsonl(REPO / res_dir / "per_item.jsonl")
        if len(rows) != 60:
            raise SystemExit(f"{label}: expected 60 per-item rows, got {len(rows)}")
        per_item[label] = rows
        manifests = _load_jsonl(REPO / out_dir / "runs.jsonl")
        runs[label] = {m["run_id"]: m for m in manifests}
        if len(runs[label]) != 60:
            raise SystemExit(f"{label}: expected 60 run manifests, got {len(runs[label])}")
    return {"bench": bench, "per_item": per_item, "runs": runs}


# ------------------------------------------------------- verification pass 1


def verify_alignment(bench: dict, per_item: dict[str, list[dict]]) -> dict[str, int]:
    """Re-derive the evaluator's content metrics; any disagreement is fatal."""
    checked = 0
    mismatches: list[str] = []
    for label, rows in per_item.items():
        for row in rows:
            item = bench[row["text_id"]]
            ref = norm_words(item["text_ref"], row["normalization_variant"])
            hyp = norm_words(row["asr_text"] or "", row["normalization_variant"])
            if not row.get("valid", True) or not row.get("transcribed", True):
                # the evaluator scores an unusable output as an empty hypothesis
                hyp = []
            al = align_words(ref, hyp)
            mask = al.ref_aligned_mask()
            n_ref = len(ref)
            cov = (sum(mask) / n_ref) if n_ref else 0.0
            end_idx = max((i for i, f in enumerate(mask) if f), default=None)
            end_cov = ((end_idx + 1) / n_ref) if (end_idx is not None and n_ref) else 0.0
            r_idx, _run, _n = last_aligned_run(mask, row["end_coverage_robust_min_run"])
            end_cov_r = ((r_idx + 1) / n_ref) if (r_idx is not None and n_ref) else 0.0
            wer = (al.substitutions + al.deletions + al.insertions) / n_ref if n_ref else 0.0
            got = {
                "hits": al.hits,
                "substitutions": al.substitutions,
                "deletions": al.deletions,
                "insertions": al.insertions,
                "wer": wer,
                "source_coverage": cov,
                "end_coverage": end_cov,
                "end_coverage_robust": end_cov_r,
                "n_ref_words": n_ref,
            }
            for key, value in got.items():
                stored = row[key]
                ok = (
                    abs(value - stored) <= 1e-9
                    if isinstance(value, float)
                    else value == stored
                )
                if not ok:
                    mismatches.append(f"{label}/{row['run_id']}: {key} {value} != {stored}")
            checked += 1
    if mismatches:
        for m in mismatches[:20]:
            print("MISMATCH", m, file=sys.stderr)
        raise SystemExit(
            f"alignment re-derivation disagrees with the evaluator on "
            f"{len(mismatches)} field(s) over {checked} items"
        )
    return {"items_checked": checked, "mismatches": 0}


# ------------------------------------------------------- verification pass 2


def verify_main_table(table3: dict, table5: dict) -> dict[str, int]:
    """Compare every cell against results/v31_sft/main_table/v31_sft.csv."""
    path = REPO / "results/v31_sft/main_table/v31_sft.csv"
    cells = 0
    bad: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            label, bucket = rec["experiment"], rec["bucket"]
            if label not in table3 or bucket not in table3[label]:
                continue
            mine = table3[label][bucket]
            pairs = [
                ("complete_rate", mine["complete_rate"]),
                ("macro_wer_all", mine["wer"]),
                ("macro_cer_all", mine["cer"]),
                ("macro_source_coverage_all", mine["coverage"]),
                ("macro_end_coverage_all", mine["end_coverage"]),
                ("macro_end_coverage_robust_all", mine["end_coverage_robust"]),
                ("macro_excess_repetition_rate_all", mine["repeat"]),
                ("macro_duration_ratio_all", mine["dur_ratio"]),
                ("macro_max_ngram_run_all", mine["max_ngram_run"]),
                ("wer_minus_floor_macro", mine["wer_minus_floor"]),
                ("floor_wer_macro", mine["floor_wer"]),
            ]
            for col, value in pairs:
                raw = rec[col]
                if raw == "":
                    if value is not None:
                        bad.append(f"{label}/{bucket}/{col}: mine {value}, csv empty")
                    continue
                if value is None or abs(float(raw) - value) > 1e-9:
                    bad.append(f"{label}/{bucket}/{col}: mine {value} != csv {raw}")
                cells += 1
            for status in STATUSES:
                col = f"n_{status}"
                if col in rec:
                    if int(rec[col]) != table5[label][bucket].get(status, 0):
                        bad.append(
                            f"{label}/{bucket}/{col}: mine "
                            f"{table5[label][bucket].get(status, 0)} != csv {rec[col]}"
                        )
                    cells += 1
    if bad:
        for b in bad[:20]:
            print("MISMATCH", b, file=sys.stderr)
        raise SystemExit(f"{len(bad)} disagreements with results/v31_sft/main_table/v31_sft.csv")
    return {"cells_checked": cells, "mismatches": 0}


# label used here -> label used by A5's final 8-checkpoint aggregate
FINAL_TABLE_LABELS = {
    "E1_native": "E1_native",
    "E0_official": "E0_official",
    "E2_epoch_0_step_200": "E2_short_byloss",
    "E2_epoch_3_step_3001": "E2_short",
    "E3_epoch_0_whole": "E3_long_byloss",
    "E3_epoch_3_step_3001": "E3_long",
    "E4_epoch_0_whole": "E4_curr",
    "E4_epoch_2_step_2001": "E4_curr_last",
}


def verify_final_table(table3: dict, table5: dict) -> dict[str, int]:
    """Second, independent cross-check against A5's FINAL 8-checkpoint aggregate.

    ``results/v31_sft/main_table/v31_sft.csv`` (checked by ``verify_main_table``) covers only
    the six pre-E4 checkpoints, so the two E4 rows would otherwise be unchecked.  A5's
    ``results/v31_final/main_table/v31_final.csv`` covers all eight and was written by a
    different script (``scripts/aggregate_results.py``); a disagreement is fatal, exactly like
    the two checks above.  This one also covers ``micro_wer_all``, which the v31_sft csv of
    2026-08-30 05:35 does not carry for every row.
    """
    path = REPO / "results/v31_final/main_table/v31_final.csv"
    if not path.exists():
        return {"cells_checked": 0, "mismatches": 0, "note": "v31_final.csv absent"}
    inv = {v: k for k, v in FINAL_TABLE_LABELS.items()}
    cells = 0
    seen: set[tuple[str, str]] = set()
    bad: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            label = inv.get(rec["experiment"])
            bucket = rec["bucket"]
            if label is None or label not in table3 or bucket not in table3[label]:
                continue
            mine = table3[label][bucket]
            seen.add((label, bucket))
            pairs = [
                ("complete_rate", mine["complete_rate"]),
                ("macro_wer_all", mine["wer"]),
                ("micro_wer_all", mine["wer_micro"]),
                ("macro_cer_all", mine["cer"]),
                ("macro_source_coverage_all", mine["coverage"]),
                ("macro_end_coverage_all", mine["end_coverage"]),
                ("macro_end_coverage_robust_all", mine["end_coverage_robust"]),
                ("macro_excess_repetition_rate_all", mine["repeat"]),
                ("macro_duration_ratio_all", mine["dur_ratio"]),
                ("macro_max_ngram_run_all", mine["max_ngram_run"]),
                ("macro_tail_deletion_rate_all", mine["tail_deletion_rate"]),
                ("wer_minus_floor_macro", mine["wer_minus_floor"]),
                ("floor_wer_macro", mine["floor_wer"]),
                ("n_attempted", float(mine["n"])),
                ("n_with_floor", float(mine["n_with_floor"])),
                ("n_loop_flag_all", float(mine["n_loop_flag"])),
            ]
            for col, value in pairs:
                raw = rec.get(col, "")
                if raw == "":
                    if value is not None:
                        bad.append(f"{label}/{bucket}/{col}: mine {value}, csv empty")
                    continue
                if value is None or abs(float(raw) - value) > 1e-9:
                    bad.append(f"{label}/{bucket}/{col}: mine {value} != csv {raw}")
                cells += 1
            for status in STATUSES:
                col = f"n_{status}"
                if col in rec:
                    if int(rec[col]) != table5[label][bucket].get(status, 0):
                        bad.append(
                            f"{label}/{bucket}/{col}: mine "
                            f"{table5[label][bucket].get(status, 0)} != csv {rec[col]}"
                        )
                    cells += 1
    expected = {(lab, b) for lab in FINAL_TABLE_LABELS for b in (*BUCKETS, "ALL")}
    missing = sorted(expected - seen)
    if missing:
        bad.append(f"{len(missing)} rows absent from v31_final.csv, first: {missing[0]}")
    if bad:
        for b in bad[:20]:
            print("MISMATCH", b, file=sys.stderr)
        raise SystemExit(
            f"{len(bad)} disagreements with results/v31_final/main_table/v31_final.csv"
        )
    return {"cells_checked": cells, "mismatches": 0, "rows_checked": len(seen)}


# ------------------------------------------------------------ macro vs micro


def wer_macro_vs_micro(table3: dict) -> dict:
    """How far the macro and micro WER of §2 diverge, per bucket and on ALL.

    Macro weighs every item equally, micro weighs it by its reference length, so the two
    separate exactly where the item lengths inside a cell differ -- i.e. on the `ALL` row.
    """
    worst = {"gap_pp": 0.0, "checkpoint": None, "bucket": None}
    for ck, buckets in table3.items():
        for b, c in buckets.items():
            if b == "ALL":
                continue
            gap = abs(100 * c["wer"] - 100 * c["wer_micro"])
            if gap > worst["gap_pp"]:
                worst = {"gap_pp": gap, "checkpoint": ck, "bucket": b}
    return {
        "max_bucket_gap": worst,
        "all_row_gap_pp": {
            ck: 100 * (buckets["ALL"]["wer_micro"] - buckets["ALL"]["wer"])
            for ck, buckets in table3.items()
        },
    }


# ------------------------------------------------------------------ Table 3


def build_table3(per_item: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for label, rows in per_item.items():
        out[label] = {}
        for bucket in (*BUCKETS, "ALL"):
            sel = rows if bucket == "ALL" else [r for r in rows if r["bucket"] == bucket]
            floor = [r for r in sel if r.get("floor_available")]
            out[label][bucket] = {
                "n": len(sel),
                "complete_rate": _mean(1.0 if r["status"] == "complete" else 0.0 for r in sel),
                "wer": _mean(r["wer"] for r in sel),
                "wer_median": _median(r["wer"] for r in sel),
                # micro-averaged WER (PLAN §12 asks for macro AND micro): one pooled
                # rate over the bucket, sum(S+D+I) / sum(n_ref_words), so a long item
                # weighs as much as its reference text is long.
                "wer_micro": (
                    sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in sel)
                    / sum(r["n_ref_words"] for r in sel)
                ),
                "cer": _mean(r["cer"] for r in sel),
                "coverage": _mean(r["source_coverage"] for r in sel),
                "coverage_median": _median(r["source_coverage"] for r in sel),
                "end_coverage": _mean(r["end_coverage"] for r in sel),
                "end_coverage_robust": _mean(r["end_coverage_robust"] for r in sel),
                "repeat": _mean(r["excess_repetition_rate"] for r in sel),
                "max_ngram_run": _mean(float(r["max_ngram_run"]) for r in sel),
                # the mean of per-item maxima is not a maximum: publish both.
                "max_ngram_run_max": max(int(r["max_ngram_run"]) for r in sel),
                "dur_ratio": _mean(r["duration_ratio"] for r in sel),
                "dur_ratio_median": _median(r["duration_ratio"] for r in sel),
                # |1 - ratio|: the direction that is monotone in quality (over- and
                # under-production are both failures, so "higher" is not "better").
                "dur_ratio_dev_from_1": abs(1.0 - _mean(r["duration_ratio"] for r in sel)),
                "wer_minus_floor": _mean(r["wer_minus_floor"] for r in floor) if floor else None,
                "floor_wer": _mean(r["floor_wer"] for r in floor) if floor else None,
                "n_with_floor": len(floor),
                "tail_deletion_rate": _mean(r["tail_deletion_rate"] for r in sel),
                "longest_deletion_run_words": _mean(float(r["longest_deletion_run"]) for r in sel),
                "raw_duration_sec": _stats([r["raw_duration_sec"] for r in sel]),
                "n_loop_flag": sum(1 for r in sel if r.get("loop_flag")),
            }
    return out


def build_table5(per_item: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for label, rows in per_item.items():
        out[label] = {}
        for bucket in (*BUCKETS, "ALL"):
            sel = rows if bucket == "ALL" else [r for r in rows if r["bucket"] == bucket]
            counts = Counter(r["status"] for r in sel)
            out[label][bucket] = {s: counts.get(s, 0) for s in STATUSES}
            out[label][bucket]["n"] = len(sel)
    return out


# ---------------------------------------------------- duration / RQ2 changes


def build_duration(bench: dict, per_item: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for label, rows in per_item.items():
        out[label] = {}
        for bucket in (*BUCKETS, "ALL"):
            sel = rows if bucket == "ALL" else [r for r in rows if r["bucket"] == bucket]
            human = [bench[r["text_id"]]["human_duration_sec"] for r in sel]
            out[label][bucket] = {
                "human": _stats(human),
                "generated": _stats([r["raw_duration_sec"] for r in sel]),
                "ratio": _stats([r["duration_ratio"] for r in sel]),
                "words_read": _stats(
                    [r["source_coverage"] * r["n_ref_words"] for r in sel]
                ),
                "n_ref_words": _stats([float(r["n_ref_words"]) for r in sel]),
                "silence_ratio": _stats([r["silence_ratio"] for r in sel]),
                "wpm_raw": _stats([r["speaking_rate_wpm_raw"] for r in sel]),
            }
    return out


# ------------------------------------------------------ residual E3 failures


def _word_times(segments: Sequence[dict] | None, variant: str) -> list[float]:
    """Mid-point time of every normalized hypothesis word.

    ``asr_segments`` carries VAD-segment start/end and the text of that segment;
    normalizing the segments separately and concatenating reproduces
    ``norm_words(asr_text)`` exactly (asserted by the caller), so word *i* of the
    hypothesis lies in a known segment and is placed at the centre of its equal
    share of that segment's span.  This is a linear interpolation inside a VAD
    segment, not a forced alignment.
    """
    out: list[float] = []
    for seg in segments or []:
        ws = norm_words(seg["text"], variant)
        if not ws:
            continue
        step = (seg["end"] - seg["start"]) / len(ws)
        out.extend(seg["start"] + step * (i + 0.5) for i in range(len(ws)))
    return out


def _first_excess_ngram(
    ref: Sequence[str], hyp: Sequence[str], min_n: int, max_n: int = NGRAM_RANGE[1]
) -> tuple[int, int, str] | None:
    """First hypothesis position that repeats an n-gram more often than the source does.

    Scanning left to right, the first index ``i`` at which the running count of
    ``hyp[i:i+n]`` exceeds ``max(1, count_ref)`` for some ``n`` in
    ``[min_n, max_n]``.  Returns ``(index, n, gram)`` of the earliest such
    position over all n, or ``None`` when the transcript never repeats anything
    the source does not.  This is the same excess-occurrence bookkeeping as
    ``src/eval/metrics.excess_repetition`` (PLAN §7.3: source repetition is not
    hallucination), read as a position instead of a rate.
    """
    best: tuple[int, int, str] | None = None
    for n in range(min_n, max_n + 1):
        if len(hyp) < n:
            continue
        ref_c = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        seen: Counter = Counter()
        for i in range(len(hyp) - n + 1):
            gram = tuple(hyp[i : i + n])
            seen[gram] += 1
            if seen[gram] > max(1, ref_c.get(gram, 0)):
                if best is None or i < best[0]:
                    best = (i, n, " ".join(gram))
                break
    return best


def residual_failures(bench: dict, per_item: dict[str, list[dict]], runs: dict) -> dict:
    """Per-item anatomy of every non-`complete` item of the selected E3 checkpoint."""
    label = "E3_epoch_3_step_3001"
    rows = per_item[label]
    items: list[dict] = []
    for row in rows:
        if row["status"] == "complete":
            continue
        item = bench[row["text_id"]]
        variant = row["normalization_variant"]
        ref = norm_words(item["text_ref"], variant)
        hyp = norm_words(row["asr_text"] or "", variant)
        times = _word_times(row["asr_segments"], variant)
        if len(times) != len(hyp):
            raise SystemExit(
                f"{row['run_id']}: segment words {len(times)} != transcript words {len(hyp)}"
            )
        al = align_words(ref, hyp)
        onsets: dict[str, dict | None] = {}
        for min_n in (3, 5, 7):
            hit = _first_excess_ngram(ref, hyp, min_n)
            if hit is None:
                onsets[f"min_n{min_n}"] = None
                continue
            idx, n, gram = hit
            # reference position the model had reached at that hypothesis word
            ref_pos = None
            for op in al.ops:
                if op["hyp_start"] <= idx < op["hyp_end"]:
                    ref_pos = op["ref_start"] + (
                        idx - op["hyp_start"] if op["op"] in ("equal", "substitute") else 0
                    )
                    break
            onsets[f"min_n{min_n}"] = {
                "hyp_word_index": idx,
                "n": n,
                "gram": gram,
                "time_sec": times[idx],
                "frac_of_audio": times[idx] / row["raw_duration_sec"]
                if row["raw_duration_sec"]
                else None,
                "ref_word_index": ref_pos,
                "ref_frac": (ref_pos / len(ref)) if (ref_pos is not None and ref) else None,
            }
        manifest = runs[label][row["run_id"]]
        items.append(
            {
                "run_id": row["run_id"],
                "text_id": row["text_id"],
                "root_id": item["root_id"],
                "source": item["source"],
                "genre": item["genre"],
                "voice_id": row["voice_id"],
                "bucket": row["bucket"],
                "status": row["status"],
                "final_status_reason": row["final_status_reason"],
                "wer": row["wer"],
                "wer_minus_floor": row["wer_minus_floor"],
                "floor_available": row["floor_available"],
                "source_coverage": row["source_coverage"],
                "end_coverage": row["end_coverage"],
                "end_coverage_robust": row["end_coverage_robust"],
                "excess_repetition_rate": row["excess_repetition_rate"],
                "excess_repetition_per_n": row["excess_repetition_per_n"],
                "max_ngram_run": row["max_ngram_run"],
                "max_ngram_run_n": row["max_ngram_run_n"],
                "max_ngram_run_gram": row["max_ngram_run_gram"],
                "max_excess_ngram_count": row["max_excess_ngram_count"],
                "max_excess_ngram_count_n": row["max_excess_ngram_count_n"],
                "max_excess_ngram_count_gram": row["max_excess_ngram_count_gram"],
                "loop_flag": row["loop_flag"],
                "loop_reason": row["loop_reason"],
                "raw_duration_sec": row["raw_duration_sec"],
                "human_duration_sec": item["human_duration_sec"],
                "duration_ratio": row["duration_ratio"],
                "n_ref_words": row["n_ref_words"],
                "n_hyp_words": row["n_hyp_words"],
                "insertions": row["insertions"],
                "substitutions": row["substitutions"],
                "deletions": row["deletions"],
                "generated_speech_tokens": manifest["generated_speech_tokens"],
                "context_occupancy": manifest.get("context_occupancy"),
                "stop_reason": manifest["stop_reason"],
                "onsets": onsets,
            }
        )
    items.sort(key=lambda d: (d["status"], d["bucket"], d["run_id"]))

    deg = [i for i in items if i["status"] == "degraded"]
    eos = [i for i in items if i["status"] == "early_eos"]
    stable = all(
        i["onsets"]["min_n3"] is not None
        and i["onsets"]["min_n5"] is not None
        and i["onsets"]["min_n7"] is not None
        and i["onsets"]["min_n3"]["hyp_word_index"]
        == i["onsets"]["min_n5"]["hyp_word_index"]
        == i["onsets"]["min_n7"]["hyp_word_index"]
        for i in deg
    )
    return {
        "items": items,
        "n_degraded": len(deg),
        "n_early_eos": len(eos),
        "onset_index_identical_for_min_n_3_5_7_on_degraded": stable,
        "degraded_summary": {
            "duration_ratio": _stats([i["duration_ratio"] for i in deg]),
            "excess_repetition_rate": _stats([i["excess_repetition_rate"] for i in deg]),
            "source_coverage": _stats([i["source_coverage"] for i in deg]),
            "onset_frac_of_audio": _stats(
                [
                    i["onsets"]["min_n3"]["frac_of_audio"]
                    for i in deg
                    if i["onsets"]["min_n3"]
                ]
            ),
            "onset_time_sec": _stats(
                [i["onsets"]["min_n3"]["time_sec"] for i in deg if i["onsets"]["min_n3"]]
            ),
            "onset_ref_frac": _stats(
                [
                    i["onsets"]["min_n3"]["ref_frac"]
                    for i in deg
                    if i["onsets"]["min_n3"] and i["onsets"]["min_n3"]["ref_frac"] is not None
                ]
            ),
            "n_loop_flag": sum(1 for i in deg if i["loop_flag"]),
            "n_dur_ratio_gt_1": sum(1 for i in deg if i["duration_ratio"] > 1.0),
        },
        "early_eos_summary": {
            "duration_ratio": _stats([i["duration_ratio"] for i in eos]),
            "source_coverage": _stats([i["source_coverage"] for i in eos]),
            "excess_repetition_rate": _stats([i["excess_repetition_rate"] for i in eos]),
            "end_coverage": _stats([i["end_coverage"] for i in eos]),
            "n_external_roots": sum(1 for i in eos if i["source"] == "external"),
        },
    }


# ------------------------------------------------------ status transitions


def status_transitions(per_item: dict[str, list[dict]]) -> dict:
    """Paired item-by-item movement of the §3.4 final status, E1 -> E2 and E1 -> E3."""
    base = {r["run_id"]: r for r in per_item["E1_native"]}
    out: dict[str, Any] = {}
    for target in ("E2_epoch_3_step_3001", "E3_epoch_3_step_3001"):
        tgt = {r["run_id"]: r for r in per_item[target]}
        if set(tgt) != set(base):
            raise SystemExit(f"run_id sets differ between E1_native and {target}")
        matrix: Counter = Counter(
            (base[k]["status"], tgt[k]["status"]) for k in base
        )
        by_bucket = {}
        for bucket in BUCKETS:
            keys = [k for k in base if base[k]["bucket"] == bucket]
            by_bucket[bucket] = {
                "n": len(keys),
                "n_complete_E1": sum(1 for k in keys if base[k]["status"] == "complete"),
                "n_complete_target": sum(1 for k in keys if tgt[k]["status"] == "complete"),
                "incomplete_to_complete": sum(
                    1
                    for k in keys
                    if base[k]["status"] != "complete" and tgt[k]["status"] == "complete"
                ),
                "complete_to_incomplete": sum(
                    1
                    for k in keys
                    if base[k]["status"] == "complete" and tgt[k]["status"] != "complete"
                ),
            }
        out[f"E1_to_{target}"] = {
            "matrix": {f"{a} -> {b}": n for (a, b), n in sorted(matrix.items(), key=lambda x: -x[1])},
            "by_bucket": by_bucket,
        }
    return out


# ------------------------------------------------ noise-buffer exposure, floor


def audio_and_floor_exposure(per_item: dict[str, list[dict]]) -> dict:
    """How far each checkpoint pushed the 300 s stock noise buffer, and where the
    WER branch of the status rule could not be applied (no ASR floor)."""
    out: dict[str, dict] = {}
    for label, rows in per_item.items():
        no_floor = [r for r in rows if not r["floor_available"]]
        complete = [r for r in rows if r["status"] == "complete"]
        out[label] = {
            "n_outputs_over_300s": sum(1 for r in rows if r["raw_duration_sec"] > 300.0),
            "max_output_sec": max(r["raw_duration_sec"] for r in rows),
            "n_without_floor": len(no_floor),
            "n_wer_branch_skipped": sum(
                1 for r in rows if r.get("final_status_note") == "no_floor_wer_criterion_skipped"
            ),
            "n_complete_without_floor_and_wer_ge_0_30": sum(
                1 for r in complete if not r["floor_available"] and r["wer"] >= 0.30
            ),
            "n_complete_with_repeat_ge_0_05": sum(
                1 for r in complete if r["excess_repetition_rate"] >= 0.05
            ),
            "complete_with_repeat_ge_0_05": [
                {
                    "text_id": r["text_id"],
                    "voice_id": r["voice_id"],
                    "bucket": r["bucket"],
                    "excess_repetition_rate": r["excess_repetition_rate"],
                }
                for r in complete
                if r["excess_repetition_rate"] >= 0.05
            ],
        }
    return out


# --------------------------------------------------------- short-form detail


def short_form_items(per_item: dict[str, list[dict]]) -> dict:
    """Every B0 and B1 item of the selected checkpoints, for the Table 6 discussion."""
    out: dict[str, list[dict]] = {}
    for label in SELECTED:
        out[label] = [
            {
                "text_id": r["text_id"],
                "voice_id": r["voice_id"],
                "bucket": r["bucket"],
                "status": r["status"],
                "wer": r["wer"],
                "cer": r["cer"],
                "source_coverage": r["source_coverage"],
                "floor_available": r["floor_available"],
                "wer_minus_floor": r["wer_minus_floor"],
                "duration_ratio": r["duration_ratio"],
            }
            for r in sorted(per_item[label], key=lambda x: (x["bucket"], -x["wer"]))
            if r["bucket"] in ("B0", "B1")
        ]
    return out


# ------------------------------------------------------------- E2 vs E1 idem


def e2_vs_e1(per_item: dict[str, list[dict]]) -> dict:
    a = {r["run_id"]: r for r in per_item["E2_epoch_3_step_3001"]}
    b = {r["run_id"]: r for r in per_item["E1_native"]}
    if set(a) != set(b):
        raise SystemExit("E2/E1 run_id sets differ")
    same_status = sum(1 for k in a if a[k]["status"] == b[k]["status"])
    both_incomplete_b2plus = sum(
        1
        for k in a
        if a[k]["bucket"] in ("B2", "B3", "B4")
        and a[k]["status"] != "complete"
        and b[k]["status"] != "complete"
    )
    return {
        "n_pairs": len(a),
        "same_final_status": same_status,
        "b2plus_both_incomplete": both_incomplete_b2plus,
        "b2plus_n": sum(1 for k in a if a[k]["bucket"] in ("B2", "B3", "B4")),
        "delta_coverage": _stats([a[k]["source_coverage"] - b[k]["source_coverage"] for k in a]),
        "delta_wer": _stats([a[k]["wer"] - b[k]["wer"] for k in a]),
        "delta_duration_sec": _stats(
            [a[k]["raw_duration_sec"] - b[k]["raw_duration_sec"] for k in a]
        ),
        "delta_duration_ratio": _stats([a[k]["duration_ratio"] - b[k]["duration_ratio"] for k in a]),
        "n_complete_E2": sum(1 for k in a if a[k]["status"] == "complete"),
        "n_complete_E1": sum(1 for k in b if b[k]["status"] == "complete"),
        "max_generated_sec_E2": max(a[k]["raw_duration_sec"] for k in a),
        "max_generated_sec_E1": max(b[k]["raw_duration_sec"] for k in b),
        "n_coverage_ge_0_95_E2_b2plus": sum(
            1 for k in a if a[k]["bucket"] in ("B2", "B3", "B4") and a[k]["source_coverage"] >= 0.95
        ),
        "n_coverage_ge_0_95_E1_b2plus": sum(
            1 for k in b if b[k]["bucket"] in ("B2", "B3", "B4") and b[k]["source_coverage"] >= 0.95
        ),
    }


# ------------------------------------------------------- checkpoint-gap view


def checkpoint_gap(table3: dict, table5: dict) -> dict:
    out = {}
    for arm, loss_ckpt, last_ckpt in (
        ("E3", "E3_epoch_0_whole", "E3_epoch_3_step_3001"),
        ("E2", "E2_epoch_0_step_200", "E2_epoch_3_step_3001"),
    ):
        out[arm] = {
            bucket: {
                "complete_by_loss": table3[loss_ckpt][bucket]["complete_rate"],
                "complete_last": table3[last_ckpt][bucket]["complete_rate"],
                "delta_complete": table3[last_ckpt][bucket]["complete_rate"]
                - table3[loss_ckpt][bucket]["complete_rate"],
                "wer_by_loss": table3[loss_ckpt][bucket]["wer"],
                "wer_last": table3[last_ckpt][bucket]["wer"],
                "coverage_by_loss": table3[loss_ckpt][bucket]["coverage"],
                "coverage_last": table3[last_ckpt][bucket]["coverage"],
                "repeat_by_loss": table3[loss_ckpt][bucket]["repeat"],
                "repeat_last": table3[last_ckpt][bucket]["repeat"],
                "degraded_by_loss": table5[loss_ckpt][bucket]["degraded"],
                "degraded_last": table5[last_ckpt][bucket]["degraded"],
            }
            for bucket in (*BUCKETS, "ALL")
        }
    return out


# ------------------------------------------------------------------ by voice


def per_voice(per_item: dict[str, list[dict]]) -> dict:
    out: dict[str, dict] = {}
    for label, rows in per_item.items():
        out[label] = {}
        for voice in sorted({r["voice_id"] for r in rows}):
            sel = [r for r in rows if r["voice_id"] == voice]
            out[label][voice] = {
                "n": len(sel),
                "complete_rate": _mean(1.0 if r["status"] == "complete" else 0.0 for r in sel),
                "wer": _mean(r["wer"] for r in sel),
                "coverage": _mean(r["source_coverage"] for r in sel),
                "repeat": _mean(r["excess_repetition_rate"] for r in sel),
                "dur_ratio": _mean(r["duration_ratio"] for r in sel),
                "generated_sec_median": _median(r["raw_duration_sec"] for r in sel),
            }
        # paired female - male on the same text
        by_text = defaultdict(dict)
        for r in rows:
            by_text[r["text_id"]][r["voice_id"]] = r
        pairs = [v for v in by_text.values() if len(v) == 2]
        f, m = "ref_female_01", "ref_male_01"
        out[label]["_paired_female_minus_male"] = {
            "n_pairs": len(pairs),
            "d_duration_sec": _stats([p[f]["raw_duration_sec"] - p[m]["raw_duration_sec"] for p in pairs]),
            "d_coverage": _stats([p[f]["source_coverage"] - p[m]["source_coverage"] for p in pairs]),
            "d_wer": _stats([p[f]["wer"] - p[m]["wer"] for p in pairs]),
            "n_female_complete": sum(1 for p in pairs if p[f]["status"] == "complete"),
            "n_male_complete": sum(1 for p in pairs if p[m]["status"] == "complete"),
        }
    return out


# ------------------------------------------------------- dataset vs external


def by_source(bench: dict, per_item: dict[str, list[dict]]) -> dict:
    out: dict[str, dict] = {}
    for label, rows in per_item.items():
        out[label] = {}
        for src in ("dataset", "external"):
            sel = [r for r in rows if bench[r["text_id"]]["source"] == src]
            out[label][src] = {
                "n": len(sel),
                "n_roots": len({bench[r["text_id"]]["root_id"] for r in sel}),
                "complete_rate": _mean(1.0 if r["status"] == "complete" else 0.0 for r in sel),
                "wer": _mean(r["wer"] for r in sel),
                "coverage": _mean(r["source_coverage"] for r in sel),
                "repeat": _mean(r["excess_repetition_rate"] for r in sel),
                "dur_ratio": _mean(r["duration_ratio"] for r in sel),
            }
        # per-root completion for the long buckets
        out[label]["_per_root_b2plus_complete"] = {}
        for root in sorted({bench[r["text_id"]]["root_id"] for r in rows}):
            sel = [
                r
                for r in rows
                if bench[r["text_id"]]["root_id"] == root and r["bucket"] in ("B2", "B3", "B4")
            ]
            out[label]["_per_root_b2plus_complete"][root] = {
                "n": len(sel),
                "n_complete": sum(1 for r in sel if r["status"] == "complete"),
                "coverage": _mean(r["source_coverage"] for r in sel),
            }
    return out


# -------------------------------------------------------------- system table


def system_table(runs: dict) -> dict:
    out: dict[str, dict] = {}
    for label, manifests in runs.items():
        rows = list(manifests.values())
        out[label] = {}
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for m in rows:
            by_bucket[m["text_id"].rsplit("__", 1)[-1]].append(m)
        for bucket in (*BUCKETS, "ALL"):
            sel = rows if bucket == "ALL" else by_bucket[bucket]
            # `context_occupancy` is absent from every official-split manifest (each of its
            # 2-34 calls has its own short context).  A missing key must be reported as
            # n/a, not as a measured 0.0.
            occ = [m["context_occupancy"] for m in sel if m.get("context_occupancy") is not None]
            tok_total = sum(m["generated_speech_tokens"] for m in sel)
            wall_total = sum(m["wall_time_sec"] for m in sel)
            out[label][bucket] = {
                "n": len(sel),
                "generated_speech_tokens": _stats([m["generated_speech_tokens"] for m in sel]),
                "generated_speech_tokens_total": tok_total,
                "wall_time_sec_total": sum(m["wall_time_sec"] for m in sel),
                "wall_time_sec_median": _median(m["wall_time_sec"] for m in sel),
                "rtf_median": _median(m["rtf"] for m in sel),
                "peak_vram_reserved_gib": max(m["peak_vram_reserved_bytes"] for m in sel) / 2**30,
                "context_occupancy_max": max(occ) if occ else None,
                "n_with_context_occupancy": len(occ),
                "stop_reasons": dict(Counter(m["stop_reason"] for m in sel)),
                "n_at_cap": sum(
                    1
                    for m in sel
                    if m.get("max_len_cap") and m["generated_speech_tokens"] >= m["max_len_cap"]
                ),
                "tok_per_sec_wall_median": _median(
                    m["generated_speech_tokens"] / m["wall_time_sec"] for m in sel
                ),
                # aggregate = total tokens / total wall.  The per-item median is dominated
                # by short items and cannot carry a decomposition of total wall time.
                "tok_per_sec_wall_aggregate": tok_total / wall_total,
            }
    return out


# ----------------------------------------------------------- training budget


CV_RE = re.compile(
    r"Epoch (?P<epoch>\d+) Step (?P<step>\d+) CV info lr \S+ 0 rank "
    r"loss (?P<loss>[\d.eE+-]+) acc (?P<acc>[\d.eE+-]+)"
)
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),")


def _cv_curve(log_path: Path) -> list[dict]:
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = CV_RE.search(line)
        if m:
            out.append(
                {
                    "epoch": int(m["epoch"]),
                    "step": int(m["step"]),
                    "loss": float(m["loss"]),
                    "acc": float(m["acc"]),
                }
            )
    return out


def _log_span_hours(log_path: Path) -> tuple[str, str, float]:
    stamps = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TS_RE.match(line)
        if m:
            stamps.append(m.group(1))
    if not stamps:
        raise SystemExit(f"no timestamps in {log_path}")
    fmt = "%Y-%m-%d %H:%M:%S"
    t0 = datetime.strptime(stamps[0], fmt)
    t1 = datetime.strptime(stamps[-1], fmt)
    return stamps[0], stamps[-1], (t1 - t0).total_seconds() / 3600.0


def training_budget() -> dict:
    out: dict[str, Any] = {"arms": {}}
    for arm, exp_dir, log_name in (
        ("long", "exp/long/pilot", "train_long.log"),
        ("short", "exp/short/pilot", "train_short.log"),
    ):
        stats = _load_jsonl(REPO / exp_dir / "train_stats.jsonl")
        info = json.loads((REPO / exp_dir / "run_info.json").read_text())
        verify = json.loads((REPO / exp_dir / "budget_verify_step200.json").read_text())["runs"][0]
        log = REPO / "logs/v31_sft" / log_name
        t0, t1, hours = _log_span_hours(log)
        cv = _cv_curve(log)
        tok = [r["speech_tokens"] for r in stats]
        step_times = [r["step_time_sec"] for r in stats if r["step_time_sec"]]
        best = min(cv, key=lambda r: r["loss"])
        out["arms"][arm] = {
            "steps": len(stats),
            "epochs_seen": max(r["epoch"] for r in stats) + 1,
            "target_tokens_total": sum(tok),
            "target_tokens_per_step_mean": sum(tok) / len(tok),
            "target_tokens_per_step_median": st.median(tok),
            "target_tokens_per_step_min": min(tok),
            "target_tokens_per_step_max": max(tok),
            "target_tokens_per_step_sd": st.pstdev(tok),
            "samples_per_step_mean": _mean(r["samples"] for r in stats),
            "step_time_sec_median": st.median(step_times),
            "step_time_sum_hours": sum(step_times) / 3600.0,
            "wall_hours_log_span": hours,
            "log_first_ts": t0,
            "log_last_ts": t1,
            "peak_vram_reserved_gb_max": max(r["peak_vram_reserved_gb"] for r in stats),
            "train_loss_first": stats[0]["loss"],
            "train_loss_last": stats[-1]["loss"],
            "cv_curve": cv,
            "cv_best": best,
            "cv_last": cv[-1],
            "verify_step200": verify,
            "config": info["args"]["config"],
            "config_sha256": info["config_sha256"],
            "train_data": info["args"]["train_data"],
            "train_data_sha256": info["train_data_sha256"],
            "cv_data": info["args"]["cv_data"],
            "cv_data_sha256": info["cv_data_sha256"],
            "init_checkpoint_sha256": info["init_checkpoint_sha256"],
            "n_trainable": info["n_trainable"],
            "start_time": info["start_time"],
            "lr": info["train_conf"]["optim_conf"]["lr"],
        }
        arm_stats = json.loads((REPO / f"data/train/{arm}/stats.json").read_text())
        out["arms"][arm]["arm_data"] = arm_stats
    lo = out["arms"]["long"]
    sh = out["arms"]["short"]
    out["ratio_step200_mean"] = (
        lo["verify_step200"]["target_tokens_mean"] / sh["verify_step200"]["target_tokens_mean"]
    )
    out["ratio_full_run_mean"] = (
        lo["target_tokens_per_step_mean"] / sh["target_tokens_per_step_mean"]
    )
    out["ratio_full_run_total"] = lo["target_tokens_total"] / sh["target_tokens_total"]
    out["cv_data_identical"] = lo["cv_data_sha256"] == sh["cv_data_sha256"]
    out["config_identical"] = lo["config_sha256"] == sh["config_sha256"]
    out["init_identical"] = lo["init_checkpoint_sha256"] == sh["init_checkpoint_sha256"]
    return out


# ------------------------------------------------------------- H4 margin


# reports/decisions.md, Lead 2026-08-30: "ПРЕРЕГИСТРАЦИЯ H4 (до hidden): non-inferiority
# margin short-form = B0 WER-floor SFT-модели <= baseline B0 WER-floor + 2.0 пп И
# B0 Complete >= baseline - 10 пп".  Pre-registered BEFORE the hidden run; applied here to
# the pilot's dev numbers, which is what the ruling itself does.
H4_BASELINE = "E1_native"
H4_WER_FLOOR_MARGIN_PP = 2.0
H4_COMPLETE_MARGIN_PP = 10.0
H4_CONTROLS = ("E0_official",)  # production control, not an SFT model (PLAN §5)


def h4_margin(table3: dict) -> dict:
    """Apply the Lead's pre-registered H4 short-form non-inferiority margin, per checkpoint."""
    base = table3[H4_BASELINE]["B0"]
    base_wf = 100.0 * base["wer_minus_floor"]
    base_cp = 100.0 * base["complete_rate"]
    out: dict[str, Any] = {
        "rule": (
            "B0 WER-floor <= baseline B0 WER-floor + 2.0 pp AND B0 Complete >= baseline - 10 pp"
        ),
        "source": "reports/decisions.md, Lead 2026-08-30 (pre-registered before the hidden run)",
        "baseline": H4_BASELINE,
        "baseline_b0_wer_minus_floor_pp": base_wf,
        "baseline_b0_complete_pp": base_cp,
        "wer_floor_margin_pp": H4_WER_FLOOR_MARGIN_PP,
        "complete_margin_pp": H4_COMPLETE_MARGIN_PP,
        "wer_floor_ceiling_pp": base_wf + H4_WER_FLOOR_MARGIN_PP,
        "complete_floor_pp": base_cp - H4_COMPLETE_MARGIN_PP,
        "checkpoints": {},
    }
    for label in ORDER:
        c = table3[label]["B0"]
        wf = 100.0 * c["wer_minus_floor"]
        cp = 100.0 * c["complete_rate"]
        wf_ok = wf <= out["wer_floor_ceiling_pp"] + 1e-9
        cp_ok = cp >= out["complete_floor_pp"] - 1e-9
        out["checkpoints"][label] = {
            "b0_wer_minus_floor_pp": wf,
            "b0_complete_pp": cp,
            "headroom_wer_floor_pp": out["wer_floor_ceiling_pp"] - wf,
            "headroom_complete_pp": cp - out["complete_floor_pp"],
            "wer_floor_ok": wf_ok,
            "complete_ok": cp_ok,
            "passes": wf_ok and cp_ok,
            "is_baseline": label == H4_BASELINE,
            "is_control": label in H4_CONTROLS,
        }
    out["n_pass"] = sum(
        1
        for k, v in out["checkpoints"].items()
        if v["passes"] and not v["is_baseline"] and not v["is_control"]
    )
    out["n_fail"] = sum(
        1
        for k, v in out["checkpoints"].items()
        if not v["passes"] and not v["is_baseline"] and not v["is_control"]
    )
    return out


# ------------------------------------------------- E4 curriculum (3 stages)


CURRICULUM_STAGES = (
    # (stage, exp dir, log, pre-registered ceiling in seconds, pre-registered max steps)
    ("S1", "exp/curriculum/s1", "train_s1.log", 90, 500),
    ("S2", "exp/curriculum/s2", "train_s2.log", 180, 500),
    ("S3", "exp/curriculum/s3", "train_s3.log", 900, 2000),
)


def curriculum_budget() -> dict | None:
    """The three E4 stages as they actually ran, from their own files.

    Separate from :func:`training_budget` on purpose: that function's output is quoted in
    Table 7 and nothing here may move one of its numbers.  Each stage reports its own step
    count, target-token total, wall clock, VRAM peak and full dev CV curve, plus the
    checkpoint it was initialised from (which is what makes the chain auditable: S1 must
    start from E2's SELECTED checkpoint per the Lead's amendment of 2026-08-30).

    The CV losses of all three stages ARE comparable to each other and to E3/E2's long-arm
    CV, because `duration_filter` is inactive on the dev pass in every stage (`active=False`,
    120/120 units kept; `reports/training_budget_comparison.md` §8.1, PLAN §11.5).
    """
    out: dict[str, Any] = {"stages": {}}
    for stage, exp_dir, log_name, ceiling, preregistered_steps in CURRICULUM_STAGES:
        d = REPO / exp_dir
        log = REPO / "logs/v31_curr" / log_name
        if not (d / "train_stats.jsonl").exists() or not log.exists():
            return None
        stats = _load_jsonl(d / "train_stats.jsonl")
        info = json.loads((d / "run_info.json").read_text())
        t0, t1, hours = _log_span_hours(log)
        cv = _cv_curve(log)
        tok = [r["speech_tokens"] for r in stats]
        step_times = [r["step_time_sec"] for r in stats if r["step_time_sec"]]
        best = min(cv, key=lambda r: r["loss"])
        out["stages"][stage] = {
            "exp_dir": exp_dir,
            "log": f"logs/v31_curr/{log_name}",
            "max_duration_sec": ceiling,
            "steps_preregistered": preregistered_steps,
            "steps": len(stats),
            "epochs_seen": max(r["epoch"] for r in stats) + 1,
            "target_tokens_total": sum(tok),
            "target_tokens_per_step_mean": sum(tok) / len(tok),
            "samples_per_step_mean": _mean(r["samples"] for r in stats),
            "step_time_sec_median": st.median(step_times),
            "wall_hours_log_span": hours,
            "log_first_ts": t0,
            "log_last_ts": t1,
            "peak_vram_reserved_gb_max": max(r["peak_vram_reserved_gb"] for r in stats),
            "train_loss_first": stats[0]["loss"],
            "train_loss_last": stats[-1]["loss"],
            "cv_curve": cv,
            "cv_best": best,
            "cv_last": cv[-1],
            "config": info["args"]["config"],
            "config_sha256": info["config_sha256"],
            "init_checkpoint": info["args"]["checkpoint"],
            "init_checkpoint_sha256": info["init_checkpoint_sha256"],
            "cv_data": info["args"]["cv_data"],
            "cv_data_sha256": info["cv_data_sha256"],
            "train_data": info["args"]["train_data"],
            "n_trainable": info["n_trainable"],
            "lr": info["train_conf"]["optim_conf"]["lr"],
        }
    stages = out["stages"]
    out["steps_total"] = sum(v["steps"] for v in stages.values())
    out["target_tokens_total"] = sum(v["target_tokens_total"] for v in stages.values())
    out["wall_hours_total"] = sum(v["wall_hours_log_span"] for v in stages.values())
    # the chain: S1 <- E2 selected, S2 <- S1 last, S3 <- S2 last
    out["init_chain"] = [stages[s]["init_checkpoint"] for s, *_ in CURRICULUM_STAGES]
    out["s1_init_is_e2_selected"] = stages["S1"]["init_checkpoint"].endswith(
        "exp/short/pilot/epoch_3_step_3001.pt"
    )
    # every stage sees the same unfiltered dev set, so the CV losses are on one scale
    out["cv_data_identical_across_stages"] = (
        len({v["cv_data_sha256"] for v in stages.values()}) == 1
    )
    out["cv_best_overall"] = min(
        ((s, v["cv_best"]) for s, v in stages.items()), key=lambda kv: kv[1]["loss"]
    )[0]
    return out


# ---------------------------------------------------------------------- RQ6


def _quantile_interp(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the numpy/`percentile` default)."""
    n = len(sorted_values)
    h = q * (n - 1)
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def rq6_length_coverage(bench: dict) -> dict:
    rows = _load_jsonl(REPO / "data/train/long/token_manifest.jsonl")
    durations = sorted(r["duration"] for r in rows)
    n = len(durations)
    total_h = sum(durations) / 3600.0
    bands: dict[str, dict] = {}
    for lo, hi, name in (
        (20, 40, "B0 20-40 s"),
        (60, 90, "B1 60-90 s"),
        (120, 180, "B2 2-3 min"),
        (240, 360, "B3 4-6 min"),
        (480, 720, "B4 8-12 min"),
    ):
        sel = [d for d in durations if lo <= d <= hi]
        bands[name] = {
            "n_units": len(sel),
            "share_units": len(sel) / n,
            "hours": sum(sel) / 3600.0,
            "share_hours": (sum(sel) / 3600.0) / total_h,
        }
    tails = {}
    for thr in (90, 180, 300, 480, 600, 720, 840, 900):
        sel = [d for d in durations if d >= thr]
        tails[f">= {thr} s"] = {
            "n_units": len(sel),
            "share_units": len(sel) / n,
            "hours": sum(sel) / 3600.0,
        }
    human = defaultdict(list)
    for item in bench.values():
        human[item["bucket"]].append(item["human_duration_sec"])
    return {
        "n_units": n,
        "hours": total_h,
        "duration_sec": {
            "min": durations[0],
            "median": st.median(durations),
            "p90": durations[int(0.90 * (n - 1))],
            # two definitions, both published: a re-deriving reader must get the same
            # number, and these two differ by 1.8 s on this sample.
            "p99_order_stat": durations[int(0.99 * (n - 1))],
            "p99": durations[int(0.99 * (n - 1))],
            "p99_interp": _quantile_interp(durations, 0.99),
            "max": durations[-1],
        },
        "bucket_bands": bands,
        "tails": tails,
        "benchmark_human_duration_sec": {
            b: _stats(human[b]) for b in BUCKETS
        },
        "max_training_unit_sec": durations[-1],
        "b4_inside_training_range": max(human["B4"]) <= durations[-1],
    }


# --------------------------------------------------- dev packing (the CV sets)


def dev_packing() -> dict:
    """The two CV sets as they were actually built, from their own files.

    The long arm trains on whole units and the short arm on 10-30 s windows cut from the
    same audio, so the two `cv_data` lists hold the same speech packed differently.  Both
    numbers come from `data/train/dev_{long,short}/` -- never from a prose line in another
    report (that is exactly how the pre-rebuild 114/1804/10.68 h/960 727 values survived).
    """
    out: dict[str, dict] = {}
    for arm, path in (("long", "data/train/dev_long"), ("short", "data/train/dev_short")):
        d = REPO / path
        stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
        rows = _load_jsonl(d / "token_manifest.jsonl")
        out[arm] = {
            "dir": path,
            "kind": stats["kind"],
            "units": len(rows),
            "units_stats_json": stats["utts"],
            "parents": stats["parents"],
            "speakers": stats["speakers"],
            "target_speech_tokens": sum(r["n_speech_token"] for r in rows),
            "hours_token_manifest": sum(r["duration"] for r in rows) / 3600.0,
            "hours_stats_json": stats["hours"],
        }
        if out[arm]["units"] != stats["utts"]:
            raise SystemExit(
                f"dev_packing: {path} token_manifest has {out[arm]['units']} rows but "
                f"stats.json says {stats['utts']} utts"
            )
    out["tokens_equal"] = (
        out["long"]["target_speech_tokens"] == out["short"]["target_speech_tokens"]
    )
    out["hours_equal_to_1e6"] = (
        abs(out["long"]["hours_token_manifest"] - out["short"]["hours_token_manifest"]) < 1e-6
    )
    return out


# ------------------------------------------------------------- voice drift (A5)


DRIFT_ROOT = REPO / "results/v31_drift"
DRIFT_ARMS = {
    "E1_native": "E1_native",
    "E0_official": "E0_official",
    "E2_epoch_3_step_3001": "E2_epoch_3_step_3001",
    "E3_epoch_3_step_3001": "E3_epoch_3_step_3001",
    "E4_epoch_0_whole": "E4_epoch_0_whole",
    "E4_epoch_2_step_2001": "E4_epoch_2_step_2001",
}
DRIFT_SPAN_MIN_SEC = 60.0
# voiced-span bands for the length-matched similarity level (see voice_drift())
DRIFT_SPAN_BANDS = (
    (0.0, 60.0),
    (60.0, 150.0),
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, float("inf")),
)


def _drift_span_sec(row: dict) -> float:
    w = row.get("windows") or []
    if len(w) < 2:
        return 0.0
    return max(x["t_end_sec"] for x in w) - min(x["t_start_sec"] for x in w)


def voice_drift() -> dict | None:
    """PLAN §9.5 voice drift, re-aggregated from A5's per-item rows.

    A5 owns the measurement (`src/eval/speaker_drift.py`, encoder and threshold frozen in
    `configs/speaker_drift.yaml`); this function only re-derives the per-bucket medians and
    the paired first-window (onset) deltas from `results/v31_drift/<exp>/per_item.jsonl`,
    and cross-checks them against A5's own `voice_drift_column.csv`.  A disagreement is a
    hard failure -- the same rule as the Table 3 / main-table cross-check.
    """
    if not DRIFT_ROOT.is_dir():
        return None
    per_exp: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    for label, exp in DRIFT_ARMS.items():
        f = DRIFT_ROOT / exp / "per_item.jsonl"
        if not f.exists():
            return None
        rows = _load_jsonl(f)
        meta[label] = rows[0] if rows and rows[0].get("_meta") else {}
        per_exp[label] = [r for r in rows if not r.get("_meta")]

    out: dict[str, Any] = {
        "source": "results/v31_drift/<exp>/per_item.jsonl (A5)",
        "owner": "A5",
        "encoder": meta["E3_epoch_3_step_3001"].get("encoder"),
        "encoder_revision": meta["E3_epoch_3_step_3001"].get("revision"),
        "encoder_sha256": meta["E3_epoch_3_step_3001"].get("model_sha256"),
        "window_sec": meta["E3_epoch_3_step_3001"].get("window_sec"),
        "hop_sec": meta["E3_epoch_3_step_3001"].get("hop_sec"),
        "t_voice_similarity_min": meta["E3_epoch_3_step_3001"].get("t_voice_similarity_min"),
        "t_voice_min_run": meta["E3_epoch_3_step_3001"].get("t_voice_min_run"),
        "span_min_sec": DRIFT_SPAN_MIN_SEC,
        "by_bucket": {},
        "onset_paired": {},
        "t_voice_fired_total": 0,
        "n_items_total": 0,
    }
    for label, rows in per_exp.items():
        out["by_bucket"][label] = {}
        for b in (*BUCKETS, "ALL"):
            sel = rows if b == "ALL" else [r for r in rows if r["bucket"] == b]
            w = [r for r in sel if (r.get("n_windows") or 0) > 0]
            lng = [r for r in w if _drift_span_sec(r) >= DRIFT_SPAN_MIN_SEC]
            fired = [r for r in w if r.get("t_voice_sec") is not None]
            out["by_bucket"][label][b] = {
                "n": len(sel),
                "n_with_windows": len(w),
                "sim_level_median": _median(r["sim_median"] for r in w) if w else None,
                "delta_median": _median(r["delta"] for r in w) if w else None,
                "span_sec_median": _median(_drift_span_sec(r) for r in w) if w else None,
                "n_span_ge_60s": len(lng),
                "slope_per_min_median": (
                    _median(r["slope_per_min"] for r in lng) if lng else None
                ),
                "t_voice_fired": len(fired),
            }
            if b == "ALL":
                out["t_voice_fired_total"] += len(fired)
                out["n_items_total"] += len(sel)
        # paired first-window similarity across the nested prefixes
        idx = {}
        for r in rows:
            if r.get("windows"):
                root = r["text_id"].rsplit("__", 1)[0]
                idx[(root, r["voice_id"], r["bucket"])] = r["windows"][0]["sim"]
        out["onset_paired"][label] = {}
        for b in BUCKETS[1:]:
            pairs = [
                (idx[(k[0], k[1], "B0")], idx[k])
                for k in idx
                if k[2] == b and (k[0], k[1], "B0") in idx
            ]
            d = [y - x for x, y in pairs]
            out["onset_paired"][label][f"B0->{b}"] = {
                "pairs": len(pairs),
                "b0_median": _median(x for x, _ in pairs) if pairs else None,
                "bk_median": _median(y for _, y in pairs) if pairs else None,
                "delta_median": _median(d) if d else None,
                "n_negative": sum(1 for x in d if x < 0),
            }
    # Length-matched levels.  Raw `Spk sim` per BUCKET is confounded with how much audio the
    # checkpoint actually produced: a checkpoint that stops early is scored on a shorter,
    # earlier stretch of its own output.  Binning the same items by their VOICED SPAN instead
    # removes that confound (A5's caveat 1 on the E4 handoff), at the cost of unequal ns.
    # How close each checkpoint came to the frozen T_voice threshold, over EVERY window
    # rather than per item: `T_voice` fires on 3 consecutive windows below 0.50, so the
    # single lowest window and the count below 0.60 say how much margin there was.
    out["window_margin"] = {}
    for label, rows in per_exp.items():
        sims = [w["sim"] for r in rows for w in (r.get("windows") or [])]
        out["window_margin"][label] = {
            "n_windows": len(sims),
            "min_window_sim": min(sims) if sims else None,
            "n_windows_below_0_60": sum(1 for x in sims if x < 0.60),
            "n_windows_below_threshold": sum(
                1 for x in sims if x < (out["t_voice_similarity_min"] or 0.5)
            ),
        }

    out["span_bands_sec"] = [list(b) for b in DRIFT_SPAN_BANDS]
    out["by_span_band"] = {}
    for label, rows in per_exp.items():
        w = [r for r in rows if (r.get("n_windows") or 0) > 0]
        out["by_span_band"][label] = {}
        for lo, hi in DRIFT_SPAN_BANDS:
            sel = [r for r in w if lo <= _drift_span_sec(r) < hi]
            name = f"{lo:g}-{hi:g}" if hi != float("inf") else f"{lo:g}+"
            out["by_span_band"][label][name] = {
                "n": len(sel),
                "sim_level_median": _median(r["sim_median"] for r in sel) if sel else None,
            }

    cal_path = DRIFT_ROOT / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        # A5's own calibration block, reproduced (not recomputed): the natural human band
        # the generated slopes/deltas have to be read against, and the same/different
        # window separation the threshold was picked from.
        nb = cal.get("natural_event_check_corpus_wide") or {}
        out["natural_band"] = {k: v for k, v in nb.items() if not isinstance(v, list)}
        out["pooled_windows"] = cal.get("pooled")
        out["chosen_threshold"] = cal.get("chosen_threshold")
        out["calibration_source"] = "results/v31_drift/calibration.json (A5)"
    out["cross_check"] = _verify_voice_drift(out["by_bucket"])
    return out


def _verify_voice_drift(by_bucket: dict) -> dict:
    """Hard cross-check of the re-aggregation against A5's published column CSV."""
    csv_path = DRIFT_ROOT / "voice_drift_column.csv"
    if not csv_path.exists():
        return {"cells_checked": 0, "mismatches": 0, "note": "voice_drift_column.csv absent"}
    checked = 0
    bad: list[str] = []
    with csv_path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            ck, b = rec["checkpoint"], rec["bucket"]
            got = by_bucket.get(ck, {}).get(b)
            if got is None:
                bad.append(f"{ck}/{b}: not re-derived")
                continue
            for col, field, tol in (
                ("n", "n", 0),
                ("n_with_windows", "n_with_windows", 0),
                ("spk_sim_level_med", "sim_level_median", 5e-5),
                ("delta_med", "delta_median", 5e-5),
                ("n_span_ge_60s", "n_span_ge_60s", 0),
                ("slope_per_min_med", "slope_per_min_median", 5e-5),
                ("t_voice_fired", "t_voice_fired", 0),
            ):
                want = rec[col]
                have = got[field]
                checked += 1
                if want in ("None", ""):
                    if have is not None:
                        bad.append(f"{ck}/{b}/{col}: csv None vs {have}")
                    continue
                if have is None:
                    bad.append(f"{ck}/{b}/{col}: csv {want} vs None")
                    continue
                if abs(float(want) - float(have)) > tol + 1e-12:
                    bad.append(f"{ck}/{b}/{col}: csv {want} vs {have}")
    if bad:
        raise SystemExit(
            "voice-drift cross-check FAILED against results/v31_drift/voice_drift_column.csv:\n  "
            + "\n  ".join(bad)
        )
    return {"cells_checked": checked, "mismatches": 0, "source": str(csv_path.relative_to(REPO))}


# ------------------------------------------------------------- Table 4 (A8)


def load_paired_deltas() -> dict:
    path = REPO / "results/v31_stats/paired_deltas.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


# ----------------------------------------------------------------- figures


# The three checkpoints of the budget-matched comparison, plus the SELECTED E4 curriculum
# checkpoint (PLAN §5 secondary).  E4 is drawn in a distinct colour and named as secondary
# in every legend, so no figure can be read as putting it in the matched comparison.
PLOTTED = SELECTED + ("E4_epoch_0_whole",)


def make_figures(
    bench: dict,
    per_item: dict[str, list[dict]],
    table3: dict,
    deltas: dict,
    budget: dict,
    out_dir: Path,
    curriculum: dict | None = None,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    colors = {
        "E1_native": "#4c72b0",
        "E0_official": "#999999",
        "E2_epoch_3_step_3001": "#dd8452",
        "E3_epoch_3_step_3001": "#55a868",
        "E2_epoch_0_step_200": "#e8b298",
        "E3_epoch_0_whole": "#a9d1b3",
        "E4_epoch_0_whole": "#8172b3",
        "E4_epoch_2_step_2001": "#c6bcdd",
    }

    # 1 -- completion by bucket, E1 / E2 / E3 (selected checkpoints)
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    width = 0.20
    xs = range(len(BUCKETS))
    off = (len(PLOTTED) - 1) / 2.0
    for k, label in enumerate(PLOTTED):
        vals = [100.0 * table3[label][b]["complete_rate"] for b in BUCKETS]
        name = ARMS[label][2] + (" — secondary, 2x budget" if label.startswith("E4") else "")
        ax.bar(
            [x + (k - off) * width for x in xs],
            vals,
            width=width,
            label=name,
            color=colors[label],
            edgecolor="black",
            linewidth=0.4,
            hatch="//" if label.startswith("E4") else None,
        )
        for x, v in zip(xs, vals):
            ax.text(x + (k - off) * width, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(BUCKETS)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Complete % (PLAN §3.4 final status)")
    ax.set_xlabel("length bucket (human reference duration)")
    ax.set_title("Completion by length, 12 items per bucket, dev pilot", fontsize=10)
    # below the axes: with four series an in-axes legend covers the 100 % bars
    ax.legend(
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "sft_completion_by_bucket.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(str(p))

    # 2 -- coverage vs human duration, per checkpoint
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    markers = {
        "E1_native": "o",
        "E2_epoch_3_step_3001": "s",
        "E3_epoch_3_step_3001": "^",
        "E4_epoch_0_whole": "D",
    }
    for label in PLOTTED:
        xs2 = [bench[r["text_id"]]["human_duration_sec"] for r in per_item[label]]
        ys2 = [r["source_coverage"] for r in per_item[label]]
        ax.scatter(
            xs2,
            ys2,
            s=26 if label != "E4_epoch_0_whole" else 22,
            marker=markers[label],
            facecolors="none" if label != "E3_epoch_3_step_3001" else colors[label],
            edgecolors=colors[label],
            linewidths=1.1,
            label=ARMS[label][2] + (" — secondary" if label.startswith("E4") else ""),
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_xlabel("human reference duration, s (log scale)")
    ax.set_ylabel("source coverage (share of source words matched)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Coverage against text length, 60 items per checkpoint", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "sft_coverage_vs_duration.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(str(p))

    # 3 -- paired deltas with 95 % CI (E3 - E2, selected checkpoints)
    comp = next(c for c in deltas["comparisons"] if c["comparison"] == "E3_vs_E2")
    metrics = [
        ("complete_rate", "Complete %, pp"),
        ("source_coverage", "Coverage"),
        ("wer", "WER-all %, pp"),
        ("end_coverage_robust", "EndCov-robust"),
        ("duration_ratio", "dur ratio"),
        ("excess_repetition_rate", "Repeat %, pp"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 3.9), sharey=True)
    cells = [*BUCKETS, "ALL"]
    ypos = list(range(len(cells)))[::-1]
    for ax, (key, name) in zip(axes, metrics):
        for y, cell in zip(ypos, cells):
            m = comp["cells"][cell]["metrics"][key]
            scale = m.get("scale", 1.0)
            d, lo, hi = m["delta"] * scale, m["ci_lo"] * scale, m["ci_hi"] * scale
            good = m["better"] == "higher"
            positive_is_good = (d > 0) == good
            col = "#55a868" if positive_is_good else "#c44e52"
            if abs(d) < 1e-12:
                col = "#888888"
            ax.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle="round")
            ax.plot([d], [y], "o", color=col, ms=5)
        ax.axvline(0.0, color="black", lw=0.8, ls="--")
        ax.set_title(name, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(cells)
    fig.suptitle(
        "E3 Long-SFT (step 3001) − E2 Short-SFT (step 3001), paired cluster bootstrap, "
        "6 roots, 10 000 resamples, 95 % percentile CI",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out_dir / "sft_paired_deltas_ci.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(str(p))

    # 4 -- CV loss curves, both arms
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm, label, col in (
        ("long", "E3 Long-SFT (cv = dev_long, 120 units)", "#55a868"),
        ("short", "E2 Short-SFT (cv = dev_short, 1448 windows)", "#dd8452"),
    ):
        cv = budget["arms"][arm]["cv_curve"]
        ax.plot([r["step"] for r in cv], [r["loss"] for r in cv], "-o", ms=3.4, color=col, label=label)
        best = budget["arms"][arm]["cv_best"]
        ax.plot([best["step"]], [best["loss"]], "*", ms=13, color=col, mec="black", mew=0.5)
        ax.annotate(
            f"min {best['loss']:.3f} @ {best['step']}",
            (best["step"], best["loss"]),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=7.5,
            color=col,
        )
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y - 0.05 * (hi_y - lo_y), hi_y + 0.03 * (hi_y - lo_y))
    ax.axvline(3001, color="black", lw=0.8, ls="--")
    ax.text(2950, ax.get_ylim()[1], "selected\n(last)", fontsize=7.5, va="top", ha="right")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("dev CV loss (cross-entropy on target speech tokens)")
    ax.set_title(
        "Dev CV loss. The two curves are on DIFFERENT dev packings and are not\n"
        "comparable to each other; only the shape within an arm is meaningful.",
        fontsize=9,
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "sft_cv_loss_curves.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(str(p))

    # 5 -- duration ratio by bucket
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label in ("E1_native", "E2_epoch_3_step_3001", "E3_epoch_3_step_3001", "E0_official"):
        ys3 = [table3[label][b]["dur_ratio"] for b in BUCKETS]
        ax.plot(
            range(len(BUCKETS)),
            ys3,
            "-o",
            ms=5,
            color=colors[label],
            label=ARMS[label][2],
            lw=1.8,
            ls="--" if label == "E0_official" else "-",
        )
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.text(4.05, 1.01, "human duration", fontsize=7.5, va="bottom", ha="right")
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels(BUCKETS)
    ax.set_ylabel("generated / human reference duration (macro mean)")
    ax.set_xlabel("length bucket")
    ax.set_title("Duration ratio by length. Above 1.0 = the model speaks longer than the human.", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "sft_duration_ratio_by_bucket.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(str(p))

    # 6 -- E4 curriculum: the dev CV curve of the three stages, on ONE x axis
    #
    # A separate figure, not a fourth line on figure 4: the E4 stages share the long arm's
    # unfiltered 120-unit dev set with E3, but each stage restarts its own step counter
    # (`--reset_steps`), so they only line up on a cumulative axis.  Figure 4 is left
    # byte-identical.
    if curriculum is not None:
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        stage_colors = {"S1": "#c6bcdd", "S2": "#a698c9", "S3": "#8172b3"}
        base = 0
        for stage, *_ in CURRICULUM_STAGES:
            v = curriculum["stages"][stage]
            cv = v["cv_curve"]
            xs4 = [base + r["step"] for r in cv]
            ys4 = [r["loss"] for r in cv]
            ax.plot(
                xs4,
                ys4,
                "-o",
                ms=3.6,
                color=stage_colors[stage],
                label=f"{stage} (<= {v['max_duration_sec']} s, {v['steps']} steps)",
            )
            if stage != "S1":
                ax.axvline(base, color="black", lw=0.7, ls=":")
            base += v["steps"]
        s3 = curriculum["stages"]["S3"]
        s3_base = curriculum["stages"]["S1"]["steps"] + curriculum["stages"]["S2"]["steps"]
        best = s3["cv_best"]
        ax.plot([s3_base + best["step"]], [best["loss"]], "*", ms=13,
                color="#8172b3", mec="black", mew=0.5)
        ax.annotate(
            f"S3 min {best['loss']:.3f} @ {best['step']} = the selected E4",
            (s3_base + best["step"], best["loss"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=7.5,
            color="#8172b3",
        )
        last = s3["cv_last"]
        ax.plot([s3_base + last["step"]], [last["loss"]], "o", ms=7,
                mfc="none", color="#8172b3", mew=1.4)
        ax.annotate(
            f"S3 last {last['loss']:.3f} @ {last['step']}",
            (s3_base + last["step"], last["loss"]),
            textcoords="offset points",
            xytext=(-10, 10),
            fontsize=7.5,
            ha="right",
            color="#8172b3",
        )
        e3 = budget["arms"]["long"]["cv_best"]
        ax.axhline(e3["loss"], color="#55a868", lw=1.2, ls="--")
        ax.text(
            base * 0.995,
            e3["loss"],
            f"E3 Long-SFT best CV {e3['loss']:.3f} @ {e3['step']}",
            fontsize=7.5,
            color="#55a868",
            va="bottom",
            ha="right",
        )
        ax.set_xlabel("cumulative optimizer step of the E4 curriculum (each stage restarts at 1)")
        ax.set_ylabel("dev CV loss (cross-entropy on target speech tokens)")
        ax.set_title(
            "E4 curriculum, dev CV loss of the three stages. Same unfiltered 120-unit dev set\n"
            "in all three and in E3, so these curves ARE on one scale (PLAN §11.5).",
            fontsize=9,
        )
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / "sft_cv_loss_e4_stages.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(str(p))
    return written


# -------------------------------------------------------------------- print


def print_tables(res: dict) -> None:
    t3, t5 = res["table3"], res["table5"]
    print("\n=== Table 3 — main results by length (PLAN §17) ===\n")
    rows = []
    for label in ORDER:
        for b in (*BUCKETS, "ALL"):
            c = t3[label][b]
            rows.append(
                [
                    label,
                    b,
                    c["n"],
                    _pct(c["complete_rate"]),
                    _pct(c["wer"], 2),
                    _pct(c["wer_micro"], 2),
                    _pct(c["cer"], 2),
                    _fmt(c["coverage"]),
                    _fmt(c["end_coverage"]),
                    _fmt(c["end_coverage_robust"]),
                    _pct(c["repeat"], 2),
                    _fmt(c["max_ngram_run"], 2),
                    str(c["max_ngram_run_max"]),
                    _fmt(c["dur_ratio"]),
                    _pct(c["wer_minus_floor"], 2),
                    c["n_with_floor"],
                ]
            )
    _print_table(
        [
            "checkpoint",
            "len",
            "n",
            "Compl%",
            "WERmac%",
            "WERmic%",
            "CER%",
            "Cov",
            "EndCov",
            "EndCovR",
            "Rep%",
            "maxrun",
            "maxrunMAX",
            "durratio",
            "WER-floor",
            "n_floor",
        ],
        rows,
    )

    print("=== Table 5 — failure taxonomy (PLAN §3.4) ===\n")
    rows = []
    for label in ORDER:
        for b in (*BUCKETS, "ALL"):
            c = t5[label][b]
            rows.append(
                [label, b, c["n"], c["complete"], c["degraded"], c["early_eos"], c["empty_or_invalid_audio"]]
            )
    _print_table(["checkpoint", "len", "n", "complete", "degraded", "early_eos", "empty/invalid"], rows)

    print("=== Table 7 — training budget ===\n")
    b = res["budget"]
    rows = []
    for arm, label in (("long", "E3 Long-SFT"), ("short", "E2 Short-SFT")):
        a = b["arms"][arm]
        rows.append(
            [
                label,
                a["steps"],
                f"{a['target_tokens_per_step_mean']:,.1f}",
                f"{a['verify_step200']['target_tokens_mean']:,.1f}",
                f"{a['target_tokens_total']:,}",
                f"{a['samples_per_step_mean']:.2f}",
                f"{a['step_time_sec_median']:.3f}",
                f"{a['wall_hours_log_span']:.3f}",
                f"{a['cv_best']['loss']:.4f}@{a['cv_best']['step']}",
                f"{a['cv_last']['loss']:.4f}",
            ]
        )
    _print_table(
        [
            "arm",
            "steps",
            "tok/step (3000)",
            "tok/step (200)",
            "total tokens",
            "samples/step",
            "step s (med)",
            "wall h",
            "CV min",
            "CV last",
        ],
        rows,
    )
    print(
        f"  tokens/step ratio long/short: first 200 steps {b['ratio_step200_mean']:.4f}, "
        f"full 3000 steps {b['ratio_full_run_mean']:.4f}, total tokens {b['ratio_full_run_total']:.4f}\n"
        f"  identical init checkpoint: {b['init_identical']}; identical config sha: {b['config_identical']}; "
        f"identical cv data: {b['cv_data_identical']}\n"
    )

    h4 = res.get("h4_margin")
    if h4:
        print("=== H4 short-form non-inferiority margin (Lead 2026-08-30, pre-registered) ===\n")
        print(
            f"  rule: {h4['rule']}\n"
            f"  baseline {h4['baseline']}: B0 WER-floor {h4['baseline_b0_wer_minus_floor_pp']:.2f} pp, "
            f"B0 Complete {h4['baseline_b0_complete_pp']:.1f} %\n"
            f"  => ceiling {h4['wer_floor_ceiling_pp']:.2f} pp, floor {h4['complete_floor_pp']:.1f} %\n"
        )
        rows = []
        for label in ORDER:
            v = h4["checkpoints"][label]
            rows.append(
                [
                    label,
                    f"{v['b0_wer_minus_floor_pp']:.2f}",
                    "ok" if v["wer_floor_ok"] else "FAIL",
                    f"{v['b0_complete_pp']:.1f}",
                    "ok" if v["complete_ok"] else "FAIL",
                    ("baseline" if v["is_baseline"] else "control" if v["is_control"]
                     else ("PASS" if v["passes"] else "FAIL")),
                ]
            )
        _print_table(
            ["checkpoint", "B0 WER-floor", "<= ceiling", "B0 Complete %", ">= floor", "H4"], rows
        )
        print(f"  SFT checkpoints passing: {h4['n_pass']}, failing: {h4['n_fail']}\n")

    cur = res.get("curriculum")
    if cur:
        print("=== E4 curriculum — the three stages as they ran ===\n")
        rows = []
        for stage, *_ in CURRICULUM_STAGES:
            v = cur["stages"][stage]
            rows.append(
                [
                    stage,
                    f"<= {v['max_duration_sec']} s",
                    v["steps"],
                    v["epochs_seen"],
                    f"{v['target_tokens_per_step_mean']:,.1f}",
                    f"{v['target_tokens_total']:,}",
                    f"{v['samples_per_step_mean']:.2f}",
                    f"{v['step_time_sec_median']:.3f}",
                    f"{v['wall_hours_log_span']:.3f}",
                    f"{v['peak_vram_reserved_gb_max']:.2f}",
                    f"{v['cv_best']['loss']:.4f}@{v['cv_best']['step']}",
                    f"{v['cv_last']['loss']:.4f}",
                ]
            )
        _print_table(
            [
                "stage",
                "ceiling",
                "steps",
                "epochs",
                "tok/step",
                "total tokens",
                "samples/step",
                "step s (med)",
                "wall h",
                "VRAM GB",
                "CV min",
                "CV last",
            ],
            rows,
        )
        print(
            f"  total {cur['steps_total']} steps, {cur['target_tokens_total']:,} target tokens, "
            f"{cur['wall_hours_total']:.3f} h wall; S1 init = E2 selected: "
            f"{cur['s1_init_is_e2_selected']}; one dev set across stages: "
            f"{cur['cv_data_identical_across_stages']}\n"
        )

    print("=== E3 (selected) residual failures ===\n")
    rows = []
    for i in res["residual"]["items"]:
        o = i["onsets"]["min_n3"]
        rows.append(
            [
                i["root_id"][:28],
                i["voice_id"],
                i["bucket"],
                i["status"],
                _pct(i["wer"], 1),
                _fmt(i["source_coverage"]),
                _pct(i["excess_repetition_rate"], 1),
                i["max_ngram_run"],
                f"{i['raw_duration_sec']:.1f}",
                _fmt(i["duration_ratio"], 2),
                f"{o['time_sec']:.1f}" if o else "-",
                _fmt(o["frac_of_audio"], 2) if o else "-",
                (o["gram"][:34] if o else "-"),
            ]
        )
    _print_table(
        [
            "root",
            "voice",
            "len",
            "status",
            "WER%",
            "cov",
            "rep%",
            "maxrun",
            "audio s",
            "ratio",
            "onset s",
            "onset frac",
            "first excess n-gram",
        ],
        rows,
    )

    print("=== RQ6 — training unit lengths vs benchmark buckets ===\n")
    r6 = res["rq6"]
    rows = [
        [
            name,
            v["n_units"],
            _pct(v["share_units"], 2),
            f"{v['hours']:.2f}",
            _pct(v["share_hours"], 2),
        ]
        for name, v in r6["bucket_bands"].items()
    ]
    _print_table(["band", "units", "% units", "hours", "% hours"], rows)
    print(
        f"  long-arm training units: {r6['n_units']} / {r6['hours']:.2f} h, "
        f"duration min {r6['duration_sec']['min']:.1f} s, median {r6['duration_sec']['median']:.1f} s, "
        f"p99 {r6['duration_sec']['p99']:.1f} s, max {r6['duration_sec']['max']:.1f} s\n"
        f"  B4 human durations {r6['benchmark_human_duration_sec']['B4']['min']:.1f}-"
        f"{r6['benchmark_human_duration_sec']['B4']['max']:.1f} s -> inside training range: "
        f"{r6['b4_inside_training_range']}\n"
        f"  p99 by definition: nearest-lower order statistic "
        f"{r6['duration_sec']['p99_order_stat']:.1f} s, linear interpolation "
        f"{r6['duration_sec']['p99_interp']:.1f} s\n"
    )

    print("=== Dev packing — the two cv_data sets as built (§6) ===\n")
    dp = res["dev_packing"]
    _print_table(
        ["arm", "dir", "units", "parents", "speakers", "hours", "target speech tokens"],
        [
            [
                arm,
                dp[arm]["dir"],
                dp[arm]["units"],
                dp[arm]["parents"],
                dp[arm]["speakers"],
                f"{dp[arm]['hours_token_manifest']:.3f}",
                dp[arm]["target_speech_tokens"],
            ]
            for arm in ("long", "short")
        ],
    )
    print(
        f"  identical target tokens on both arms: {dp['tokens_equal']}; "
        f"identical hours: {dp['hours_equal_to_1e6']}\n"
    )

    print("=== System — aggregate vs median throughput (§7.12) ===\n")
    rows = []
    for label in ORDER:
        sy = res["system"][label]["ALL"]
        rows.append(
            [
                label,
                sy["generated_speech_tokens_total"],
                f"{sy['wall_time_sec_total']:.1f}",
                f"{sy['tok_per_sec_wall_aggregate']:.1f}",
                f"{sy['tok_per_sec_wall_median']:.1f}",
                _fmt(sy["rtf_median"]),
                "n/a" if sy["context_occupancy_max"] is None else _fmt(sy["context_occupancy_max"], 4),
            ]
        )
    _print_table(
        ["checkpoint", "tokens", "wall s", "tok/s agg", "tok/s med", "RTF med", "ctx occ max"],
        rows,
    )
    rows = []
    for label in ("E1_native", "E3_epoch_3_step_3001"):
        for b in BUCKETS:
            sy = res["system"][label][b]
            rows.append(
                [
                    label,
                    b,
                    sy["generated_speech_tokens_total"],
                    f"{sy['wall_time_sec_total']:.1f}",
                    f"{sy['tok_per_sec_wall_aggregate']:.1f}",
                    f"{sy['tok_per_sec_wall_median']:.1f}",
                    _fmt(sy["rtf_median"]),
                ]
            )
    _print_table(
        ["checkpoint", "len", "tokens", "wall s", "tok/s agg", "tok/s med", "RTF med"], rows
    )
    e1, e3 = res["system"]["E1_native"]["ALL"], res["system"]["E3_epoch_3_step_3001"]["ALL"]
    print(
        f"  E3/E1 wall {e3['wall_time_sec_total'] / e1['wall_time_sec_total']:.1f}x = "
        f"tokens {e3['generated_speech_tokens_total'] / e1['generated_speech_tokens_total']:.2f}x"
        f" x aggregate throughput "
        f"{e1['tok_per_sec_wall_aggregate'] / e3['tok_per_sec_wall_aggregate']:.2f}x\n"
    )

    vd = res.get("voice_drift")
    if vd:
        print("=== Voice drift (PLAN §9.5, A5's per-item rows re-aggregated) ===\n")
        rows = []
        for label in (
            "E1_native",
            "E0_official",
            "E2_epoch_3_step_3001",
            "E3_epoch_3_step_3001",
            "E4_epoch_0_whole",
            "E4_epoch_2_step_2001",
        ):
            for b in (*BUCKETS, "ALL"):
                c = vd["by_bucket"][label][b]
                rows.append(
                    [
                        label,
                        b,
                        c["n"],
                        c["n_with_windows"],
                        _fmt(c["sim_level_median"], 4),
                        _fmt(c["delta_median"], 4),
                        c["n_span_ge_60s"],
                        "n/a" if c["slope_per_min_median"] is None else _fmt(c["slope_per_min_median"], 4),
                        f"{c['t_voice_fired']}/{c['n_with_windows']}",
                    ]
                )
        _print_table(
            [
                "checkpoint",
                "len",
                "n",
                "windows",
                "sim level med",
                "delta med",
                "n span>=60s",
                "slope/min med",
                "T_voice",
            ],
            rows,
        )
        rows = []
        for label in (
            "E0_official",
            "E1_native",
            "E2_epoch_3_step_3001",
            "E3_epoch_3_step_3001",
            "E4_epoch_0_whole",
            "E4_epoch_2_step_2001",
        ):
            for pair, c in vd["onset_paired"][label].items():
                rows.append(
                    [
                        label,
                        pair,
                        c["pairs"],
                        _fmt(c["b0_median"], 4),
                        _fmt(c["bk_median"], 4),
                        _fmt(c["delta_median"], 4),
                        f"{c['n_negative']}/{c['pairs']}",
                    ]
                )
        _print_table(
            ["checkpoint", "vs", "pairs", "first-win B0", "first-win Bk", "delta med", "neg"],
            rows,
        )
        print(
            f"  T_voice fired on {vd['t_voice_fired_total']} of {vd['n_items_total']} items; "
            f"cross-check vs A5's voice_drift_column.csv: "
            f"{vd['cross_check']['cells_checked']} cells, "
            f"{vd['cross_check']['mismatches']} mismatches\n"
        )


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-json",
        default=str(REPO / "results/v31_sft_analysis/sft_analysis.json"),
    )
    ap.add_argument(
        "--figures-dir",
        default=str(REPO / "reports/figures/v31_sft"),
    )
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    data = load_inputs()
    bench, per_item, runs = data["bench"], data["per_item"], data["runs"]

    v1 = verify_alignment(bench, per_item)
    table3 = build_table3(per_item)
    table5 = build_table5(per_item)
    v2 = verify_main_table(table3, table5)
    v3 = verify_final_table(table3, table5)

    res = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script": "src/stats/sft_analysis.py",
            "plan": "PLAN.md §2, §9, §12, §17",
            "arms": {k: {"per_item": v[0], "runs": v[1], "name": v[2]} for k, v in ARMS.items()},
            "selected_checkpoints": list(SELECTED),
            "verification": {"alignment": v1, "main_table": v2, "final_table": v3},
        },
        "table3": table3,
        "wer_macro_vs_micro": wer_macro_vs_micro(table3),
        "table5": table5,
        "duration": build_duration(bench, per_item),
        "system": system_table(runs),
        "residual": residual_failures(bench, per_item, runs),
        "e2_vs_e1": e2_vs_e1(per_item),
        "status_transitions": status_transitions(per_item),
        "audio_and_floor_exposure": audio_and_floor_exposure(per_item),
        "short_form_items": short_form_items(per_item),
        "checkpoint_gap": checkpoint_gap(table3, table5),
        "per_voice": per_voice(per_item),
        "by_source": by_source(bench, per_item),
        "budget": training_budget(),
        "curriculum": curriculum_budget(),
        "h4_margin": h4_margin(table3),
        "rq6": rq6_length_coverage(bench),
        "dev_packing": dev_packing(),
        "voice_drift": voice_drift(),
        "paired_deltas_source": "results/v31_stats/paired_deltas.json (A8)",
    }

    deltas = load_paired_deltas()
    res["paired_deltas_meta"] = deltas["meta"]

    figures: list[str] = []
    if not args.no_figures:
        figures = make_figures(
            bench,
            per_item,
            table3,
            deltas,
            res["budget"],
            Path(args.figures_dir),
            res["curriculum"],
        )
    res["figures"] = figures

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print_tables(res)
    print(f"alignment re-derivation: {v1['items_checked']} items, {v1['mismatches']} mismatches")
    print(f"main-table cross-check:  {v2['cells_checked']} cells, {v2['mismatches']} mismatches")
    print(f"final-table cross-check: {v3['cells_checked']} cells, {v3['mismatches']} mismatches")
    print(f"json  -> {out_json}")
    for f in figures:
        print(f"figure-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
