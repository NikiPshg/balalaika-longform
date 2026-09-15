"""RQ1/H1 analysis of the M2 pilot baseline (PLAN.md §2, §9, §17).

Every number quoted in ``reports/pilot_baseline_v31.md`` is produced by this script
from four immutable inputs and nothing else:

* ``data/benchmark/pilot.jsonl``            (A2, canonical §7.5 fields)
* ``results/v31_base/{E1,E0}/per_item.jsonl``  (A4, evaluator output)
* ``outputs/v31_base/{E1_native,E0_official}/runs.jsonl`` (A3/A7, run manifests §10)

The script never reads the aggregate tables, so its output is an independent
recomputation of them; ``--verify`` (on by default) re-runs the word alignment
from ``text_ref`` + ``asr_text`` and asserts that the stored per-item
``wer``/``hits``/``substitutions``/``deletions``/``insertions``/
``source_coverage``/``end_coverage`` are reproduced exactly.  Any mismatch is a
hard failure: this file must never silently disagree with the evaluator.

Usage::

    .venv-eval/bin/python src/stats/pilot_analysis.py                 # tables + json + figures
    .venv-eval/bin/python src/stats/pilot_analysis.py --no-figures
    .venv-eval/bin/python src/stats/pilot_analysis.py --out-json PATH

No GPU, no model, no network.  Runs in ~2 s.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.alignment import align_words  # noqa: E402
from src.eval.metrics import last_aligned_run  # noqa: E402
from src.eval.normalize import words as norm_words  # noqa: E402

BUCKETS = ("B0", "B1", "B2", "B3", "B4")
# PLAN.md §3.4 taxonomy, in the order the failure table prints them.
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
# A sentence of ``text_tts`` ends at one of these followed by whitespace.
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


# ---------------------------------------------------------------- small utils


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    """min / median / max / mean / n, with None for an empty sample."""
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


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]], align_right_from: int = 1) -> str:
    header = list(header)
    body = [[str(c) for c in row] for row in rows]
    sep = ["---" if i < align_right_from else "---:" for i in range(len(header))]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(out)


# ------------------------------------------------------------------- loading


def load_all(args: argparse.Namespace) -> dict[str, Any]:
    bench = {r["text_id"]: r for r in _load_jsonl(args.benchmark)}
    data: dict[str, Any] = {"benchmark": bench, "sets": {}}
    for label, per_item_path, runs_path in (
        ("E1_native", args.e1_per_item, args.e1_runs),
        ("E0_official", args.e0_per_item, args.e0_runs),
    ):
        per_item = _load_jsonl(per_item_path)
        runs = {r["run_id"]: r for r in _load_jsonl(runs_path)}
        missing = [r["run_id"] for r in per_item if r["run_id"] not in runs]
        if missing:
            raise SystemExit(f"{label}: {len(missing)} per-item rows have no run manifest, e.g. {missing[:3]}")
        for row in per_item:
            row["_run"] = runs[row["run_id"]]
            if row["text_id"] not in bench:
                raise SystemExit(f"{label}: text_id {row['text_id']} absent from the benchmark (PLAN §7.5)")
            row["_bench"] = bench[row["text_id"]]
        data["sets"][label] = {
            "per_item": per_item,
            "runs": runs,
            "per_item_path": str(per_item_path),
            "runs_path": str(runs_path),
        }
    return data


def by_bucket(rows: Sequence[dict]) -> dict[str, list[dict]]:
    out = {b: [] for b in BUCKETS}
    for row in rows:
        out[row["bucket"]].append(row)
    return out


# -------------------------------------------------- independent recomputation


def verify_alignment(rows: Sequence[dict], label: str) -> dict[str, Any]:
    """Re-derive the content metrics from text_ref + asr_text and compare."""
    checked = 0
    mismatches: list[str] = []
    for row in rows:
        variant = row["normalization_variant"]
        ref = norm_words(row["_bench"]["text_ref"], variant)
        hyp = norm_words(row.get("asr_text") or "", variant)
        al = align_words(ref, hyp)
        n_ref = len(ref)
        wer = (al.substitutions + al.deletions + al.insertions) / n_ref if n_ref else float("inf")
        mask = al.ref_aligned_mask()
        cov = sum(mask) / n_ref if n_ref else 0.0
        last = al.last_aligned_ref_index()
        endcov = (last + 1) / n_ref if (last is not None and n_ref) else 0.0
        end_idx, _run_len, _n_runs = last_aligned_run(mask, min_run=row["end_coverage_robust_min_run"])
        robust = (end_idx + 1) / n_ref if (end_idx is not None and n_ref) else 0.0
        pairs = (
            ("n_ref_words", n_ref, row["n_ref_words"]),
            ("hits", al.hits, row["hits"]),
            ("substitutions", al.substitutions, row["substitutions"]),
            ("deletions", al.deletions, row["deletions"]),
            ("insertions", al.insertions, row["insertions"]),
            ("wer", wer, row["wer"]),
            ("source_coverage", cov, row["source_coverage"]),
            ("end_coverage", endcov, row["end_coverage"]),
            ("end_coverage_robust", robust, row["end_coverage_robust"]),
        )
        for name, got, stored in pairs:
            ok = math.isclose(got, stored, rel_tol=1e-9, abs_tol=1e-9) if isinstance(got, float) else got == stored
            if not ok:
                mismatches.append(f"{label} {row['run_id']} {name}: recomputed {got} vs stored {stored}")
        checked += 1
    return {"checked": checked, "n_mismatch": len(mismatches), "mismatches": mismatches[:10]}


# ------------------------------------------------------------ sentence ends


def sentence_boundaries(bench_row: dict) -> list[int] | None:
    """Cumulative reference-word counts at every sentence end of ``text_tts``.

    Returns None when the sentence split does not reproduce ``text_ref`` word
    for word (then the boundary question cannot be answered for that text).
    """
    variant = bench_row.get("normalization_variant", "lenient")
    ref_n = len(norm_words(bench_row["text_ref"], variant))
    sents = [s for s in SENT_SPLIT.split(bench_row["text_tts"].strip()) if s.strip()]
    cum, total = [], 0
    for sent in sents:
        total += len(norm_words(sent, variant))
        cum.append(total)
    if total != ref_n:
        return None
    return cum


def stop_analysis(rows: Sequence[dict], coherent_min_prefix_coverage: float = 0.8) -> dict[str, Any]:
    """Where does the native reading end, and is that a sentence end?

    For every item the reference words are aligned to the ASR transcript; the
    *stop index* is the position after the last aligned reference word (that is
    ``end_coverage x N_ref``).  ``prefix_coverage`` is the share of reference
    words BEFORE the stop that were aligned — it separates "read a prefix and
    stopped" from "matched a word by accident deep in the text".
    """
    items: list[dict] = []
    for row in rows:
        bench = row["_bench"]
        bounds = sentence_boundaries(bench)
        variant = row["normalization_variant"]
        ref = norm_words(bench["text_ref"], variant)
        hyp = norm_words(row.get("asr_text") or "", variant)
        n_ref = len(ref)
        al = align_words(ref, hyp)
        mask = al.ref_aligned_mask()
        last = al.last_aligned_ref_index()
        stop = (last + 1) if last is not None else 0
        prefix_cov = (sum(mask[:stop]) / stop) if stop else 0.0
        end_idx, run_len, n_runs = last_aligned_run(mask, min_run=row["end_coverage_robust_min_run"])
        robust_stop = (end_idx + 1) if end_idx is not None else 0
        robust_prefix_cov = (sum(mask[:robust_stop]) / robust_stop) if robust_stop else 0.0
        rec: dict[str, Any] = {
            "run_id": row["run_id"],
            "bucket": row["bucket"],
            "voice_id": row["voice_id"],
            "n_ref_words": n_ref,
            "n_sentences": len(bounds) if bounds else None,
            "stop_word_index": stop,
            "stop_frac": stop / n_ref if n_ref else 0.0,
            # Every benchmark text ends on a sentence end (PLAN §7.2 nested prefixes),
            # so a run that reached the last word is on a boundary BY CONSTRUCTION and
            # carries no information about where the model chooses to stop.
            "is_full_read": bool(n_ref) and stop == n_ref,
            "prefix_coverage": prefix_cov,
            "robust_stop_word_index": robust_stop,
            "robust_prefix_coverage": robust_prefix_cov,
            "robust_run_words": run_len,
            "n_aligned_runs": n_runs,
            "boundaries_usable": bounds is not None,
            "dist_to_boundary": None,
            "dist_to_boundary_robust": None,
            "chance_within_0": None,
            "chance_within_2": None,
        }
        if bounds and stop:
            rec["dist_to_boundary"] = min(abs(stop - b) for b in bounds)
            # chance that a uniformly random stop lands within +-k words of one
            # of the sentence ends of THIS text (the null the counts are read against)
            for k in (0, 2):
                covered = len({p for b in bounds for p in range(max(1, b - k), min(n_ref, b + k) + 1)})
                rec[f"chance_within_{k}"] = covered / n_ref
        if bounds and robust_stop:
            rec["dist_to_boundary_robust"] = min(abs(robust_stop - b) for b in bounds)
        items.append(rec)

    coherent = [i for i in items if i["prefix_coverage"] >= coherent_min_prefix_coverage and i["boundaries_usable"]]
    # A run that emitted 2-4 correct words trivially has prefix coverage 1.0; the
    # sentence-end question is only interesting once the model produced a real
    # stretch of the text, so the same statistics are repeated on that subset.
    substantial = [i for i in coherent if i["stop_word_index"] >= 20]
    summary = {
        "coherent_min_prefix_coverage": coherent_min_prefix_coverage,
        "n_items": len(items),
        "n_coherent": len(coherent),
        "n_coherent_by_bucket": {b: sum(1 for i in coherent if i["bucket"] == b) for b in BUCKETS},
        "substantial_min_stop_words": 20,
        "n_substantial": len(substantial),
        "n_substantial_by_bucket": {b: sum(1 for i in substantial if i["bucket"] == b) for b in BUCKETS},
        "substantial_within_0": sum(1 for i in substantial if i["dist_to_boundary"] == 0),
        "substantial_within_2": sum(1 for i in substantial if (i["dist_to_boundary"] or 0) <= 2),
        "substantial_dist": _stats([i["dist_to_boundary"] for i in substantial if i["dist_to_boundary"] is not None]),
    }
    # Same question without the coherence filter: every item that aligned at all.
    with_stop = [i for i in items if i["boundaries_usable"] and i["dist_to_boundary"] is not None]
    summary["n_with_stop"] = len(with_stop)
    summary["all_within_0"] = sum(1 for i in with_stop if i["dist_to_boundary"] == 0)
    summary["all_within_2"] = sum(1 for i in with_stop if i["dist_to_boundary"] <= 2)
    summary["all_dist"] = _stats([i["dist_to_boundary"] for i in with_stop])
    for k in (0, 2):
        summary[f"all_chance_within_{k}_mean"] = (
            sum(i[f"chance_within_{k}"] for i in with_stop) / len(with_stop) if with_stop else None
        )
    for k in (0, 1, 2, 3):
        summary[f"coherent_within_{k}"] = sum(1 for i in coherent if i["dist_to_boundary"] is not None and i["dist_to_boundary"] <= k)
    for k in (0, 2):
        summary[f"coherent_chance_within_{k}_mean"] = (
            sum(i[f"chance_within_{k}"] for i in coherent if i[f"chance_within_{k}"] is not None) / len(coherent)
            if coherent
            else None
        )
    summary["coherent_dist"] = _stats([i["dist_to_boundary"] for i in coherent if i["dist_to_boundary"] is not None])

    # ---- full-read / partial-read split (the confound the raw counts hide) ----
    # Each benchmark text ends on a sentence end, so an item whose stop index is
    # N_ref sits on a boundary by construction.  Only the items that stopped
    # BEFORE the last word carry information about where the model chooses to stop.
    texts = {}
    for row in rows:
        bench = row["_bench"]
        if bench["text_id"] in texts:
            continue
        bounds = sentence_boundaries(bench)
        n_ref = len(norm_words(bench["text_ref"], bench.get("normalization_variant", "lenient")))
        texts[bench["text_id"]] = (bounds[-1] == n_ref) if bounds else None
    summary["n_texts"] = len(texts)
    summary["n_texts_with_usable_boundaries"] = sum(1 for v in texts.values() if v is not None)
    summary["n_texts_ending_on_a_sentence_end"] = sum(1 for v in texts.values() if v)

    def _split(pop: Sequence[dict], name: str) -> None:
        full = [i for i in pop if i["is_full_read"]]
        partial = [i for i in pop if not i["is_full_read"]]
        summary[f"{name}_n_full_read"] = len(full)
        summary[f"{name}_n_partial_read"] = len(partial)
        summary[f"{name}_full_within_0"] = sum(1 for i in full if i["dist_to_boundary"] == 0)
        for k in (0, 2):
            summary[f"{name}_partial_within_{k}"] = sum(
                1 for i in partial if i["dist_to_boundary"] is not None and i["dist_to_boundary"] <= k
            )
            summary[f"{name}_partial_chance_within_{k}_mean"] = (
                sum(i[f"chance_within_{k}"] for i in partial if i[f"chance_within_{k}"] is not None) / len(partial)
                if partial
                else None
            )
        summary[f"{name}_partial_dist"] = _stats(
            [i["dist_to_boundary"] for i in partial if i["dist_to_boundary"] is not None]
        )
        groups = {"B0B1": ("B0", "B1"), "B2plus": ("B2", "B3", "B4")}
        for gname, buckets in groups.items():
            grp = [i for i in partial if i["bucket"] in buckets]
            summary[f"{name}_partial_{gname}"] = {
                "n": len(grp),
                "within_0": sum(1 for i in grp if i["dist_to_boundary"] == 0),
                "within_2": sum(1 for i in grp if i["dist_to_boundary"] is not None and i["dist_to_boundary"] <= 2),
                "chance_within_0_mean": (
                    sum(i["chance_within_0"] for i in grp if i["chance_within_0"] is not None) / len(grp)
                    if grp
                    else None
                ),
            }
        summary[f"{name}_partial_by_bucket"] = {
            b: {
                "n": sum(1 for i in partial if i["bucket"] == b),
                "within_0": sum(1 for i in partial if i["bucket"] == b and i["dist_to_boundary"] == 0),
            }
            for b in BUCKETS
        }

    _split(coherent, "coherent")
    _split(substantial, "substantial")
    _split(with_stop, "all")
    return {"items": items, "summary": summary}


# ------------------------------------------------------------------ sections


def duration_section(sets: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, blob in sets.items():
        per_bucket = by_bucket(blob["per_item"])
        out[label] = {}
        for b in BUCKETS:
            rows = per_bucket[b]
            out[label][b] = {
                "n": len(rows),
                "generated_sec": _stats([r["raw_duration_sec"] for r in rows]),
                "human_sec": _stats([r["reference_duration_sec"] for r in rows]),
                "ratio": _stats([r["duration_ratio"] for r in rows]),
                "voiced_sec": _stats([r["voiced_duration_sec"] for r in rows]),
                "silence_ratio": _stats([r["silence_ratio"] for r in rows]),
                "speaking_rate_wpm": _stats([r["speaking_rate_wpm_voiced"] for r in rows]),
            }
    return out


def token_section(sets: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, blob in sets.items():
        per_bucket = by_bucket(blob["per_item"])
        out[label] = {}
        for b in BUCKETS:
            rows = per_bucket[b]
            runs = [r["_run"] for r in rows]
            gen = [r["generated_speech_tokens"] for r in runs]
            txt = [r["text_tokens"] for r in runs]
            out[label][b] = {
                "n": len(rows),
                "text_tokens": _stats(txt),
                "generated_speech_tokens": _stats(gen),
                "gen_per_text_token": _stats([g / t for g, t in zip(gen, txt) if t]),
                # 25 Hz speech-token rate (audit §5): tokens the human takes
                "human_tokens_25hz": _stats([25.0 * r["reference_duration_sec"] for r in rows]),
                "gen_over_human_tokens": _stats(
                    [g / (25.0 * r["reference_duration_sec"]) for g, r in zip(gen, rows) if r["reference_duration_sec"]]
                ),
                "context_tokens_total": _stats([r["context_tokens_total"] for r in runs]),
                "context_occupancy": _stats([r.get("context_occupancy") for r in runs if r.get("context_occupancy") is not None]),
                "max_len_cap": _stats([r["max_len_cap"] for r in runs]),
                "n_at_cap": sum(1 for g, r in zip(gen, runs) if g >= r["max_len_cap"]),
                "n_below_min_len": sum(1 for g, r in zip(gen, runs) if g < r["min_len"]),
            }
    return out


def _throughput_field_check(runs: Sequence[dict], tol: float = 0.02) -> dict[str, Any]:
    """Which denominator does the manifest's ``speech_tokens_per_sec_wall`` really use?

    The field is written by ``src/adapters/cosyvoice3_native.py:154`` as
    ``n_yielded / llm_seconds``.  This recomputes both candidate rates from the
    manifest's own token count and times and reports how many runs each explains,
    so the report never has to take the field's *name* on trust.  It also checks
    that ``llm + flow_hift + frontend`` accounts for the wall time, i.e. that the
    time the LM-based rate leaves out is real work.
    """
    n = matches_llm = matches_wall = 0
    residuals: list[float] = []
    for r in runs:
        stored = r.get("speech_tokens_per_sec_wall")
        if stored is None or not r.get("wall_time_sec") or not r.get("llm_time_sec"):
            continue
        n += 1
        gen = r["generated_speech_tokens"]
        if abs(stored - gen / r["llm_time_sec"]) <= tol:
            matches_llm += 1
        if abs(stored - gen / r["wall_time_sec"]) <= tol:
            matches_wall += 1
        parts = sum(float(r.get(k) or 0.0) for k in ("llm_time_sec", "flow_hift_time_sec", "frontend_time_sec"))
        residuals.append(abs(r["wall_time_sec"] - parts))
    return {
        "n": n,
        "tol": tol,
        "n_matching_llm_time_denominator": matches_llm,
        "n_matching_wall_time_denominator": matches_wall,
        "max_abs_residual_wall_minus_components_sec": max(residuals) if residuals else None,
    }


def system_section(sets: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, blob in sets.items():
        per_bucket = by_bucket(blob["per_item"])
        out[label] = {}
        for b in list(BUCKETS) + ["ALL"]:
            rows = blob["per_item"] if b == "ALL" else per_bucket[b]
            runs = [r["_run"] for r in rows]
            out[label][b] = {
                "n": len(rows),
                "wall_sec": _stats([r["wall_time_sec"] for r in runs]),
                "wall_sec_total": sum(r["wall_time_sec"] for r in runs),
                "llm_sec": _stats([r.get("llm_time_sec") for r in runs if r.get("llm_time_sec") is not None]),
                "flow_hift_sec": _stats([r.get("flow_hift_time_sec") for r in runs if r.get("flow_hift_time_sec") is not None]),
                "rtf": _stats([r["rtf"] for r in runs]),
                "peak_vram_gb": _stats([r["peak_vram_bytes"] / 2**30 for r in runs]),
                "peak_vram_reserved_gb": _stats([r["peak_vram_reserved_bytes"] / 2**30 for r in runs]),
                "generated_speech_tokens": _stats([r["generated_speech_tokens"] for r in runs]),
                # The manifest field is NAMED ``speech_tokens_per_sec_wall`` but is computed
                # as n_yielded / llm_seconds (src/adapters/cosyvoice3_native.py:154), i.e. its
                # denominator is LM time, not wall time.  It is republished here under the
                # name it actually has, and the wall-time rate is recomputed next to it.
                "speech_tokens_per_sec_llm": _stats(
                    [r.get("speech_tokens_per_sec_wall") for r in runs if r.get("speech_tokens_per_sec_wall") is not None]
                ),
                "speech_tokens_per_sec_wall_true": _stats(
                    [r["generated_speech_tokens"] / r["wall_time_sec"] for r in runs if r.get("wall_time_sec")]
                ),
                "throughput_field_check": _throughput_field_check(runs),
                "llm_share_of_wall": _stats(
                    [r["llm_time_sec"] / r["wall_time_sec"] for r in runs
                     if r.get("llm_time_sec") is not None and r.get("wall_time_sec")]
                ),
                "generated_sec_total": sum(r["raw_duration_sec"] for r in rows),
                "flow_noise_buffer_sec": sorted({r["flow_noise_buffer_sec"] for r in runs}),
                "hift_noise_buffer_sec": sorted({r["hift_noise_buffer_sec"] for r in runs}),
                "stop_reasons": {s: sum(1 for r in runs if r["stop_reason"] == s) for s in sorted({r["stop_reason"] for r in runs})},
                "n_oom": sum(1 for r in runs if r.get("exception_type") == "OutOfMemoryError"),
                "n_exception": sum(1 for r in runs if r.get("exception_type")),
                "n_nan": sum(1 for r in runs if r.get("audio", {}).get("has_nan")),
                "peak_amplitude_max": max((r.get("audio", {}).get("peak", 0.0) for r in runs), default=None),
                # E0 only: the official adapter splits the text, so this is how
                # many independent TTS calls the "production control" needed.
                "n_chunks": _stats([r["official_split"]["n_chunks"] for r in runs if r.get("official_split")]),
                "llm_inference_calls": _stats(
                    [r["native_contract"]["llm_inference_calls"] for r in runs if r.get("native_contract")]
                ),
            }
    return out


def speaking_rate_section(sets: dict[str, Any], bench: dict[str, dict]) -> dict[str, Any]:
    """Words per minute of the human readings and of the generated audio.

    §5.9 compares the two, so both denominators are published side by side: the
    human rate can only be computed on RAW duration (there is no VAD pass over
    the human slices in the pilot), while the evaluator stores both a raw and a
    voiced rate for the generated audio.  Restricted to the four dataset roots,
    whose ``human_duration_sec`` is measured rather than estimated at 103.332 wpm.
    """
    ds_texts = {tid: r for tid, r in bench.items() if r["root_id"].startswith("ds")}
    human = {
        "n": len(ds_texts),
        "wpm_raw": _stats([60.0 * r["ref_words"] / r["human_duration_sec"] for r in ds_texts.values()]),
        "duration_source": sorted({r.get("human_duration_source", "?") for r in ds_texts.values()}),
    }
    out: dict[str, Any] = {"human_dataset_roots": human}
    for label, blob in sets.items():
        rows = [r for r in blob["per_item"] if r["text_id"] in ds_texts]
        out[label] = {
            "dataset_roots": {
                "n": len(rows),
                "wpm_raw": _stats([r["speaking_rate_wpm_raw"] for r in rows if r.get("speaking_rate_wpm_raw")]),
                "wpm_voiced": _stats([r["speaking_rate_wpm_voiced"] for r in rows if r.get("speaking_rate_wpm_voiced")]),
                "silence_ratio": _stats([r["silence_ratio"] for r in rows if r.get("silence_ratio") is not None]),
            },
            "dataset_roots_B4": {
                "n": len([r for r in rows if r["bucket"] == "B4"]),
                "wpm_raw": _stats([r["speaking_rate_wpm_raw"] for r in rows if r["bucket"] == "B4" and r.get("speaking_rate_wpm_raw")]),
                "wpm_voiced": _stats([r["speaking_rate_wpm_voiced"] for r in rows if r["bucket"] == "B4" and r.get("speaking_rate_wpm_voiced")]),
            },
            "all_roots_B4": {
                "n": len([r for r in blob["per_item"] if r["bucket"] == "B4"]),
                "wpm_raw": _stats([r["speaking_rate_wpm_raw"] for r in blob["per_item"] if r["bucket"] == "B4" and r.get("speaking_rate_wpm_raw")]),
                "wpm_voiced": _stats([r["speaking_rate_wpm_voiced"] for r in blob["per_item"] if r["bucket"] == "B4" and r.get("speaking_rate_wpm_voiced")]),
            },
        }
    return out


def split_context_section(path: Path) -> dict[str, Any] | None:
    """Hours per split unit, for the §7.4 selection-bias caveat.

    Reads ``data/manifests/split_channels.json`` (A1's split record).  A
    *catalogue* channel is one whose ``channel_id`` is a catalogue label, i.e.
    not an ``other:<youtube id>`` key (`reports/split_rule_v3.md` §0).
    """
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    chans = blob["channels"]
    total = sum(c["hours_total"] for c in chans.values())
    cat = {k: c for k, c in chans.items() if not k.startswith("other:")}
    cat_train = {k: c for k, c in cat.items() if c["split"] == "train"}
    cat_held = {k: c for k, c in cat.items() if c["split"] != "train"}
    return {
        "source": str(path),
        "total_hours": total,
        "total_hours_reported": blob.get("total_hours"),
        "n_channels": len(chans),
        "n_catalogue": len(cat),
        "catalogue_hours": sum(c["hours_total"] for c in cat.values()),
        "catalogue_share": sum(c["hours_total"] for c in cat.values()) / total,
        "n_catalogue_train_only": len(cat_train),
        "catalogue_train_only_hours": sum(c["hours_total"] for c in cat_train.values()),
        "catalogue_train_only_share": sum(c["hours_total"] for c in cat_train.values()) / total,
        "catalogue_held_out": {k: {"split": c["split"], "hours": c["hours_total"]} for k, c in cat_held.items()},
        "hours_by_split": {
            s: sum(c["hours_total"] for c in chans.values() if c["split"] == s)
            for s in sorted({c["split"] for c in chans.values()})
        },
    }


def content_section(sets: dict[str, Any]) -> dict[str, Any]:
    """Macro means and medians of the content metrics, recomputed here."""
    out: dict[str, Any] = {}
    for label, blob in sets.items():
        per_bucket = by_bucket(blob["per_item"])
        out[label] = {}
        for b in list(BUCKETS) + ["ALL"]:
            rows = blob["per_item"] if b == "ALL" else per_bucket[b]
            with_floor = [r for r in rows if r.get("floor_available")]
            out[label][b] = {
                "n": len(rows),
                "complete_rate": sum(1 for r in rows if r["status"] == "complete") / len(rows),
                "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES},
                "wer": _stats([r["wer"] for r in rows if math.isfinite(r["wer"])]),
                "cer": _stats([r["cer"] for r in rows if math.isfinite(r["cer"])]),
                "coverage": _stats([r["source_coverage"] for r in rows]),
                "end_coverage": _stats([r["end_coverage"] for r in rows]),
                "end_coverage_robust": _stats([r["end_coverage_robust"] for r in rows]),
                "tail_deletion_rate": _stats([r["tail_deletion_rate"] for r in rows]),
                "longest_deletion_run_frac": _stats([r["longest_deletion_run_frac"] for r in rows]),
                "excess_repetition_rate": _stats([r["excess_repetition_rate"] for r in rows]),
                "max_ngram_run": _stats([r["max_ngram_run"] for r in rows]),
                "n_loop_flag": sum(1 for r in rows if r.get("loop_flag")),
                "floor_wer": _stats([r["floor_wer"] for r in with_floor]),
                "wer_minus_floor": _stats([r["wer_minus_floor"] for r in with_floor if r["wer_minus_floor"] is not None]),
                "n_with_floor": len(with_floor),
            }
    return out


def voice_section(sets: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, blob in sets.items():
        rows = blob["per_item"]
        voices = sorted({r["voice_id"] for r in rows})
        out[label] = {}
        for voice in voices:
            vr = [r for r in rows if r["voice_id"] == voice]
            per_bucket = by_bucket(vr)
            out[label][voice] = {
                "n": len(vr),
                "complete_rate": sum(1 for r in vr if r["status"] == "complete") / len(vr),
                "wer": _stats([r["wer"] for r in vr if math.isfinite(r["wer"])]),
                "coverage": _stats([r["source_coverage"] for r in vr]),
                "duration_ratio": _stats([r["duration_ratio"] for r in vr]),
                "generated_sec": _stats([r["raw_duration_sec"] for r in vr]),
                "generated_speech_tokens": _stats([r["_run"]["generated_speech_tokens"] for r in vr]),
                "by_bucket": {
                    b: {
                        "n": len(per_bucket[b]),
                        "complete": sum(1 for r in per_bucket[b] if r["status"] == "complete"),
                        "generated_sec_median": _stats([r["raw_duration_sec"] for r in per_bucket[b]])["median"],
                        "coverage_median": _stats([r["source_coverage"] for r in per_bucket[b]])["median"],
                        "wer_median": _stats([r["wer"] for r in per_bucket[b] if math.isfinite(r["wer"])])["median"],
                    }
                    for b in BUCKETS
                },
            }
        # paired difference on the same (text_id, seed): female - male
        paired = []
        by_text = {}
        for r in rows:
            by_text.setdefault(r["text_id"], {})[r["voice_id"]] = r
        for text_id, d in by_text.items():
            if "ref_female_01" in d and "ref_male_01" in d:
                paired.append(
                    {
                        "text_id": text_id,
                        "bucket": d["ref_female_01"]["bucket"],
                        "d_generated_sec": d["ref_female_01"]["raw_duration_sec"] - d["ref_male_01"]["raw_duration_sec"],
                        "d_coverage": d["ref_female_01"]["source_coverage"] - d["ref_male_01"]["source_coverage"],
                        "d_wer": d["ref_female_01"]["wer"] - d["ref_male_01"]["wer"],
                        "d_tokens": d["ref_female_01"]["_run"]["generated_speech_tokens"]
                        - d["ref_male_01"]["_run"]["generated_speech_tokens"],
                    }
                )
        out[label]["_paired_female_minus_male"] = {
            "n_pairs": len(paired),
            "d_generated_sec": _stats([p["d_generated_sec"] for p in paired]),
            "d_coverage": _stats([p["d_coverage"] for p in paired]),
            "d_wer": _stats([p["d_wer"] for p in paired]),
            "d_tokens": _stats([p["d_tokens"] for p in paired]),
            "n_female_longer": sum(1 for p in paired if p["d_generated_sec"] > 0),
        }
    return out


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation, average ranks for ties, stdlib only."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None

    def ranks(vals: Sequence[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def length_effect_section(sets: dict[str, Any], stop_items: dict[str, Sequence[dict]]) -> dict[str, Any]:
    """H1 / RQ1: does a longer input make the model stop EARLIER in absolute terms?

    Everything here is in ABSOLUTE units (source words actually read, seconds of
    audio, speech tokens) rather than the ratios of the main table, because the
    claim under test is that the model reads *fewer words of the same document*
    when it is handed more of it.  The chain is (root_id, voice_id); B0..B4 of one
    chain are nested prefixes of one text read by one voice, so the comparison is
    within-item and needs no model.
    """
    out: dict[str, Any] = {}
    for label, blob in sets.items():
        stops = {i["run_id"]: i for i in stop_items[label]}
        rows = blob["per_item"]
        recs = []
        for r in rows:
            st_i = stops[r["run_id"]]
            recs.append(
                {
                    "run_id": r["run_id"],
                    "root_id": r["_bench"]["root_id"],
                    "bucket": r["bucket"],
                    "voice_id": r["voice_id"],
                    "source": r["_bench"]["source"],
                    "n_ref_words": r["n_ref_words"],
                    "text_tokens": r["_run"]["text_tokens"],
                    "words_read": r["source_coverage"] * r["n_ref_words"],
                    "stop_word_index": st_i["stop_word_index"],
                    "robust_stop_word_index": st_i["robust_stop_word_index"],
                    "generated_sec": r["raw_duration_sec"],
                    "generated_speech_tokens": r["_run"]["generated_speech_tokens"],
                    "human_sec": r["reference_duration_sec"],
                }
            )
        per_bucket = {
            b: {
                "n": sum(1 for x in recs if x["bucket"] == b),
                "text_tokens": _stats([x["text_tokens"] for x in recs if x["bucket"] == b]),
                "words_read_abs": _stats([x["words_read"] for x in recs if x["bucket"] == b]),
                "stop_word_index": _stats([x["stop_word_index"] for x in recs if x["bucket"] == b]),
                "robust_stop_word_index": _stats([x["robust_stop_word_index"] for x in recs if x["bucket"] == b]),
                "generated_sec": _stats([x["generated_sec"] for x in recs if x["bucket"] == b]),
                "generated_speech_tokens": _stats([x["generated_speech_tokens"] for x in recs if x["bucket"] == b]),
            }
            for b in BUCKETS
        }
        # within-chain monotonicity: root x voice, B0 -> B4
        chains: dict[tuple[str, str], dict[str, dict]] = {}
        for x in recs:
            chains.setdefault((x["root_id"], x["voice_id"]), {})[x["bucket"]] = x
        chain_rows = []
        for (root, voice), by_b in sorted(chains.items()):
            if len(by_b) != len(BUCKETS):
                continue
            words = [by_b[b]["words_read"] for b in BUCKETS]
            secs = [by_b[b]["generated_sec"] for b in BUCKETS]
            chain_rows.append(
                {
                    "root_id": root,
                    "voice_id": voice,
                    "words_read": words,
                    "generated_sec": secs,
                    "stop_index": [by_b[b]["stop_word_index"] for b in BUCKETS],
                    "argmax_bucket_words": BUCKETS[max(range(len(words)), key=lambda i: words[i])],
                    "b4_lt_b0_words": words[-1] < words[0],
                    "b4_lt_b1_words": words[-1] < words[1],
                    "monotone_nonincreasing_from_peak": all(
                        words[i] >= words[i + 1]
                        for i in range(max(range(len(words)), key=lambda i: words[i]), len(words) - 1)
                    ),
                    "b4_lt_b0_sec": secs[-1] < secs[0],
                }
            )
        out[label] = {
            "per_bucket": per_bucket,
            "chains": chain_rows,
            "n_chains": len(chain_rows),
            "n_chains_b4_lt_b0_words": sum(1 for c in chain_rows if c["b4_lt_b0_words"]),
            "n_chains_b4_lt_b1_words": sum(1 for c in chain_rows if c["b4_lt_b1_words"]),
            "n_chains_b4_lt_b0_sec": sum(1 for c in chain_rows if c["b4_lt_b0_sec"]),
            "argmax_bucket_counts": {
                b: sum(1 for c in chain_rows if c["argmax_bucket_words"] == b) for b in BUCKETS
            },
            "n_chains_nonincreasing_from_peak": sum(
                1 for c in chain_rows if c["monotone_nonincreasing_from_peak"]
            ),
            # rank correlations over all items of the set
            "spearman_texttok_vs_words_read": _spearman(
                [x["text_tokens"] for x in recs], [x["words_read"] for x in recs]
            ),
            "spearman_texttok_vs_generated_sec": _spearman(
                [x["text_tokens"] for x in recs], [x["generated_sec"] for x in recs]
            ),
            "spearman_texttok_vs_generated_tokens": _spearman(
                [x["text_tokens"] for x in recs], [x["generated_speech_tokens"] for x in recs]
            ),
            "items": recs,
        }
    return out


def case_section(sets: dict[str, Any]) -> dict[str, Any]:
    """The individual items the report has to explain by name."""
    cases: dict[str, Any] = {}
    for label, blob in sets.items():
        for row in blob["per_item"]:
            if row["status"] in ("degraded", "empty_or_invalid_audio", "loop_cap", "timeout", "oom", "context_limit"):
                run = row["_run"]
                bench = row["_bench"]
                cases[f"{label}:{row['run_id']}"] = {
                    "status": row["status"],
                    "final_status_reason": row["final_status_reason"],
                    "gen_status": row["gen_status"],
                    "stop_reason": row["stop_reason"],
                    "bucket": row["bucket"],
                    "voice_id": row["voice_id"],
                    "n_ref_words": row["n_ref_words"],
                    "n_hyp_words": row["n_hyp_words"],
                    "hits": row["hits"],
                    "substitutions": row["substitutions"],
                    "deletions": row["deletions"],
                    "insertions": row["insertions"],
                    "wer": row["wer"],
                    "floor_wer": row["floor_wer"],
                    "wer_minus_floor": row["wer_minus_floor"],
                    "coverage": row["source_coverage"],
                    "end_coverage": row["end_coverage"],
                    "end_coverage_robust": row["end_coverage_robust"],
                    "longest_deletion_run": row["longest_deletion_run"],
                    "longest_deletion_run_frac": row["longest_deletion_run_frac"],
                    "excess_repetition_rate": row["excess_repetition_rate"],
                    "max_ngram_run": row["max_ngram_run"],
                    "max_ngram_run_gram": row["max_ngram_run_gram"],
                    "max_excess_ngram_count": row["max_excess_ngram_count"],
                    "max_excess_ngram_count_gram": row["max_excess_ngram_count_gram"],
                    "loop_flag": row["loop_flag"],
                    "raw_duration_sec": row["raw_duration_sec"],
                    "reference_duration_sec": row["reference_duration_sec"],
                    "silence_ratio": row["silence_ratio"],
                    "longest_silence_sec": row["longest_silence_sec"],
                    "text_tokens": run["text_tokens"],
                    "generated_speech_tokens": run["generated_speech_tokens"],
                    "audio_rms": run.get("audio", {}).get("rms"),
                    "audio_peak": run.get("audio", {}).get("peak"),
                    "audio_has_nan": run.get("audio", {}).get("has_nan"),
                    "asr_ok": row["asr_ok"],
                    "asr_error": row["asr_error"],
                    "invalid_reason": row["invalid_reason"],
                    "n_asr_segments": len(row.get("asr_segments") or []),
                    "asr_text_head": (row.get("asr_text") or "")[:300],
                    "asr_text_tail": (row.get("asr_text") or "")[-300:],
                    "source_contains_tail_words": None,
                    "bench_words": bench["words"],
                    "bench_sentences": bench["sentences"],
                }
                # is the tail of the transcript actually present in the source text?
                tail = norm_words((row.get("asr_text") or ""), row["normalization_variant"])[-8:]
                ref_set = set(norm_words(bench["text_ref"], row["normalization_variant"]))
                if tail:
                    cases[f"{label}:{row['run_id']}"]["source_contains_tail_words"] = {
                        "tail_words": tail,
                        "n_in_source": sum(1 for w in tail if w in ref_set),
                    }
    return cases


# ------------------------------------------------------------------- figures


def make_figures(analysis: dict[str, Any], data: dict[str, Any], outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    def _log_ticks(ax: Any) -> None:
        """Bucket-centre ticks; the default log minor labels collide at this size."""
        ticks = [20, 30, 50, 75, 150, 300, 600]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    e1 = data["sets"]["E1_native"]["per_item"]
    e0 = data["sets"]["E0_official"]["per_item"]
    colors = {"B0": "#1f77b4", "B1": "#2ca02c", "B2": "#ff7f0e", "B3": "#d62728", "B4": "#9467bd"}

    # (a) generated vs human duration, log-log, colour = bucket, marker = system
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for b in BUCKETS:
        xs = [r["reference_duration_sec"] for r in e1 if r["bucket"] == b]
        ys = [r["raw_duration_sec"] for r in e1 if r["bucket"] == b]
        ax.scatter(xs, ys, c=colors[b], marker="o", s=34, label=f"E1 native {b}")
        xs0 = [r["reference_duration_sec"] for r in e0 if r["bucket"] == b]
        ys0 = [r["raw_duration_sec"] for r in e0 if r["bucket"] == b]
        ax.scatter(xs0, ys0, facecolors="none", edgecolors=colors[b], marker="s", s=34)
    lim = [20, 800]
    ax.plot(lim, lim, "k--", lw=1, label="y = x (human duration)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(3, 800)
    _log_ticks(ax)
    ax.set_xlabel("human reference duration, s (log)")
    ax.set_ylabel("generated duration, s (log)")
    ax.set_title("Generated vs human duration\nfilled circle = E1 native, open square = E0 official")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    p = outdir / "pilot_duration_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # (b) status stacked bars per bucket, E1 and E0 side by side
    present = [s for s in STATUSES if any(analysis["content"][lbl][b]["status_counts"][s] for lbl in ("E1_native", "E0_official") for b in BUCKETS)]
    stat_colors = {"complete": "#2ca02c", "degraded": "#ff7f0e", "early_eos": "#d62728", "empty_or_invalid_audio": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.38
    xs = list(range(len(BUCKETS)))
    for off, lbl in ((-width / 2, "E1_native"), (width / 2, "E0_official")):
        bottom = [0.0] * len(BUCKETS)
        for s in present:
            vals = [analysis["content"][lbl][b]["status_counts"][s] for b in BUCKETS]
            ax.bar(
                [x + off for x in xs], vals, width, bottom=bottom,
                color=stat_colors.get(s, "#8c564b"),
                edgecolor="black", linewidth=0.4,
                label=s if lbl == "E1_native" else None,
                hatch="" if lbl == "E1_native" else "//",
            )
            bottom = [a + b_ for a, b_ in zip(bottom, vals)]
    ax.set_xticks(xs)
    ax.set_xticklabels(BUCKETS)
    ax.set_ylabel("items (12 per bucket)")
    ax.set_ylim(0, 15.6)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_title("PLAN §3.4 final status per bucket\nleft bar = E1 native, right bar (hatched) = E0 official")
    ax.legend(fontsize=8, ncol=4, loc="upper center", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = outdir / "pilot_status_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # (c) WER-all per bucket with the ASR floor line
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for lbl, colour, marker in (("E1_native", "#d62728", "o"), ("E0_official", "#1f77b4", "s")):
        ys = [100 * analysis["content"][lbl][b]["wer"]["mean"] for b in BUCKETS]
        ax.plot(xs, ys, marker=marker, color=colour, label=f"{lbl} WER-all (macro)")
    floor = [100 * analysis["content"]["E1_native"][b]["floor_wer"]["mean"] for b in BUCKETS]
    ax.plot(xs, floor, "k--", marker="^", label="ASR floor on human audio (4 texts/bucket)")
    ax.set_xticks(xs)
    ax.set_xticklabels(BUCKETS)
    ax.set_ylabel("WER-all, % (macro mean, not capped)")
    ax.set_title("WER-all per length bucket vs the ASR error floor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "pilot_wer_by_bucket.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # (d) per-item coverage vs human duration, log x
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for b in BUCKETS:
        ax.scatter([r["reference_duration_sec"] for r in e1 if r["bucket"] == b],
                   [r["source_coverage"] for r in e1 if r["bucket"] == b],
                   c=colors[b], marker="o", s=34, label=f"E1 {b}")
        ax.scatter([r["reference_duration_sec"] for r in e0 if r["bucket"] == b],
                   [r["source_coverage"] for r in e0 if r["bucket"] == b],
                   facecolors="none", edgecolors=colors[b], marker="s", s=34)
    ax.set_xscale("log")
    _log_ticks(ax)
    ax.set_xlim(20, 800)
    ax.set_xlabel("human reference duration, s (log)")
    ax.set_ylabel("source coverage (share of reference words read)")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Per-item source coverage vs text length\nfilled circle = E1 native, open square = E0 official")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2, loc="center left")
    fig.tight_layout()
    p = outdir / "pilot_coverage_vs_length.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))
    return written


# -------------------------------------------------------------------- report


def print_report(analysis: dict[str, Any]) -> None:
    dur, tok, sysm, cont, voice = (
        analysis["duration"], analysis["tokens"], analysis["system"], analysis["content"], analysis["voice"]
    )
    print("# pilot_analysis.py — RQ1/H1 numbers\n")

    print("## Verification of the stored per-item metrics")
    for label, v in analysis["verification"].items():
        print(f"- {label}: {v['checked']} items re-aligned, {v['n_mismatch']} mismatches")
        for m in v["mismatches"]:
            print(f"    ! {m}")
    print()

    for label in ("E1_native", "E0_official"):
        print(f"## Duration, {label} (seconds)\n")
        rows = []
        for b in BUCKETS:
            d = dur[label][b]
            rows.append([
                b, d["n"],
                f"{_fmt(d['human_sec']['min'],1)} / {_fmt(d['human_sec']['median'],1)} / {_fmt(d['human_sec']['max'],1)}",
                f"{_fmt(d['generated_sec']['min'],1)} / {_fmt(d['generated_sec']['median'],1)} / {_fmt(d['generated_sec']['max'],1)}",
                f"{_fmt(d['ratio']['min'],3)} / {_fmt(d['ratio']['median'],3)} / {_fmt(d['ratio']['max'],3)}",
                _fmt(d["silence_ratio"]["median"], 4),
                _fmt(d["speaking_rate_wpm"]["median"], 1),
            ])
        print(_table(["bucket", "n", "human min/med/max", "generated min/med/max", "ratio min/med/max", "silence med", "wpm med"], rows))
        print()

    for label in ("E1_native", "E0_official"):
        print(f"## Tokens, {label}\n")
        rows = []
        for b in BUCKETS:
            t = tok[label][b]
            rows.append([
                b, t["n"],
                f"{_fmt(t['text_tokens']['min'],0)} / {_fmt(t['text_tokens']['median'],0)} / {_fmt(t['text_tokens']['max'],0)}",
                f"{_fmt(t['generated_speech_tokens']['min'],0)} / {_fmt(t['generated_speech_tokens']['median'],0)} / {_fmt(t['generated_speech_tokens']['max'],0)}",
                _fmt(t["gen_per_text_token"]["median"], 3),
                _fmt(t["human_tokens_25hz"]["median"], 0),
                _fmt(t["gen_over_human_tokens"]["median"], 3),
                _fmt(t["context_occupancy"]["median"], 4),
                f"{t['n_at_cap']} / {t['n_below_min_len']}",
            ])
        print(_table(
            ["bucket", "n", "text tok min/med/max", "gen tok min/med/max", "gen/text med",
             "human tok 25Hz med", "gen/human med", "ctx occ med", "at cap / below min_len"], rows))
        print()

    for label in ("E1_native", "E0_official"):
        print(f"## System, {label} (PLAN §9.6)\n")
        rows = []
        for b in list(BUCKETS) + ["ALL"]:
            s = sysm[label][b]
            rows.append([
                b, s["n"],
                f"{_fmt(s['wall_sec']['median'],1)} ({_fmt(s['wall_sec']['min'],1)}-{_fmt(s['wall_sec']['max'],1)})",
                _fmt(s["wall_sec_total"] / 60.0, 1),
                _fmt(s["rtf"]["median"], 3),
                _fmt(s["peak_vram_gb"]["max"], 2),
                _fmt(s["peak_vram_reserved_gb"]["max"], 2),
                _fmt(s["generated_speech_tokens"]["median"], 0),
                _fmt(s["speech_tokens_per_sec_llm"]["median"], 1),
                _fmt(s["speech_tokens_per_sec_wall_true"]["median"], 1),
                ",".join(f"{k}:{v}" for k, v in s["stop_reasons"].items()),
                f"{s['n_oom']}/{s['n_exception']}/{s['n_nan']}",
            ])
        print(_table(
            ["bucket", "n", "wall s med (min-max)", "wall total min", "RTF med", "peak alloc GiB",
             "peak reserved GiB", "gen tok med", "tok/s LLM med", "tok/s wall med", "stop reasons",
             "oom/exc/nan"], rows))
        c = sysm[label]["ALL"]["throughput_field_check"]
        print(f"\n- manifest field `speech_tokens_per_sec_wall` explained by generated/llm_time on "
              f"{c['n_matching_llm_time_denominator']} of {c['n']} runs and by generated/wall_time on "
              f"{c['n_matching_wall_time_denominator']} of {c['n']} (tol {c['tol']}); "
              f"llm+flow_hift+frontend accounts for wall time to within "
              f"{_fmt(c['max_abs_residual_wall_minus_components_sec'], 3)} s; "
              f"LM share of wall time median {_fmt(sysm[label]['ALL']['llm_share_of_wall']['median'], 3)}")
        print()

    for label in ("E1_native", "E0_official"):
        print(f"## Content, {label}\n")
        rows = []
        for b in list(BUCKETS) + ["ALL"]:
            c = cont[label][b]
            rows.append([
                b, c["n"], _fmt(100 * c["complete_rate"], 1),
                f"{_fmt(100*c['wer']['mean'],2)} / {_fmt(100*c['wer']['median'],2)}",
                f"{_fmt(c['coverage']['mean'],3)} / {_fmt(c['coverage']['median'],3)}",
                f"{_fmt(c['end_coverage']['mean'],3)} / {_fmt(c['end_coverage']['median'],3)}",
                f"{_fmt(c['end_coverage_robust']['mean'],3)} / {_fmt(c['end_coverage_robust']['median'],3)}",
                _fmt(c["tail_deletion_rate"]["mean"], 3),
                _fmt(c["longest_deletion_run_frac"]["median"], 3),
                _fmt(100 * c["floor_wer"]["mean"], 2) if c["n_with_floor"] else "n/a",
                _fmt(100 * c["wer_minus_floor"]["mean"], 2) if c["n_with_floor"] else "n/a",
                c["n_with_floor"],
            ])
        print(_table(
            ["bucket", "n", "complete %", "WER mean/med %", "cov mean/med", "endcov mean/med",
             "robust mean/med", "tail del", "longest del run frac med", "floor %", "WER-floor %", "n floor"], rows))
        print()

    print("## Failure taxonomy (PLAN §17 Table 5)\n")
    present = [s for s in STATUSES if any(cont[l][b]["status_counts"][s] for l in cont for b in BUCKETS)]
    rows = []
    for label in ("E1_native", "E0_official"):
        for b in list(BUCKETS) + ["ALL"]:
            c = cont[label][b]
            rows.append([label, b, c["n"]] + [c["status_counts"][s] for s in present])
    print(_table(["checkpoint", "bucket", "n"] + list(present), rows, align_right_from=2))
    print()

    sr = analysis.get("speaking_rate")
    if sr:
        print("## Speaking rate (dataset roots only; §5.9)\n")
        h = sr["human_dataset_roots"]
        print(f"- human readings ({h['n']} texts, duration source {h['duration_source']}): raw-basis wpm "
              f"min {_fmt(h['wpm_raw']['min'],1)} / median {_fmt(h['wpm_raw']['median'],1)} / max {_fmt(h['wpm_raw']['max'],1)}")
        for label in ("E1_native", "E0_official"):
            d = sr[label]
            for key in ("dataset_roots", "dataset_roots_B4", "all_roots_B4"):
                k = d[key]
                print(f"- {label} {key} (n={k['n']}): raw wpm "
                      f"{_fmt(k['wpm_raw']['min'],1)} / {_fmt(k['wpm_raw']['median'],1)} / {_fmt(k['wpm_raw']['max'],1)}"
                      f"   voiced wpm {_fmt(k['wpm_voiced']['min'],1)} / {_fmt(k['wpm_voiced']['median'],1)} / "
                      f"{_fmt(k['wpm_voiced']['max'],1)}")
            s_ = d["dataset_roots"]["silence_ratio"]
            print(f"    silence ratio on the dataset roots: {_fmt(s_['min'],3)} / {_fmt(s_['median'],3)} / {_fmt(s_['max'],3)}")
        print()

    sc = analysis.get("split_context")
    if sc:
        print("## Split context (§7.4)\n")
        print(f"- {sc['n_channels']} channels, {_fmt(sc['total_hours'],2)} h total; hours by split "
              + ", ".join(f"{k} {_fmt(v,2)}" for k, v in sc["hours_by_split"].items()))
        print(f"- {sc['n_catalogue']} catalogue channels carry {_fmt(sc['catalogue_hours'],2)} h "
              f"({_fmt(100*sc['catalogue_share'],1)} %); the {sc['n_catalogue_train_only']} that stay in train carry "
              f"{_fmt(sc['catalogue_train_only_hours'],2)} h ({_fmt(100*sc['catalogue_train_only_share'],1)} %)")
        print(f"- catalogue channels NOT in train: {sc['catalogue_held_out']}")
        print()

    print("## Where the native reading stops\n")
    s = analysis["stop"]["summary"]
    print(f"- coherent items (prefix coverage >= {s['coherent_min_prefix_coverage']}): {s['n_coherent']} / {s['n_items']}"
          f"  by bucket {s['n_coherent_by_bucket']}")
    print(f"- distance from the stop word to the nearest sentence end (coherent items): "
          f"min {_fmt(s['coherent_dist']['min'],1)} / median {_fmt(s['coherent_dist']['median'],1)} / max {_fmt(s['coherent_dist']['max'],1)}")
    for k in (0, 1, 2, 3):
        print(f"- stop within +-{k} words of a sentence end: {s[f'coherent_within_{k}']} / {s['n_coherent']}")
    print(f"- chance baseline, mean over the same texts: exactly on a sentence end "
          f"{_fmt(s['coherent_chance_within_0_mean'],3)}, within +-2 words "
          f"{_fmt(s['coherent_chance_within_2_mean'],3)}")
    print(f"- ALL items that aligned at all ({s['n_with_stop']} / {s['n_items']}), no coherence filter: "
          f"on a sentence end {s['all_within_0']} / {s['n_with_stop']} (chance {_fmt(s['all_chance_within_0_mean'],3)}), "
          f"within +-2 {s['all_within_2']} / {s['n_with_stop']} (chance {_fmt(s['all_chance_within_2_mean'],3)}), "
          f"distance median {_fmt(s['all_dist']['median'],1)} max {_fmt(s['all_dist']['max'],1)}")
    print(f"- restricted to stops after >= {s['substantial_min_stop_words']} source words "
          f"({s['n_substantial']} items, by bucket {s['n_substantial_by_bucket']}): "
          f"on a sentence end {s['substantial_within_0']} / {s['n_substantial']}, "
          f"within +-2 {s['substantial_within_2']} / {s['n_substantial']}, "
          f"distance median {_fmt(s['substantial_dist']['median'],1)} max {_fmt(s['substantial_dist']['max'],1)}")
    print(f"- ALL {s['n_texts']} benchmark texts end on a sentence end "
          f"({s['n_texts_ending_on_a_sentence_end']} of {s['n_texts_with_usable_boundaries']} with usable boundaries), "
          f"so a run that read to the last word is on a boundary BY CONSTRUCTION.")
    print(f"- full readings among the coherent items: {s['coherent_n_full_read']} / {s['n_coherent']}, "
          f"of which {s['coherent_full_within_0']} on a sentence end (trivially)")
    print(f"- PARTIAL readings (stopped before the last word), coherent: {s['coherent_n_partial_read']}, "
          f"on a sentence end {s['coherent_partial_within_0']} / {s['coherent_n_partial_read']} "
          f"(chance {_fmt(s['coherent_partial_chance_within_0_mean'],3)}), "
          f"within +-2 {s['coherent_partial_within_2']} (chance {_fmt(s['coherent_partial_chance_within_2_mean'],3)}), "
          f"distance median {_fmt(s['coherent_partial_dist']['median'],1)} max {_fmt(s['coherent_partial_dist']['max'],1)}")
    for gname in ("B0B1", "B2plus"):
        g = s[f"coherent_partial_{gname}"]
        print(f"    {gname}: {g['within_0']} / {g['n']} on a sentence end "
              f"(chance {_fmt(g['chance_within_0_mean'],3)}), within +-2 {g['within_2']} / {g['n']}")
    print(f"    partial by bucket: "
          + ", ".join(f"{b} {v['within_0']}/{v['n']}" for b, v in s["coherent_partial_by_bucket"].items()))
    print(f"- same split on the >= 20-word subset ({s['n_substantial']} items): "
          f"full {s['substantial_n_full_read']} (on a boundary {s['substantial_full_within_0']}), "
          f"partial {s['substantial_n_partial_read']} with {s['substantial_partial_within_0']} on a sentence end "
          f"(chance {_fmt(s['substantial_partial_chance_within_0_mean'],3)}), within +-2 "
          f"{s['substantial_partial_within_2']}")
    print(f"- same split without the coherence filter ({s['n_with_stop']} items): "
          f"full {s['all_n_full_read']} (on a boundary {s['all_full_within_0']}), "
          f"partial {s['all_n_partial_read']} with {s['all_partial_within_0']} on a sentence end "
          f"(chance {_fmt(s['all_partial_chance_within_0_mean'],3)})")
    s0 = analysis["stop_E0"]["summary"]
    print(f"- E0 official for comparison: coherent {s0['n_coherent']} / {s0['n_items']}, "
          f"on a sentence end {s0['coherent_within_0']} / {s0['n_coherent']}, "
          f"within +-2 {s0['coherent_within_2']} / {s0['n_coherent']}; "
          f"full readings {s0['coherent_n_full_read']} (on a boundary {s0['coherent_full_within_0']}), "
          f"partial {s0['coherent_n_partial_read']} with {s0['coherent_partial_within_0']} on a sentence end")
    print()
    print("- noise buffers and calls per system (PLAN §3.2 operational change, §3.1 contract):")
    for label in ("E1_native", "E0_official"):
        a = sysm[label]["ALL"]
        print(f"    {label}: flow buffer {a['flow_noise_buffer_sec']} s, HiFT buffer {a['hift_noise_buffer_sec']} s, "
              f"llm.inference calls per item min/med/max {_fmt(a['llm_inference_calls']['min'],0)}/"
              f"{_fmt(a['llm_inference_calls']['median'],0)}/{_fmt(a['llm_inference_calls']['max'],0)}, "
              f"official chunks per item min/med/max {_fmt(a['n_chunks']['min'],0)}/"
              f"{_fmt(a['n_chunks']['median'],0)}/{_fmt(a['n_chunks']['max'],0)}")
    print()
    rows = []
    for it in analysis["stop"]["items"]:
        rows.append([
            it["run_id"], it["bucket"], it["n_ref_words"], it["n_sentences"],
            it["stop_word_index"], _fmt(it["stop_frac"], 3), _fmt(it["prefix_coverage"], 3),
            _fmt(it["dist_to_boundary"], 0), it["robust_stop_word_index"], _fmt(it["dist_to_boundary_robust"], 0),
        ])
    print(_table(["run_id", "bucket", "N_ref", "n_sent", "stop idx", "stop frac", "prefix cov",
                  "dist to sent end", "robust stop idx", "robust dist"], rows))
    print()

    print("## Longer input, earlier stop (absolute units)\n")
    le = analysis["length_effect"]
    for label in ("E1_native", "E0_official"):
        d = le[label]
        rows = []
        for b in BUCKETS:
            pb = d["per_bucket"][b]
            rows.append([
                b, pb["n"],
                _fmt(pb["text_tokens"]["median"], 0),
                f"{_fmt(pb['words_read_abs']['min'],1)} / {_fmt(pb['words_read_abs']['median'],1)} / {_fmt(pb['words_read_abs']['max'],1)}",
                f"{_fmt(pb['stop_word_index']['min'],0)} / {_fmt(pb['stop_word_index']['median'],0)} / {_fmt(pb['stop_word_index']['max'],0)}",
                f"{_fmt(pb['generated_sec']['min'],1)} / {_fmt(pb['generated_sec']['median'],1)} / {_fmt(pb['generated_sec']['max'],1)}",
                f"{_fmt(pb['generated_speech_tokens']['min'],0)} / {_fmt(pb['generated_speech_tokens']['median'],0)} / {_fmt(pb['generated_speech_tokens']['max'],0)}",
            ])
        print(f"### {label}\n")
        print(_table(
            ["bucket", "n", "text tok med", "source words READ min/med/max",
             "stop word idx min/med/max", "generated s min/med/max", "gen tok min/med/max"], rows))
        print(
            f"\n- chains (root x voice, 5 nested prefixes each): {d['n_chains']}; "
            f"B4 reads fewer source words than B0 in {d['n_chains_b4_lt_b0_words']}/{d['n_chains']}, "
            f"fewer than B1 in {d['n_chains_b4_lt_b1_words']}/{d['n_chains']}; "
            f"B4 audio shorter than B0 audio in {d['n_chains_b4_lt_b0_sec']}/{d['n_chains']}"
        )
        print(f"- bucket where the chain reads the most source words: {d['argmax_bucket_counts']}; "
              f"non-increasing after that peak in {d['n_chains_nonincreasing_from_peak']}/{d['n_chains']} chains")
        print(f"- Spearman over all {len(d['items'])} items: text tokens vs source words read "
              f"{_fmt(d['spearman_texttok_vs_words_read'],3)}, vs generated seconds "
              f"{_fmt(d['spearman_texttok_vs_generated_sec'],3)}, vs generated speech tokens "
              f"{_fmt(d['spearman_texttok_vs_generated_tokens'],3)}")
        print()
    print("### E1 native chains, source words read at B0..B4\n")
    rows = []
    for c in le["E1_native"]["chains"]:
        rows.append([c["root_id"], c["voice_id"]] + [_fmt(w, 1) for w in c["words_read"]]
                    + [_fmt(s, 1) for s in c["generated_sec"]])
    print(_table(["root_id", "voice"] + [f"words {b}" for b in BUCKETS] + [f"sec {b}" for b in BUCKETS],
                 rows, align_right_from=2))
    print()

    print("## Per voice\n")
    rows = []
    for label in ("E1_native", "E0_official"):
        for v, d in voice[label].items():
            if v.startswith("_"):
                continue
            rows.append([
                label, v, d["n"], _fmt(100 * d["complete_rate"], 1),
                _fmt(100 * d["wer"]["mean"], 2), _fmt(d["coverage"]["mean"], 3),
                _fmt(d["generated_sec"]["median"], 1), _fmt(d["generated_speech_tokens"]["median"], 0),
                _fmt(d["duration_ratio"]["mean"], 3),
            ])
    print(_table(["checkpoint", "voice", "n", "complete %", "WER %", "coverage", "gen s med", "gen tok med", "dur ratio"], rows, align_right_from=2))
    print()
    for label in ("E1_native", "E0_official"):
        p = voice[label]["_paired_female_minus_male"]
        print(f"- {label} paired (female - male, same text, n={p['n_pairs']}): "
              f"generated s median {_fmt(p['d_generated_sec']['median'],2)} "
              f"(female longer in {p['n_female_longer']}/{p['n_pairs']}), "
              f"coverage median {_fmt(p['d_coverage']['median'],3)}, "
              f"WER median {_fmt(p['d_wer']['median'],4)}, tokens median {_fmt(p['d_tokens']['median'],1)}")
    print()

    print("## Named cases\n")
    for key, c in analysis["cases"].items():
        print(f"### {key} — {c['status']}")
        print(f"- reason: {c['final_status_reason']}")
        print(f"- N_ref {c['n_ref_words']} words, hypothesis {c['n_hyp_words']} words; "
              f"hits {c['hits']}, sub {c['substitutions']}, del {c['deletions']}, ins {c['insertions']}")
        pct = lambda v: "n/a" if v is None else _fmt(100 * v, 2)  # noqa: E731 (external texts have no floor)
        print(f"- WER {pct(c['wer'])} %, floor {pct(c['floor_wer'])} %, WER-floor {pct(c['wer_minus_floor'])} pp; "
              f"coverage {_fmt(c['coverage'],3)}, EndCoverage {_fmt(c['end_coverage'],3)}, robust {_fmt(c['end_coverage_robust'],3)}")
        print(f"- longest deletion run {c['longest_deletion_run']} words ({_fmt(c['longest_deletion_run_frac'],3)} of the text); "
              f"excess repetition {_fmt(c['excess_repetition_rate'],4)}, max n-gram run {c['max_ngram_run']} ({c['max_ngram_run_gram']}), loop flag {c['loop_flag']}")
        print(f"- audio {c['raw_duration_sec']} s vs human {_fmt(c['reference_duration_sec'],1)} s; "
              f"rms {_fmt(c['audio_rms'],4)}, peak {_fmt(c['audio_peak'],4)}, NaN {c['audio_has_nan']}, "
              f"silence ratio {_fmt(c['silence_ratio'],4)}, ASR segments {c['n_asr_segments']}, "
              f"asr_ok {c['asr_ok']}, invalid_reason {c['invalid_reason']}")
        print(f"- text tokens {c['text_tokens']}, generated speech tokens {c['generated_speech_tokens']}")
        if c["source_contains_tail_words"]:
            t = c["source_contains_tail_words"]
            print(f"- last 8 transcript words {t['tail_words']} — {t['n_in_source']}/8 occur in the source text")
        print(f"- transcript head: {c['asr_text_head']!r}")
        print(f"- transcript tail: {c['asr_text_tail']!r}")
        print()


# ---------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, default=REPO / "data/benchmark/pilot.jsonl")
    ap.add_argument("--e1-per-item", type=Path, default=REPO / "results/v31_base/E1/per_item.jsonl")
    ap.add_argument("--e0-per-item", type=Path, default=REPO / "results/v31_base/E0/per_item.jsonl")
    ap.add_argument("--e1-runs", type=Path, default=REPO / "outputs/v31_base/E1_native/runs.jsonl")
    ap.add_argument("--e0-runs", type=Path, default=REPO / "outputs/v31_base/E0_official/runs.jsonl")
    ap.add_argument("--split-channels", type=Path, default=REPO / "data/manifests/split_channels.json")
    ap.add_argument("--out-json", type=Path, default=REPO / "results/v31_base/analysis/pilot_analysis.json")
    ap.add_argument("--figures-dir", type=Path, default=REPO / "reports/figures/v31")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    data = load_all(args)
    analysis: dict[str, Any] = {
        "inputs": {
            "benchmark": str(args.benchmark),
            "E1_native": {"per_item": str(args.e1_per_item), "runs": str(args.e1_runs)},
            "E0_official": {"per_item": str(args.e0_per_item), "runs": str(args.e0_runs)},
        },
        "verification": {},
    }
    if not args.no_verify:
        for label, blob in data["sets"].items():
            analysis["verification"][label] = verify_alignment(blob["per_item"], label)
        bad = sum(v["n_mismatch"] for v in analysis["verification"].values())
        if bad:
            for label, v in analysis["verification"].items():
                for m in v["mismatches"]:
                    print(f"MISMATCH {m}", file=sys.stderr)
            raise SystemExit(f"recomputation disagrees with per_item.jsonl in {bad} places — refusing to report")

    analysis["duration"] = duration_section(data["sets"])
    analysis["tokens"] = token_section(data["sets"])
    analysis["system"] = system_section(data["sets"])
    analysis["content"] = content_section(data["sets"])
    analysis["voice"] = voice_section(data["sets"])
    analysis["cases"] = case_section(data["sets"])
    analysis["speaking_rate"] = speaking_rate_section(data["sets"], data["benchmark"])
    analysis["split_context"] = split_context_section(args.split_channels)
    analysis["stop"] = stop_analysis(data["sets"]["E1_native"]["per_item"])
    analysis["stop_E0"] = stop_analysis(data["sets"]["E0_official"]["per_item"])
    analysis["length_effect"] = length_effect_section(
        data["sets"],
        {"E1_native": analysis["stop"]["items"], "E0_official": analysis["stop_E0"]["items"]},
    )

    if not args.no_figures:
        analysis["figures"] = make_figures(analysis, data, args.figures_dir)

    print_report(analysis)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(analysis, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[written] {args.out_json}")
    for p in analysis.get("figures", []):
        print(f"[written] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
