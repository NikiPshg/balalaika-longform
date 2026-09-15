#!/usr/bin/env python
"""Merge several evaluated experiments into the PLAN.md §17 main table.

Input
    one or more ``results/<exp>/per_item.jsonl`` files, or the directories that
    contain them.  ``LABEL=PATH`` overrides the row label (default: the
    ``checkpoint`` / ``experiment_id`` recorded in the rows themselves).

Output (under --output-dir)
    <name>.md    §17 main table (rows = experiment x bucket) + §17 Table 5
                 failure taxonomy + provenance
    <name>.csv   the same main table, machine-readable, unrounded

The numbers are produced by exactly the same :func:`eval.metrics.aggregate` call
that ``scripts/run_evaluation.py`` uses for one experiment, so a row here is
identical to the corresponding row of that experiment's ``summary.md``.  Nothing
is recomputed from audio and no threshold is applied a second time: the final
PLAN.md §3.4 status was already frozen into ``per_item.jsonl`` by the evaluator
(configs/eval.yaml).

Hard failures
    * a per_item.jsonl that is missing (or empty);
    * the same ``run_id`` appearing twice under one label -- that would count one
      generation twice in "Attempted" and in every macro mean;
    * a row without ``bucket`` / ``status`` / ``checkpoint``.

``RMST words`` (PLAN.md §9.3 degradation events, A4 part 2) is still a
placeholder and prints as ``n/a`` unless the per-item rows already carry it.

Voice drift and speaker similarity (PLAN.md §9.5, A5; Lead ruling 2026-08-30)
    A5's measurement lives in its own per-item file,
    ``results/v31_drift/<exp>/per_item.jsonl`` -- one row per generated item,
    carrying the cosine similarity of every 10 s voiced window to the
    enrollment reference.  Point this script at it and two §17 columns fill in::

        --drift LABEL=results/v31_drift/<exp>         (repeatable), or
        --drift-root [DIR]    auto-discovery: DIR/<label>/per_item.jsonl for
                              every label in the table (DIR defaults to
                              results/v31_drift)

    ``Voice drift`` = *median ``delta`` / median ``slope_per_min``*, and the new
    ``Spk sim`` = *median ``sim_median``*, per experiment x bucket -- exactly the
    definitions A5 published in ``results/v31_drift/voice_drift_column.csv``:
    level and delta are medians over the items that have at least one window,
    the slope is a median over the items whose voiced span reaches
    ``DRIFT_SPAN_MIN_SEC`` (60 s), because a per-minute slope fitted on a 20 s
    output is an extrapolation rather than a measurement.  Without ``--drift``
    both columns stay ``n/a``, exactly as before.  ``Spk sim`` is what makes the
    §9.5 finding visible in the main table: E3's similarity LEVEL falls with
    length while its slope does not, i.e. an onset offset, not drift.

    Nothing is recomputed from audio here either -- this is A5's per-item output
    re-aggregated, and a label named in ``--drift`` that is not in the table is a
    hard failure, so a typo cannot quietly become an ``n/a`` column.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.final_status import KNOWN_STATUSES  # noqa: E402
from eval.metrics import aggregate  # noqa: E402

BUCKET_ORDER = ("B0", "B1", "B2", "B3", "B4")
ALL = "ALL"

# per-item keys that would carry the two not-yet-implemented §17 columns
RMST_KEYS = ("rmst_words", "rmst_stable_words")
DRIFT_KEYS = ("voice_drift_slope_per_min", "voice_drift", "speaker_drift_slope_per_min")

# A5's voice-drift files (PLAN §9.5).  A slope "per minute" fitted on an output that
# does not last a minute is an extrapolation, so the slope median is taken only over
# items whose voiced span reaches this; the level and delta medians use every item
# that produced at least one window.  Frozen by A5, mirrored here.
DRIFT_SPAN_MIN_SEC = 60.0
DRIFT_ROOT_DEFAULT = REPO_ROOT / "results" / "v31_drift"


class AggregateError(RuntimeError):
    """An input cannot be aggregated without silently distorting the table."""


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AggregateError(f"{path}:{i}: bad json: {exc}") from exc
    return rows


def resolve_input(spec: str) -> tuple[str | None, Path]:
    """``[LABEL=]PATH`` -> (label or None, path to per_item.jsonl)."""
    label = None
    raw = spec
    if "=" in spec and not Path(spec).exists():
        label, raw = spec.split("=", 1)
    p = Path(raw)
    if p.is_dir():
        p = p / "per_item.jsonl"
    if not p.exists():
        raise AggregateError(f"{spec}: no per_item.jsonl at {p}")
    return label, p


def load_inputs(specs: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """Return (rows, provenance). Every row is tagged with its label and source."""
    rows: list[dict] = []
    provenance: list[dict] = []
    seen: dict[tuple[str, str], str] = {}
    for spec in specs:
        label, path = resolve_input(spec)
        items = _read_jsonl(path)
        if not items:
            raise AggregateError(f"{path}: no rows")
        labels = {label or str(r.get("checkpoint") or r.get("experiment_id")) for r in items}
        for r in items:
            lab = label or str(r.get("checkpoint") or r.get("experiment_id"))
            for key in ("bucket", "status"):
                if r.get(key) is None:
                    raise AggregateError(f"{path}: a row of {lab} has no {key!r}")
            if lab == "None":
                raise AggregateError(f"{path}: a row has neither checkpoint nor experiment_id")
            rid = str(r.get("run_id"))
            if (lab, rid) in seen:
                raise AggregateError(
                    f"duplicate run_id {rid!r} for {lab!r}: {seen[(lab, rid)]} and {path}; "
                    "aggregating it twice would double-count one generation"
                )
            seen[(lab, rid)] = str(path)
            r["__label__"] = lab
            rows.append(r)
        meta = {}
        summary_json = path.parent / "summary.json"
        if summary_json.exists():
            try:
                meta = json.load(open(summary_json, "rt", encoding="utf-8")).get("meta", {})
            except Exception:  # provenance is best-effort, never fatal
                meta = {}
        provenance.append({
            "spec": spec, "path": str(path), "labels": sorted(labels), "n_rows": len(items),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
            "eval_config": meta.get("eval_config"),
            "eval_config_version": meta.get("eval_config_version"),
            "asr_model_id": meta.get("asr_model_id"),
            "normalization_variant": meta.get("normalization_variant"),
            "final_status_thresholds": meta.get("final_status_thresholds"),
        })
    return rows, provenance


# ------------------------------------------------------------ voice drift (PLAN §9.5)


def _median(values: Sequence[float]) -> float | None:
    xs = sorted(values)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def drift_span_sec(row: dict) -> float:
    """Voiced span the windows of one item cover, in seconds (0 if fewer than two)."""
    w = row.get("windows") or []
    if len(w) < 2:
        return 0.0
    return max(x["t_end_sec"] for x in w) - min(x["t_start_sec"] for x in w)


def resolve_drift(spec: str) -> tuple[str, Path]:
    """``LABEL=PATH`` -> (label, path to per_item.jsonl).  The label is required."""
    if "=" not in spec:
        raise AggregateError(
            f"--drift {spec!r}: expected LABEL=PATH (the label must name a row of the table)"
        )
    label, raw = spec.split("=", 1)
    if not label:
        raise AggregateError(f"--drift {spec!r}: empty label")
    p = Path(raw)
    if p.is_dir():
        p = p / "per_item.jsonl"
    if not p.exists():
        raise AggregateError(f"--drift {spec!r}: no per_item.jsonl at {p}")
    return label, p


def load_drift(specs: Sequence[str], root: Path | None,
               labels: Sequence[str]) -> tuple[dict[str, dict], list[dict]]:
    """Load A5's per-item drift rows for the labels of the table.

    ``specs`` are explicit ``LABEL=PATH`` pairs; ``root`` additionally auto-discovers
    ``root/<label>/per_item.jsonl`` for any label not named explicitly.  A label that is
    not a row of the table is an error -- silently producing an all-``n/a`` column is the
    failure mode this guard exists for.  Auto-discovery, by contrast, is allowed to find
    nothing: a checkpoint A5 did not measure simply keeps its ``n/a``.
    """
    known = set(labels)
    chosen: "OrderedDict[str, Path]" = OrderedDict()
    for spec in specs:
        label, path = resolve_drift(spec)
        if label not in known:
            raise AggregateError(
                f"--drift {spec!r}: {label!r} is not a row of this table "
                f"(labels: {', '.join(labels)})"
            )
        chosen[label] = path
    if root is not None:
        if not root.is_dir():
            raise AggregateError(f"--drift-root {root}: not a directory")
        for label in labels:
            if label in chosen:
                continue
            cand = root / label / "per_item.jsonl"
            if cand.exists():
                chosen[label] = cand

    out: dict[str, dict] = {}
    provenance: list[dict] = []
    for label, path in chosen.items():
        raw = _read_jsonl(path)
        meta = raw[0] if raw and raw[0].get("_meta") else {}
        items = [r for r in raw if not r.get("_meta")]
        if not items:
            raise AggregateError(f"{path}: no drift rows for {label!r}")
        for r in items:
            if r.get("bucket") is None:
                raise AggregateError(f"{path}: a drift row of {label!r} has no 'bucket'")
        out[label] = {"path": path, "rows": items, "meta": meta,
                      "by_bucket": drift_by_bucket(items)}
        provenance.append({
            "label": label,
            "path": str(path),
            "n_rows": len(items),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
            "encoder": meta.get("encoder"),
            "revision": meta.get("revision"),
            "window_sec": meta.get("window_sec"),
            "hop_sec": meta.get("hop_sec"),
            "t_voice_similarity_min": meta.get("t_voice_similarity_min"),
            "t_voice_min_run": meta.get("t_voice_min_run"),
        })
    return out, provenance


def drift_by_bucket(rows: Sequence[dict]) -> dict[str, dict]:
    """A5's per-bucket (and ALL) medians, re-derived from the per-item rows.

    Buckets outside the frozen B0-B4 grid (the E11 short-tail S0/S1) are appended
    after it in name order, mirroring ``build_table``'s ``order``; for benchmarks
    that only use B0-B4 the result is byte-identical to before (A11-short,
    2026-08-31).
    """
    present = {str(r["bucket"]) for r in rows}
    buckets = ([b for b in BUCKET_ORDER if b in present]
               + sorted(present - set(BUCKET_ORDER)))
    out: dict[str, dict] = {}
    for b in (*buckets, ALL):
        sel = rows if b == ALL else [r for r in rows if str(r["bucket"]) == b]
        with_windows = [r for r in sel if (r.get("n_windows") or 0) > 0]
        long_enough = [r for r in with_windows if drift_span_sec(r) >= DRIFT_SPAN_MIN_SEC]
        out[b] = {
            "n": len(sel),
            "n_with_windows": len(with_windows),
            "n_span_ge_min": len(long_enough),
            "spk_sim_median": _median([r["sim_median"] for r in with_windows]),
            # A17-qe3p-eval 2026-09-01: an item can have windows yet a None delta/slope
            # (QE3P robust rv02 B4: no voiced window inside the last 30 s -> last30_median
            # None). Medians skip those items; byte-identical for every older set, where
            # no such rows exist.
            "delta_median": _median([r["delta"] for r in with_windows
                                     if r["delta"] is not None]),
            "slope_per_min_median": _median([r["slope_per_min"] for r in long_enough
                                             if r["slope_per_min"] is not None]),
            "t_voice_fired": sum(1 for r in with_windows if r.get("t_voice_sec") is not None),
        }
    return out


def _first_present(rows: Sequence[dict], keys: Sequence[str]) -> str | None:
    for k in keys:
        if any(r.get(k) is not None for r in rows):
            return k
    return None


def build_table(rows: Sequence[dict], drift: dict[str, dict] | None = None) -> list[dict]:
    """One aggregated record per (experiment, bucket), plus an ALL row per experiment.

    ``drift`` is the optional ``{label: {...}}`` from :func:`load_drift`; a label it does
    not cover keeps ``None`` in the two §9.5 columns, which render as ``n/a``.
    """
    groups: "OrderedDict[tuple[str, str], list[dict]]" = OrderedDict()
    for r in rows:
        groups.setdefault((r["__label__"], str(r["bucket"])), []).append(r)
    labels = list(dict.fromkeys(r["__label__"] for r in rows))
    for lab in labels:
        groups[(lab, ALL)] = [r for r in rows if r["__label__"] == lab]

    def order(key: tuple[str, str]) -> tuple:
        lab, bucket = key
        b = BUCKET_ORDER.index(bucket) if bucket in BUCKET_ORDER else len(BUCKET_ORDER) + 1
        return (labels.index(lab), bucket == ALL, b, bucket)

    out: list[dict] = []
    for key in sorted(groups, key=order):
        lab, bucket = key
        sub = groups[key]
        agg = aggregate(sub)
        floors = [r["floor_wer"] for r in sub if r.get("floor_wer") is not None]
        wmf = [r["wer_minus_floor"] for r in sub if r.get("wer_minus_floor") is not None]
        rmst_key = _first_present(sub, RMST_KEYS)
        drift_key = _first_present(sub, DRIFT_KEYS)
        rec = {
            "experiment": lab,
            "bucket": bucket,
            "n_attempted": agg["n_attempted"],
            "complete_rate": agg.get("complete_rate"),
            "gen_complete_rate": (
                sum(1 for r in sub if r.get("gen_status_complete")) / len(sub)) if sub else None,
            "valid_rate": agg.get("valid_rate"),
            "n_valid": agg.get("n_valid"),
            "macro_wer_all": agg.get("macro_wer_all"),
            "macro_wer_all_n_inf": agg.get("macro_wer_all_n_inf"),
            "micro_wer_all": agg.get("micro_wer_all"),
            "macro_wer_valid": agg.get("macro_wer_valid"),
            "macro_cer_all": agg.get("macro_cer_all"),
            "macro_end_coverage_all": agg.get("macro_end_coverage_all"),
            "macro_end_coverage_robust_all": agg.get("macro_end_coverage_robust_all"),
            "macro_source_coverage_all": agg.get("macro_source_coverage_all"),
            "macro_tail_deletion_rate_all": agg.get("macro_tail_deletion_rate_all"),
            "macro_excess_repetition_rate_all": agg.get("macro_excess_repetition_rate_all"),
            "macro_max_ngram_run_all": agg.get("macro_max_ngram_run_all"),
            "n_loop_flag_all": agg.get("n_loop_flag_all"),
            "macro_duration_ratio_all": agg.get("macro_duration_ratio_all"),
            "floor_wer_macro": (sum(floors) / len(floors)) if floors else None,
            "wer_minus_floor_macro": (sum(wmf) / len(wmf)) if wmf else None,
            "n_with_floor": len(floors),
            "rmst_words": (
                sum(r[rmst_key] for r in sub if r.get(rmst_key) is not None)
                / sum(1 for r in sub if r.get(rmst_key) is not None)) if rmst_key else None,
            "voice_drift": (
                sum(r[drift_key] for r in sub if r.get(drift_key) is not None)
                / sum(1 for r in sub if r.get(drift_key) is not None)) if drift_key else None,
        }
        d = ((drift or {}).get(lab) or {}).get("by_bucket", {}).get(bucket)
        if d is not None and d["n"] > rec["n_attempted"]:
            raise AggregateError(
                f"{lab}/{bucket}: the drift file has {d['n']} rows but the table has "
                f"{rec['n_attempted']} attempted items -- these are not the same run"
            )
        rec.update({
            "spk_sim_median": d["spk_sim_median"] if d else None,
            "voice_drift_delta_median": d["delta_median"] if d else None,
            "voice_drift_slope_per_min_median": d["slope_per_min_median"] if d else None,
            "drift_n": d["n"] if d else None,
            "drift_n_with_windows": d["n_with_windows"] if d else None,
            "drift_n_span_ge_min": d["n_span_ge_min"] if d else None,
            "drift_t_voice_fired": d["t_voice_fired"] if d else None,
        })
        for status in KNOWN_STATUSES:
            rec[f"n_{status}"] = sum(1 for r in sub if r.get("status") == status)
        out.append(rec)
    return out


def _fmt(v: Any, pct: bool = False, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if v == float("inf"):
            return "inf"
        return f"{v * 100:.{digits}f}" if pct else f"{v:.{digits}f}"
    return str(v)


def _fmt_drift(rec: dict) -> str:
    """Table 3's `Voice drift` cell: "<median delta> / <median slope per minute>"."""
    if rec.get("voice_drift_delta_median") is None:
        legacy = rec.get("voice_drift")
        return _fmt(legacy, digits=4) if legacy is not None else "n/a"
    delta = _fmt(rec["voice_drift_delta_median"], digits=4)
    slope = _fmt(rec["voice_drift_slope_per_min_median"], digits=4)
    return f"{delta} / {slope}"


def render_markdown(table: Sequence[dict], provenance: Sequence[dict], rows: Sequence[dict],
                    title: str, drift_provenance: Sequence[dict] = ()) -> str:
    out: list[str] = [f"# {title}", ""]
    out.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"- {len(rows)} per-item rows from {len(provenance)} result set(s)")
    out.append("")
    out.append("## Table 3 — main results by length (PLAN.md §17)")
    out.append("")
    out.append("| Checkpoint | Length | Attempted | Complete % | WER-all | CER-all | Coverage | "
               "EndCoverage | EndCov-robust | RMST words | Repeat % | Voice drift | Spk sim | "
               "gen complete % | WER-valid | n valid | "
               "WER−floor | floor WER | max run | dur ratio |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in table:
        inf = f" (+{r['macro_wer_all_n_inf']} inf)" if r.get("macro_wer_all_n_inf") else ""
        out.append(
            f"| {r['experiment']} | {r['bucket']} | {r['n_attempted']} | "
            f"{_fmt(r['complete_rate'], pct=True, digits=1)} | {_fmt(r['macro_wer_all'], pct=True)}{inf} | "
            f"{_fmt(r['macro_cer_all'], pct=True)} | "
            f"{_fmt(r['macro_source_coverage_all'], digits=3)} | "
            f"{_fmt(r['macro_end_coverage_all'], digits=3)} | "
            f"{_fmt(r['macro_end_coverage_robust_all'], digits=3)} | "
            f"{_fmt(r['rmst_words'], digits=1)} | {_fmt(r['macro_excess_repetition_rate_all'], pct=True)} | "
            f"{_fmt_drift(r)} | {_fmt(r.get('spk_sim_median'), digits=4)} | "
            f"{_fmt(r['gen_complete_rate'], pct=True, digits=1)} | "
            f"{_fmt(r['macro_wer_valid'], pct=True)} | {r['n_valid']} | "
            f"{_fmt(r['wer_minus_floor_macro'], pct=True)} | {_fmt(r['floor_wer_macro'], pct=True)} | "
            f"{_fmt(r['macro_max_ngram_run_all'], digits=2)} | "
            f"{_fmt(r['macro_duration_ratio_all'], digits=3)} |"
        )
    out.append("")
    out.append("- **Complete %** is the share of the FINAL PLAN.md §3.4 status `complete` "
               "(evaluator-side rule, configs/eval.yaml); **gen complete %** is the generator's "
               "provisional label. WER is never capped; `(+N inf)` counts items with an empty "
               "reference that are outside the macro mean.")
    out.append("- **Coverage** = macro `source_coverage`: the share of source words matched by a "
               "hit or a substitution anywhere. **EndCoverage** = position of the LAST such word "
               "(pre-registered, gates the §3.4 status). **EndCov-robust** = the same position "
               "but only counting the last run of >= 3 consecutive aligned source words "
               "(Lead 2026-08-28), so a single coincidence deep in the text cannot pretend the "
               "model read that far. Coverage far below EndCoverage means the reading is sparse.")
    out.append("- **RMST words** (§9.3 degradation events) is `n/a` until that metric is "
               "computed, and **Voice drift** / **Spk sim** (§9.5) are `n/a` for any checkpoint "
               "with no drift file; this table never fills them with a guess.")
    if drift_provenance:
        out.append("- **Voice drift** (§9.5) is *median `delta` / median `slope_per_min`* of the "
                   "cosine similarity to the enrollment reference, and **Spk sim** is the median "
                   "`sim_median`, both re-aggregated from A5's per-item drift files. The level and "
                   "the delta use every item with at least one window; the slope uses only items "
                   f"whose voiced span reaches {DRIFT_SPAN_MIN_SEC:.0f} s, so a cell whose items "
                   "are all shorter prints `n/a` for the slope. A checkpoint with no drift file "
                   "keeps `n/a` in both columns. Read the two together: a flat slope with a falling "
                   "**Spk sim** across buckets is an onset offset, not drift within the output.")
    else:
        out.append("- **Voice drift** and **Spk sim** (§9.5) are `n/a`: no drift file was passed "
                   "(`--drift LABEL=results/v31_drift/<exp>` or `--drift-root`).")
    out.append("")
    if drift_provenance:
        out.append("### Voice-drift sources (PLAN.md §9.5, A5)")
        out.append("")
        out.append("| label | source | rows | modified | encoder | window / hop | T_voice |")
        out.append("|---|---|---:|---|---|---|---|")
        for d in drift_provenance:
            win = (f"{d['window_sec']} / {d['hop_sec']} s"
                   if d.get("window_sec") is not None else "-")
            tv = (f"sim < {d['t_voice_similarity_min']} x{d['t_voice_min_run']}"
                  if d.get("t_voice_similarity_min") is not None else "-")
            out.append(f"| {d['label']} | `{d['path']}` | {d['n_rows']} | {d['mtime']} | "
                       f"{d.get('encoder') or '-'} | {win} | {tv} |")
        out.append("")
        fired = sum(r.get("drift_t_voice_fired") or 0 for r in table if r["bucket"] == ALL)
        scanned = sum(r.get("drift_n") or 0 for r in table if r["bucket"] == ALL)
        out.append(f"- `T_voice` fired on **{fired} of {scanned}** scanned items.")
        out.append("")
    out.append("## Table 5 — failure taxonomy (PLAN.md §3.4, final status)")
    out.append("")
    present = [s for s in KNOWN_STATUSES if any(r.get(f"n_{s}") for r in table)]
    out.append("| Checkpoint | Length | n | " + " | ".join(present) + " |")
    out.append("|---|---|---:|" + "---:|" * len(present))
    for r in table:
        out.append(f"| {r['experiment']} | {r['bucket']} | {r['n_attempted']} | "
                   + " | ".join(str(r[f"n_{s}"]) for s in present) + " |")
    out.append("")
    out.append("## Provenance")
    out.append("")
    out.append("| source | labels | rows | modified | ASR | normalization | eval config |")
    out.append("|---|---|---:|---|---|---|---|")
    for p in provenance:
        out.append(f"| `{p['path']}` | {', '.join(p['labels'])} | {p['n_rows']} | {p['mtime']} | "
                   f"{p.get('asr_model_id') or '-'} | {p.get('normalization_variant') or '-'} | "
                   f"{p.get('eval_config') or '-'} v{p.get('eval_config_version') or '-'} |")
    thresholds = {json.dumps(p.get("final_status_thresholds"), sort_keys=True, ensure_ascii=False)
                  for p in provenance if p.get("final_status_thresholds")}
    if len(thresholds) > 1:
        out.append("")
        out.append("**WARNING: the merged experiments were scored with different frozen "
                   "thresholds.** They are not comparable until they are re-evaluated with one "
                   "configs/eval.yaml:")
        for t in sorted(thresholds):
            out.append(f"- `{t}`")
    out.append("")
    return "\n".join(out)


CSV_COLUMNS = (
    "experiment", "bucket", "n_attempted", "complete_rate", "gen_complete_rate", "valid_rate",
    "n_valid", "macro_wer_all", "macro_wer_all_n_inf", "micro_wer_all", "macro_wer_valid",
    "macro_cer_all", "macro_end_coverage_all", "macro_end_coverage_robust_all",
    "macro_source_coverage_all",
    "macro_tail_deletion_rate_all", "macro_excess_repetition_rate_all", "macro_max_ngram_run_all",
    "n_loop_flag_all", "macro_duration_ratio_all", "wer_minus_floor_macro", "floor_wer_macro",
    "n_with_floor", "rmst_words", "voice_drift",
    "spk_sim_median", "voice_drift_delta_median", "voice_drift_slope_per_min_median",
    "drift_n", "drift_n_with_windows", "drift_n_span_ge_min", "drift_t_voice_fired",
) + tuple(f"n_{s}" for s in KNOWN_STATUSES)


def write_csv(path: Path, table: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in table:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge per_item.jsonl files into the §17 main table")
    ap.add_argument("inputs", nargs="+",
                    help="results/<exp>[/per_item.jsonl], optionally prefixed with LABEL=")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "reports" / "tables"))
    ap.add_argument("--name", default="main_table")
    ap.add_argument("--title", default="RuLongTTS main results (PLAN.md §17)")
    ap.add_argument("--drift", action="append", default=[], metavar="LABEL=PATH",
                    help="A5's voice-drift per_item.jsonl for one row of the table "
                         "(PLAN §9.5); repeatable. Fills 'Voice drift' and 'Spk sim'.")
    ap.add_argument("--drift-root", nargs="?", const=str(DRIFT_ROOT_DEFAULT), default=None,
                    metavar="DIR",
                    help="auto-discover DIR/<label>/per_item.jsonl for every label in the "
                         f"table (default DIR: {DRIFT_ROOT_DEFAULT})")
    args = ap.parse_args(argv)

    rows, provenance = load_inputs(args.inputs)
    labels = list(dict.fromkeys(r["__label__"] for r in rows))
    drift, drift_provenance = load_drift(
        args.drift, Path(args.drift_root) if args.drift_root else None, labels)
    table = build_table(rows, drift)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.name}.md"
    csv_path = out_dir / f"{args.name}.csv"
    md_path.write_text(
        render_markdown(table, provenance, rows, args.title, drift_provenance), encoding="utf-8")
    write_csv(csv_path, table)
    drift_note = (f", drift for {len(drift)}/{len(labels)} label(s)" if drift
                  else ", no drift file")
    print(f"[aggregate] {len(rows)} rows -> {len(table)} table rows{drift_note}: "
          f"{md_path}, {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AggregateError as exc:
        print(f"[aggregate] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
