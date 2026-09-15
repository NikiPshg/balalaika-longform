#!/usr/bin/env python3
"""Paired, all-attempt statistics for a frozen expanded evaluation.

No GPU, transcript recomputation, changed completion rule, or implicit pair
selection. The spec names every arm, contrast and planned text/voice/seed row.
Missing score rows are an unfinished evaluation, not missing-at-random data.

Primary uncertainty: sample source/book clusters, conditional on the selected
reference voices. Sensitivity: independently resample sources and voices and
multiply their multiplicities (the crossed/pigeonhole bootstrap). Both use the
same weights for all arms, metrics and duration subsets. Intervals condition on
the fitted checkpoints and inference seeds; they do not include retraining.

Usage: python scripts/expanded_eval_statistics.py --spec SPEC.json --out-dir DIR

Spec paths resolve relative to SPEC.json; example structure is in
``example_spec()``. The design is one JSONL row per intended text/voice/seed
combination. It must supply the configured cluster field, root_id and bucket.
Scored rows use the existing run_evaluation.py / f5_evaluate.py schema.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PAIR_FIELDS = ("text_id", "voice_id", "seed")
METRICS = {
    "wer_all_pct": {"unit": "%", "delta_unit": "pp", "better": "lower"},
    "correct_word_recall_pct": {"unit": "%", "delta_unit": "pp", "better": "higher"},
    "frozen_complete_pct": {"unit": "%", "delta_unit": "pp", "better": "higher"},
}
METHODS = ("source_conditional", "source_voice_crossed")
METHOD_REFERENCE = "https://arxiv.org/abs/0712.1111"


def example_spec() -> dict:
    return {
        "design": "data/attempts.jsonl",
        "source_cluster_field": "book_id",
        "resamples": 10000,
        "seed": 20260913,
        "alpha": 0.05,
        "primary_cells": ["B4"],
        "arms": [
            {"arm_id": "cosy_short", "backbone": "CosyVoice3", "family": "ar",
             "protocol": "continuous", "results": "results/cosy_short/per_item.jsonl"},
            {"arm_id": "cosy_long", "backbone": "CosyVoice3", "family": "ar",
             "protocol": "continuous", "results": "results/cosy_long/per_item.jsonl"},
        ],
        "contrasts": [
            {"arm_a": "cosy_long", "arm_b": "cosy_short", "role": "primary"}
        ],
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_key(row: dict) -> tuple[str, str, int]:
    for field in PAIR_FIELDS:
        if field not in row or row[field] is None:
            raise ValueError(f"Missing non-null pairing field: {field}")
    if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
        raise ValueError("Pairing seed must be an integer")
    if not all(isinstance(row[f], str) and row[f] for f in PAIR_FIELDS[:2]):
        raise ValueError("text_id and voice_id must be nonempty strings")
    return tuple(row[f] for f in PAIR_FIELDS)


def keyed(rows: list[dict], description: str) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = pair_key(row)
        if key in result:
            raise ValueError(f"Duplicate paired item in {description}: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"Empty {description}")
    return result


def metric_values(row: dict) -> np.ndarray:
    """Unconditional metrics; failed runs retain their evaluator-assigned values."""
    required = ("wer", "hits", "n_ref_words", "substitutions", "deletions", "insertions", "status")
    missing = [f for f in required if f not in row or row[f] is None]
    if missing:
        raise ValueError(f"Incomplete scored row {row.get('run_id')}: {missing}")
    counts = {}
    for field in required[1:-1]:
        number = row[field]
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            raise ValueError(f"Nonnegative integer count required for {field}")
        counts[field] = number
    nref = counts["n_ref_words"]
    if nref <= 0 or counts["hits"] + counts["substitutions"] + counts["deletions"] != nref:
        raise ValueError("Scored word counts do not partition the nonempty reference")
    wer = float(row["wer"])
    expected_wer = sum(counts[k] for k in ("substitutions", "deletions", "insertions")) / nref
    if not math.isfinite(wer) or wer < 0 or not math.isclose(wer, expected_wer, abs_tol=1e-10):
        raise ValueError("WER must be a finite nonnegative fraction consistent with word counts")
    if not isinstance(row["status"], str) or not row["status"]:
        raise ValueError("A frozen final status is required")
    complete = row["status"] == "complete"
    if "status_complete" in row and row["status_complete"] != complete:
        raise ValueError("status_complete disagrees with frozen final status")
    # WER may exceed 1.0 because of insertions: never clip it to 100%.
    return np.array([100.0 * wer, 100.0 * counts["hits"] / nref, 100.0 * complete])


def validate_inputs(design_rows: list[dict], arms: list[dict], source_field: str):
    design = keyed(design_rows, "planned design")
    text_metadata = {}
    root_sources = {}
    for key, item in design.items():
        for field in (source_field, "root_id", "bucket"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ValueError(f"Design item {key} needs a nonempty {field}")
        identity = (item[source_field], item["root_id"], item["bucket"])
        if key[0] in text_metadata and text_metadata[key[0]] != identity:
            raise ValueError(f"Inconsistent source/root/bucket for text_id {key[0]}")
        text_metadata[key[0]] = identity
        if item["root_id"] in root_sources and root_sources[item["root_id"]] != item[source_field]:
            raise ValueError("Nested versions of one root must remain in one source cluster")
        root_sources[item["root_id"]] = item[source_field]
    keys = sorted(design)
    arm_ids = [arm["arm_id"] for arm in arms]
    if not arm_ids or len(set(arm_ids)) != len(arm_ids):
        raise ValueError("At least one uniquely named arm is required")
    values = {}
    aligned = {}
    for arm in arms:
        arm_id = arm["arm_id"]
        if arm.get("family") not in ("ar", "flow"):
            raise ValueError(f"Declare generation family ar/flow for {arm_id}")
        if arm.get("protocol") not in ("continuous", "chunked"):
            raise ValueError(f"Declare inference protocol continuous/chunked for {arm_id}")
        rows = keyed(arm["rows"], arm_id)
        missing, extra = set(design) - set(rows), set(rows) - set(design)
        if missing or extra:
            raise ValueError(f"{arm_id}: planned pairing mismatch; missing={len(missing)}, extra={len(extra)}")
        for key in keys:
            row = rows[key]
            for field in (source_field, "root_id", "bucket"):
                if field in row and row[field] != design[key][field]:
                    raise ValueError(f"{arm_id} metadata disagrees with design: {field} at {key}")
            if "bucket" not in row:
                raise ValueError(f"{arm_id} scored row lacks bucket")
            if arm["family"] == "flow":
                if row.get("status") == "early_eos" or row.get("stop_reason") == "eos":
                    raise ValueError("Flow generation must not be labelled an EOS event")
                if row.get("generation_family") != "non_autoregressive_flow_matching":
                    raise ValueError("Flow arm must retain the flow evaluator's generation_family")
        aligned[arm_id] = [rows[k] for k in keys]
        values[arm_id] = np.stack([metric_values(rows[k]) for k in keys])
    # Source word counts, including failures, must be equal in paired runs.
    ref_counts = {tuple(r["n_ref_words"] for r in aligned[arm_id]) for arm_id in arm_ids}
    if len(ref_counts) != 1:
        raise ValueError("Paired arms were scored against different reference word counts")
    return [design[k] for k in keys], aligned, values


def multiplicities(n_levels: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    if n_levels < 1:
        raise ValueError("A factor must contain at least one level")
    # Exactly n_levels draws with replacement, not independent item weights.
    draws = rng.integers(0, n_levels, size=(n_resamples, n_levels))
    weights = np.zeros((n_resamples, n_levels), dtype=np.int32)
    np.add.at(weights, (np.arange(n_resamples)[:, None], draws), 1)
    return weights


def bootstrap_weights(design: list[dict], source_field: str, n_resamples: int, seed: int):
    if n_resamples < 1 or seed < 0:
        raise ValueError("Positive resamples and nonnegative seed required")
    source_levels = sorted({r[source_field] for r in design})
    voice_levels = sorted({r["voice_id"] for r in design})
    source_lookup = {value: i for i, value in enumerate(source_levels)}
    voice_lookup = {value: i for i, value in enumerate(voice_levels)}
    source_idx = np.array([source_lookup[r[source_field]] for r in design])
    voice_idx = np.array([voice_lookup[r["voice_id"]] for r in design])
    source_seed, voice_seed = np.random.SeedSequence(seed).spawn(2)
    source = multiplicities(len(source_levels), n_resamples, np.random.default_rng(source_seed))
    voice = multiplicities(len(voice_levels), n_resamples, np.random.default_rng(voice_seed))
    item_source = source[:, source_idx].astype(np.float64)
    weights = {
        "source_conditional": item_source,
        "source_voice_crossed": item_source * voice[:, voice_idx],
    }
    return weights, {"sources": source_levels, "voices": voice_levels}


def weighted_means(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    numerator = weights @ values
    denominator = weights.sum(axis=1)[:, None]
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def interval(distribution: np.ndarray, alpha: float) -> dict:
    values = np.asarray(distribution)
    valid = values[np.isfinite(values)]
    n_empty = int(values.size - valid.size)
    if not valid.size:
        return {"lo": None, "hi": None, "n_valid": 0, "n_empty": n_empty, "degenerate": None}
    lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "n_valid": int(valid.size),
            "n_empty": n_empty, "degenerate": bool(np.allclose(valid, valid[0], rtol=0, atol=1e-12))}


def support(design: list[dict], source_field: str) -> dict:
    return {
        "attempts": len(design),
        "source_clusters": len({r[source_field] for r in design}),
        "root_passages": len({r["root_id"] for r in design}),
        "distinct_texts": len({r.get("source_text_id", r["text_id"]) for r in design}),
        "reference_voices": len({r["voice_id"] for r in design}),
        "inference_seeds": sorted({r["seed"] for r in design}),
        "attempts_per_source": dict(sorted(Counter(r[source_field] for r in design).items())),
        "attempts_per_voice": dict(sorted(Counter(r["voice_id"] for r in design).items())),
    }


def analyze(design_rows: list[dict], arms: list[dict], contrasts: list[dict],
            source_field: str = "book_id", n_resamples: int = 10000,
            seed: int = 20260913, alpha: float = 0.05, primary_cells: list[str] | None = None,
            descriptive_cohort_cells: list[dict] | None = None) -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    design, rows, values = validate_inputs(design_rows, arms, source_field)
    arm_ids = set(values)
    comparison_ids = []
    for contrast in contrasts:
        a, b = contrast["arm_a"], contrast["arm_b"]
        if a not in arm_ids or b not in arm_ids or a == b:
            raise ValueError(f"Invalid prespecified contrast: {a} minus {b}")
        comparison_ids.append((a, b))
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError("Duplicate prespecified contrasts")
    weights, levels = bootstrap_weights(design, source_field, n_resamples, seed)
    cells = sorted({r["bucket"] for r in design}) + ["ALL"]
    cell_masks = {cell: np.array([cell == "ALL" or row["bucket"] == cell for row in design]) for cell in cells}
    for definition in descriptive_cohort_cells or []:
        bucket, cohort = definition["bucket"], definition["voice_cohort"]
        label = bucket + "::" + cohort
        if label in cell_masks:
            raise ValueError("Duplicate descriptive cohort cell")
        mask = np.array([r["bucket"] == bucket and r.get("voice_cohort") == cohort for r in design])
        if not mask.any():
            raise ValueError(f"Descriptive cohort cell is absent from design: {label}")
        cell_masks[label] = mask
    primary_cells = ["B4"] if primary_cells is None else primary_cells
    if set(primary_cells) - set(cells):
        raise ValueError("A primary analysis cell is absent from the design")
    output: dict[str, Any] = {
        "method": {
            "source_cluster_field": source_field, "resamples": n_resamples, "seed": seed,
            "alpha": alpha, "pair_fields": list(PAIR_FIELDS), "primary_cells": primary_cells,
            "descriptive_cohort_cells": descriptive_cohort_cells or [],
            "point_estimand": "Equal weight per planned attempt within each reported input-length subset",
            "source_conditional": "Resample source/book clusters; selected reference voices fixed",
            "source_voice_crossed": "Independently resample source and voice levels; product multiplicities",
            "reference": METHOD_REFERENCE,
            "intervals": "Pointwise percentile intervals; no multiplicity-adjusted hypothesis tests",
            "no_outcome_filtering": True,
            "failure_rule": "Use every frozen evaluator row, including missing audio and empty ASR",
            "correct_word_recall": "100 * hits / n_ref_words from the frozen alignment; excludes substitutions",
            "excluded_uncertainty": "Checkpoint training, repeated inference, ASR error, corpus selection bias",
            "empty_draw_policy": "Exclude zero-weight subset draws and report their count; never drop observed attempts",
        },
        "design": {**support(design, source_field), "levels": levels},
        "arms": [{k: v for k, v in arm.items() if k != "rows"} for arm in arms],
        "contrasts": contrasts,
        "cells": {},
        "warnings": [
            "Nested lengths and repeated voices are dependent; attempt count is not an independent sample size.",
            "Sampling intervals condition on the chosen checkpoints and seeds; chosen texts/voices are not a probability sample of all Russian speech.",
            "Crossed source/voice intervals are a sensitivity analysis; sparse assignments and few levels can make percentile coverage inaccurate.",
            "Frozen completion without a human ASR floor omits its WER-minus-floor condition; consult WER-all and correct-word recall.",
            "A zero-width bootstrap interval at 0% or 100% completion does not prove a population rate of 0% or 100%.",
            "Pointwise intervals across multiple backbones, metrics and lengths do not establish familywise significance.",
        ],
    }
    if len(levels["sources"]) < 20 or len(levels["voices"]) < 10:
        output["warnings"].append("Few source or voice clusters: interpret resampling intervals cautiously.")
    for cell, mask in cell_masks.items():
        cell_design = [row for row, keep in zip(design, mask) if keep]
        cw = {method: matrix[:, mask] for method, matrix in weights.items()}
        arm_distributions = {}
        summary = {"support": support(cell_design, source_field), "arms": {}, "contrasts": []}
        for arm_id in values:
            outcomes = values[arm_id][mask]
            selected = [row for row, keep in zip(rows[arm_id], mask) if keep]
            point = outcomes.mean(axis=0)
            distributions = {method: weighted_means(outcomes, cw[method]) for method in METHODS}
            arm_distributions[arm_id] = distributions
            metrics = {}
            for j, (metric, metadata) in enumerate(METRICS.items()):
                metrics[metric] = {"estimate": float(point[j]), **metadata,
                    "intervals": {method: interval(distributions[method][:, j], alpha) for method in METHODS}}
            summary["arms"][arm_id] = {
                "n_attempted": len(selected), "n_complete": sum(r["status"] == "complete" for r in selected),
                "n_floor_available": sum(bool(r.get("floor_available")) for r in selected),
                "n_asr_error": sum(bool(r.get("asr_error")) for r in selected),
                "n_audio_missing": sum(r.get("output_exists") is False for r in selected),
                "status_counts": dict(sorted(Counter(r["status"] for r in selected).items())),
                "metrics": metrics,
            }
        for contrast in contrasts:
            a, b = contrast["arm_a"], contrast["arm_b"]
            delta = (values[a][mask] - values[b][mask]).mean(axis=0)
            metrics = {}
            for j, (metric, metadata) in enumerate(METRICS.items()):
                metrics[metric] = {"delta_a_minus_b": float(delta[j]), **metadata,
                    "intervals": {method: interval(
                        arm_distributions[a][method][:, j] - arm_distributions[b][method][:, j], alpha
                    ) for method in METHODS}}
            summary["contrasts"].append({**contrast, "metrics": metrics})
        output["cells"][cell] = summary
    return output


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def content_only_payload(payload: dict) -> dict:
    """Public text-only metrics must not inherit the floor-dependent label."""
    public = deepcopy(payload)
    public["method"]["report_content_only"] = True
    public["method"]["reported_metrics"] = ["wer_all_pct", "correct_word_recall_pct"]
    for summary in public["cells"].values():
        for arm in summary["arms"].values():
            for field in ("n_complete", "n_floor_available", "status_counts"):
                arm.pop(field, None)
            arm["metrics"].pop("frozen_complete_pct", None)
        for contrast in summary["contrasts"]:
            contrast["metrics"].pop("frozen_complete_pct", None)
    public["warnings"] = [warning for warning in public["warnings"] if "completion" not in warning.lower()]
    return public


def write_outputs(payload: dict, out_dir: Path, report_content_only: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    if report_content_only:
        internal = {**payload, "report_role": "Internal evaluator diagnostics; floor-dependent labels are excluded from the public text-only report"}
        (out_dir / "statistics_internal.json").write_text(json.dumps(internal, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        payload = content_only_payload(payload)
    (out_dir / "statistics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    estimates, differences = [], []
    for cell, summary in payload["cells"].items():
        base = {"cell": cell, **{k: summary["support"][k] for k in
                ("attempts", "source_clusters", "root_passages", "reference_voices")}}
        for arm_id, arm in summary["arms"].items():
            for name, result in arm["metrics"].items():
                for method, bounds in result["intervals"].items():
                    counts = {} if report_content_only else {"complete_count": f"{arm['n_complete']}/{arm['n_attempted']}"}
                    estimates.append({**base, "arm_id": arm_id, "metric": name,
                        "estimate": result["estimate"], "method": method, **bounds, **counts})
        for contrast in summary["contrasts"]:
            for name, result in contrast["metrics"].items():
                for method, bounds in result["intervals"].items():
                    differences.append({**base, "arm_a": contrast["arm_a"], "arm_b": contrast["arm_b"],
                        "role": contrast.get("role", "unspecified"), "metric": name,
                        "delta_a_minus_b": result["delta_a_minus_b"], "method": method, **bounds})
    write_csv(out_dir / "arm_estimates.csv", estimates)
    write_csv(out_dir / "paired_differences.csv", differences)
    lines = ["# Expanded evaluation: paired all-attempt results", "",
             "Every number includes the full frozen attempt grid. WER and recall are macro averages.", "",
             "Intervals condition on fitted checkpoints and inference seeds. The primary bootstrap resamples "
             "source/book clusters with selected voices fixed; crossed source/voice resampling is reported separately.", ""]
    for cell, summary in payload["cells"].items():
        count = summary["support"]
        lines += [f"## {cell}", "", f"{count['attempts']} attempts per arm; {count['source_clusters']} source clusters, "
                  f"{count['root_passages']} root passages, {count['reference_voices']} reference voices.", "",
                  "| Arm | WER-all % | Correct-word recall % |" if report_content_only else "| Arm | Complete x/N | WER-all % | Correct-word recall % |",
                  "|---|---:|---:|" if report_content_only else "|---|---:|---:|---:|"]
        for arm_id, arm in summary["arms"].items():
            metrics = arm["metrics"]
            complete = "" if report_content_only else f"{arm['n_complete']}/{arm['n_attempted']} | "
            lines.append(f"| {arm_id} | {complete}"
                         f"{metrics['wer_all_pct']['estimate']:.2f} | {metrics['correct_word_recall_pct']['estimate']:.2f} |")
        lines += ["", "Paired differences A − B, in percentage points. Intervals are pointwise.", "",
                  "| A − B | Metric | Difference | Source CI | Crossed source/voice CI |",
                  "|---|---|---:|---:|---:|"]
        for contrast in summary["contrasts"]:
            for name, result in contrast["metrics"].items():
                def show(method):
                    bounds = result["intervals"][method]
                    return "undefined" if bounds["lo"] is None else f"[{bounds['lo']:.2f}, {bounds['hi']:.2f}]"
                lines.append(f"| {contrast['arm_a']} − {contrast['arm_b']} | {name} | "
                             f"{result['delta_a_minus_b']:.2f} | {show(METHODS[0])} | {show(METHODS[1])} |")
        lines.append("")
    lines += ["## Interpretation limits", ""] + [f"- {warning}" for warning in payload["warnings"]]
    lines += ["", f"Crossed resampling reference: [Owen, The pigeonhole bootstrap]({METHOD_REFERENCE}).", ""]
    (out_dir / "statistics.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def run(spec_path: Path, out_dir: Path, report_content_only: bool = False):
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    resolve = lambda path: (spec_path.parent / path).resolve()
    design_path = resolve(spec["design"])
    arms = []
    provenance = {"spec": {"path": str(spec_path), "sha256": sha256(spec_path)},
                  "design": {"path": str(design_path), "sha256": sha256(design_path)},
                  "script_sha256": sha256(Path(__file__)), "results": {}}
    for arm in spec["arms"]:
        result_path = resolve(arm["results"])
        provenance["results"][arm["arm_id"]] = {"path": str(result_path), "sha256": sha256(result_path)}
        arms.append({**arm, "rows": read_jsonl(result_path)})
    design_rows = read_jsonl(design_path)
    # The frozen text-only row contract itself also suppresses this label, so
    # forgetting a presentation flag cannot accidentally publish it.
    report_content_only = bool(report_content_only or spec.get("report_content_only") or any(r.get("primary_report_content_only") for r in design_rows))
    payload = analyze(design_rows, arms, spec["contrasts"],
        source_field=spec.get("source_cluster_field", "book_id"),
        n_resamples=spec.get("resamples", 10000), seed=spec.get("seed", 20260913),
        alpha=spec.get("alpha", 0.05), primary_cells=spec.get("primary_cells", ["B4"]),
        descriptive_cohort_cells=spec.get("descriptive_cohort_cells", []))
    payload["provenance"] = provenance
    return write_outputs(payload, out_dir, report_content_only=report_content_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--print-example-spec", action="store_true")
    parser.add_argument("--report-content-only", action="store_true", help="Publish WER and recall only; preserve legacy evaluator labels in a separate internal JSON")
    args = parser.parse_args()
    if args.print_example_spec:
        print(json.dumps(example_spec(), indent=2))
        return
    if args.spec is None or args.out_dir is None:
        parser.error("--spec and --out-dir are required")
    payload = run(args.spec, args.out_dir, report_content_only=args.report_content_only)
    print(json.dumps({"out_dir": str(args.out_dir), "design": payload["design"], "arm_count": len(payload["arms"])}))


if __name__ == "__main__":
    main()
