#!/usr/bin/env python3
"""Assemble auditable expanded-evaluation artifacts only from complete grids.

This CPU-only wrapper never changes the manuscript, frozen inputs, scorer rows,
or statistical implementation. Synthetic tests belong in temporary directories.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expanded_eval_statistics as stats

DEFAULT_ROOT = Path("artifacts/expanded_source")
BACKBONES = {"cosy": "CosyVoice3", "qwen": "Qwen3-TTS", "vox": "VoxCPM2", "f5": "F5-TTS"}
VARIANTS = {"base": "Base", "short": "Short-SFT", "long": "Long-SFT", "long_punct": "Long-SFT-Punct", "base_chunked": "Base, chunked"}
CONTINUOUS = [f"{b}_{v}" for b in BACKBONES for v in ("base", "short", "long")]
CONTINUOUS.insert(3, "cosy_long_punct")
CONTROLS = ["cosy_base_chunked", "f5_base_chunked"]
SECONDARY = [f"{b}_{v}" for b in BACKBONES for v in ("short", "long")]
LENGTHS = {"B0": 75, "B2": 300, "B4": 1200}


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def evidence(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": stats.sha256(path)}


def read_stable_rows(path: Path):
    entry = evidence(path)
    rows = stats.read_jsonl(path)
    if stats.sha256(path) != entry["sha256"]:
        raise ValueError(f"Scored input changed while it was being read: {path}")
    return rows, entry


def ensure_grid(design: list[dict], stage: str):
    stats.keyed(design, "report design")
    counts = Counter(row["bucket"] for row in design)
    expected = {"B0": 30, "B2": 30, "B4": 60} if stage == "primary" else {"B4": 60}
    if dict(counts) != expected:
        raise ValueError(f"{stage} requires the full frozen grid {expected}; found {dict(counts)}")
    if len({r["root_id"] for r in design}) != 30 or len({r["book_id"] for r in design}) < 20:
        raise ValueError("Frozen design must have 30 source passages in at least 20 work clusters")
    for root in {r["root_id"] for r in design}:
        group = [r for r in design if r["root_id"] == root]
        if Counter(r["bucket"] for r in group) != Counter({"B0": 1, "B2": 1, "B4": 2} if stage == "primary" else {"B4": 2}):
            raise ValueError("Each source passage must retain every prescribed length and voice attempt")
    if len({r["voice_id"] for r in design}) != 21:
        raise ValueError("The full grid requires all 21 frozen reference voices")
    if any(not r.get("primary_report_content_only") or r.get("human_audio_path") is not None for r in design):
        raise ValueError("This report requires the frozen text-only content-metric contract")
    for row in design:
        if row.get("word_target") != LENGTHS[row["bucket"]]:
            raise ValueError("Internal bucket does not match its frozen word-length label")


def load_context(root: Path, spec_path: Path, stage: str):
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text())
    design_path = (spec_path.parent / spec["design"]).resolve()
    freeze_path = root / "data/BENCHMARK_FREEZE.json"
    frozen = json.loads(freeze_path.read_text())
    references = root / "data/references.jsonl"
    if stats.sha256(design_path) != frozen["benchmark_sha256"] or stats.sha256(references) != frozen["references_sha256"]:
        raise ValueError("Main benchmark/reference hash disagrees with the pre-generation freeze")
    if not frozen.get("text_only") or not frozen.get("report_content_only"):
        raise ValueError("Not the frozen text-only expanded evaluation")
    design = stats.read_jsonl(design_path)
    ensure_grid(design, "primary")
    if {a["arm_id"] for a in spec["arms"]} != set(CONTINUOUS + CONTROLS):
        raise ValueError("The primary specification must contain exactly 13 continuous and 2 chunked arms")
    paths = [spec_path, design_path, freeze_path, references, Path(stats.__file__), Path(__file__)]
    arms, scored_inputs = [], []
    for arm in spec["arms"]:
        path = (spec_path.parent / arm["results"]).resolve()
        rows, entry = read_stable_rows(path)
        arms.append({**arm, "rows": rows, "source_path": str(path)})
        scored_inputs.append(entry)
    # A final secondary report still requires the complete primary experiment.
    stats.validate_inputs(design, arms, spec.get("source_cluster_field", "book_id"))
    return spec, design, arms, [evidence(path) for path in paths] + scored_inputs


def label(arm_id: str):
    backbone, variant = arm_id.split("_", 1)
    if backbone != "cosy" and variant == "long":
        variant = "long_punct"
    return BACKBONES[backbone], VARIANTS[variant]


def metric_record(arm_id: str, summary: dict):
    metrics = summary["metrics"]
    wer, recall = metrics["wer_all_pct"], metrics["correct_word_recall_pct"]
    bounds = wer["intervals"]["source_conditional"]
    backbone, variant = label(arm_id)
    return dict(arm_id=arm_id, model=backbone, training=variant, attempts=summary["n_attempted"],
                wer_pct=wer["estimate"], wer_ci_low=bounds["lo"], wer_ci_high=bounds["hi"],
                correct_word_recall_pct=recall["estimate"])


def write_table(out: Path, stem: str, rows: list[dict]):
    stats.write_csv(out / f"{stem}.csv", rows)
    markdown = ["| Model | Training / protocol | WER % [95% CI] | Correct-word recall % |", "|---|---|---:|---:|"]
    tex = [r"\begin{tabular}{llrr}", r"\toprule", r"Model & Training / protocol & WER (\%) [95\% CI] & Recall (\%) \\", r"\midrule"]
    for row in rows:
        value = f"{row['wer_pct']:.1f} [{row['wer_ci_low']:.1f}, {row['wer_ci_high']:.1f}]"
        markdown.append(f"| {row['model']} | {row['training']} | {value} | {row['correct_word_recall_pct']:.1f} |")
        tex.append(f"{row['model']} & {row['training']} & {value} & {row['correct_word_recall_pct']:.1f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / f"{stem}.md").write_text("\n".join(markdown) + "\n")
    (out / f"{stem}.tex").write_text("\n".join(tex) + "\n")


def plot_lengths(payload: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    styles = {"base": ("#4C78A8", "o", "-"), "short": ("#E28B31", "s", "-"),
              "long": ("#218C74", "^", "-"), "long_punct": ("#9467BD", "v", "-"),
              "base_chunked": ("#666666", "D", "--")}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharey=True, layout="constrained")
    axes = axes.ravel()
    all_highs, handles = [], {}
    for axis, backbone in zip(axes, BACKBONES):
        variants = ["base", "short", "long"] + (["long_punct"] if backbone == "cosy" else [])
        variants += ["base_chunked"] if backbone in ("cosy", "f5") else []
        for variant in variants:
            arm = backbone + "_" + variant
            metrics = [payload["cells"][b]["arms"][arm]["metrics"]["wer_all_pct"] for b in LENGTHS]
            means = np.array([m["estimate"] for m in metrics])
            low = np.array([m["intervals"]["source_conditional"]["lo"] for m in metrics])
            high = np.array([m["intervals"]["source_conditional"]["hi"] for m in metrics])
            color, marker, line = styles[variant]
            # Draw interval endpoints directly: a percentile interval need not
            # contain the point estimate in every small sample.
            handle, = axis.plot(range(3), means, color=color, marker=marker, linestyle=line, linewidth=1.5,
                                markersize=4.5, label=VARIANTS[variant])
            axis.vlines(range(3), low, high, color=color, alpha=.65, linewidth=1)
            handles[variant] = handle
            all_highs.extend(high)
            all_highs.extend(means)
        axis.set_title(BACKBONES[backbone], fontsize=11)
        axis.set_xticks(range(3), [str(n) for n in LENGTHS.values()])
        axis.set_xlabel("Input length (words)")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("WER (%) ↓")
    axes[2].set_ylabel("WER (%) ↓")
    axes[0].set_ylim(0, max(1, max(all_highs)) * 1.08)  # Never clip WER at 100%.
    fig.legend(list(handles.values()), [VARIANTS[v] for v in handles], loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(out / "wer_by_length.pdf", bbox_inches="tight")
    fig.savefig(out / "wer_by_length.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def speaker_rows(root: Path, design: list[dict], primary_arms: list[dict]):
    expected = set(stats.keyed(design, "speaker design"))
    records, inputs = [], []
    for arm in primary_arms:
        path = root / "results" / arm["arm_id"] / "speaker_drift.jsonl"
        rows, entry = read_stable_rows(path)
        headers = [r for r in rows if r.get("_meta")]
        if len(headers) != 1 or headers[0].get("per_item_sha256") != stats.sha256(Path(arm["source_path"])):
            raise ValueError(f"Speaker source provenance mismatch: {arm['arm_id']}")
        mapping = stats.keyed([r for r in rows if not r.get("_meta")], "speaker " + arm["arm_id"])
        if set(mapping) != expected:
            raise ValueError(f"Incomplete speaker grid for {arm['arm_id']}")
        measured = []
        for item in design:
            if item["bucket"] != "B4":
                continue
            row = mapping[stats.pair_key(item)]
            value = row.get("sim_median")
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(value) or row.get("n_windows", 0) < 1:
                    raise ValueError("Voice similarity must come from measured voiced windows")
                measured.append(float(value))
        records.append(dict(arm_id=arm["arm_id"], sim_median=float(np.median(measured)) if measured else None,
                            measured_n=len(measured), attempted_n=60,
                            statistic="median of available per-item 10-second voiced-window median cosine similarities"))
        inputs.append(entry)
    return records, inputs


def primary_report(root, spec_path, spec, design, arms, out, include_speaker):
    payload = stats.run(spec_path, out / "statistics", report_content_only=True)
    b4 = payload["cells"]["B4"]
    write_table(out, "main_b4", [metric_record(a, b4["arms"][a]) for a in CONTINUOUS])
    write_table(out, "chunked_controls_b4", [metric_record(a, b4["arms"][a]) for a in CONTROLS])
    plot_lengths(payload, out)
    extra_inputs = []
    if include_speaker:
        records, extra_inputs = speaker_rows(root, design, arms)
        stats.write_csv(out / "speaker_b4.csv", records)
    support = b4["support"]
    lines = ["# Expanded text-only evaluation", "", "The complete frozen grid was scored: 120 attempts × 15 systems. "
             f"The longest condition contains 60 attempts from {support['root_passages']} passages, "
             f"{support['source_clusters']} conservative work clusters and {support['reference_voices']} voices.", "",
             "Input lengths target 75, 300 and 1200 normalized words. Every attempt, including missing audio and empty ASR, "
             "contributes to macro WER and correct-word recall. WER values above 100% are retained.", "",
             "The main table has 13 continuous systems. The two chunked Base controls are separate diagnostics and dashed lines in the figure. "
             "The figure shows macro WER with pointwise source-cluster 95% intervals; selected voices are fixed. "
             "The statistical supplement also includes crossed source/voice intervals and prespecified new-voice sensitivity.", "",
             "## Long versus Short at 1200 words", ""]
    for contrast in b4["contrasts"]:
        if contrast.get("role") != "primary":
            continue
        metric = contrast["metrics"]["wer_all_pct"]
        ci = metric["intervals"]["source_conditional"]
        lines.append(f"- {label(contrast['arm_a'])[0]}: Long minus Short WER {metric['delta_a_minus_b']:+.1f} pp "
                     f"[{ci['lo']:+.1f}, {ci['hi']:+.1f}].")
    lines += ["", "These are ASR-based content measures. No human MOS, naturalness, preference or human-reference-duration result is inferred. "
              "No floor-dependent completion statistic is published. Speaker similarity, when requested, is a separate conditional statistic with measured N; it is not a listener score.", "",
              "Intervals condition on fixed checkpoints and inference seeds, and do not include retraining or ASR systematic uncertainty. "
              "Nested texts and repeated voices are dependent; 120 attempts are not 120 independent sources. "
              "Pointwise intervals do not imply multiplicity-adjusted significance.", "",
              "Artifacts only: the manuscript has not been updated by this script. This report describes completed scored files and makes no claim about a future secondary evaluation.", ""]
    (out / "report.md").write_text("\n".join(lines))
    return extra_inputs


def secondary_view(secondary_rows, primary_rows, design, arm_id):
    expected = set(stats.keyed(design, "secondary design"))
    primary = stats.keyed(primary_rows, "primary " + arm_id)
    secondary = stats.keyed(secondary_rows, "secondary " + arm_id)
    if set(secondary) != expected or not expected <= set(primary):
        raise ValueError(f"Secondary {arm_id} requires all 60 planned B4 pairs")
    derived = []
    for key in sorted(expected):
        source, row = primary[key], dict(secondary[key])
        for field in ("run_id", "status", "status_complete", "output_path", "n_ref_words"):
            if row.get(field) != source.get(field):
                raise ValueError(f"Secondary did not preserve primary {field}: {arm_id}")
        if row.get("completion_status_source") != "unchanged_primary_rnnt":
            raise ValueError("Secondary status provenance is missing")
        if row.get("asr_model_id") != "t-tech/t-one" or row.get("asr_role") != "prespecified_secondary_wer_audit":
            raise ValueError("Unexpected secondary recognizer identity or analysis role")
        if row.get("wer_primary") != source["wer"] or row.get("wer_secondary") != row["wer"]:
            raise ValueError("Secondary WER provenance disagrees with original primary/scored WER")
        for field in ("generation_family", "stop_reason", "output_exists"):
            if field in source:
                if field in row and row[field] != source[field]:
                    raise ValueError(f"Secondary generation metadata conflicts with primary {field}")
                row[field] = source[field]
        row["derived_report_view"] = {"metadata_from": "original primary paired row", "original_run_id": source["run_id"]}
        derived.append(row)
    return derived


def secondary_report(root, spec_path, spec, design, primary_arms, out):
    design = [r for r in design if r["bucket"] == "B4"]
    ensure_grid(design, "secondary")
    original = {a["arm_id"]: a for a in primary_arms}
    inputs, derived_arms, primary_b4 = [], [], []
    for arm_id in SECONDARY:
        path = root / "results_secondary" / arm_id / "per_item.jsonl"
        original_secondary, entry = read_stable_rows(path)
        derived = secondary_view(original_secondary, original[arm_id]["rows"], design, arm_id)
        file = out / "derived_views" / arm_id / "per_item.jsonl"
        write_jsonl(file, derived)
        derived_arms.append({**original[arm_id], "results": str(file.relative_to(out)), "rows": derived})
        primary_b4.append({**original[arm_id], "rows": [r for r in original[arm_id]["rows"] if r["bucket"] == "B4"]})
        inputs.append(entry)
    contrasts = [c for c in spec["contrasts"] if c.get("role") == "primary"]
    kwargs = dict(source_field=spec.get("source_cluster_field", "book_id"), n_resamples=spec["resamples"],
                  seed=spec["seed"], alpha=spec["alpha"], primary_cells=["B4"],
                  descriptive_cohort_cells=spec.get("descriptive_cohort_cells", []))
    # The frozen implementation performs all paired validation and inference.
    secondary = stats.analyze(design, derived_arms, contrasts, **kwargs)
    primary = stats.analyze(design, primary_b4, contrasts, **kwargs)
    secondary["provenance"] = {"role": "Prespecified secondary recognizer sensitivity", "original_secondary_inputs": inputs,
                               "original_primary_inputs": [evidence(Path(original[a]["source_path"])) for a in SECONDARY],
                               "frozen_statistical_source": evidence(Path(stats.__file__))}
    stats.write_outputs(secondary, out / "statistics", report_content_only=True)
    write_jsonl(out / "derived_views/benchmark_b4.jsonl", design)
    write_json(out / "derived_views/view_manifest.json", {"original_secondary_inputs": inputs,
               "metadata_recovered": ["generation_family", "stop_reason", "output_exists"],
               "rule": "Restore only matching frozen generation metadata from the same primary pair; preserve secondary transcript, errors, counts and WER",
               "derived_file_sha256": {a: stats.sha256(out / "derived_views" / a / "per_item.jsonl") for a in SECONDARY}})
    comparisons, rows = [], []
    for arm_id in SECONDARY:
        first = metric_record(arm_id, primary["cells"]["B4"]["arms"][arm_id])
        second = metric_record(arm_id, secondary["cells"]["B4"]["arms"][arm_id])
        rows.append({"arm_id": arm_id, "model": first["model"], "training": first["training"], "attempts": 60,
                     "primary_wer_pct": first["wer_pct"], "secondary_wer_pct": second["wer_pct"],
                     "primary_recall_pct": first["correct_word_recall_pct"], "secondary_recall_pct": second["correct_word_recall_pct"]})
    ordered, _, secondary_values = stats.validate_inputs(design, derived_arms, kwargs["source_field"])
    _, _, primary_values = stats.validate_inputs(design, primary_b4, kwargs["source_field"])
    weights, _ = stats.bootstrap_weights(ordered, kwargs["source_field"], kwargs["n_resamples"], kwargs["seed"])
    for a, b in zip(primary["cells"]["B4"]["contrasts"], secondary["cells"]["B4"]["contrasts"]):
        for index, metric in enumerate(("wer_all_pct", "correct_word_recall_pct")):
            av, bv = a["metrics"][metric], b["metrics"][metric]
            arm_a, arm_b = a["arm_a"], a["arm_b"]
            delta = (secondary_values[arm_a][:, index] - secondary_values[arm_b][:, index]) - (primary_values[arm_a][:, index] - primary_values[arm_b][:, index])
            record = {"model": label(arm_a)[0], "contrast": "Long minus Short", "metric": metric,
                      "primary_delta_pp": av["delta_a_minus_b"], "secondary_delta_pp": bv["delta_a_minus_b"],
                      "change_in_delta_pp": float(delta.mean())}
            for method in stats.METHODS:
                for role, metric_value in (("primary", av), ("secondary", bv)):
                    for bound in ("lo", "hi"):
                        record[f"{role}_{method}_{bound}"] = metric_value["intervals"][method][bound]
                bounds = stats.interval(stats.weighted_means(delta, weights[method])[:, 0], kwargs["alpha"])
                for field in ("lo", "hi", "n_empty"):
                    record[f"change_{method}_{field}"] = bounds[field]
            comparisons.append(record)
    stats.write_csv(out / "secondary_b4.csv", rows)
    stats.write_csv(out / "secondary_paired.csv", comparisons)
    write_json(out / "secondary_paired.json", comparisons)
    lines = ["# Prespecified secondary-recognizer sensitivity", "",
             "All 60 B4 attempts × 8 Short/Long systems are present. The same generated recordings and published target words are scored with primary RNNT and secondary T-one. "
             "Missing audio and ASR failures remain in both all-attempt estimates; no paired rows are dropped.", "",
             "| Model | Long − Short primary WER, pp | Long − Short secondary WER, pp | Change in paired gap [source 95% CI], pp |",
             "|---|---:|---:|---:|"]
    for row in comparisons:
        if row["metric"] == "wer_all_pct":
            lines.append(f"| {row['model']} | {row['primary_delta_pp']:+.1f} | {row['secondary_delta_pp']:+.1f} | "
                         f"{row['change_in_delta_pp']:+.1f} [{row['change_source_conditional_lo']:+.1f}, {row['change_source_conditional_hi']:+.1f}] |")
    lines += ["", "The change in gap is (secondary Long − Short) − (primary Long − Short), calculated per paired attempt before applying the same frozen source/crossed resampling weights. "
              "It diagnoses recognizer sensitivity and is not a new model-training experiment. Complete within-recognizer WER/recall contrasts and crossed intervals are in CSV/JSON.", "",
              "Original primary/secondary files remain unchanged. A derived view restores generation-family, stop reason and audio-existence metadata from the matching primary row; secondary transcript/counts/WER remain untouched. "
              "Legacy primary statuses are not reclassified or publicly reported. The manuscript has not been updated by this script.", ""]
    (out / "report.md").write_text("\n".join(lines))
    return inputs


def run_report(stage: str, root: Path, spec_path: Path, out_dir: Path, include_speaker: bool = False):
    root, spec_path, out_dir = root.resolve(), spec_path.resolve(), out_dir.resolve()
    spec, design, arms, inputs = load_context(root, spec_path, stage)
    if out_dir.exists():
        raise ValueError(f"Refusing to replace existing report directory: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expanded-report-", dir=out_dir.parent) as folder:
        work = Path(folder) / "report"
        work.mkdir()
        extra = (primary_report(root, spec_path, spec, design, arms, work, include_speaker) if stage == "primary"
                 else secondary_report(root, spec_path, spec, design, arms, work))
        inputs += extra
        for item in inputs:
            if stats.sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Scored input changed during report assembly: {item['path']}")
        manifest = {"state": "complete", "stage": stage, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "manuscript_updated": False, "all_attempts_retained": True, "report_content_only": True,
                    "input_evidence": inputs, "speaker_requested": include_speaker,
                    "scored_pairs_per_arm": 120 if stage == "primary" else 60, "arm_count": 15 if stage == "primary" else 8,
                    "output_sha256": {str(p.relative_to(work)): stats.sha256(p) for p in work.rglob("*") if p.is_file()}}
        write_json(work / "REPORT_COMPLETE.json", manifest)
        shutil.move(str(work), out_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["primary", "secondary"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--spec", type=Path, help="Defaults to ROOT/statistics_spec.json")
    parser.add_argument("--out-dir", type=Path, help="Defaults to ROOT/reports/STAGE")
    parser.add_argument("--speaker", choices=["off", "required"], default="off")
    args = parser.parse_args()
    try:
        result = run_report(args.stage, args.root, args.spec or args.root / "statistics_spec.json",
                            args.out_dir or args.root / "reports" / args.stage, args.speaker == "required")
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        print(json.dumps({"state": "not_reported", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({k: result[k] for k in ("state", "stage", "scored_pairs_per_arm", "arm_count", "manuscript_updated")}))


if __name__ == "__main__":
    main()
