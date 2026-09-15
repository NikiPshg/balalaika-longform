#!/usr/bin/env python3
"""Report a separately frozen 50-attempt evaluation within the deadline.

Final reports require every planned pair for an explicit, predefined arm group.
Descriptive progress snapshots are separate and never receive final intervals,
paper tables, or a completion marker. No frozen legacy file is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expanded_eval_statistics as stats
from expanded_eval_report import (
    BACKBONES, CONTINUOUS, CONTROLS, LENGTHS, SECONDARY, VARIANTS,
    evidence, label, metric_record, read_stable_rows, write_json, write_table,
)

GROUPS = {"main13": CONTINUOUS, "primary8": SECONDARY}
DEFAULT_ROOT = Path("artifacts/published50")


def load_plan(spec_path: Path):
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text())
    resolve = lambda p: (spec_path.parent / p).resolve()
    design_path = resolve(spec["design"])
    design, design_evidence = read_stable_rows(design_path)
    if design_evidence["sha256"] != spec.get("design_sha256"):
        raise ValueError("The limited plan must pin its exact design_sha256 before reporting")
    if spec.get("planned_attempts") != 50 or len(design) != 50:
        raise ValueError("This separate limited plan requires exactly 50 prescribed attempts")
    mapping = stats.keyed(design, "frozen limited design")
    if set(r["bucket"] for r in design) != set(LENGTHS):
        raise ValueError("All three prespecified input-word conditions must be represented")
    if spec.get("resamples", 10000) != 10000 or spec.get("alpha", .05) != .05:
        raise ValueError("Keep the prespecified 10,000-resample, 95% uncertainty protocol")
    for row in design:
        if row.get("word_target") != LENGTHS[row["bucket"]] or not row.get("primary_report_content_only"):
            raise ValueError("Limited rows must preserve the published-text word-length and content-only contract")
    arms = {a["arm_id"]: a for a in spec["arms"]}
    if len(arms) != len(spec["arms"]):
        raise ValueError("Duplicate arm identities in limited specification")
    groups = spec.get("report_groups", GROUPS)
    for group, wanted in GROUPS.items():
        if set(groups.get(group, [])) != set(wanted):
            raise ValueError(f"Report group {group} must be predefined as its complete fixed arm list")
    source_design = None
    inputs = [evidence(spec_path), design_evidence, evidence(Path(__file__)), evidence(Path(stats.__file__)),
              evidence(ROOT / "scripts/expanded_eval_report.py")]
    protocol_path = resolve(spec.get("selection_protocol", "data/selection_protocol.json"))
    protocol = json.loads(protocol_path.read_text())
    protocol_evidence = evidence(protocol_path)
    if spec.get("selection_protocol_sha256", protocol_evidence["sha256"]) != protocol_evidence["sha256"]:
        raise ValueError("Limited selection-protocol checksum changed")
    if protocol.get("subset_benchmark_sha256") != design_evidence["sha256"]:
        raise ValueError("Selection protocol does not identify this fixed50 design")
    if protocol.get("selection_before_any_tts") is not False or protocol.get("selection_rule_uses_outcomes") is not False:
        raise ValueError("Preserve the post-start, outcome-independent scope-amendment disclosure")
    spec["_selection_timing_disclosure"] = protocol["timing_disclosure"]
    inputs.append(protocol_evidence)
    freeze_path = resolve(spec.get("subset_freeze", "data/SUBSET_FREEZE.json"))
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text())
        if frozen.get("benchmark_sha256") != design_evidence["sha256"]:
            raise ValueError("Limited subset freeze and design disagree")
        inputs.append(evidence(freeze_path))
    if spec.get("allow_source_superset"):
        source_path = resolve(spec["source_design"])
        source_rows, entry = read_stable_rows(source_path)
        if entry["sha256"] != spec.get("source_design_sha256"):
            raise ValueError("Projection requires the pinned original source-design hash")
        source_design = stats.keyed(source_rows, "original source design")
        if not set(mapping) <= set(source_design):
            raise ValueError("Limited design contains a pair absent from its claimed original design")
        for key, row in mapping.items():
            for field in ("source_text_id", "root_id", "book_id", "bucket", "text_ref", "text_tts", "word_target"):
                if row.get(field) != source_design[key].get(field):
                    raise ValueError(f"Limited subset altered original pair {field}")
        inputs.append(entry)
    return spec, design, arms, source_design, inputs


def load_selected_arms(spec_path, spec, design, arm_specs, source_design, group, controls, descriptive):
    ids = list(GROUPS[group]) + (CONTROLS if controls else [])
    missing_specs = set(ids) - set(arm_specs)
    if missing_specs:
        raise ValueError(f"Missing prescribed arm specifications: {sorted(missing_specs)}")
    planned = set(stats.keyed(design, "limited design"))
    selected, inputs, coverage = [], [], []
    for arm_id in ids:
        arm = arm_specs[arm_id]
        path = (spec_path.parent / arm["results"]).resolve()
        if descriptive and not path.exists():
            rows, entry = [], None
        else:
            rows, entry = read_stable_rows(path)
        keyed = stats.keyed(rows, arm_id) if rows else {}
        allowed = set(source_design) if source_design is not None else planned
        if set(keyed) - allowed:
            raise ValueError(f"Foreign scored pair in {arm_id}")
        projected = [keyed[key] for key in sorted(planned & set(keyed))]
        if not descriptive and len(projected) != len(planned):
            raise ValueError(f"Incomplete predefined {group} grid: {arm_id} has {len(projected)}/50 scored pairs")
        selected.append({**arm, "rows": projected})
        coverage.append({"arm_id": arm_id, "planned": len(planned), "scored": len(projected),
                         "source_rows": len(rows), "outside_fixed_subset": len(rows) - len(projected)})
        if entry:
            inputs.append(entry)
    return selected, inputs, coverage


def contrasts_for(spec, selected):
    ids = {a["arm_id"] for a in selected}
    contrasts = [c for c in spec["contrasts"] if c["arm_a"] in ids and c["arm_b"] in ids]
    primary = {(c["arm_a"], c["arm_b"]) for c in contrasts if c.get("role") == "primary"}
    if primary != {(f"{b}_long", f"{b}_short") for b in BACKBONES}:
        raise ValueError("All four predefined Long-minus-Short primary contrasts are required")
    return contrasts


def plot(payload, selected, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    styles = {"base": ("#4C78A8", "o", "-"), "short": ("#E28B31", "s", "-"),
              "long": ("#218C74", "^", "-"), "long_punct": ("#9467BD", "v", "-"),
              "base_chunked": ("#666666", "D", "--")}
    ids = {a["arm_id"] for a in selected}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharey=True, layout="constrained")
    handles, heights = {}, []
    for axis, backbone in zip(axes.ravel(), BACKBONES):
        for variant, (color, marker, line) in styles.items():
            arm = backbone + "_" + variant
            if arm not in ids:
                continue
            metrics = [payload["cells"][b]["arms"][arm]["metrics"]["wer_all_pct"] for b in LENGTHS]
            means = [m["estimate"] for m in metrics]
            bounds = [m["intervals"]["source_conditional"] for m in metrics]
            handle, = axis.plot(range(3), means, color=color, marker=marker, linestyle=line, linewidth=1.5, markersize=4)
            axis.vlines(range(3), [b["lo"] for b in bounds], [b["hi"] for b in bounds], color=color, linewidth=1, alpha=.65)
            handles[variant] = handle
            heights.extend(means + [b["hi"] for b in bounds])
        axis.set_title(BACKBONES[backbone])
        axis.set_xticks(range(3), [str(v) for v in LENGTHS.values()])
        axis.set_xlabel("Input length (words)")
        axis.grid(axis="y", alpha=.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("WER (%) ↓")
    axes[0, 0].set_ylim(0, max(1, max(heights)) * 1.08)
    fig.legend(list(handles.values()), [VARIANTS[v] for v in handles], loc="outside lower center", ncol=min(3, len(handles)), frameon=False)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(out / f"wer_by_length.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def complete_artifacts(spec, design, arms, group, out, inputs, coverage):
    contrasts = contrasts_for(spec, arms)
    payload = stats.analyze(design, arms, contrasts,
        source_field=spec.get("source_cluster_field", "book_id"), n_resamples=10000,
        seed=spec.get("seed", 20260913), alpha=.05, primary_cells=spec.get("primary_cells", ["B4"]),
        descriptive_cohort_cells=spec.get("descriptive_cohort_cells", []))
    payload["provenance"] = {"report_scope": "Separately fixed 50-attempt deadline evaluation", "inputs": inputs,
                             "explicit_arm_group": group, "primary_target": "main13", "coverage": coverage}
    public = stats.write_outputs(payload, out / "statistics", report_content_only=True)
    longest = public["cells"]["B4"]
    continuous = [a for a in GROUPS[group]]
    write_table(out, "main_b4", [metric_record(a, longest["arms"][a]) for a in continuous])
    controls = [a for a in CONTROLS if a in longest["arms"]]
    if controls:
        write_table(out, "chunked_controls_b4", [metric_record(a, longest["arms"][a]) for a in controls])
    plot(public, arms, out)
    support = public["design"]
    length_counts = dict(Counter(r["bucket"] for r in design))
    omitted = [a for a in CONTINUOUS + CONTROLS if a not in {a["arm_id"] for a in arms}]
    lines = ["# Limited evaluation: complete predefined 50-attempt grid", "",
             f"This report covers the explicit **{group}** group: {len(continuous)} continuous systems and {len(controls)} chunked controls. "
             "Each included system has all 50 fixed text–voice pairs. The larger 120-pair evaluation is deferred and is not described as complete.", "",
             spec["_selection_timing_disclosure"] + " The selector used source metadata only; the timing of the scope amendment remains a limitation that the bootstrap does not remove.", "",
             f"Coverage: {support['source_clusters']} conservative work clusters, {support['root_passages']} passages and {support['reference_voices']} reference voices. "
             f"Length-specific attempts per system: " + "; ".join(f"{LENGTHS[b]} words: {length_counts[b]}" for b in LENGTHS) + ".", "",
             "WER and correct-word recall include every planned attempt, including generation or recognition failures. Values above 100% WER are retained. "
             "The longest-input table therefore has its actual B4 subset denominator, not 50 independent long passages.", "",
             "The four Long-minus-Short comparisons remain predefined. Pointwise 95% intervals use the unchanged 10,000-resample source-cluster bootstrap; "
             "crossed source/voice intervals are reported separately. Repeated speakers and nested texts are dependent. "
             "Intervals condition on checkpoints and inference seeds and exclude ASR systematic and retraining uncertainty.", "",
             "Omitted from this report: " + (", ".join(omitted) if omitted else "none") + ". "
             "An omitted arm is not an observed synthesis failure. Its omission is not evidence of model performance.", "",
             "No floor-dependent completion, human MOS or listener-preference claim is published. Word-length labels do not imply human-recording durations. "
             "The manuscript is not modified by this script.", "",
             "## Primary contrasts at the longest word condition", ""]
    for row in longest["contrasts"]:
        if row.get("role") == "primary":
            metric = row["metrics"]["wer_all_pct"]
            ci = metric["intervals"]["source_conditional"]
            lines.append(f"- {label(row['arm_a'])[0]}: Long − Short WER {metric['delta_a_minus_b']:+.1f} pp [{ci['lo']:+.1f}, {ci['hi']:+.1f}].")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return {"state": "complete_fixed_subset", "arm_group": group, "attempts_per_arm": 50,
            "continuous_arm_count": len(continuous), "control_arm_count": len(controls),
            "full_120_evaluation_complete": False, "coverage": coverage,
            "source_support": {k: support[k] for k in ("attempts", "source_clusters", "root_passages", "reference_voices")},
            "length_counts": length_counts, "omitted_arms": omitted}


def descriptive_artifacts(spec, design, arms, group, out, coverage):
    planned = stats.keyed(design, "planned descriptive design")
    summaries, by_arm = [], {}
    for arm in arms:
        rows = arm["rows"]
        if rows:
            # Validate each observed pair without imputing unrun planned pairs.
            stats.validate_inputs([planned[stats.pair_key(r)] for r in rows], [arm], spec.get("source_cluster_field", "book_id"))
        by_arm[arm["arm_id"]] = stats.keyed(rows, arm["arm_id"]) if rows else {}
        for bucket in [*LENGTHS, "ALL"]:
            observed = [r for r in rows if bucket == "ALL" or r["bucket"] == bucket]
            means = np.mean([stats.metric_values(r)[:2] for r in observed], axis=0) if observed else [None, None]
            summaries.append({"arm_id": arm["arm_id"], "bucket": bucket, "scored_n": len(observed),
                              "planned_n": sum(bucket == "ALL" or r["bucket"] == bucket for r in design),
                              "observed_only_wer_pct": None if means[0] is None else float(means[0]),
                              "observed_only_correct_word_recall_pct": None if means[1] is None else float(means[1])})
    paired = []
    for contrast in contrasts_for(spec, arms):
        a, b = contrast["arm_a"], contrast["arm_b"]
        for bucket in [*LENGTHS, "ALL"]:
            keys = sorted(k for k in set(by_arm[a]) & set(by_arm[b]) if bucket == "ALL" or planned[k]["bucket"] == bucket)
            if any(by_arm[a][k]["n_ref_words"] != by_arm[b][k]["n_ref_words"] for k in keys):
                raise ValueError("Observed paired rows have different reference word counts")
            differences = [stats.metric_values(by_arm[a][k])[:2] - stats.metric_values(by_arm[b][k])[:2] for k in keys]
            mean = np.mean(differences, axis=0) if differences else [None, None]
            paired.append(dict(arm_a=a, arm_b=b, bucket=bucket, observed_paired_n=len(keys),
                               observed_only_wer_delta_pp=None if mean[0] is None else float(mean[0]),
                               observed_only_recall_delta_pp=None if mean[1] is None else float(mean[1])))
    stats.write_csv(out / "descriptive_scored_rows.csv", summaries)
    stats.write_csv(out / "descriptive_observed_pairs.csv", paired)
    (out / "DESCRIPTIVE_ONLY.md").write_text(
        "# Incomplete-grid progress snapshot — descriptive only\n\n"
        "These observed-row means and observed-pair differences depend on which computations finished. "
        "They are not the final fixed-50 estimands and cannot substitute for the complete planned grid. "
        "No confidence intervals, paper tables, figures or completion marker are emitted. "
        "Unrun pairs are listed as missing, not imputed as successes or failures. Actual scored/planned N accompanies every cell.\n")
    return {"state": "descriptive_only", "arm_group": group, "coverage": coverage,
            "full_120_evaluation_complete": False, "usable_as_final_table": False}


def run(mode, spec_path, out_dir, group="main13", controls=False):
    spec_path, out_dir = spec_path.resolve(), out_dir.resolve()
    if out_dir.exists():
        raise ValueError("Refusing to replace an existing output snapshot")
    spec, design, arms, source_design, inputs = load_plan(spec_path)
    selected, score_inputs, coverage = load_selected_arms(spec_path, spec, design, arms, source_design, group, controls, mode == "descriptive")
    inputs += score_inputs
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".limited-report-", dir=out_dir.parent) as folder:
        work = Path(folder) / "report"
        work.mkdir()
        result = (complete_artifacts(spec, design, selected, group, work, inputs, coverage) if mode == "final" else
                  descriptive_artifacts(spec, design, selected, group, work, coverage))
        for item in inputs:
            if stats.sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Input changed during report assembly: {item['path']}")
        result.update(created_at_utc=datetime.now(timezone.utc).isoformat(), input_evidence=inputs, manuscript_updated=False,
                      output_sha256={str(p.relative_to(work)): stats.sha256(p) for p in work.rglob("*") if p.is_file()})
        write_json(work / ("REPORT_COMPLETE.json" if mode == "final" else "DESCRIPTIVE_SNAPSHOT.json"), result)
        shutil.move(str(work), out_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["final", "descriptive"])
    parser.add_argument("--spec", type=Path, default=DEFAULT_ROOT / "statistics_spec.json")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--group", choices=GROUPS, default="main13")
    parser.add_argument("--controls", choices=["none", "chunked"], default="none")
    args = parser.parse_args()
    suffix = args.group + ("_with_controls" if args.controls == "chunked" else "")
    out = args.out_dir or args.spec.parent / "reports" / (suffix if args.mode == "final" else "descriptive_" + suffix)
    try:
        result = run(args.mode, args.spec, out, args.group, args.controls == "chunked")
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
        print(json.dumps({"state": "not_reported", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"state": result["state"], "out_dir": str(out), "arm_group": args.group}))


if __name__ == "__main__":
    main()
