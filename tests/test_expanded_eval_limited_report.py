"""The deadline subset is fixed independently of which result rows exist."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import expanded_eval_limited_report as limited
from test_expanded_eval_report import full_fixture


def fixture(tmp_path):
    parent = tmp_path / "parent"
    all_rows, original = full_fixture(parent)
    selected = []
    for bucket, target in (("B0", 14), ("B2", 14), ("B4", 22)):
        seen = set()
        for row in all_rows:
            if row["bucket"] == bucket and row["book_id"] not in seen:
                selected.append(row)
                seen.add(row["book_id"])
                if len(seen) == target:
                    break
    root = tmp_path / "limited50"
    root.mkdir()
    path = root / "benchmark.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in selected))
    spec = deepcopy(original)
    spec.update(design="benchmark.jsonl", design_sha256=limited.stats.sha256(path),
                planned_attempts=50, report_groups=limited.GROUPS, resamples=10000,
                allow_source_superset=True, source_design=str(parent / "data/benchmark.jsonl"),
                source_design_sha256=limited.stats.sha256(parent / "data/benchmark.jsonl"),
                descriptive_cohort_cells=[])
    for arm in spec["arms"]:
        arm["results"] = str(parent / arm["results"])
    spec_path = root / "statistics_spec.json"
    limited.write_json(spec_path, spec)
    (root / "data").mkdir()
    limited.write_json(root / "data/selection_protocol.json", dict(
        subset_benchmark_sha256=spec["design_sha256"], selection_before_any_tts=False,
        selection_rule_uses_outcomes=False,
        timing_disclosure="Synthetic fixture: a fixed metadata-only subset after parent-run start."))
    return root, spec_path, spec, selected


def test_full50_main13_exports_all_formats_and_uses_true_b4_denominator(tmp_path):
    root, spec_path, _, _ = fixture(tmp_path)
    out = root / "reports/main13"
    result = limited.run("final", spec_path, out)
    assert result["state"] == "complete_fixed_subset"
    assert result["continuous_arm_count"] == 13
    assert result["full_120_evaluation_complete"] is False
    assert result["length_counts"] == {"B0": 14, "B2": 14, "B4": 22}
    assert result["source_support"]["source_clusters"] == 22
    table = list(csv.DictReader((out / "main_b4.csv").open()))
    assert len(table) == 13 and {r["attempts"] for r in table} == {"22"}
    assert max(float(r["wer_pct"]) for r in table) > 200
    assert (out / "wer_by_length.pdf").read_bytes().startswith(b"%PDF")
    assert "<svg" in (out / "wer_by_length.svg").read_text()
    assert (out / "wer_by_length.png").read_bytes().startswith(b"\x89PNG")
    assert (out / "REPORT_COMPLETE.json").exists()
    assert all(r["scored"] == 50 and r["outside_fixed_subset"] == 70 for r in result["coverage"])
    assert "frozen_complete" not in (out / "statistics/statistics.json").read_text()


def test_primary8_is_explicit_and_does_not_read_missing_base_scores(tmp_path):
    root, spec_path, spec, _ = fixture(tmp_path)
    Path(next(a["results"] for a in spec["arms"] if a["arm_id"] == "cosy_base")).unlink()
    with pytest.raises(FileNotFoundError):
        limited.run("final", spec_path, root / "reports/main13")
    assert not (root / "reports/main13").exists()
    result = limited.run("final", spec_path, root / "reports/primary8", group="primary8")
    assert result["continuous_arm_count"] == 8 and "cosy_base" in result["omitted_arms"]
    assert result["arm_group"] == "primary8"


def test_missing_planned_score_rejects_final_but_descriptive_labels_counts_and_has_no_intervals(tmp_path):
    root, spec_path, spec, selected = fixture(tmp_path)
    path = Path(next(a["results"] for a in spec["arms"] if a["arm_id"] == "f5_long"))
    key = limited.stats.pair_key(selected[0])
    rows = [r for r in limited.stats.read_jsonl(path) if limited.stats.pair_key(r) != key]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="49/50"):
        limited.run("final", spec_path, root / "reports/final")
    out = root / "reports/descriptive"
    result = limited.run("descriptive", spec_path, out)
    assert result["state"] == "descriptive_only" and result["usable_as_final_table"] is False
    assert not (out / "REPORT_COMPLETE.json").exists()
    assert not list(out.glob("*.pdf")) and not (out / "main_b4.csv").exists()
    summaries = list(csv.DictReader((out / "descriptive_scored_rows.csv").open()))
    cell = next(r for r in summaries if r["arm_id"] == "f5_long" and r["bucket"] == "ALL")
    assert cell["scored_n"] == "49" and cell["planned_n"] == "50"
    assert not any("ci" in field or "interval" in field for field in cell)


def test_superset_projection_requires_pinned_parent_and_rejects_foreign_row(tmp_path):
    root, spec_path, spec, _ = fixture(tmp_path)
    spec["source_design_sha256"] = "wrong"
    limited.write_json(spec_path, spec)
    with pytest.raises(ValueError, match="source-design hash"):
        limited.load_plan(spec_path)
    spec["source_design_sha256"] = limited.stats.sha256(Path(spec["source_design"]))
    limited.write_json(spec_path, spec)
    path = Path(spec["arms"][0]["results"])
    rows = limited.stats.read_jsonl(path)
    rows.append(dict(rows[0], text_id="foreign"))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="Foreign scored pair"):
        limited.run("final", spec_path, root / "reports/foreign")


def test_changed50_design_and_data_driven_group_reduction_are_rejected(tmp_path):
    _, spec_path, spec, _ = fixture(tmp_path)
    spec["design_sha256"] = "wrong"
    limited.write_json(spec_path, spec)
    with pytest.raises(ValueError, match="exact design_sha256"):
        limited.load_plan(spec_path)
    spec["design_sha256"] = limited.stats.sha256(spec_path.parent / spec["design"])
    spec["report_groups"] = deepcopy(limited.GROUPS)
    spec["report_groups"]["primary8"].pop()
    limited.write_json(spec_path, spec)
    with pytest.raises(ValueError, match="complete fixed arm list"):
        limited.load_plan(spec_path)


def test_original_text_cannot_be_rewritten_when_projecting_parent_scores(tmp_path):
    root, spec_path, spec, selected = fixture(tmp_path)
    selected[0]["text_ref"] = "Изменённый текст."
    path = root / "benchmark.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in selected))
    spec["design_sha256"] = limited.stats.sha256(path)
    limited.write_json(spec_path, spec)
    protocol_path = root / "data/selection_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["subset_benchmark_sha256"] = spec["design_sha256"]
    limited.write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="altered original pair text_ref"):
        limited.load_plan(spec_path)


def test_descriptive_pairs_still_require_identical_reference_counts(tmp_path):
    root, spec_path, spec, selected = fixture(tmp_path)
    path = Path(next(a["results"] for a in spec["arms"] if a["arm_id"] == "f5_long"))
    rows = limited.stats.read_jsonl(path)
    row = next(r for r in rows if r["text_id"] == selected[0]["text_id"])
    row["n_ref_words"] += 1
    row["hits"] += 1
    row["wer"] = (row["deletions"] + row["substitutions"] + row["insertions"]) / row["n_ref_words"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="different reference word counts"):
        limited.run("descriptive", spec_path, root / "reports/bad_descriptive")


def test_scope_amendment_cannot_be_relabelled_pre_tts(tmp_path):
    root, spec_path, _, _ = fixture(tmp_path)
    path = root / "data/selection_protocol.json"
    protocol = json.loads(path.read_text())
    protocol["selection_before_any_tts"] = True
    limited.write_json(path, protocol)
    with pytest.raises(ValueError, match="scope-amendment disclosure"):
        limited.load_plan(spec_path)
