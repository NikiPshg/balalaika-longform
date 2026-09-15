"""Report integrity tests use synthetic scored rows only inside pytest tmp_path."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expanded_eval_report as report


def full_fixture(root):
    design = []
    refs = [{"voice_id": f"n{i}"} for i in range(8)] + [{"voice_id": f"p{i}"} for i in range(13)]
    for i in range(30):
        pair = [f"n{i % 8}", f"p{i % 13}"]
        shorter = pair if i < 15 else list(reversed(pair))
        for bucket, voices in (("B0", shorter[:1]), ("B2", shorter[1:]), ("B4", pair)):
            for voice in voices:
                design.append(dict(text_id=f"r{i}_{bucket}_{voice}", source_text_id=f"r{i}_{bucket}",
                    root_id=f"r{i}", book_id=f"book{i % 22}", voice_id=voice, seed=0, bucket=bucket,
                    voice_cohort="new_external_heldout" if voice[0] == "n" else "previously_evaluated_heldout",
                    word_target=report.LENGTHS[bucket], primary_report_content_only=True, human_audio_path=None))
    report.write_jsonl(root / "data/benchmark.jsonl", design)
    report.write_jsonl(root / "data/references.jsonl", refs)
    report.write_json(root / "data/BENCHMARK_FREEZE.json", dict(text_only=True, report_content_only=True,
        benchmark_sha256=report.stats.sha256(root / "data/benchmark.jsonl"),
        references_sha256=report.stats.sha256(root / "data/references.jsonl")))
    arms = []
    for arm_id in report.CONTINUOUS + report.CONTROLS:
        family = "flow" if arm_id.startswith("f5_") else "ar"
        metadata = dict(arm_id=arm_id, backbone=report.label(arm_id)[0], family=family,
                        protocol="chunked" if arm_id.endswith("chunked") else "continuous",
                        results=f"results/{arm_id}/per_item.jsonl")
        arms.append(metadata)
        rows = []
        for item in design:
            nref = item["word_target"]
            deleted = int(nref * (.1 if "long" in arm_id else .3))
            inserted = nref * 2 if arm_id.endswith("base") else 0
            missing = item["text_id"] == "r0_B4_n0"
            if missing:
                deleted, inserted = nref, 0
            row = dict(item, run_id=arm_id + item["text_id"], status="empty_or_invalid_audio" if missing else "complete",
                status_complete=not missing, output_path=None if missing else f"/synthetic/{arm_id}/{item['text_id']}.wav",
                output_exists=not missing, n_ref_words=nref, n_hyp_words=nref-deleted+inserted,
                hits=nref-deleted, deletions=deleted, substitutions=0, insertions=inserted,
                wer=(deleted+inserted)/nref, stop_reason="flow_completed" if family == "flow" else "eos")
            if family == "flow":
                row["generation_family"] = "non_autoregressive_flow_matching"
            rows.append(row)
        report.write_jsonl(root / metadata["results"], rows)
        if arm_id in report.SECONDARY:
            derived = []
            for original in rows:
                if original["bucket"] != "B4":
                    continue
                row = dict(original)
                for field in ("generation_family", "stop_reason", "output_exists"):
                    row.pop(field, None)
                row.update(wer_primary=original["wer"], wer_secondary=original["wer"],
                    completion_status_source="unchanged_primary_rnnt", asr_model_id="t-tech/t-one",
                    asr_role="prespecified_secondary_wer_audit")
                derived.append(row)
            report.write_jsonl(root / "results_secondary" / arm_id / "per_item.jsonl", derived)
    spec = dict(design="data/benchmark.jsonl", arms=arms, resamples=100, seed=20260913, alpha=.05,
                report_content_only=True, primary_cells=["B4"], source_cluster_field="book_id",
                descriptive_cohort_cells=[dict(bucket="B4", voice_cohort="new_external_heldout")],
                contrasts=[dict(arm_a=f"{b}_long", arm_b=f"{b}_short", role="primary") for b in report.BACKBONES])
    report.write_json(root / "statistics_spec.json", spec)
    return design, spec


def test_primary_complete_grid_writes_13_continuous_rows_two_controls_and_unclipped_figure(tmp_path):
    full_fixture(tmp_path)
    out = tmp_path / "reports/primary"
    result = report.run_report("primary", tmp_path, tmp_path / "statistics_spec.json", out)
    assert result["arm_count"] == 15 and result["scored_pairs_per_arm"] == 120
    import csv
    main = list(csv.DictReader((out / "main_b4.csv").open()))
    controls = list(csv.DictReader((out / "chunked_controls_b4.csv").open()))
    assert len(main) == 13 and len(controls) == 2
    assert max(float(row["wer_pct"]) for row in main) > 200
    assert all(row["attempts"] == "60" for row in main + controls)
    assert (out / "wer_by_length.pdf").read_bytes().startswith(b"%PDF")
    assert (out / "wer_by_length.png").read_bytes().startswith(b"\x89PNG")
    for file in ("main_b4.tex", "main_b4.md", "statistics/statistics.json"):
        assert "frozen_complete" not in (out / file).read_text()
    assert result["manuscript_updated"] is False


def test_missing_primary_attempt_refuses_entire_report_directory(tmp_path):
    full_fixture(tmp_path)
    path = tmp_path / "results/f5_long/per_item.jsonl"
    report.write_jsonl(path, report.stats.read_jsonl(path)[:-1])
    out = tmp_path / "reports/primary"
    with pytest.raises(ValueError, match="planned pairing mismatch"):
        report.run_report("primary", tmp_path, tmp_path / "statistics_spec.json", out)
    assert not out.exists()


def test_secondary_recovers_flow_metadata_keeps_originals_and_paired_gap_is_zero_for_same_transcripts(tmp_path):
    full_fixture(tmp_path)
    paths = [tmp_path / "results_secondary" / a / "per_item.jsonl" for a in report.SECONDARY]
    original_hashes = {p: report.stats.sha256(p) for p in paths}
    out = tmp_path / "reports/secondary"
    report.run_report("secondary", tmp_path, tmp_path / "statistics_spec.json", out)
    assert all(report.stats.sha256(p) == sha for p, sha in original_hashes.items())
    flow = report.stats.read_jsonl(out / "derived_views/f5_long/per_item.jsonl")
    assert all(r["generation_family"] == "non_autoregressive_flow_matching" for r in flow)
    assert any(r["output_exists"] is False and r["wer"] == 1 for r in flow)
    comparisons = json.loads((out / "secondary_paired.json").read_text())
    assert len(comparisons) == 8
    for row in comparisons:
        assert row["change_in_delta_pp"] == pytest.approx(0)
        assert row["change_source_conditional_lo"] == row["change_source_conditional_hi"] == 0
    assert len(report.stats.read_jsonl(out / "derived_views/benchmark_b4.jsonl")) == 60


@pytest.mark.parametrize("mutation", ["missing", "status", "wer_provenance", "recognizer"])
def test_secondary_rejects_missing_or_reclassified_or_untraceable_pairs(tmp_path, mutation):
    design, _ = full_fixture(tmp_path)
    secondary = report.stats.read_jsonl(tmp_path / "results_secondary/f5_long/per_item.jsonl")
    primary = report.stats.read_jsonl(tmp_path / "results/f5_long/per_item.jsonl")
    if mutation == "missing":
        secondary.pop()
    elif mutation == "status":
        secondary[0]["status"] = "reclassified_by_secondary"
    elif mutation == "wer_provenance":
        secondary[0]["wer_primary"] += .1
    else:
        secondary[0]["asr_model_id"] = "different-asr"
    with pytest.raises(ValueError):
        report.secondary_view(secondary, primary, [r for r in design if r["bucket"] == "B4"], "f5_long")


def test_speaker_statistic_has_available_n_and_rejects_partial_speaker_grid(tmp_path):
    design, spec = full_fixture(tmp_path)
    arms = []
    for arm in spec["arms"]:
        source_path = tmp_path / arm["results"]
        rows = report.stats.read_jsonl(source_path)
        header = dict(_meta=True, per_item_sha256=report.stats.sha256(source_path))
        speaker = [dict(r, sim_median=.8 if r["output_exists"] else None, n_windows=3 if r["output_exists"] else 0) for r in rows]
        report.write_jsonl(tmp_path / "results" / arm["arm_id"] / "speaker_drift.jsonl", [header] + speaker)
        arms.append(dict(arm, source_path=str(source_path)))
    measured, _ = report.speaker_rows(tmp_path, design, arms)
    assert len(measured) == 15
    assert all(r["measured_n"] == 59 and r["attempted_n"] == 60 and r["sim_median"] == .8 for r in measured)
    path = tmp_path / "results/cosy_base/speaker_drift.jsonl"
    report.write_jsonl(path, report.stats.read_jsonl(path)[:-1])
    with pytest.raises(ValueError, match="Incomplete speaker grid"):
        report.speaker_rows(tmp_path, design, arms)


def test_changed_frozen_design_is_rejected(tmp_path):
    full_fixture(tmp_path)
    path = tmp_path / "data/benchmark.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash disagrees"):
        report.load_context(tmp_path, tmp_path / "statistics_spec.json", "primary")


def test_a_writer_modifying_scores_during_read_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    report.write_jsonl(path, [{"value": 1}])
    original = report.stats.read_jsonl
    def mutate_after_read(source):
        rows = original(source)
        source.write_text(source.read_text() + "\n")
        return rows
    monkeypatch.setattr(report.stats, "read_jsonl", mutate_after_read)
    with pytest.raises(ValueError, match="changed while it was being read"):
        report.read_stable_rows(path)
