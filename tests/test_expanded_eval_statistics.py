"""Integrity and clustered-pairing checks for the expanded evaluation report."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.expanded_eval_statistics import (
    analyze, bootstrap_weights, interval, metric_values, run, weighted_means, write_outputs,
)


def fixture_grid():
    """Three books, two crossed voices, two lengths: 12 dependent attempts."""
    design, short, long = [], [], []
    for source in range(3):
        for bucket in ("B0", "B4"):
            for voice in range(2):
                item = dict(text_id=f"s{source}_{bucket}_v{voice}", source_text_id=f"s{source}_{bucket}",
                            root_id=f"s{source}", book_id=f"book{source}", voice_id=f"v{voice}",
                            seed=0, bucket=bucket)
                design.append(item)
                base = dict(item, status="complete", status_complete=True, floor_available=False,
                            output_exists=True, n_ref_words=100, substitutions=0, insertions=0,
                            stop_reason="eos", run_id=item["text_id"])
                deletions = 30 + source * 10 + voice * 5
                short.append(dict(base, hits=100 - deletions, deletions=deletions, wer=deletions / 100))
                long.append(dict(base, hits=110 - deletions, deletions=deletions - 10, wer=(deletions - 10) / 100))
    arms = [dict(arm_id="short", family="ar", protocol="continuous", rows=short),
            dict(arm_id="long", family="ar", protocol="continuous", rows=long)]
    contrast = [dict(arm_a="long", arm_b="short", role="primary")]
    return design, arms, contrast


def test_all_attempts_include_failed_generation_and_wer_above_100_percent():
    design, arms, contrasts = fixture_grid()
    # A completely missing waveform remains a deletion of all reference words.
    arms[0]["rows"][0].update(status="empty_or_invalid_audio", status_complete=False,
                             hits=0, deletions=100, wer=1.0, output_exists=False)
    # Insertions can produce WER > 100% while reference words remain present.
    arms[0]["rows"][1].update(hits=100, deletions=0, insertions=150, wer=1.5)
    result = analyze(design, arms, contrasts, n_resamples=400)
    short = result["cells"]["ALL"]["arms"]["short"]
    assert short["n_attempted"] == 12
    assert short["n_complete"] == 11
    assert short["n_audio_missing"] == 1
    assert short["metrics"]["wer_all_pct"]["estimate"] == pytest.approx(
        np.mean([r["wer"] * 100 for r in arms[0]["rows"]]))
    assert metric_values(arms[0]["rows"][1])[0] == 150


def test_exact_paired_constant_effect_and_nested_source_support():
    design, arms, contrasts = fixture_grid()
    result = analyze(design, arms, contrasts, n_resamples=600)
    assert result["cells"]["ALL"]["support"]["attempts"] == 12
    assert result["cells"]["ALL"]["support"]["source_clusters"] == 3
    assert result["cells"]["ALL"]["support"]["distinct_texts"] == 6
    assert result["cells"]["B4"]["support"]["attempts"] == 6
    assert result["cells"]["B4"]["support"]["source_clusters"] == 3
    for cell in result["cells"].values():
        metrics = cell["contrasts"][0]["metrics"]
        assert metrics["wer_all_pct"]["delta_a_minus_b"] == pytest.approx(-10)
        for bounds in metrics["wer_all_pct"]["intervals"].values():
            assert bounds["lo"] == pytest.approx(-10)
            assert bounds["hi"] == pytest.approx(-10)
            assert bounds["degenerate"]
        assert metrics["correct_word_recall_pct"]["delta_a_minus_b"] == pytest.approx(10)


def test_reference_voice_variance_is_absent_from_conditional_but_present_in_crossed_ci():
    design, arms, contrasts = fixture_grid()
    # With a balanced source x voice grid and outcomes depending only on voice,
    # resampling sources cannot detect uncertainty from the choice of voices.
    for row in arms[0]["rows"]:
        errors = 10 if row["voice_id"] == "v0" else 90
        row.update(hits=100 - errors, deletions=errors, wer=errors / 100)
    result = analyze(design, arms, contrasts, n_resamples=2000)
    bounds = result["cells"]["B4"]["arms"]["short"]["metrics"]["wer_all_pct"]["intervals"]
    assert bounds["source_conditional"]["lo"] == pytest.approx(50)
    assert bounds["source_conditional"]["hi"] == pytest.approx(50)
    assert bounds["source_voice_crossed"]["lo"] == pytest.approx(10)
    assert bounds["source_voice_crossed"]["hi"] == pytest.approx(90)


def test_draws_preserve_nested_items_and_repeated_crossed_factors():
    design, _, _ = fixture_grid()
    weights, _ = bootstrap_weights(design, "book_id", 200, 20260913)
    for method, matrix in weights.items():
        for i, a in enumerate(design):
            for j, b in enumerate(design):
                if a["book_id"] == b["book_id"] and (method == "source_conditional" or a["voice_id"] == b["voice_id"]):
                    np.testing.assert_array_equal(matrix[:, i], matrix[:, j])
    assert not np.array_equal(weights["source_conditional"], weights["source_voice_crossed"])


def test_determinism_and_order_invariance():
    design, arms, contrasts = fixture_grid()
    a = analyze(design, arms, contrasts, n_resamples=300, seed=71)
    shuffled = deepcopy(arms)
    for arm in shuffled:
        arm["rows"] = list(reversed(arm["rows"]))
    b = analyze(list(reversed(design)), shuffled, contrasts, n_resamples=300, seed=71)
    assert a == b


def test_unbalanced_cluster_means_use_attempt_weights_not_equal_cluster_means():
    # Draw each source once: source A contributes 3 items and B contributes 1.
    values = np.array([0, 0, 0, 100])
    weights = np.array([[1, 1, 1, 1], [2, 2, 2, 0], [0, 0, 0, 2]])
    np.testing.assert_allclose(weighted_means(values, weights).ravel(), [25, 0, 100])


def test_zero_weight_sparse_resamples_are_counted_not_silently_reported_as_zero():
    values = weighted_means(np.array([3, 5]), np.array([[0, 0], [1, 0], [1, 1]]))[:, 0]
    result = interval(values, 0.05)
    assert np.isnan(values[0])
    assert result["n_empty"] == 1
    assert result["n_valid"] == 2
    assert 3 <= result["lo"] <= result["hi"] <= 4


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "wrong_seed"])
def test_rejects_unpaired_grid_instead_of_intersection_filtering(problem):
    design, arms, contrasts = fixture_grid()
    if problem == "missing":
        arms[0]["rows"].pop()
    elif problem == "extra":
        arms[0]["rows"].append(dict(arms[0]["rows"][0], text_id="unknown"))
    elif problem == "duplicate":
        arms[0]["rows"].append(deepcopy(arms[0]["rows"][0]))
    else:
        arms[0]["rows"][0]["seed"] = 9
    with pytest.raises(ValueError, match="pair|Duplicate"):
        analyze(design, arms, contrasts, n_resamples=100)


def test_rejects_changed_source_mapping_and_inconsistent_wer_units():
    design, arms, contrasts = fixture_grid()
    bad = deepcopy(arms)
    bad[0]["rows"][0]["book_id"] = "different"
    with pytest.raises(ValueError, match="metadata"):
        analyze(design, bad, contrasts, n_resamples=100)
    bad = deepcopy(arms)
    bad[0]["rows"][0]["wer"] *= 100
    with pytest.raises(ValueError, match="WER"):
        analyze(design, bad, contrasts, n_resamples=100)


def test_flow_failures_keep_non_eos_labels_and_full_denominator():
    design, arms, contrasts = fixture_grid()
    arms[0]["family"] = "flow"
    for row in arms[0]["rows"]:
        row["generation_family"] = "non_autoregressive_flow_matching"
        row["stop_reason"] = "flow_completed"
    arms[0]["rows"][0].update(status="incomplete_text", status_complete=False)
    result = analyze(design, arms, contrasts, n_resamples=100)
    assert result["cells"]["ALL"]["arms"]["short"]["status_counts"]["incomplete_text"] == 1
    assert result["cells"]["ALL"]["arms"]["short"]["n_attempted"] == 12
    arms[0]["rows"][0]["status"] = "early_eos"
    with pytest.raises(ValueError, match="EOS"):
        analyze(design, arms, contrasts, n_resamples=100)


def test_file_integration_writes_auditable_json_csv_and_markdown(tmp_path):
    design, arms, contrasts = fixture_grid()
    def jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    jsonl(tmp_path / "design.jsonl", design)
    specs = []
    for arm in arms:
        path = tmp_path / f"{arm['arm_id']}.jsonl"
        jsonl(path, arm["rows"])
        specs.append({**{k: v for k, v in arm.items() if k != "rows"}, "results": path.name})
    spec = dict(design="design.jsonl", arms=specs, contrasts=contrasts, resamples=100)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    result = run(spec_path, tmp_path / "out")
    saved = json.loads((tmp_path / "out/statistics.json").read_text())
    assert saved == result
    assert len(saved["provenance"]["design"]["sha256"]) == 64
    assert (tmp_path / "out/arm_estimates.csv").stat().st_size > 100
    assert (tmp_path / "out/paired_differences.csv").stat().st_size > 100
    assert "12 attempts per arm; 3 source clusters" in (tmp_path / "out/statistics.md").read_text()


def test_content_only_report_suppresses_floor_dependent_labels_but_keeps_every_failure(tmp_path):
    design, arms, contrasts = fixture_grid()
    arms[0]["rows"][0].update(status="empty_or_invalid_audio", status_complete=False,
                             hits=0, deletions=100, wer=1, output_exists=False)
    internal = analyze(design, arms, contrasts, n_resamples=100)
    public = write_outputs(internal, tmp_path, report_content_only=True)
    assert public == json.loads((tmp_path / "statistics.json").read_text())
    assert public["cells"]["ALL"]["arms"]["short"]["n_attempted"] == 12
    assert public["cells"]["ALL"]["arms"]["short"]["n_audio_missing"] == 1
    assert internal["cells"]["ALL"]["arms"]["short"]["n_complete"] == 11
    assert "status_counts" in json.loads((tmp_path / "statistics_internal.json").read_text())["cells"]["ALL"]["arms"]["short"]
    for name in ("statistics.json", "statistics.md", "arm_estimates.csv", "paired_differences.csv"):
        content = (tmp_path / name).read_text().lower()
        assert "frozen_complete" not in content and "n_complete" not in content
        assert "complete_count" not in content and "complete x/n" not in content
        assert "status_counts" not in content


def test_prespecified_new_voice_cell_is_descriptive_and_keeps_every_paired_source():
    design, arms, contrasts = fixture_grid()
    for row in design:
        row["voice_cohort"] = "new" if row["voice_id"] == "v0" else "old"
    result = analyze(design, arms, contrasts, n_resamples=100,
                     descriptive_cohort_cells=[{"bucket": "B4", "voice_cohort": "new"}])
    assert result["method"]["primary_cells"] == ["B4"]
    assert result["cells"]["B4::new"]["support"]["attempts"] == 3
    assert result["cells"]["B4::new"]["support"]["source_clusters"] == 3
    assert result["cells"]["B4::new"]["support"]["reference_voices"] == 1
    assert result["cells"]["B4::new"]["contrasts"][0]["metrics"]["wer_all_pct"]["delta_a_minus_b"] == pytest.approx(-10)
    with pytest.raises(ValueError, match="absent from design"):
        analyze(design, arms, contrasts, n_resamples=100,
                descriptive_cohort_cells=[{"bucket": "B4", "voice_cohort": "missing"}])
