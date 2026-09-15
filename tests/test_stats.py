"""Analytic bootstrap tests; scripts/reproduce.py checks the released observations."""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stats.bootstrap import (  # noqa: E402
    BUCKETS,
    CELLS,
    METRICS,
    Arm,
    aligned_keys,
    bootstrap_p_two_sided,
    build_clusters,
    cluster_bootstrap_mean,
    cluster_bootstrap_ols,
    compare,
    load_arm,
    load_jsonl,
    ols_fit,
    paired_delta,
    percentile_ci,
    resample_matrix,
    rmst_table,
    stable_words,
)

BENCH = REPO_ROOT / "data/benchmark/pilot.jsonl"
ARMS = {
    "E3": REPO_ROOT / "results/v31_sft/E3_epoch_3_step_3001/per_item.jsonl",
    "E2": REPO_ROOT / "results/v31_sft/E2_epoch_3_step_3001/per_item.jsonl",
    "E1": REPO_ROOT / "results/v31_base/E1/per_item.jsonl",
    "E3loss": REPO_ROOT / "results/v31_sft/E3_epoch_0_whole/per_item.jsonl",
    "E2loss": REPO_ROOT / "results/v31_sft/E2_epoch_0_step_200/per_item.jsonl",
    "E0": REPO_ROOT / "results/v31_base/E0/per_item.jsonl",
}


# --------------------------------------------------------------- synthetic


def test_resample_matrix_is_seed_reproducible_and_in_range():
    a = resample_matrix(6, 500, seed=0)
    b = resample_matrix(6, 500, seed=0)
    c = resample_matrix(6, 500, seed=1)
    assert a.shape == (500, 6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0 and a.max() <= 5


def test_single_cluster_bootstrap_is_degenerate():
    """One cluster: every resample is the same data, so the CI collapses."""
    values = np.array([1.0, 2.0, 3.0, 10.0])
    ci = np.zeros(4, dtype=np.int64)
    idx = resample_matrix(1, 100, seed=0)
    point, dist = cluster_bootstrap_mean(values, ci, idx, 1)
    assert point == pytest.approx(4.0)
    assert np.allclose(dist, 4.0)
    lo, hi, n = percentile_ci(dist)
    assert lo == pytest.approx(4.0) and hi == pytest.approx(4.0) and n == 100


def test_resampled_mean_lies_between_cluster_means_when_clusters_are_balanced():
    """Balanced clusters: any resample mean is a convex combination of them."""
    values = np.array([0.0, 0.0, 5.0, 5.0, 9.0, 9.0])
    ci = np.array([0, 0, 1, 1, 2, 2])
    idx = resample_matrix(3, 2000, seed=0)
    point, dist = cluster_bootstrap_mean(values, ci, idx, 3)
    assert point == pytest.approx(np.mean([0.0, 5.0, 9.0]))
    assert dist.min() >= 0.0 - 1e-12
    assert dist.max() <= 9.0 + 1e-12
    # all three clusters drawn once each reproduces the point estimate
    assert np.isclose(dist, point).any()


def test_cluster_bootstrap_pools_items_not_cluster_means():
    """Unequal cluster sizes: the estimator is the pooled item mean."""
    values = np.array([1.0, 1.0, 1.0, 7.0])  # cluster 0 has 3 items, cluster 1 has 1
    ci = np.array([0, 0, 0, 1])
    idx = np.array([[0, 1], [0, 0], [1, 1]], dtype=np.int64)
    point, dist = cluster_bootstrap_mean(values, ci, idx, 2)
    assert point == pytest.approx(2.5)
    assert dist[0] == pytest.approx((3 + 7) / 4)  # clusters {0,1}
    assert dist[1] == pytest.approx(1.0)  # cluster 0 twice
    assert dist[2] == pytest.approx(7.0)  # cluster 1 twice


def test_paired_delta_equals_difference_of_means():
    rng = np.random.default_rng(7)
    a = rng.normal(size=40)
    b = rng.normal(size=40)
    ci = np.repeat(np.arange(4), 10)
    idx = resample_matrix(4, 500, seed=0)
    res = paired_delta(a, b, ci, idx, 4)
    assert res["delta"] == pytest.approx(a.mean() - b.mean())
    assert res["mean_a"] == pytest.approx(a.mean())
    assert res["mean_b"] == pytest.approx(b.mean())
    assert res["n_items"] == 40
    assert res["n_clusters"] == 4
    assert res["ci_lo"] <= res["delta"] <= res["ci_hi"]


def test_paired_delta_on_binary_outcome_is_a_risk_difference():
    a = np.array([1.0, 1.0, 1.0, 0.0])  # 75 %
    b = np.array([0.0, 0.0, 1.0, 0.0])  # 25 %
    ci = np.array([0, 0, 1, 1])
    idx = resample_matrix(2, 100, seed=0)
    res = paired_delta(a, b, ci, idx, 2)
    assert res["delta"] == pytest.approx(0.50)


def test_paired_delta_rejects_unpaired_input():
    with pytest.raises(ValueError):
        paired_delta(np.zeros(3), np.zeros(4), np.zeros(3, dtype=np.int64), resample_matrix(1, 5, 0), 1)


def test_identical_arms_give_a_zero_delta_and_a_zero_width_ci():
    x = np.array([0.3, 0.9, 0.1, 0.7, 0.5, 0.5])
    ci = np.array([0, 0, 1, 1, 2, 2])
    idx = resample_matrix(3, 500, seed=0)
    res = paired_delta(x, x.copy(), ci, idx, 3)
    assert res["delta"] == pytest.approx(0.0)
    assert res["ci_lo"] == pytest.approx(0.0)
    assert res["ci_hi"] == pytest.approx(0.0)
    assert res["excludes_zero"] is False


def test_percentile_ci_matches_numpy_and_drops_nan():
    d = np.concatenate([np.arange(1000, dtype=float), [np.nan]])
    lo, hi, n = percentile_ci(d, alpha=0.05)
    ref_lo, ref_hi = np.percentile(np.arange(1000, dtype=float), [2.5, 97.5])
    assert n == 1000
    assert lo == pytest.approx(ref_lo)
    assert hi == pytest.approx(ref_hi)


def test_bootstrap_p_is_floored_at_one_over_b_and_capped_at_one():
    assert bootstrap_p_two_sided(np.ones(100)) == pytest.approx(0.01)
    assert bootstrap_p_two_sided(np.zeros(100)) == pytest.approx(1.0)
    mixed = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)])
    assert bootstrap_p_two_sided(mixed) == pytest.approx(1.0)


def test_ols_recovers_known_coefficients_exactly():
    rng = np.random.default_rng(3)
    n = 200
    a = rng.integers(0, 2, n).astype(float)
    c = rng.normal(size=n)
    X = np.column_stack([np.ones(n), a, c, a * c])
    beta = np.array([0.5, -0.2, 0.3, 0.7])
    y = X @ beta  # noiseless
    assert ols_fit(X, y) == pytest.approx(beta, abs=1e-9)


def test_cluster_bootstrap_ols_is_exact_on_noiseless_data():
    """With a perfect fit every resample must return the same coefficients."""
    n_clusters = 5
    per = 20
    rng = np.random.default_rng(11)
    a = np.tile(np.repeat([0.0, 1.0], per // 2), n_clusters)
    c = rng.normal(size=n_clusters * per)
    X = np.column_stack([np.ones_like(c), a, c, a * c])
    beta = np.array([1.0, -0.4, 0.25, 0.6])
    y = X @ beta
    ci = np.repeat(np.arange(n_clusters), per)
    idx = resample_matrix(n_clusters, 300, seed=0)
    point, dist = cluster_bootstrap_ols(X, y, ci, idx, n_clusters)
    assert point == pytest.approx(beta, abs=1e-9)
    assert np.allclose(dist, beta, atol=1e-8)


def test_cluster_bootstrap_ols_matches_a_brute_force_resample():
    """The per-cluster X'X / X'y sum must equal refitting the stacked design."""
    n_clusters = 4
    per = 6
    rng = np.random.default_rng(5)
    c = rng.normal(size=n_clusters * per)
    a = np.tile(np.repeat([0.0, 1.0], per // 2), n_clusters)
    X = np.column_stack([np.ones_like(c), a, c, a * c])
    y = rng.normal(size=c.size)
    ci = np.repeat(np.arange(n_clusters), per)
    idx = resample_matrix(n_clusters, 25, seed=2)
    _, dist = cluster_bootstrap_ols(X, y, ci, idx, n_clusters)
    for i in range(idx.shape[0]):
        rows = np.concatenate([np.where(ci == c_)[0] for c_ in idx[i]])
        assert dist[i] == pytest.approx(ols_fit(X[rows], y[rows]), abs=1e-8)


def test_stable_words_is_bounded_by_tau():
    row = {"end_coverage_robust": 0.5, "n_ref_words": 400}
    assert stable_words(row) == pytest.approx(200.0)
    full = {"end_coverage_robust": 1.0, "n_ref_words": 400}
    assert stable_words(full) == pytest.approx(400.0)
    none = {"end_coverage_robust": 0.0, "n_ref_words": 400}
    assert stable_words(none) == pytest.approx(0.0)


def test_build_clusters_groups_by_root():
    text_to_root = {"r1__B0": "r1", "r1__B1": "r1", "r2__B0": "r2"}
    keys = [("r1__B0", "v", 0), ("r2__B0", "v", 0), ("r1__B1", "v", 0)]
    ci, roots = build_clusters(keys, text_to_root)
    assert roots == ["r1", "r2"]
    assert ci.tolist() == [0, 1, 0]


