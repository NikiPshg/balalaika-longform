"""Analytic bootstrap tests; scripts/reproduce.py checks the released observations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stats.bootstrap import (  # noqa: E402
    METRICS,
    load_jsonl,
    paired_delta,
    resample_matrix,
)
from src.stats.bootstrap_pooled import (  # noqa: E402
    ARM_SPECS_POOLED,
    COMPARISONS_POOLED,
    NAME,
    STRATA,
    STRATUM_PREFIX,
    run,
    stratified_resample_matrix,
    stratum_seed,
)

RESAMPLES = 400  # plenty for structural tests; the real run uses 10 000


# ----------------------------------------------------------- stratification


def test_stratified_matrix_shape_and_stratum_ranges():
    """Pilot columns draw only pilot cluster indices, robust columns only robust."""
    idx = stratified_resample_matrix((6, 13), RESAMPLES, 0)
    assert idx.shape == (RESAMPLES, 19)
    pilot_block, robust_block = idx[:, :6], idx[:, 6:]
    # a pilot cluster (index < 6) never appears in a robust draw and vice versa
    assert pilot_block.min() >= 0 and pilot_block.max() < 6
    assert robust_block.min() >= 6 and robust_block.max() < 19
    # both strata actually vary (with replacement, not identity)
    assert len(np.unique(pilot_block)) == 6
    assert len(np.unique(robust_block)) == 13


def test_stratified_matrix_is_deterministic_and_blocks_use_disjoint_seeds():
    a = stratified_resample_matrix((6, 13), RESAMPLES, 0)
    b = stratified_resample_matrix((6, 13), RESAMPLES, 0)
    assert np.array_equal(a, b)
    # block s is exactly bootstrap.resample_matrix under the derived stratum seed
    assert np.array_equal(a[:, :6], resample_matrix(6, RESAMPLES, stratum_seed(0, 0)))
    assert np.array_equal(a[:, 6:], resample_matrix(13, RESAMPLES, stratum_seed(0, 1)) + 6)
    assert stratum_seed(0, 0) != stratum_seed(0, 1)


def test_stratum_seeds_are_disjoint_for_any_base_seed():
    seeds = set()
    for base in range(50):
        for pos in range(len(STRATA)):
            seeds.add(stratum_seed(base, pos))
    assert len(seeds) == 50 * len(STRATA)


# -------------------------------------------------------------- synthetic


def _synthetic(delta_pilot: float, delta_robust: float, noise: float, seed: int = 7):
    """Two-strata paired data with known per-stratum effects.

    Pilot: 6 clusters × 10 items; robust: 13 clusters × 4 items (52 items) — close
    to the real 60/53 geometry.  Returns (a, b, cluster_index, idx, n_clusters).
    """
    rng = np.random.default_rng(seed)
    sizes = {"pilot": (6, 10), "robust": (13, 4)}
    deltas = {"pilot": delta_pilot, "robust": delta_robust}
    a_parts, b_parts, ci_parts = [], [], []
    offset = 0
    for s in STRATA:
        n_cl, per = sizes[s]
        cluster_effect = rng.normal(0.0, noise, size=n_cl)
        for c in range(n_cl):
            base = rng.normal(0.0, 1.0, size=per)
            item_noise = rng.normal(0.0, noise, size=per)
            b_parts.append(base)
            a_parts.append(base + deltas[s] + cluster_effect[c] + item_noise)
            ci_parts.append(np.full(per, offset + c, dtype=np.int64))
        offset += n_cl
    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    cluster_index = np.concatenate(ci_parts)
    idx = stratified_resample_matrix((6, 13), 2000, 0)
    return a, b, cluster_index, idx, 19


def test_constant_paired_difference_gives_a_zero_width_ci_at_the_truth():
    a, b, cluster_index, idx, n_cl = _synthetic(0.0, 0.0, noise=0.5)
    d = 2.5
    res = paired_delta(b + d, b, cluster_index, idx, n_cl)
    assert res["delta"] == pytest.approx(d)
    assert res["ci_lo"] == pytest.approx(d)
    assert res["ci_hi"] == pytest.approx(d)
    assert res["excludes_zero"]


def test_known_delta_falls_inside_the_pooled_ci():
    true = 1.0
    a, b, cluster_index, idx, n_cl = _synthetic(true, true, noise=0.3, seed=0)
    res = paired_delta(a, b, cluster_index, idx, n_cl)
    # point estimate is exactly the sample mean of the paired differences
    assert res["delta"] == pytest.approx(float((a - b).mean()))
    assert res["ci_lo"] <= true <= res["ci_hi"]
    assert res["excludes_zero"]  # a 1.0 effect at 0.3 noise is unmissable


def test_ci_coverage_of_the_true_delta_over_many_simulations():
    """Deterministic coverage check: the 95 % CI catches the true effect in the
    large majority of 40 fixed-seed simulations (nominal 95 %; a percentile
    cluster bootstrap at 6+13 clusters undercovers slightly, so >= 80 % is the
    bar — the honest sanity, instead of cherry-picking one lucky seed)."""
    true = 1.0
    hits = 0
    for data_seed in range(40):
        a, b, cluster_index, idx, n_cl = _synthetic(true, true, noise=0.3,
                                                    seed=data_seed)
        res = paired_delta(a, b, cluster_index, idx, n_cl)
        hits += int(res["ci_lo"] <= true <= res["ci_hi"])
    assert hits >= 32, f"coverage {hits}/40 below 80 %"


def test_pooled_point_is_the_item_weighted_stratum_average_on_synthetic():
    a, b, cluster_index, idx, n_cl = _synthetic(2.0, 0.5, noise=0.2)
    res = paired_delta(a, b, cluster_index, idx, n_cl)
    d = a - b
    pilot_mask = cluster_index < 6
    n_p, n_r = int(pilot_mask.sum()), int((~pilot_mask).sum())
    expected = (n_p * d[pilot_mask].mean() + n_r * d[~pilot_mask].mean()) / (n_p + n_r)
    assert res["delta"] == pytest.approx(float(expected))


