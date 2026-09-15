"""Tests for src/eval/spk_sim_windows.py (A18-simtime).

Covers the window-grid identity with A12 (same spans/centres), the
min-voiced scoring rule, the per-item aggregation, and — when the scored
artefacts exist — batch integrity of results/v31_sim (row counts, window
counts vs per-item sums, sim range, no NaN).

Importable from `.venv-eval` (numpy-only paths; the ONNX encoder is never
loaded here).
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.eval.distillmos_windows import (
    HOP_SAMPLES,
    MIN_TAIL_SAMPLES,
    WIN_SAMPLES,
    window_spans,
)
from src.eval.spk_sim_windows import (
    MIN_VOICED_SAMPLES,
    MIN_VOICED_SEC,
    SR,
    aggregate_windows,
    voiced_selection,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v31_sim"


# ------------------------------------------------------- grid identity (A12)

def test_grid_is_the_a12_grid():
    assert SR == 16_000
    assert WIN_SAMPLES == 80_000 and HOP_SAMPLES == 40_000
    assert MIN_TAIL_SAMPLES == 40_000
    # spans come straight from distillmos_windows -> identical by construction;
    # assert the contract anyway on a representative length
    n = int(round(17.72 * SR))
    spans = window_spans(n)
    assert spans[0] == (0, WIN_SAMPLES) and spans[-1] == (6 * HOP_SAMPLES, n)


def test_min_voiced_constant():
    assert MIN_VOICED_SEC == 1.0
    assert MIN_VOICED_SAMPLES == 16_000


# ------------------------------------------------------------ voiced windows

def test_voiced_selection_all_voiced():
    n = WIN_SAMPLES + HOP_SAMPLES
    vmask = np.ones(n, dtype=bool)
    sel = voiced_selection(vmask, window_spans(n))
    assert len(sel) == 3
    assert all(ok for _, _, ok in sel)
    assert sel[0][1] == WIN_SAMPLES  # full window fully voiced


def test_voiced_selection_silent_window_skipped():
    n = WIN_SAMPLES + HOP_SAMPLES
    vmask = np.ones(n, dtype=bool)
    vmask[:WIN_SAMPLES] = False          # first window fully silent
    sel = voiced_selection(vmask, window_spans(n))
    assert sel[0][2] is False and sel[0][1] == 0
    assert sel[-1][2] is True            # tail window [2*HOP, n) fully voiced


def test_voiced_selection_threshold_boundary():
    n = WIN_SAMPLES
    spans = [(0, n)]
    vmask = np.zeros(n, dtype=bool)
    vmask[:MIN_VOICED_SAMPLES] = True    # exactly 1.0 s voiced -> scored
    assert voiced_selection(vmask, spans)[0][2] is True
    vmask[MIN_VOICED_SAMPLES - 1] = False  # one sample less -> skipped
    assert voiced_selection(vmask, spans)[0][2] is False


# ---------------------------------------------------------------- aggregation

def test_aggregate_empty():
    agg = aggregate_windows([], [])
    assert agg["n_windows_total"] == 0 and agg["n_windows"] == 0
    assert agg["mean_sim"] is None and agg["min_sim"] is None


def test_aggregate_with_skipped_windows():
    ts = [2.5, 5.0, 100.0, 300.0, 302.5]
    ss = [0.8, None, 0.7, 0.1, None]
    agg = aggregate_windows(ts, ss)
    assert agg["n_windows_total"] == 5
    assert agg["n_windows"] == 3
    assert agg["n_windows_skipped_low_voiced"] == 2
    assert agg["mean_sim"] == pytest.approx((0.8 + 0.7 + 0.1) / 3)
    assert agg["min_sim"] == pytest.approx(0.1)
    assert agg["sim_first_30s"] == pytest.approx(0.8)
    assert agg["sim_after_300s"] == pytest.approx(0.1)


def test_aggregate_all_skipped():
    agg = aggregate_windows([2.5, 5.0], [None, None])
    assert agg["n_windows"] == 0 and agg["n_windows_total"] == 2
    assert agg["mean_sim"] is None and agg["sim_after_300s"] is None


def test_aggregate_rejects_nan_and_mismatch():
    with pytest.raises(ValueError):
        aggregate_windows([2.5], [float("nan")])
    with pytest.raises(ValueError):
        aggregate_windows([2.5, 5.0], [0.5])


# ------------------------------------------------- batch integrity (v31_sim)

def _scoring_finished() -> bool:
    per_item = RESULTS / "per_item.jsonl"
    tasks = RESULTS / "tasks.jsonl"
    if not per_item.exists() or not tasks.exists():
        return False
    n_items = sum(1 for l in per_item.open() if l.strip())
    n_tasks = sum(1 for l in tasks.open() if l.strip())
    return n_items >= n_tasks


needs_results = pytest.mark.skipif(
    not _scoring_finished(),
    reason="results/v31_sim not scored yet (or scoring still in progress)",
)


@needs_results
def test_per_item_integrity():
    items = [json.loads(l) for l in (RESULTS / "per_item.jsonl").open()]
    # A25: 9 systems x (60+53) + human (20 pilot roots x 2 pilot voices
    # + 53 robust)
    assert len(items) == 1110
    from collections import Counter
    c = Counter((r["set"], r["system"]) for r in items)
    assert c[("pilot", "human")] == 40 and c[("robust", "human")] == 53
    for sysname in ("E1", "E3", "E7", "QE1", "QE2P", "QE3P",
                    "VCE1", "VCE2P", "VCE3P"):
        assert c[("pilot", sysname)] == 60
        assert c[("robust", sysname)] == 53
    for r in items:
        assert r["n_windows_total"] > 0, r["text_id"]
        assert (r["n_windows"] + r["n_windows_skipped_low_voiced"]
                == r["n_windows_total"])
        if r["n_windows"] == 0:
            continue
        for k in ("mean_sim", "min_sim"):
            assert r[k] is not None and not math.isnan(r[k]), (r["text_id"], k)
            assert -1.0 <= r[k] <= 1.0
        n = int(round(r["scored_duration_sec"] * SR))
        cands = {len(window_spans(m)) for m in (n - 1, n, n + 1)}
        assert r["n_windows_total"] in cands, (r["text_id"], r["n_windows_total"])


@needs_results
def test_per_window_integrity():
    pq = pytest.importorskip("pyarrow.parquet")
    tbl = pq.read_table(RESULTS / "per_window.parquet")
    assert tbl.column_names == ["set", "system", "text_id", "voice_id",
                                "bucket", "t_center", "sim", "voiced_sec"]
    sim = np.asarray(tbl.column("sim"))
    assert not np.isnan(sim).any()
    assert ((sim >= -1.0) & (sim <= 1.0)).all()
    vs = np.asarray(tbl.column("voiced_sec"))
    assert (vs >= MIN_VOICED_SEC - 1e-9).all()   # only scored windows stored
    items = [json.loads(l) for l in (RESULTS / "per_item.jsonl").open()]
    assert len(tbl) == sum(r["n_windows"] for r in items)
    # per-item mean recomputed from windows matches the stored aggregate
    rows = tbl.to_pylist()
    acc = {}
    for r in rows:
        key = (r["set"], r["system"], r["text_id"], r["voice_id"])
        acc.setdefault(key, []).append(r["sim"])
    for r in items[:50]:
        if r["n_windows"] == 0:
            continue
        key = (r["set"], r["system"], r["text_id"], r["voice_id"])
        assert np.mean(acc[key]) == pytest.approx(r["mean_sim"], abs=1e-9)
