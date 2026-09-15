"""Tests for src/eval/distillmos_windows.py (A12-mos).

Covers the pure window math (spans, centres, tail keep/drop contract), the
per-item aggregation, and — when the scored artefacts exist — batch integrity
of results/v31_mos (row counts, window counts vs durations, no NaN).

Importable from the torch-less .venv-eval: the module lazy-imports torch.
"""
import json
import math
from pathlib import Path

import pytest

from src.eval.distillmos_windows import (
    DISTILLMOS_SAMPLE_RATE,
    HOP_SAMPLES,
    MIN_TAIL_SAMPLES,
    WIN_SAMPLES,
    aggregate_windows,
    t_center,
    window_spans,
)

SR = DISTILLMOS_SAMPLE_RATE
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v31_mos"


# ---------------------------------------------------------------- window math

def test_constants():
    assert SR == 16_000
    assert WIN_SAMPLES == 80_000   # 5.0 s
    assert HOP_SAMPLES == 40_000   # 2.5 s
    assert MIN_TAIL_SAMPLES == 40_000  # 2.5 s


def test_too_short_no_windows():
    assert window_spans(0) == []
    assert window_spans(MIN_TAIL_SAMPLES - 1) == []


def test_partial_only_clip():
    # 2.5 s <= n < 5 s -> single partial window covering the whole clip
    for n in (MIN_TAIL_SAMPLES, WIN_SAMPLES - 1):
        assert window_spans(n) == [(0, n)]


def test_exact_five_seconds():
    # n = 5.0 s: full window [0,5) plus the truncated grid window at 2.5 s
    # covering the final 2.5 s (kept: length == MIN_TAIL). Documented contract:
    # with WIN = 2*HOP the truncated final window of any clip >= 5 s always
    # has length in [2.5, 5.0) s and is always kept.
    assert window_spans(WIN_SAMPLES) == [
        (0, WIN_SAMPLES),
        (HOP_SAMPLES, WIN_SAMPLES),
    ]


def test_grid_start_below_min_tail_dropped():
    # 7.4 s: windows at 0 (full 5 s) and 2.5 s (truncated 4.9 s, kept);
    # the next grid start (5.0 s) would leave 2.4 s < 2.5 s -> dropped.
    n = WIN_SAMPLES + MIN_TAIL_SAMPLES - 1600
    assert window_spans(n) == [(0, WIN_SAMPLES), (HOP_SAMPLES, n)]
    # last-start check: nothing starts at or after 5.0 s
    assert all(a < WIN_SAMPLES for a, _ in window_spans(n))


def test_tail_kept_at_min():
    # 5 s + exactly 2.5 s tail after the last full window start
    n = WIN_SAMPLES + HOP_SAMPLES  # 7.5 s: full [0,5), tail [2.5*sr? ...]
    # full windows: starts 0 (0+5<=7.5) and 2.5 (2.5+5=7.5<=7.5) -> 2 full,
    # tail start=5.0s len 2.5s -> kept
    assert window_spans(n) == [
        (0, WIN_SAMPLES),
        (HOP_SAMPLES, HOP_SAMPLES + WIN_SAMPLES),
        (2 * HOP_SAMPLES, n),
    ]


def test_seventeen_second_clip():
    # 17.72 s @16k (the E1 smoke clip): full starts 0..12.5 s, tail 2.72 s kept
    n = int(round(17.72 * SR))
    spans = window_spans(n)
    full = [s for s in spans if s[1] - s[0] == WIN_SAMPLES]
    assert len(full) == 6  # starts 0, 2.5, 5, 7.5, 10, 12.5
    assert spans[-1] == (6 * HOP_SAMPLES, n)
    assert (spans[-1][1] - spans[-1][0]) / SR == pytest.approx(2.72, abs=1e-4)


def test_spans_sorted_disjoint_starts_and_coverage():
    n = int(600.4 * SR)
    spans = window_spans(n)
    starts = [a for a, _ in spans]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    assert all(a % HOP_SAMPLES == 0 for a in starts)
    assert spans[-1][1] <= n
    # every sample from 0 to the last covered point is inside some window
    assert spans[0][0] == 0
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 <= b1  # overlap or contiguity, no holes


def test_t_center():
    assert t_center((0, WIN_SAMPLES)) == pytest.approx(2.5)
    assert t_center((HOP_SAMPLES, HOP_SAMPLES + WIN_SAMPLES)) == pytest.approx(5.0)
    assert t_center((0, MIN_TAIL_SAMPLES)) == pytest.approx(1.25)


# ---------------------------------------------------------------- aggregation

def test_aggregate_empty():
    agg = aggregate_windows([], [])
    assert agg["n_windows"] == 0
    assert agg["mean_mos"] is None and agg["min_mos"] is None
    assert agg["mos_first_30s"] is None and agg["mos_after_300s"] is None


def test_aggregate_basic():
    ts = [2.5, 29.9, 30.0, 100.0, 300.0, 305.5]
    vs = [4.0, 3.0, 2.0, 5.0, 1.0, 2.0]
    agg = aggregate_windows(ts, vs)
    assert agg["n_windows"] == 6
    assert agg["mean_mos"] == pytest.approx(sum(vs) / 6)
    assert agg["min_mos"] == 1.0
    assert agg["mos_first_30s"] == pytest.approx((4.0 + 3.0 + 2.0) / 3)
    assert agg["mos_after_300s"] == pytest.approx((1.0 + 2.0) / 2)


def test_aggregate_short_clip_has_no_after_300():
    agg = aggregate_windows([2.5, 5.0], [4.0, 4.2])
    assert agg["mos_after_300s"] is None
    assert agg["mos_first_30s"] == pytest.approx(4.1)


def test_aggregate_rejects_nan():
    with pytest.raises(ValueError):
        aggregate_windows([2.5], [float("nan")])
    with pytest.raises(ValueError):
        aggregate_windows([2.5, 5.0], [4.0])  # length mismatch


# ------------------------------------------------------- batch integrity (v31_mos)

def _scoring_finished() -> bool:
    """True when per_item.jsonl covers every task (scoring run complete)."""
    per_item = RESULTS / "per_item.jsonl"
    tasks = RESULTS / "tasks.jsonl"
    if not per_item.exists() or not tasks.exists():
        return False
    n_items = sum(1 for l in per_item.open() if l.strip())
    n_tasks = sum(1 for l in tasks.open() if l.strip())
    return n_items >= n_tasks


needs_results = pytest.mark.skipif(
    not _scoring_finished(),
    reason="results/v31_mos not scored yet (or scoring still in progress)",
)


@needs_results
def test_per_item_integrity():
    items = [json.loads(l) for l in (RESULTS / "per_item.jsonl").open()]
    # A25: 9 systems (CosyVoice3 x3, Qwen3-TTS x3, VoxCPM2 x3) x (60+53)
    # + human (20+53)
    assert len(items) == 1090
    from collections import Counter
    c = Counter((r["set"], r["system"]) for r in items)
    assert c[("pilot", "human")] == 20 and c[("robust", "human")] == 53
    for sysname in ("E1", "E3", "E7", "QE1", "QE2P", "QE3P",
                    "VCE1", "VCE2P", "VCE3P"):
        assert c[("pilot", sysname)] == 60
        assert c[("robust", sysname)] == 53
    for r in items:
        assert r["n_windows"] > 0, r["text_id"]
        for k in ("mean_mos", "min_mos", "mos_first_30s"):
            assert r[k] is not None and not math.isnan(r[k]), (r["text_id"], k)
            assert 1.0 <= r[k] <= 5.0
        # window count must match the documented span rule for the duration
        n = int(round(r["scored_duration_sec"] * SR))
        # scored_duration is a float; allow +-1 sample of rounding
        cands = {len(window_spans(m)) for m in (n - 1, n, n + 1)}
        assert r["n_windows"] in cands, (r["text_id"], r["n_windows"], cands)


@needs_results
def test_per_window_integrity():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    df = pd.read_parquet(RESULTS / "per_window.parquet")
    assert list(df.columns) == ["set", "system", "text_id", "voice_id",
                                "bucket", "t_center", "mos"]
    assert not df["mos"].isna().any()
    assert not df["t_center"].isna().any()
    assert df["mos"].between(1.0, 5.0).all()
    items = [json.loads(l) for l in (RESULTS / "per_item.jsonl").open()]
    assert len(df) == sum(r["n_windows"] for r in items)
    # per-item mean recomputed from windows matches the stored aggregate
    g = df.groupby(["set", "system", "text_id", "voice_id"])["mos"].mean()
    for r in items[:50]:
        key = (r["set"], r["system"], r["text_id"], r["voice_id"])
        assert g[key] == pytest.approx(r["mean_mos"], abs=1e-9)
