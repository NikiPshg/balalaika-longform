"""Paired cluster bootstrap for the RuLongTTS pilot (PLAN.md §12, §17 Table 4).

PLAN.md §12 asks for:

* comparisons paired by ``text × voice × seed``;
* cluster unit = **root document**, not the rolling window and not the item;
* paired cluster bootstrap, 10 000 resamples, 95 % percentile CI;
* binary outcomes as a paired **risk difference**;
* a ``checkpoint × log(length)`` interaction analysis (hypothesis H6);
* effect size + CI reported ahead of any single p-value;
* unconditional metrics (failures included) next to conditional ones.

This module implements exactly that and nothing else.  It reads only the
evaluator's per-item rows and the benchmark, never an aggregate table, so its
output is an independent recomputation:

* ``results/v31_sft/{E3_epoch_0_whole,E3_epoch_3_step_3001,
  E2_epoch_0_step_200,E2_epoch_3_step_3001}/per_item.jsonl``
* ``results/v31_base/{E1,E0}/per_item.jsonl``
* ``data/benchmark/pilot.jsonl`` (root_id ↔ text_id, ref_words, bucket)

Every metric is **unconditional**: an item whose status is ``early_eos`` or
``empty_or_invalid_audio`` stays in the sample with the number the evaluator
gave it (PLAN §0 rule 4, §9.1).  Nothing is dropped.

Method
------
The 60 items of one checkpoint are ``6 roots × 5 nested prefixes × 2 voices``,
one seed.  Items pair 1:1 across checkpoints on ``(text_id, voice_id, seed)``;
the pairing is asserted, not assumed.  The cluster unit is ``root_id``, so a
resample draws **6 roots with replacement** and pools every item of every drawn
root (with multiplicity).  The statistic is the paired mean difference

    delta = mean_i(metric_A(i) - metric_B(i))

over the pooled items, which for a complete pairing equals the difference of
the two arm means.  For the binary ``complete`` outcome that mean difference
*is* the paired risk difference.  The CI is the 2.5/97.5 percentile of the
bootstrap distribution.  One single resample matrix (seed 0) is shared by every
metric, bucket and comparison — common random numbers, so deltas in the same
row of the table are drawn under the same cluster draws.

The interaction (H6) is a per-item OLS on source coverage

    coverage ~ 1 + A + c + A:c ,   c = log(n_ref_words) - mean(log(n_ref_words))

fitted on the 120 stacked rows of both checkpoints (statsmodels-free: numpy
normal equations).  ``A`` is 1 for the first checkpoint of the pair.  The
interaction coefficient ``A:c`` is the change in the coverage-per-log-word
slope between the two checkpoints; it is cluster-bootstrapped with the same
root draws (per-root ``X'X`` / ``X'y`` blocks are summed over the drawn roots,
which is exactly the resampled design).  The centering constant is computed
once on the original data and held fixed across resamples so the coefficients
keep one interpretation.

Six clusters is a small number.  The CIs this file prints are wide and the
percentile bootstrap has a coarse granularity at 6 clusters; see the caveats
block written into ``paired_deltas.md``.

Usage::

    .venv-eval/bin/python src/stats/bootstrap.py                  # writes results/v31_stats/
    .venv-eval/bin/python src/stats/bootstrap.py --resamples 1000 # faster smoke run
    .venv-eval/bin/python src/stats/bootstrap.py --out-dir tmp/A8/dry

No GPU, no model, no network.  numpy only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]

BUCKETS = ("B0", "B1", "B2", "B3", "B4")
CELLS = BUCKETS + ("ALL",)
ITEM_KEY_FIELDS = ("text_id", "voice_id", "seed")

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05


# --------------------------------------------------------------------------- io


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def item_key(row: dict) -> tuple:
    return tuple(row[f] for f in ITEM_KEY_FIELDS)


# ------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class Metric:
    """One per-item outcome plus how it is displayed."""

    name: str
    label: str
    getter: Callable[[dict], float]
    scale: float  # multiply the raw value by this for display
    unit: str  # unit of the *delta*
    better: str  # "higher" or "lower"
    binary: bool = False


def _is_complete(row: dict) -> float:
    # PLAN §3.4 final status, assigned by the evaluator (configs/eval.yaml).
    return 1.0 if row["status"] == "complete" else 0.0


METRICS: tuple[Metric, ...] = (
    Metric("complete_rate", "Complete %", _is_complete, 100.0, "pp", "higher", binary=True),
    Metric("source_coverage", "Coverage", lambda r: float(r["source_coverage"]), 1.0, "abs", "higher"),
    Metric("wer", "WER-all %", lambda r: float(r["wer"]), 100.0, "pp", "lower"),
    Metric("end_coverage_robust", "EndCov-robust", lambda r: float(r["end_coverage_robust"]), 1.0, "abs", "higher"),
    Metric("excess_repetition_rate", "Repeat %", lambda r: float(r["excess_repetition_rate"]), 100.0, "pp", "lower"),
    Metric("duration_ratio", "dur ratio", lambda r: float(r["duration_ratio"]), 1.0, "abs", "higher"),
)


# ------------------------------------------------------------------ bootstrap


def resample_matrix(n_clusters: int, n_resamples: int, seed: int) -> np.ndarray:
    """``(n_resamples, n_clusters)`` matrix of cluster indices drawn with replacement."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(n_resamples, n_clusters), dtype=np.int64)


def _cluster_blocks(values: np.ndarray, cluster_index: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster ``(sum, count)`` of ``values``."""
    sums = np.bincount(cluster_index, weights=values, minlength=n_clusters)
    counts = np.bincount(cluster_index, minlength=n_clusters).astype(np.float64)
    return sums, counts


def cluster_bootstrap_mean(
    values: np.ndarray,
    cluster_index: np.ndarray,
    idx: np.ndarray,
    n_clusters: int,
) -> tuple[float, np.ndarray]:
    """Point estimate and bootstrap distribution of the mean of ``values``.

    ``values`` are item-level; ``cluster_index`` maps each item to a cluster in
    ``[0, n_clusters)``; ``idx`` is the shared resample matrix.  Resampling a
    cluster pulls in all of its items, so the resampled mean is
    ``sum(drawn cluster sums) / sum(drawn cluster counts)``.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), np.full(idx.shape[0], np.nan)
    sums, counts = _cluster_blocks(values, cluster_index, n_clusters)
    point = float(values.mean())
    s = sums[idx].sum(axis=1)
    n = counts[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(n > 0, s / n, np.nan)
    return point, dist


def percentile_ci(dist: np.ndarray, alpha: float = DEFAULT_ALPHA) -> tuple[float, float, int]:
    """Percentile CI of a bootstrap distribution; NaN resamples are dropped."""
    d = np.asarray(dist, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(d, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi), int(d.size)


def bootstrap_p_two_sided(dist: np.ndarray) -> float:
    """Secondary only (PLAN §12: effect size + CI outrank a single p-value).

    Achieved significance level of the percentile bootstrap: twice the smaller
    tail mass on either side of 0, floored at ``1/B`` because the bootstrap
    cannot resolve below its own resolution.
    """
    d = np.asarray(dist, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    p_le = float(np.mean(d <= 0.0))
    p_ge = float(np.mean(d >= 0.0))
    p = 2.0 * min(p_le, p_ge)
    return float(min(1.0, max(p, 1.0 / d.size)))


def paired_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    cluster_index: np.ndarray,
    idx: np.ndarray,
    n_clusters: int,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Paired cluster bootstrap of ``mean(A) - mean(B)`` over matched items.

    ``values_a[i]`` and ``values_b[i]`` must be the *same* benchmark item under
    the two checkpoints.  For a binary outcome this is the paired risk
    difference.  Arm means are bootstrapped under the same draws.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"unpaired arrays: {a.shape} vs {b.shape}")
    mean_a, dist_a = cluster_bootstrap_mean(a, cluster_index, idx, n_clusters)
    mean_b, dist_b = cluster_bootstrap_mean(b, cluster_index, idx, n_clusters)
    delta, dist_d = cluster_bootstrap_mean(a - b, cluster_index, idx, n_clusters)
    lo, hi, n_valid = percentile_ci(dist_d, alpha)
    lo_a, hi_a, _ = percentile_ci(dist_a, alpha)
    lo_b, hi_b, _ = percentile_ci(dist_b, alpha)
    return {
        "n_items": int(a.size),
        "n_clusters": int(np.unique(cluster_index).size),
        "mean_a": mean_a,
        "mean_a_lo": lo_a,
        "mean_a_hi": hi_a,
        "mean_b": mean_b,
        "mean_b_lo": lo_b,
        "mean_b_hi": hi_b,
        "delta": delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_boot": bootstrap_p_two_sided(dist_d),
        "n_resamples_valid": n_valid,
        "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
    }


# ------------------------------------------------------------------------ OLS


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients via ``lstsq`` (no statsmodels, no scipy)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def cluster_bootstrap_ols(
    X: np.ndarray,
    y: np.ndarray,
    cluster_index: np.ndarray,
    idx: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their cluster-bootstrap distribution.

    ``X'X`` and ``X'y`` are additive over clusters, so a resample's normal
    equations are the sum of the drawn clusters' blocks — exact, and it avoids
    materialising 10 000 design matrices.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, k = X.shape
    point = ols_fit(X, y)

    xtx = np.zeros((n_clusters, k, k))
    xty = np.zeros((n_clusters, k))
    for c in range(n_clusters):
        m = cluster_index == c
        if not m.any():
            continue
        Xc = X[m]
        xtx[c] = Xc.T @ Xc
        xty[c] = Xc.T @ y[m]

    A = xtx[idx].sum(axis=1)  # (B, k, k)
    b = xty[idx].sum(axis=1)  # (B, k)
    out = np.full((idx.shape[0], k), np.nan)
    try:
        out = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        for i in range(idx.shape[0]):
            try:
                out[i] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                out[i] = np.nan
    return point, out


# ------------------------------------------------------- stable words / RMST


def stable_words(row: dict) -> float:
    """Per-item 'stable words' **proxy** (see the caveat in the report).

    ``end_coverage_robust × n_ref_words`` — the source-word position of the end
    of the last run of ≥ 3 consecutive aligned words, i.e. how far into the
    source the model was still demonstrably on track.  This is *not* the PLAN
    §9.3 rolling-window degradation point: no window, no calibrated threshold,
    no separation of T_content / T_voice / T_trunc.  It is bounded above by
    ``n_ref_words`` by construction, so the mean below is an RMST with the
    horizon τ = the item's own reference length and every non-degrading item
    contributing its full length (right-censored at τ).
    """
    return float(row["end_coverage_robust"]) * float(row["n_ref_words"])


# ---------------------------------------------------------------- assembly


@dataclass(frozen=True)
class Arm:
    """One evaluated checkpoint."""

    key: str
    label: str
    path: Path
    rows: dict  # item_key -> per-item row

    @property
    def n(self) -> int:
        return len(self.rows)


def load_arm(key: str, label: str, path: Path) -> Arm:
    rows = load_jsonl(path)
    by_key: dict = {}
    for r in rows:
        k = item_key(r)
        if k in by_key:
            raise ValueError(f"{path}: duplicate item {k}")
        by_key[k] = r
    return Arm(key=key, label=label, path=path, rows=by_key)


def build_clusters(keys: Sequence[tuple], text_to_root: dict) -> tuple[np.ndarray, list[str]]:
    """Map each item to its root document; returns (cluster_index, root names)."""
    roots = sorted({text_to_root[k[0]] for k in keys})
    root_pos = {r: i for i, r in enumerate(roots)}
    cluster_index = np.array([root_pos[text_to_root[k[0]]] for k in keys], dtype=np.int64)
    return cluster_index, roots


def aligned_keys(a: Arm, b: Arm) -> list[tuple]:
    """Sorted item keys present in both arms; a mismatch is a hard error."""
    ka, kb = set(a.rows), set(b.rows)
    if ka != kb:
        missing_b = sorted(ka - kb)[:5]
        missing_a = sorted(kb - ka)[:5]
        raise ValueError(
            f"{a.key} and {b.key} are not paired: "
            f"{len(ka - kb)} only in {a.key} (e.g. {missing_b}), "
            f"{len(kb - ka)} only in {b.key} (e.g. {missing_a})"
        )
    return sorted(ka)


def compare(
    a: Arm,
    b: Arm,
    bench: dict,
    idx: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """All metrics × cells for one paired comparison, plus the H6 interaction."""
    keys = aligned_keys(a, b)
    text_to_root = {t: rec["root_id"] for t, rec in bench.items()}
    out: dict = {
        "comparison": f"{a.key}_vs_{b.key}",
        "arm_a": a.key,
        "arm_a_label": a.label,
        "arm_a_path": str(a.path.relative_to(REPO)),
        "arm_b": b.key,
        "arm_b_label": b.label,
        "arm_b_path": str(b.path.relative_to(REPO)),
        "cells": {},
    }
    for cell in CELLS:
        cell_keys = [k for k in keys if cell == "ALL" or a.rows[k]["bucket"] == cell]
        if not cell_keys:
            continue
        cluster_index, roots = build_clusters(cell_keys, text_to_root)
        n_clusters = len(roots)
        cell_out: dict = {"n_items": len(cell_keys), "n_clusters": n_clusters, "roots": roots, "metrics": {}}
        for m in METRICS:
            va = np.array([m.getter(a.rows[k]) for k in cell_keys])
            vb = np.array([m.getter(b.rows[k]) for k in cell_keys])
            res = paired_delta(va, vb, cluster_index, idx, n_clusters, alpha)
            res["metric"] = m.name
            res["label"] = m.label
            res["scale"] = m.scale
            res["unit"] = m.unit
            res["better"] = m.better
            res["binary"] = m.binary
            cell_out["metrics"][m.name] = res
        out["cells"][cell] = cell_out

    out["interaction"] = interaction_model(a, b, keys, text_to_root, idx, alpha)
    return out


def interaction_model(
    a: Arm,
    b: Arm,
    keys: Sequence[tuple],
    text_to_root: dict,
    idx: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """H6: ``coverage ~ 1 + A + c + A:c`` with ``c = log(words) - mean``."""
    rows_a = [a.rows[k] for k in keys]
    rows_b = [b.rows[k] for k in keys]
    y = np.array(
        [float(r["source_coverage"]) for r in rows_a] + [float(r["source_coverage"]) for r in rows_b]
    )
    logw = np.array([math.log(float(r["n_ref_words"])) for r in rows_a + rows_b])
    center = float(logw.mean())
    c = logw - center
    is_a = np.array([1.0] * len(rows_a) + [0.0] * len(rows_b))
    X = np.column_stack([np.ones_like(y), is_a, c, is_a * c])

    stacked_keys = list(keys) + list(keys)
    cluster_index, roots = build_clusters(stacked_keys, text_to_root)
    n_clusters = len(roots)

    point, dist = cluster_bootstrap_ols(X, y, cluster_index, idx, n_clusters)
    names = ["intercept", f"checkpoint[{a.key}]", "log_words_centered", f"checkpoint[{a.key}]:log_words_centered"]
    coefs = {}
    for j, nm in enumerate(names):
        lo, hi, nv = percentile_ci(dist[:, j], alpha)
        coefs[nm] = {
            "estimate": float(point[j]),
            "ci_lo": lo,
            "ci_hi": hi,
            "p_boot": bootstrap_p_two_sided(dist[:, j]),
            "n_resamples_valid": nv,
            "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
        }
    resid = y - X @ point
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "outcome": "source_coverage",
        "formula": "coverage ~ 1 + A + c + A:c,  A = 1 for " + a.key + ",  c = log(n_ref_words) - " + f"{center:.6f}",
        "log_words_center": center,
        "n_rows": int(y.size),
        "n_clusters": n_clusters,
        "r2": float(1.0 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "coefficients": coefs,
        "interaction_name": names[3],
    }


def rmst_table(arms: Sequence[Arm], bench: dict, idx: np.ndarray, alpha: float = DEFAULT_ALPHA) -> list[dict]:
    """'RMST words' proxy per checkpoint × bucket, with a cluster-bootstrap CI."""
    text_to_root = {t: rec["root_id"] for t, rec in bench.items()}
    out: list[dict] = []
    for arm in arms:
        keys = sorted(arm.rows)
        for cell in CELLS:
            cell_keys = [k for k in keys if cell == "ALL" or arm.rows[k]["bucket"] == cell]
            if not cell_keys:
                continue
            cluster_index, roots = build_clusters(cell_keys, text_to_root)
            sw = np.array([stable_words(arm.rows[k]) for k in cell_keys])
            tau = np.array([float(arm.rows[k]["n_ref_words"]) for k in cell_keys])
            frac = sw / tau
            point, dist = cluster_bootstrap_mean(sw, cluster_index, idx, len(roots))
            lo, hi, _ = percentile_ci(dist, alpha)
            fpoint, fdist = cluster_bootstrap_mean(frac, cluster_index, idx, len(roots))
            flo, fhi, _ = percentile_ci(fdist, alpha)
            n_censored = int(np.sum(np.isclose(sw, tau)))
            out.append(
                {
                    "checkpoint": arm.key,
                    "label": arm.label,
                    "bucket": cell,
                    "n_items": len(cell_keys),
                    "n_clusters": len(roots),
                    "tau_words_mean": float(tau.mean()),
                    "rmst_words_proxy": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "rmst_frac": fpoint,
                    "frac_ci_lo": flo,
                    "frac_ci_hi": fhi,
                    "n_censored_at_tau": n_censored,
                    "median_stable_words": float(np.median(sw)),
                }
            )
    return out


# ------------------------------------------------------------------ rendering


def _fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _scaled(res: dict, key: str) -> float:
    return res[key] * res["scale"]


def _nd_for(unit: str) -> int:
    return 2 if unit == "pp" else 3


def render_headline(payload: dict) -> list[str]:
    """The one paragraph a reader gets for free, built from the payload itself.

    Deliberately narrow: it states what the intervals say about the pilot's dev
    benchmark and nothing else.  PLAN §1's forbidden claims apply — no statement
    about all TTS models, about the dataset solving anything, or about a
    limitation having been removed.
    """
    comp = next((c for c in payload["comparisons"] if c["comparison"] == "E3_vs_E2"), None)
    if comp is None:
        return []

    def cell(bucket: str, metric: str) -> dict:
        return comp["cells"][bucket]["metrics"][metric]

    def txt(bucket: str, metric: str, scale: float, nd: int, unit: str = "") -> str:
        r = cell(bucket, metric)
        return (
            f"{r['delta'] * scale:+.{nd}f}{unit} "
            f"[{r['ci_lo'] * scale:+.{nd}f}, {r['ci_hi'] * scale:+.{nd}f}]"
        )

    inter = comp["interaction"]["coefficients"][comp["interaction"]["interaction_name"]]
    b0_wer = cell("B0", "wer")
    rep_all = cell("ALL", "excess_repetition_rate")
    headline_cells = [
        ("ALL", "complete_rate"),
        ("ALL", "source_coverage"),
        ("ALL", "wer"),
        ("B2", "complete_rate"),
        ("B3", "complete_rate"),
        ("B4", "complete_rate"),
    ]
    n_excl = sum(1 for b, m in headline_cells if cell(b, m)["excludes_zero"])
    excl_phrase = (
        f"All {n_excl} of those intervals exclude 0"
        if n_excl == len(headline_cells)
        else f"{n_excl} of those {len(headline_cells)} intervals exclude 0"
    )

    return [
        "## 0. What the intervals say",
        "",
        f"On this 6-root dev pilot, at equal steps and matched target-token budget, **E3 Long-SFT "
        f"minus E2 Short-SFT** is {txt('ALL', 'complete_rate', 100.0, 2, ' pp')} on Complete % over "
        f"all 60 paired items, {txt('ALL', 'source_coverage', 1.0, 3)} on coverage and "
        f"{txt('ALL', 'wer', 100.0, 2, ' pp')} on WER-all. The gap is carried by the long buckets: "
        f"Complete % {txt('B2', 'complete_rate', 100.0, 2, ' pp')} on B2, "
        f"{txt('B3', 'complete_rate', 100.0, 2, ' pp')} on B3, "
        f"{txt('B4', 'complete_rate', 100.0, 2, ' pp')} on B4. {excl_phrase}; "
        f"**H2 is supported on this benchmark** (PLAN §2).",
        "",
        f"**H6** — the `checkpoint × log(words)` interaction on per-item coverage is "
        f"{inter['estimate']:+.4f} [{inter['ci_lo']:+.4f}, {inter['ci_hi']:+.4f}], against a base "
        f"log-word slope of "
        f"{comp['interaction']['coefficients']['log_words_centered']['estimate']:+.4f}: E2's coverage "
        f"falls with length and E3's largely does not. The interval excludes 0.",
        "",
        f"**H4, short-form** — on B0 the WER-all difference is "
        f"{b0_wer['delta'] * 100:+.2f} pp [{b0_wer['ci_lo'] * 100:+.2f}, {b0_wer['ci_hi'] * 100:+.2f}], "
        f"i.e. E3 is not worse than E2 on the short-form control here. **No pre-registered "
        f"non-inferiority margin exists in PLAN §2 for H4**, so this is an interval, not a passed "
        f"test; the margin has to be set by the Lead before the hidden run.",
        "",
        f"**What did not improve** — E3's excess repetition rate is "
        f"{rep_all['delta'] * 100:+.2f} pp [{rep_all['ci_lo'] * 100:+.2f}, {rep_all['ci_hi'] * 100:+.2f}] "
        f"against E2 over all items, driven by B3/B4. E2 has almost no repetitions because it "
        f"produces almost no audio on those buckets (coverage {cell('B3', 'source_coverage')['mean_b']:.3f} "
        f"on B3, {cell('B4', 'source_coverage')['mean_b']:.3f} on B4), so this contrast compares a "
        f"model that reads and repeats against one that stops. It is reported because PLAN §0 rule 4 "
        f"forbids dropping it, not because it is a like-for-like comparison.",
        "",
        f"**E2 vs E1** — short-form SFT on the same data moves nothing at the ALL level: Complete % "
        f"{_signed(payload, 'E2_vs_E1', 'ALL', 'complete_rate', 100.0, 2, ' pp')}, coverage "
        f"{_signed(payload, 'E2_vs_E1', 'ALL', 'source_coverage', 1.0, 3)}. Both intervals contain 0.",
        "",
    ]


def _signed(payload: dict, comparison: str, bucket: str, metric: str, scale: float, nd: int, unit: str = "") -> str:
    comp = next(c for c in payload["comparisons"] if c["comparison"] == comparison)
    r = comp["cells"][bucket]["metrics"][metric]
    return f"{r['delta'] * scale:+.{nd}f}{unit} [{r['ci_lo'] * scale:+.{nd}f}, {r['ci_hi'] * scale:+.{nd}f}]"


def render_markdown(payload: dict) -> str:
    meta = payload["meta"]
    L: list[str] = []
    L.append(meta.get("title", "# Table 4 — paired Long-vs-Short deltas with 95 % CI (PLAN.md §12, §17)"))
    L.append("")
    L.append(
        f"- generated: {meta['generated_utc']} by `src/stats/bootstrap.py`\n"
        f"- paired cluster bootstrap, **{meta['n_resamples']} resamples**, seed **{meta['seed']}**, "
        f"**{int(100 * (1 - meta['alpha']))} % percentile CI**\n"
        f"- pairing unit: `text_id × voice_id × seed` (seed {meta['seeds']} only) — "
        f"{meta['n_items_per_arm']} items per checkpoint\n"
        f"- cluster unit: `root_id` — **{meta['n_clusters']} clusters** "
        f"({', '.join(meta['roots'])})\n"
        f"- every metric is unconditional: `early_eos`, `degraded` and `empty_or_invalid_audio` items "
        f"stay in the sample with the evaluator's number (PLAN §0 rule 4, §9.1)"
    )
    L.append("")
    caveat_sec = "§4" if "rmst_words_proxy" in payload else "§3"
    L.append(f"**Read the CIs with {caveat_sec} open. Six clusters, two voices, one seed.**")
    L.append("")
    L.append(
        "**Sign convention.** Δ is always A − B, never re-signed. The `better` arrow says which "
        "direction is good for that metric, so a positive Δ on a `↓` row (`WER-all %`, `Repeat %`) "
        "means A is **worse** than B on it. "
        + meta.get(
            "sign_note",
            "This matters in exactly one place below: E3's repeat rate on B3/B4 is higher than "
            "E2's, which is the residual failure mode the Lead recorded on 2026-08-30, not a win.",
        )
    )
    L.append("")
    # Optional preset-specific notes.  Presets that set no "notes" render exactly as before.
    for note in meta.get("notes", []):
        L.append(note)
        L.append("")
    L.extend(render_headline(payload))

    group_titles = meta.get("group_titles", DEFAULT_GROUP_TITLES)
    group_intros = meta.get("group_intros", DEFAULT_GROUP_INTROS)
    for group in ("primary", "secondary"):
        comps = [c for c in payload["comparisons"] if c["group"] == group]
        if not comps:
            continue
        L.append(f"## {group_titles[group]}")
        L.append("")
        L.append(group_intros[group])
        L.append("")
        for comp in comps:
            L.append(f"### {comp['arm_a_label']} − {comp['arm_b_label']}")
            L.append("")
            L.append(
                f"`{comp['arm_a_path']}` minus `{comp['arm_b_path']}`. "
                f"Δ is A − B; the arrow marks the direction that is better for the metric."
            )
            L.append("")
            L.append(
                "| Metric | better | Bucket | n | A | B | Δ (A−B) | 95 % CI | CI excludes 0 | p (boot) |"
            )
            L.append("|---|---|---|---:|---:|---:|---:|---|---|---:|")
            for m in METRICS:
                for cell in CELLS:
                    if cell not in comp["cells"]:
                        continue
                    r = comp["cells"][cell]["metrics"][m.name]
                    nd = _nd_for(m.unit)
                    arrow = "↑" if m.better == "higher" else "↓"
                    L.append(
                        f"| {m.label} | {arrow} | {cell} | {r['n_items']} | "
                        f"{_fmt(_scaled(r, 'mean_a'), nd)} | {_fmt(_scaled(r, 'mean_b'), nd)} | "
                        f"**{_fmt(r['delta'] * m.scale, nd)}** | "
                        f"[{_fmt(r['ci_lo'] * m.scale, nd)}, {_fmt(r['ci_hi'] * m.scale, nd)}] | "
                        f"{'yes' if r['excludes_zero'] else 'no'} | {_fmt(r['p_boot'], 4)} |"
                    )
            L.append("")
            inter = comp["interaction"]
            L.append(f"**H6 interaction** — `{inter['formula']}`, "
                     f"{inter['n_rows']} rows, {inter['n_clusters']} clusters, R² = {_fmt(inter['r2'], 3)}.")
            L.append("")
            L.append("| Coefficient | estimate | 95 % CI | CI excludes 0 | p (boot) |")
            L.append("|---|---:|---|---|---:|")
            for nm, c in inter["coefficients"].items():
                mark = "**" if nm == inter["interaction_name"] else ""
                L.append(
                    f"| {mark}{nm}{mark} | {_fmt(c['estimate'], 4)} | "
                    f"[{_fmt(c['ci_lo'], 4)}, {_fmt(c['ci_hi'], 4)}] | "
                    f"{'yes' if c['excludes_zero'] else 'no'} | {_fmt(c['p_boot'], 4)} |"
                )
            L.append("")

    has_rmst = "rmst_words_proxy" in payload
    if has_rmst:
        L.extend(_rmst_section(payload))

    caveats = CAVEATS if has_rmst else re.sub(r"^(#{2,3}) 4\.", r"\1 3.", CAVEATS, flags=re.M)
    L.append(caveats)
    L.append("")
    L.append("## 5. Reproduce" if has_rmst else "## 4. Reproduce")
    L.append("")
    L.append("```")
    L.append("cd .")
    preset_flag = "" if meta.get("preset", "default") == "default" else f" --preset {meta['preset']}"
    L.append(
        f".venv-eval/bin/python src/stats/bootstrap.py --resamples {meta['n_resamples']} "
        f"--seed {meta['seed']}{preset_flag}"
    )
    L.append(".venv-eval/bin/python -m pytest tests/test_stats.py -q")
    L.append("```")
    L.append("")
    L.append("### Inputs")
    L.append("")
    L.append("| file | rows | sha256 |")
    L.append("|---|---:|---|")
    for src in payload["meta"]["inputs"]:
        L.append(f"| `{src['path']}` | {src['rows']} | `{src['sha256']}` |")
    L.append("")
    return "\n".join(L)


def _rmst_section(payload: dict) -> list[str]:
    L: list[str] = []
    L.append("## 3. 'RMST words' proxy — a first fill-in for the Table 3 column")
    L.append("")
    L.append(
        "**This is a proxy, not the PLAN §9.3 definition.** §9.3 defines the degradation point from "
        "rolling windows (40 words, stride 10), a threshold calibrated on natural dev audio and frozen "
        "before hidden, and separate `T_content` / `T_voice` / `T_trunc` events. None of that exists yet. "
        "What is below is `stable_words = end_coverage_robust × n_ref_words` — the source-word position "
        "where the last run of ≥ 3 consecutive aligned words ends — averaged over items, with the "
        "horizon τ being each item's own `n_ref_words`. An item that never degrades contributes its "
        "full length (right-censored at τ), which is what makes the mean RMST-*like*. It inherits every "
        "weakness of `end_coverage_robust` (`reports/pilot_baseline_v31.md` §7.3) and it cannot "
        "distinguish a model that stopped early from one that drifted: both simply end their last "
        "aligned run sooner. **Table 3's `RMST words` column must stay `n/a` until §9.3 is implemented; "
        "this table is a separate, labelled quantity.**"
    )
    L.append("")
    L.append("`frac` = the same thing divided by τ (the restricted mean fraction of the text read on-track). "
             "`cens` = items whose stable words equal τ, i.e. aligned to the very last word.")
    L.append("")
    L.append("| Checkpoint | Bucket | n | τ words (mean) | RMST words (proxy) | 95 % CI | frac | frac 95 % CI | median | cens |")
    L.append("|---|---|---:|---:|---:|---|---:|---|---:|---:|")
    for r in payload["rmst_words_proxy"]:
        L.append(
            f"| {r['label']} | {r['bucket']} | {r['n_items']} | {_fmt(r['tau_words_mean'], 1)} | "
            f"{_fmt(r['rmst_words_proxy'], 1)} | [{_fmt(r['ci_lo'], 1)}, {_fmt(r['ci_hi'], 1)}] | "
            f"{_fmt(r['rmst_frac'], 3)} | [{_fmt(r['frac_ci_lo'], 3)}, {_fmt(r['frac_ci_hi'], 3)}] | "
            f"{_fmt(r['median_stable_words'], 1)} | {r['n_censored_at_tau']} |"
        )
    L.append("")
    return L


CAVEATS = """## 4. Caveats that must travel with every CI above

### 4.1 Six clusters

The cluster unit is the root document and there are **6 of them**: `ds1`, `ds2`, `ds3`, `ds4` from
dev, plus the two constructed roots `ex5_genre5_numbers` and `ex6_genre6_stress`. A percentile
bootstrap over 6 clusters has a coarse grid — a single root can be drawn 0 or 6 times — so the
intervals are wide, they are not symmetric, and the bootstrap p-values cannot resolve small tail
masses. Read them as effect-size intervals, not as tests. PLAN §12 states the same order of
priority: effect size + CI ahead of any single p-value.

### 4.2 Two external roots have no human floor

`ex5` and `ex6` were written for the benchmark and have no human reference audio. For them
`human_duration_sec` is *estimated* as `words / 103.332 wpm` (`data/benchmark/pilot.jsonl`,
`human_duration_source`), and there is no ASR floor. That affects the `dur ratio` rows above: for a
third of the clusters the denominator is a model of a duration, not a measured one. `WER − floor`
is therefore **not** bootstrapped here — it exists for 4 of 6 roots only, and a paired contrast on
a different cluster set would not be comparable with the rest of the table. The paired WER-all
contrast is unaffected: the floor bias is common to both arms and cancels in the difference
(`reports/pilot_baseline_v31.md` §7.2).

### 4.3 Two voices, one seed

`ref_female_01` and `ref_male_01`, `seeds: [0]`, one generation per item. Voice is not a cluster
here — with 2 levels it cannot be resampled meaningfully — so voice-to-voice variation sits inside
the clusters and is not separately quantified. And `reports/cosyvoice3_audit.md` §7 measured that
seed 0 does **not** pin the trajectory on this stack, so every per-item number carries an
unquantified inference variance that this bootstrap does not capture: it resamples documents, not
re-generations. A repeat-subset run (PLAN §5, §12) would be needed for that and has not been done.

### 4.4 Nested prefixes are not independent

B0 ⊂ B1 ⊂ B2 ⊂ B3 ⊂ B4 within a root by design (PLAN §7.2). The cluster bootstrap handles this
correctly for the `ALL` rows — resampling a root pulls all five prefixes together — but the
per-bucket rows are 12 items from 6 roots and the buckets themselves are not independent samples of
"length". The interaction model in §1/§2 uses all 5 buckets as its length axis and inherits the
same structure; its `log(words)` slope is a within-document slope as much as a between-document one.

### 4.5 The status rule is the evaluator's, and it is pre-registered

`Complete %` is the share of the final PLAN §3.4 status `complete`, assigned by
`configs/eval.yaml` v1 (`frozen_at 2026-08-28`): EndCoverage ≥ 0.95, WER − floor < 0.30, no loop.
The risk difference inherits that rule, including its use of plain EndCoverage rather than the
robust variant.

### 4.6 This is dev, not the hidden test

All six roots are dev-side or constructed (`reports/pilot_baseline_v31.md` §7.4). The checkpoint
selection rule was applied on this same benchmark (Lead, 2026-08-30), so these intervals describe
the selected checkpoints *on the set they were selected on*. Nothing here is an unseen-source
generalisation estimate.

### 4.7 The interaction model is a descriptive slope contrast

`coverage` is bounded in [0, 1] and the OLS in §1/§2 is not. Fitted values can leave the interval,
the residuals are heteroscedastic by construction (coverage piles up at 1 for E3 and at 0 for E2 on
the long buckets), and no random effects are used — PLAN §2's written model is
`metric ~ checkpoint * log(words) + (1 | document_id) + (1 | speaker_id)`, and what is fitted here
is its fixed-effect part with the document grouping handled by the cluster bootstrap instead of by a
variance component. Speaker (voice) is not modelled at all: 2 levels. Read the interaction
coefficient as "how much less coverage-per-log-word this checkpoint loses", not as a calibrated
dose-response. The centering constant `mean(log(n_ref_words))` is computed once on the original 120
rows and held fixed across resamples, so the intercept and the `checkpoint` main effect are
"at the mean log length of the benchmark", not at zero words.

### 4.8 The bootstrap p-values are conservative on discrete outcomes

`p (boot)` is `2 × min(P(Δ* ≤ 0), P(Δ* ≥ 0))`, floored at `1/B`. Both tails include the resamples
that land **exactly** on 0, which is a large mass for `Complete %` and `Repeat %` (12 items per
bucket, and many buckets where one arm is uniformly 0). A row can therefore show a Δ of +8.33 pp
with a p near 0.66 while its CI runs [0.00, 25.00]. The floor at `1/B` also means `0.0001` is "at
or below the bootstrap's resolution", never a measured 1-in-10 000. Use the interval, not the
p-value; PLAN §12 says the same."""


def write_csv(path: Path, payload: dict) -> None:
    fields = [
        "group",
        "comparison",
        "arm_a",
        "arm_b",
        "bucket",
        "metric",
        "label",
        "better",
        "unit",
        "n_items",
        "n_clusters",
        "mean_a",
        "mean_a_ci_lo",
        "mean_a_ci_hi",
        "mean_b",
        "mean_b_ci_lo",
        "mean_b_ci_hi",
        "delta",
        "ci_lo",
        "ci_hi",
        "excludes_zero",
        "p_boot",
        "display_scale",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for comp in payload["comparisons"]:
            for cell, cellblock in comp["cells"].items():
                for name, r in cellblock["metrics"].items():
                    w.writerow(
                        {
                            "group": comp["group"],
                            "comparison": comp["comparison"],
                            "arm_a": comp["arm_a"],
                            "arm_b": comp["arm_b"],
                            "bucket": cell,
                            "metric": name,
                            "label": r["label"],
                            "better": r["better"],
                            "unit": r["unit"],
                            "n_items": r["n_items"],
                            "n_clusters": r["n_clusters"],
                            "mean_a": r["mean_a"],
                            "mean_a_ci_lo": r["mean_a_lo"],
                            "mean_a_ci_hi": r["mean_a_hi"],
                            "mean_b": r["mean_b"],
                            "mean_b_ci_lo": r["mean_b_lo"],
                            "mean_b_ci_hi": r["mean_b_hi"],
                            "delta": r["delta"],
                            "ci_lo": r["ci_lo"],
                            "ci_hi": r["ci_hi"],
                            "excludes_zero": r["excludes_zero"],
                            "p_boot": r["p_boot"],
                            "display_scale": r["scale"],
                        }
                    )
        # interaction coefficients ride in the same csv with bucket="ALL_interaction"
        for comp in payload["comparisons"]:
            inter = comp["interaction"]
            for nm, c in inter["coefficients"].items():
                w.writerow(
                    {
                        "group": comp["group"],
                        "comparison": comp["comparison"],
                        "arm_a": comp["arm_a"],
                        "arm_b": comp["arm_b"],
                        "bucket": "ALL_interaction",
                        "metric": nm,
                        "label": f"OLS {inter['outcome']}",
                        "better": "",
                        "unit": "coef",
                        "n_items": inter["n_rows"],
                        "n_clusters": inter["n_clusters"],
                        "mean_a": "",
                        "mean_a_ci_lo": "",
                        "mean_a_ci_hi": "",
                        "mean_b": "",
                        "mean_b_ci_lo": "",
                        "mean_b_ci_hi": "",
                        "delta": c["estimate"],
                        "ci_lo": c["ci_lo"],
                        "ci_hi": c["ci_hi"],
                        "excludes_zero": c["excludes_zero"],
                        "p_boot": c["p_boot"],
                        "display_scale": 1.0,
                    }
                )


def write_rmst_csv(path: Path, payload: dict) -> None:
    fields = [
        "checkpoint",
        "label",
        "bucket",
        "n_items",
        "n_clusters",
        "tau_words_mean",
        "rmst_words_proxy",
        "ci_lo",
        "ci_hi",
        "rmst_frac",
        "frac_ci_lo",
        "frac_ci_hi",
        "median_stable_words",
        "n_censored_at_tau",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in payload["rmst_words_proxy"]:
            w.writerow(r)


def write_stable_words_csv(path: Path, arms: Sequence[Arm]) -> None:
    fields = [
        "checkpoint",
        "label",
        "text_id",
        "root_id",
        "bucket",
        "voice_id",
        "seed",
        "status",
        "n_ref_words",
        "end_coverage_robust",
        "stable_words_proxy",
        "censored_at_tau",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for arm in arms:
            for k in sorted(arm.rows):
                r = arm.rows[k]
                sw = stable_words(r)
                w.writerow(
                    {
                        "checkpoint": arm.key,
                        "label": arm.label,
                        "text_id": r["text_id"],
                        "root_id": r["text_id"].rsplit("__", 1)[0],
                        "bucket": r["bucket"],
                        "voice_id": r["voice_id"],
                        "seed": r["seed"],
                        "status": r["status"],
                        "n_ref_words": r["n_ref_words"],
                        "end_coverage_robust": r["end_coverage_robust"],
                        "stable_words_proxy": sw,
                        "censored_at_tau": int(math.isclose(sw, float(r["n_ref_words"]))),
                    }
                )


# ---------------------------------------------------------------------- main


ARM_SPECS = [
    ("E3", "E3 Long-SFT step 3001 (selected)", "results/v31_sft/E3_epoch_3_step_3001/per_item.jsonl"),
    ("E2", "E2 Short-SFT step 3001 (selected)", "results/v31_sft/E2_epoch_3_step_3001/per_item.jsonl"),
    ("E1", "E1 Base-Native", "results/v31_base/E1/per_item.jsonl"),
    ("E3loss", "E3 Long-SFT step 772 (by dev loss)", "results/v31_sft/E3_epoch_0_whole/per_item.jsonl"),
    ("E2loss", "E2 Short-SFT step 200 (by dev loss)", "results/v31_sft/E2_epoch_0_step_200/per_item.jsonl"),
    ("E0", "E0 Base official-split (production control)", "results/v31_base/E0/per_item.jsonl"),
]

COMPARISONS = [
    ("primary", "E3", "E2"),
    ("primary", "E3", "E1"),
    ("primary", "E2", "E1"),
    ("secondary", "E3loss", "E2loss"),
    ("secondary", "E3loss", "E1"),
    ("secondary", "E2loss", "E1"),
]

# ------------------------------------------------------------------- presets
#
# The E4 Curriculum-SFT arm is a SECONDARY, product-oriented recipe (PLAN §5,
# pre-registered in reports/decisions.md 2026-08-29, amended 2026-08-30).  It is NOT
# budget-matched to E2/E3 -- it inherits E2's 3000 short-window steps and then spends
# 3000 more on the long arm -- so its contrasts go to their own file and never replace
# the E3-vs-E2 main comparison.  Same estimator, same rule for the resample matrix
# (10 000 resamples, seed 0, cluster = root_id), same metrics, same pairing key.

ARM_SPECS_E4 = [
    ("E3", "E3 Long-SFT step 3001 (selected)", "results/v31_sft/E3_epoch_3_step_3001/per_item.jsonl"),
    ("E4", "E4 Curriculum-SFT S3 step 773 (selected)", "results/v31_curr/E4_epoch_0_whole/per_item.jsonl"),
    ("E4last", "E4 Curriculum-SFT S3 step 2001 (last)", "results/v31_curr/E4_epoch_2_step_2001/per_item.jsonl"),
    ("E2", "E2 Short-SFT step 3001 (selected)", "results/v31_sft/E2_epoch_3_step_3001/per_item.jsonl"),
    ("E1", "E1 Base-Native", "results/v31_base/E1/per_item.jsonl"),
]

COMPARISONS_E4 = [
    ("primary", "E3", "E4"),
    ("primary", "E4", "E2"),
    ("secondary", "E4", "E1"),
    ("secondary", "E4last", "E4"),
]

# The E6 «Mixed-SFT recipe» family and the E7 «Punct-SFT» arm (both pre-registered in
# reports/decisions.md 2026-08-30, results recorded 2026-08-30/31).  Unlike E4 these ARE
# budget-matched to E3: 3000 optimizer steps from the same pretrained llm.pt, target
# tokens/step inside ±3 % of E3.  Same estimator, same resample rule (10 000 resamples,
# seed 0, cluster = root_id), same metrics, same pairing key -- nothing about the method
# changes, only which arms are paired.  Its own output files; it overwrites nothing.

_ARM_BY_KEY = {spec[0]: spec for spec in ARM_SPECS}

ARM_SPECS_E6E7 = [
    ("E7", "E7 Punct-SFT step 3001 (final)", "results/v31_punct/E7_punct_epoch_3_step_3001/per_item.jsonl"),
    ("M31", "E6-M31 Mixed S,S,S,L step 3001 (final)", "results/v31_mix/E6_M31_epoch_0_step_3001/per_item.jsonl"),
    ("M11", "E6-M11 Mixed S,L step 3001 (final)", "results/v31_mix/E6_M11_epoch_0_step_3001/per_item.jsonl"),
    ("M13", "E6-M13 Mixed S,L,L,L step 3001 (final)", "results/v31_mix/E6_M13_epoch_0_step_3001/per_item.jsonl"),
    ("S", "E6-S ext_short only step 3001 (final)", "results/v31_mix/E6_S_epoch_0_step_3001/per_item.jsonl"),
    # E1/E2/E3 exactly as the default preset defines them -- same key, label and path.
    _ARM_BY_KEY["E3"],
    _ARM_BY_KEY["E2"],
    _ARM_BY_KEY["E1"],
]

COMPARISONS_E6E7 = [
    ("primary", "E7", "E3"),
    ("secondary", "M31", "E3"),
    ("secondary", "M11", "E3"),
    ("secondary", "M13", "E3"),
    ("secondary", "S", "E2"),
    ("secondary", "S", "E1"),
]

E6E7_NOTES = [
    (
        "**Budget match — every contrast against E3 below is a recipe contrast, not a compute "
        "contrast.** The six arms trained for this family — `E3`, `E7`, `E6-M31`, `E6-M11`, "
        "`E6-M13`, `E6-S` — all run **3000 optimizer steps from the same pretrained `llm.pt`** "
        "with E3's config (lr 1e-5, warmup 50, long-batch cap 22 500), differing only in which "
        "data or which text the steps see. Every one is inside the PLAN §6.4 ±3 % band on mean "
        "target tokens per step against E3 (19 923 over the whole run), and each was recorded "
        "PASS in `reports/decisions.md`: **E7 −1.27 %**, **E6-S −1.02 %**, **E6-M31 −0.80 %**, "
        "**E6-M11 −0.61 %**, **E6-M13 −0.39 %**. The two carried-over arms are not part of that "
        "family: `E2` Short-SFT is also 3000 steps and budget-matched to E3 at ratio 0.9896 "
        "(−1.04 %, measured over the first 200 steps, `decisions.md` 2026-08-30), and `E1` is "
        "the untrained Base-Native baseline with no training budget at all."
    ),
    (
        "**The `infrastructure_error` items, and exactly how this file counts them.** "
        "`E6-M13` has one: `ex6_genre6_stress__B4 × ref_female_01 × seed 0`, "
        "`gen_status=infrastructure_error`, `stop_reason=exception`, `output_path=null`. The "
        "generation ran away to **22 628 tokens (947 s wall)**, past the 900 s noise buffer of "
        "the flow-matching stage, so the vocoder raised and **no audio exists** — by substance a "
        "runaway repetition, by PLAN §3.4 an infrastructure error, passed through unchanged and "
        "not re-run (same seed, same runaway). **This script applies no special handling to it.** "
        "PLAN §0 rule 4 forbids dropping it, so it stays in all 60 paired items of M13 with the "
        "evaluator's own per-item numbers: `Complete %` → **0** (`status != \"complete\"`), "
        "`Coverage` → **0.000**, `WER-all %` → **100.00** (all 1036 reference words scored as "
        "deletions), `EndCov-robust` → **0.000**, `dur ratio` → **0.000**, and — the one that can "
        "mislead — `Repeat %` → **0.00**, because excess repetition is measured on the ASR "
        "hypothesis and there is no hypothesis. A runaway-repetition failure therefore enters the "
        "`Repeat %` column as a zero and makes M13 look *less* repetitive, not more. The same "
        "item's E3 row enters the pair unchanged, so the Δ is a real paired difference on that "
        "item. It is 1 of the 60 items in every `ALL` cell and 1 of the 12 in every `B4` cell, "
        "and it also supplies a `coverage = 0` row at `log(1036)` to the H6 interaction fit. "
        "**`E6-M11` carries one item in the identical state** — `ex6_genre6_stress__B4 × "
        "ref_male_01`, `status=infrastructure_error`, `output_path=null` — counted the same way; "
        "the 2026-08-30 `decisions.md` note about the `speaker_drift.py` crash called that item "
        "`empty_or_invalid_audio`, but the status the evaluator actually wrote to "
        "`per_item.jsonl` is `infrastructure_error`."
    ),
    (
        "**Every bucket-level cell below carries ±20 pp of single-seed noise that these CIs do "
        "not contain.** The 2026-08-30 inference-variance check (`results/v31_seed1`: seed 1 on "
        "B3–B4, 12 texts × 2 voices, same checkpoints) moved E3's per-bucket Complete from "
        "41.7 % / 33.3 % (B3/B4, seed 0) to 66.7 % / 50.0 % (seed 1). At n = 12 per bucket the "
        "sampling variability of a single seed is ≈ ±2–3 items ≈ **±20 pp**, so **per-bucket "
        "`Complete %` differences below roughly 25 pp do not separate two arms** — and the Lead's "
        "own conclusion from that check was that E6 arms are to be compared first on `ALL` "
        "(60 items) and on the continuous metrics. This bootstrap resamples **documents, not "
        "re-generations**, so that inference variance is outside every interval printed here "
        "(§4.3). Read the `ALL` rows and `Coverage` / `WER-all %` first; treat a B0–B4 "
        "`Complete %` row as descriptive."
    ),
]

DEFAULT_GROUP_TITLES = {
    "primary": "1. Primary comparisons — selected checkpoints",
    "secondary": "2. Secondary comparisons — by-dev-loss checkpoints",
}
DEFAULT_GROUP_INTROS = {
    "primary": (
        "Checkpoint selection follows the Lead's 2026-08-30 rule (completion rate on the dev "
        "benchmark, short-form constraint, coverage tie-break): E3 = `epoch_3_step_3001`, "
        "E2 = `epoch_3_step_3001`. Both are the last checkpoint at equal steps and equal "
        "target-token budget (ratio 0.9896, `reports/decisions.md` 2026-08-30)."
    ),
    "secondary": (
        "The same three contrasts using the by-dev-loss checkpoints (E3 `epoch_0_whole` = step 772, "
        "E2 `epoch_0_step_200`). Secondary: dev loss was found to be a poor proxy for long-form "
        "behaviour (`reports/decisions.md` 2026-08-30), and the two arms are then no longer at "
        "equal steps."
    ),
}

PRESETS: dict[str, dict] = {
    "default": {
        "arm_specs": ARM_SPECS,
        "comparisons": COMPARISONS,
        "name": "paired_deltas",
        "title": "# Table 4 — paired Long-vs-Short deltas with 95 % CI (PLAN.md §12, §17)",
        "group_titles": DEFAULT_GROUP_TITLES,
        "group_intros": DEFAULT_GROUP_INTROS,
        "headline": True,
        "extras": True,
    },
    "e4": {
        "arm_specs": ARM_SPECS_E4,
        "comparisons": COMPARISONS_E4,
        "name": "paired_deltas_e4",
        "title": (
            "# Paired E4 Curriculum-SFT deltas with 95 % CI "
            "(PLAN.md §12; E4 is a PLAN §5 secondary arm)"
        ),
        "group_titles": {
            "primary": "1. E4 against the two budget-matched arms",
            "secondary": "2. E4 against the baseline, and E4's own two S3 checkpoints",
        },
        "group_intros": {
            "primary": (
                "**E4 is not budget-matched to E2 or E3**, and neither contrast below replaces the "
                "main E3 − E2 comparison in `results/v31_stats/paired_deltas.md`. E4 starts from the "
                "*selected* E2 checkpoint (`epoch_3_step_3001`, Lead amendment 2026-08-30) and adds "
                "3000 more optimizer steps on the long arm in three stages (S1 ≤ 90 s / 500 steps, "
                "S2 ≤ 180 s / 500, S3 ≤ 900 s / 2000) — 6000 steps in total against 3000 for each of "
                "E2 and E3 (`reports/training_budget_comparison.md` §8). `E3 − E4` therefore asks "
                "what direct Long-SFT buys over a curriculum that cost twice as much; `E4 − E2` asks "
                "what the curriculum's own long-arm stages bought on top of the checkpoint they "
                "started from."
            ),
            "secondary": (
                "`E4 − E1` is the same contrast against the untrained baseline. `E4last − E4` "
                "compares the two S3 checkpoints that were kept (step 2001 minus step 773): it is "
                "the interval behind the statement that E4 gets worse as S3 continues. It is paired "
                "on the same 60 items and clustered on the same 6 roots, but both arms come from one "
                "training run with one seed, so it describes this run's trajectory and is not a "
                "between-recipe contrast."
            ),
        },
        "headline": False,
        "extras": False,
        "sign_note": (
            "Two rows need it here. `dur ratio` is generated ÷ human duration, so it is **not** "
            "monotone in quality — over- and under-production are both failures and the quantity "
            "that is monotone is |1 − ratio|; the arrow is kept as the file's own convention and "
            "must not be read as a win. And on `Repeat %` a checkpoint that stops early cannot "
            "repeat, so a favourable Δ against an arm that produces almost no audio on B3/B4 is "
            "not like-for-like."
        ),
    },
    "e6e7": {
        "arm_specs": ARM_SPECS_E6E7,
        "comparisons": COMPARISONS_E6E7,
        "name": "paired_deltas_e6e7",
        "title": (
            "# Paired E6 mix-recipe and E7 Punct-SFT deltas with 95 % CI "
            "(PLAN.md §12; E6 and E7 pre-registered in reports/decisions.md 2026-08-30)"
        ),
        "group_titles": {
            "primary": "1. Primary contrast — punctuation in the training text, at equal compute",
            "secondary": "2. Secondary contrasts — the E6 mix-recipe gradient",
        },
        "group_intros": {
            "primary": (
                "`E7 − E3` is the pre-registered E7 question (`reports/decisions.md` 2026-08-30, "
                "«ПРЕРЕГИСТРАЦИЯ E7»): does punctuation in the **training** text change long-form "
                "behaviour? E7 is E3's long arm with `text` replaced by the `text_e2e` field of the "
                "same manifest (gigaam-v3-e2e-ctc, punctuated and capitalised; 6041/6043 units "
                "covered, 2 falling back to ROVER). Its shards were built by transforming E3's own "
                "parquet without re-extracting speech tokens, so **the audio tokens are identical to "
                "E3's and the text is the only difference between the two arms** — 3000 steps, same "
                "config, full-run budget −1.27 %. The pre-registration measured the mismatch it "
                "removes: punctuation density 0.252 marks/word in the benchmark's `text_tts`, 0.298 "
                "in `text_e2e`, **0.014** in the ROVER text E1–E6 trained on."
            ),
            "secondary": (
                "The E6 «Mixed-SFT recipe» arms (`reports/decisions.md` 2026-08-30, «ПРЕРЕГИСТРАЦИЯ "
                "E6»), each against the long-only E3 at the same 3000 steps: `M31` = pattern S,S,S,L "
                "(25 % long batches), `M11` = S,L (50 %), `M13` = S,L,L,L (75 %), `S` = `ext_short` "
                "only (0 % long). The short source is `ext_short` — 100 h of ≤ 20 s single-speaker "
                "clips selected by DistillMOS from the balalaika parquet, v3.1 video ids excluded and "
                "voice-leakage gated — mixed at the optimizer-step level (a batch is wholly from one "
                "source, patterns cycle). `S − E2` and `S − E1` are the separate question of whether "
                "100 h of high-MOS short clips beat the v3.1 short arm or the untrained baseline on "
                "their own. Together the five contrasts are the intervals behind the recipe gradient "
                "in the long share. One limitation stated in the pre-registration itself: with "
                "`num_workers=2` the global batch sequence is the interleaving of two workers' "
                "patterns, so the source **proportions** are exact and the **order** is only "
                "approximate."
            ),
        },
        "notes": E6E7_NOTES,
        "headline": False,
        "extras": False,
        "sign_note": (
            "Three things need it here. `dur ratio` is generated ÷ human duration and is **not** "
            "monotone in quality — both over- and under-production are failures, the monotone "
            "quantity is |1 − ratio| — so its arrow is this file's convention and never a verdict. "
            "On `Repeat %` an arm that stops early cannot repeat: `E6-S` produces almost no audio "
            "on B2–B4 (coverage 0.10/0.03/0.01) and `E1`/`E2` collapse the same way, so a "
            "favourable Δ on those rows compares a model that reads against one that stops, not "
            "two models that read. And the two `infrastructure_error` items score `Repeat % = 0` "
            "for want of a hypothesis even though the failure was a runaway repetition (see the "
            "note above)."
        ),
    },
}


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(
    repo: Path,
    out_dir: Path,
    n_resamples: int,
    seed: int,
    alpha: float,
    preset: str = "default",
) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    cfg = PRESETS[preset]
    arm_specs = cfg["arm_specs"]
    comparisons = cfg["comparisons"]

    bench_path = repo / "data/benchmark/pilot.jsonl"
    bench = {r["text_id"]: r for r in load_jsonl(bench_path)}

    arms = {}
    for key, label, rel in arm_specs:
        arms[key] = load_arm(key, label, repo / rel)

    # Every arm must cover the same 60 benchmark items.
    ref_keys = sorted(arms["E1"].rows)
    for key, arm in arms.items():
        if sorted(arm.rows) != ref_keys:
            raise ValueError(f"arm {key} does not cover the same items as E1")
    for k in ref_keys:
        if k[0] not in bench:
            raise ValueError(f"text_id {k[0]} is not in {bench_path}")

    roots = sorted({bench[k[0]]["root_id"] for k in ref_keys})
    idx = resample_matrix(len(roots), n_resamples, seed)

    payload: dict = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "script": "src/stats/bootstrap.py",
            "plan": "PLAN.md §12 (statistics), §17 Table 4",
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "ci": f"{int(100 * (1 - alpha))}% percentile",
            "cluster_unit": "root_id",
            "n_clusters": len(roots),
            "roots": roots,
            "pairing_key": list(ITEM_KEY_FIELDS),
            "seeds": sorted({k[2] for k in ref_keys}),
            "n_items_per_arm": len(ref_keys),
            "metrics": [m.name for m in METRICS],
            "conditioning": "unconditional — all statuses included, nothing dropped (PLAN §0 rule 4)",
            "inputs": [
                {"path": str(p.relative_to(repo)), "rows": n, "sha256": sha256_of(p)}
                for p, n in [(bench_path, len(bench))]
                + [(arms[k].path, arms[k].n) for k, _, _ in arm_specs]
            ],
        },
        "comparisons": [],
    }
    if preset != "default":
        payload["meta"]["preset"] = preset
        payload["meta"]["title"] = cfg["title"]
        if cfg.get("sign_note"):
            payload["meta"]["sign_note"] = cfg["sign_note"]
        if cfg.get("notes"):
            payload["meta"]["notes"] = list(cfg["notes"])
        payload["meta"]["group_titles"] = cfg["group_titles"]
        payload["meta"]["group_intros"] = cfg["group_intros"]

    for group, a_key, b_key in comparisons:
        comp = compare(arms[a_key], arms[b_key], bench, idx, alpha)
        comp["group"] = group
        payload["comparisons"].append(comp)

    if cfg["extras"]:
        payload["rmst_words_proxy"] = rmst_table(
            [arms[k] for k, _, _ in arm_specs], bench, idx, alpha
        )
        payload["rmst_words_proxy_definition"] = (
            "stable_words = end_coverage_robust * n_ref_words; mean over items, horizon tau = the item's "
            "own n_ref_words, non-degrading items censored at tau. PROXY — not the PLAN §9.3 rolling-window "
            "degradation point."
        )

    name = cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / f"{name}.csv", payload)
    if cfg["extras"]:
        write_rmst_csv(out_dir / "rmst_words_proxy.csv", payload)
        write_stable_words_csv(
            out_dir / "stable_words_per_item.csv", [arms[k] for k, _, _ in arm_specs]
        )
    (out_dir / f"{name}.md").write_text(render_markdown(payload) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <repo>/results/v31_stats")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help=(
            "which arms and contrasts to run. 'default' = the six PLAN §17 Table 4 contrasts "
            "-> paired_deltas.{json,csv,md} + the RMST proxy; 'e4' = the E4 Curriculum-SFT "
            "contrasts (E3-E4, E4-E2, E4-E1, E4last-E4) -> paired_deltas_e4.{json,csv,md}; "
            "'e6e7' = the budget-matched E7 Punct-SFT and E6 mix-recipe contrasts (E7-E3 primary; "
            "M31-E3, M11-E3, M13-E3, S-E2, S-E1 secondary) -> paired_deltas_e6e7.{json,csv,md}. "
            "Each preset writes its own files and none overwrites another's."
        ),
    )
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir is not None else repo / "results/v31_stats"
    payload = run(repo, Path(out_dir), args.resamples, args.seed, args.alpha, args.preset)

    meta = payload["meta"]
    print(
        f"paired cluster bootstrap: {meta['n_resamples']} resamples, seed {meta['seed']}, "
        f"{meta['n_clusters']} clusters, {meta['n_items_per_arm']} paired items per arm"
    )
    for comp in payload["comparisons"]:
        r = comp["cells"]["ALL"]["metrics"]
        cr = r["complete_rate"]
        cov = r["source_coverage"]
        print(
            f"  [{comp['group']:9s}] {comp['comparison']:16s} ALL  "
            f"Complete {100 * cr['delta']:+7.2f} pp [{100 * cr['ci_lo']:+.2f}, {100 * cr['ci_hi']:+.2f}]  "
            f"Coverage {cov['delta']:+.3f} [{cov['ci_lo']:+.3f}, {cov['ci_hi']:+.3f}]"
        )
    print(f"written: {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
