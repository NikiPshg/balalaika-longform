#!/usr/bin/env python
"""Pooled pilot+robust STRATIFIED paired cluster bootstrap (reviewer #2, «small sample»).

Answer to reviewer #2's small-sample concern: the pilot (60 items) and the E8
«Robust-20» benchmark (53 items) evaluate the SAME three checkpoints
(E1 Base-Native, E3 Long-SFT step 3001, E7 Punct-SFT step 3001), so their paired
per-item deltas can be pooled into one estimate with a narrower CI — provided each
design keeps its own cluster structure.  Pre-registered in reports/decisions.md
2026-08-31, «ПРЕРЕГИСТРАЦИЯ pooled-бутстрэпа» (A8b-pooled).

SAME machinery as src/stats/bootstrap.py — this file adds no estimator of its own.
`resample_matrix`, `paired_delta`, `percentile_ci`, `build_clusters`, `METRICS`,
`load_arm`, `aligned_keys` and `write_csv` are imported and called unchanged (the
convention established by src/stats/bootstrap_robust.py).  The one new piece is
orchestration: a stratified resample matrix glued from per-stratum matrices.

Method
------
* **Stratum «pilot»** — the 60 pilot items (30 nested texts × 2 reference voices,
  seed 0); cluster unit = ``root_id`` (**6 root documents**), exactly as in
  bootstrap.py.
* **Stratum «robust»** — the 53 Robust-20 items (each voice reads only its own
  texts, seed 0); cluster unit = ``voice_id`` (**13 voices**), exactly as in
  bootstrap_robust.py.
* In every one of the 10 000 resamples the two strata are resampled
  **independently, each within itself**: 6 pilot roots are drawn with replacement
  from the 6 pilot roots, 13 robust voices from the 13 robust voices.  A pilot
  cluster can never be drawn into the robust part of a resample nor vice versa —
  the designs differ, and independent within-stratum resampling is the honest way
  to combine them.  Implementation: clusters get pooled indices with the pilot
  block first (``[0, n_pilot)``) and the robust block after it
  (``[n_pilot, n_pilot + n_robust)``); the resample matrix is the column-wise
  concatenation of ``resample_matrix(n_pilot, B, 2*seed)`` and
  ``resample_matrix(n_robust, B, 2*seed + 1) + n_pilot`` — disjoint derived seeds,
  one fixed matrix per cluster configuration ``(n_pilot, n_robust)`` of a cell,
  shared across metrics, cells and comparisons (deterministic).
* The statistic is the paired mean difference ``mean_i(A(i) − B(i))`` over ALL
  items of the pooled resample.  **Stratum weights are «as the items fall»**: every
  drawn item counts once, so the strata enter in proportion to their drawn item
  counts (≈ 60:53 around the point estimate, which is exactly
  ``(60·Δ_pilot + 53·Δ_robust)/113``); there is NO reweighting to equal strata.
  The estimand is therefore the item-weighted average paired effect over the union
  of the two benchmarks, not the effect of either design alone.
* 10 000 resamples, base seed 0, 95 % percentile CI, pairing on
  ``(text_id, voice_id, seed)`` within each stratum; every metric is unconditional
  (PLAN §0 rule 4) and the metric set is bootstrap.py's frozen ``METRICS``.
* Cells: **ALL (n = 113, 6 + 13 = 19 clusters) is the headline**; B0–B4 cells are
  descriptive only (single-seed ±20 pp per-bucket noise, see the note in the md).
* The H6 interaction fit is NOT pooled — the two strata have different text
  designs (nested prefixes vs independent per-voice texts); the per-design fits
  live in paired_deltas_e6e7.md and paired_deltas_robust.md.

Contrasts: primary E7−E3; secondary E3−E1, E7−E1.

Usage::

    .venv-eval/bin/python src/stats/bootstrap_pooled.py       # -> results/v31_stats/
    .venv-eval/bin/python src/stats/bootstrap_pooled.py --resamples 1000 --out-dir tmp/x

Writes results/v31_stats/paired_deltas_pooled.{md,csv,json} and overwrites nothing
else.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from bootstrap import (  # noqa: E402,F401  (same directory)
    CELLS,
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    ITEM_KEY_FIELDS,
    METRICS,
    Arm,
    _fmt,
    _nd_for,
    _scaled,
    aligned_keys,
    build_clusters,
    load_arm,
    load_jsonl,
    paired_delta,
    resample_matrix,
    sha256_of,
    write_csv,
)

REPO = Path(__file__).resolve().parents[2]

NAME = "paired_deltas_pooled"

# Stratum order is load-bearing: build_clusters sorts cluster names, and the
# prefixes are chosen so that every "pilot:*" name sorts before every "robust:*"
# name — the pilot block therefore always occupies the leading cluster indices.
STRATA = ("pilot", "robust")
STRATUM_PREFIX = {"pilot": "pilot:", "robust": "robust:"}

BENCH_PATHS = {
    "pilot": "data/benchmark/pilot.jsonl",
    "robust": "data/benchmark/robust.jsonl",
}

# Same checkpoints evaluated on both benchmarks (per_item.jsonl each).
ARM_SPECS_POOLED = [
    ("E7", "E7 Punct-SFT step 3001 (pilot+robust)", {
        "pilot": "results/v31_punct/E7_punct_epoch_3_step_3001/per_item.jsonl",
        "robust": "results/v31_robust/E7/per_item.jsonl",
    }),
    ("E3", "E3 Long-SFT step 3001 (pilot+robust)", {
        "pilot": "results/v31_sft/E3_epoch_3_step_3001/per_item.jsonl",
        "robust": "results/v31_robust/E3/per_item.jsonl",
    }),
    ("E1", "E1 Base-Native (pilot+robust)", {
        "pilot": "results/v31_base/E1/per_item.jsonl",
        "robust": "results/v31_robust/E1/per_item.jsonl",
    }),
]

COMPARISONS_POOLED = [
    ("primary", "E7", "E3"),
    ("secondary", "E3", "E1"),
    ("secondary", "E7", "E1"),
]


# ----------------------------------------------------------- stratified matrix


def stratum_seed(base_seed: int, stratum_pos: int) -> int:
    """Disjoint deterministic per-stratum seeds: 2*seed + position.

    Guarantees the two strata never share an RNG stream for any base seed
    (pilot -> 2*seed, robust -> 2*seed + 1).
    """
    return 2 * base_seed + stratum_pos


def stratified_resample_matrix(strata_sizes: tuple[int, ...], n_resamples: int,
                               base_seed: int) -> np.ndarray:
    """``(n_resamples, sum(sizes))`` matrix of pooled cluster indices.

    Column block ``s`` (width ``sizes[s]``) is ``bootstrap.resample_matrix`` for
    that stratum alone, offset into the pooled index space: its values lie in
    ``[offset, offset + sizes[s])`` where ``offset = sum(sizes[:s])``.  Row ``b``
    is therefore one resample in which every stratum is resampled with
    replacement independently, within itself.  Deterministic per
    ``(strata_sizes, base_seed)``.
    """
    blocks: list[np.ndarray] = []
    offset = 0
    for pos, n in enumerate(strata_sizes):
        blocks.append(resample_matrix(n, n_resamples, stratum_seed(base_seed, pos)) + offset)
        offset += n
    return np.concatenate(blocks, axis=1)


# ------------------------------------------------------------------ comparison


def _split_clusters(clusters: list[str]) -> tuple[int, int]:
    """(n_pilot, n_robust) of a sorted pooled cluster list; asserts the partition.

    ``build_clusters`` returns the names sorted, and the prefixes sort
    "pilot:" < "robust:", so the pilot block must be a contiguous leading run.
    """
    n_pilot = sum(1 for c in clusters if c.startswith(STRATUM_PREFIX["pilot"]))
    n_robust = len(clusters) - n_pilot
    for i, c in enumerate(clusters):
        want = STRATUM_PREFIX["pilot"] if i < n_pilot else STRATUM_PREFIX["robust"]
        if not c.startswith(want):
            raise AssertionError(f"cluster order broken at {i}: {clusters}")
    return n_pilot, n_robust


def compare_pooled(a: Arm, b: Arm, arm_paths: dict, text_to_cluster: dict,
                   text_to_stratum: dict, idx_by_sizes,
                   alpha: float = DEFAULT_ALPHA) -> dict:
    """bootstrap.compare with stratified orchestration, no estimator change.

    ``text_to_cluster`` maps text_id -> prefixed cluster name ("pilot:<root_id>"
    or "robust:<voice_id>"); ``build_clusters`` then gives pooled indices with
    the pilot block first, ``idx_by_sizes((n_pilot, n_robust))`` supplies the
    stratified resample matrix, and ``paired_delta`` — bootstrap.py's own,
    unchanged — does everything statistical.
    """
    keys = aligned_keys(a, b)
    out: dict = {
        "comparison": f"{a.key}_vs_{b.key}",
        "arm_a": a.key,
        "arm_a_label": a.label,
        "arm_a_paths": {s: arm_paths[a.key][s] for s in STRATA},
        "arm_b": b.key,
        "arm_b_label": b.label,
        "arm_b_paths": {s: arm_paths[b.key][s] for s in STRATA},
        "cells": {},
    }
    for cell in CELLS:
        cell_keys = [k for k in keys if cell == "ALL" or a.rows[k]["bucket"] == cell]
        if not cell_keys:
            continue
        cluster_index, clusters = build_clusters(cell_keys, text_to_cluster)
        n_pilot, n_robust = _split_clusters(clusters)
        if n_pilot == 0 or n_robust == 0:
            raise AssertionError(f"cell {cell}: a stratum is empty ({n_pilot}, {n_robust})")
        idx = idx_by_sizes((n_pilot, n_robust))
        n_items_pilot = sum(1 for k in cell_keys if text_to_stratum[k[0]] == "pilot")
        cell_out: dict = {
            "n_items": len(cell_keys),
            "n_items_pilot": n_items_pilot,
            "n_items_robust": len(cell_keys) - n_items_pilot,
            "n_clusters": n_pilot + n_robust,
            "n_clusters_pilot": n_pilot,
            "n_clusters_robust": n_robust,
            "clusters": clusters,
            "metrics": {},
        }
        for m in METRICS:
            va = np.array([m.getter(a.rows[k]) for k in cell_keys])
            vb = np.array([m.getter(b.rows[k]) for k in cell_keys])
            res = paired_delta(va, vb, cluster_index, idx, n_pilot + n_robust, alpha)
            res.update({"metric": m.name, "label": m.label, "scale": m.scale,
                        "unit": m.unit, "better": m.better, "binary": m.binary})
            cell_out["metrics"][m.name] = res
        out["cells"][cell] = cell_out

    # No pooled H6 fit (pre-registered): the strata have different text designs
    # (nested pilot prefixes vs independent per-voice robust texts).  The empty
    # coefficient dict keeps bootstrap.write_csv reusable unchanged.
    out["interaction"] = {
        "coefficients": {},
        "skipped": ("not pooled by design — per-design H6 fits live in "
                    "results/v31_stats/paired_deltas_e6e7.md and "
                    "results/v31_robust/paired_deltas_robust.md"),
    }
    return out


# ------------------------------------------------------------------- markdown


POOLED_BUCKET_NOTE = (
    "**Per-bucket cells are descriptive only — they carry single-seed noise on the "
    "order of ±20 pp that these CIs do not contain.** The 2026-08-30 "
    "inference-variance check (`results/v31_seed1`: seed 1 on B3–B4, same "
    "checkpoints) moved E3's per-bucket Complete by up to 25 pp against seed 0; at "
    "12–17 items per pooled bucket a single seed is worth ≈ ±2–3 items ≈ ±20 pp, so "
    "per-bucket `Complete %` differences below roughly 25 pp do not separate two "
    "arms. This bootstrap resamples documents and voices, not re-generations, so "
    "that inference variance sits outside every interval printed here. Read the "
    "`ALL` rows (n = 113) and the continuous metrics (`Coverage`, `WER-all %`) "
    "first; treat any B0–B4 row as descriptive."
)


def render_markdown_pooled(payload: dict) -> str:
    meta = payload["meta"]
    L: list[str] = []
    L.append("# Pooled pilot+robust paired deltas with 95 % CI — stratified cluster "
             "bootstrap (reviewer #2 «small sample»; prereg reports/decisions.md "
             "2026-08-31, A8b-pooled)")
    L.append("")
    L.append(
        f"- generated: {meta['generated_utc']} by `src/stats/bootstrap_pooled.py` "
        f"(estimator imported unchanged from `src/stats/bootstrap.py`)\n"
        f"- stratified paired cluster bootstrap, **{meta['n_resamples']} resamples**, "
        f"base seed **{meta['seed']}**, **{int(100 * (1 - meta['alpha']))} % percentile "
        f"CI**\n"
        f"- pairing unit: `text_id × voice_id × seed` within each stratum — "
        f"**{meta['n_items_per_arm']} paired items per checkpoint** "
        f"({meta['n_items_pilot']} pilot + {meta['n_items_robust']} robust)\n"
        f"- **stratum «pilot»**: cluster = `root_id`, {meta['n_clusters_pilot']} root "
        f"documents (as `bootstrap.py`); **stratum «robust»**: cluster = `voice_id`, "
        f"{meta['n_clusters_robust']} voices (as `bootstrap_robust.py`); ALL cell = "
        f"{meta['n_clusters_pilot'] + meta['n_clusters_robust']} clusters\n"
        f"- every metric is unconditional: failures stay in with the evaluator's "
        f"numbers (PLAN §0 rule 4, §9.1)"
    )
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(
        "In every resample the two strata are resampled **independently, each within "
        "itself**: 6 pilot roots are drawn with replacement from the 6 pilot roots and "
        "13 robust voices from the 13 robust voices (per-stratum matrices from "
        "`bootstrap.resample_matrix` with disjoint derived seeds `2·seed` / "
        "`2·seed + 1`, one fixed matrix per cluster configuration of a cell, shared "
        "across metrics, cells and comparisons). A pilot cluster can never land in the "
        "robust part of a draw nor vice versa — the two designs differ, and "
        "independent within-stratum resampling is the honest way to pool them. The "
        "statistic is the paired mean difference `mean_i(A(i) − B(i))` over **all "
        "items of the pooled resample**, so **stratum weights are «as the items "
        "fall»**: each drawn item counts once, the strata enter in proportion to "
        "their drawn item counts (≈ 60:53 around the point estimate, which equals "
        "`(60·Δ_pilot + 53·Δ_robust)/113` exactly), and nothing is reweighted to "
        "equal strata. The estimand is the item-weighted average paired effect over "
        "the union of the two benchmarks — not the effect of either design alone. "
        "All statistical calls (`paired_delta`, `percentile_ci`, `build_clusters`) "
        "are `bootstrap.py`'s own, unchanged."
    )
    L.append("")
    L.append("**Sign convention.** Δ is always A − B. The `better` arrow says which "
             "direction is good; a positive Δ on a `↓` row (WER-all %, Repeat %) means "
             "A is worse.")
    L.append("")
    L.append(POOLED_BUCKET_NOTE)
    L.append("")
    for comp in payload["comparisons"]:
        L.append(f"## {comp['arm_a_label']} − {comp['arm_b_label']} [{comp['group']}]")
        L.append("")
        L.append(f"A = `{comp['arm_a_paths']['pilot']}` + `{comp['arm_a_paths']['robust']}`, "
                 f"B = `{comp['arm_b_paths']['pilot']}` + `{comp['arm_b_paths']['robust']}`.")
        L.append("")
        L.append("| Metric | better | Bucket | n (pilot+robust) | clusters (p+r) | A | B "
                 "| Δ (A−B) | 95 % CI | CI excludes 0 | p (boot) |")
        L.append("|---|---|---|---:|---:|---:|---:|---:|---|---|---:|")
        for m in METRICS:
            for cell in CELLS:
                if cell not in comp["cells"]:
                    continue
                c = comp["cells"][cell]
                r = c["metrics"][m.name]
                nd = _nd_for(m.unit)
                arrow = "↑" if m.better == "higher" else "↓"
                L.append(
                    f"| {m.label} | {arrow} | {cell} | "
                    f"{c['n_items']} ({c['n_items_pilot']}+{c['n_items_robust']}) | "
                    f"{c['n_clusters']} ({c['n_clusters_pilot']}+{c['n_clusters_robust']}) | "
                    f"{_fmt(_scaled(r, 'mean_a'), nd)} | {_fmt(_scaled(r, 'mean_b'), nd)} | "
                    f"**{_fmt(r['delta'] * m.scale, nd)}** | "
                    f"[{_fmt(r['ci_lo'] * m.scale, nd)}, {_fmt(r['ci_hi'] * m.scale, nd)}] | "
                    f"{'yes' if r['excludes_zero'] else 'no'} | {_fmt(r['p_boot'], 4)} |"
                )
        L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- The two strata are different populations by design: pilot = 30 nested "
             "prefixes of 6 documents read by 2 fixed reference voices; robust = 13 "
             "voices each reading its own independent texts (voice and text "
             "deliberately confounded within an arm; the paired contrast across arms "
             "is unaffected). Pooling answers «does the effect hold over the union», "
             "with item-count weights — it does not make the designs equivalent.")
    L.append("- One seed, one generation per item in both strata: inference variance "
             "is NOT inside these intervals (the bootstrap resamples clusters, not "
             "re-generations).")
    L.append("- No pooled H6 interaction fit (pre-registered): the text-length designs "
             "differ; per-design fits live in `results/v31_stats/paired_deltas_e6e7.md` "
             "and `results/v31_robust/paired_deltas_robust.md`.")
    L.append("- Per-design results this file pools (same estimator, same seeds "
             "convention): `results/v31_stats/paired_deltas_e6e7.md` (pilot E7−E3), "
             "`results/v31_stats/paired_deltas.md` (pilot E3−E1), "
             "`results/v31_robust/paired_deltas_robust.md` (robust E7−E3, E3−E1).")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```")
    L.append("cd .")
    L.append(f".venv-eval/bin/python src/stats/bootstrap_pooled.py --resamples "
             f"{meta['n_resamples']} --seed {meta['seed']}")
    L.append("```")
    L.append("")
    L.append("### Inputs")
    L.append("")
    L.append("| file | rows | sha256 |")
    L.append("|---|---:|---|")
    for src in meta["inputs"]:
        L.append(f"| `{src['path']}` | {src['rows']} | `{src['sha256']}` |")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------ run


def _load_pooled_arm(key: str, label: str, rel_paths: dict, repo: Path,
                     stratum_texts: dict) -> tuple[Arm, dict]:
    """One merged Arm from the pilot and robust per_item files of a checkpoint."""
    sub = {s: load_arm(f"{key}_{s}", label, repo / rel_paths[s]) for s in STRATA}
    merged: dict = {}
    for s in STRATA:
        for k, row in sub[s].rows.items():
            if k in merged:
                raise ValueError(f"{key}: item key {k} present in both strata")
            if k[0] not in stratum_texts[s]:
                raise ValueError(f"{key}/{s}: text_id {k[0]} not in the {s} benchmark")
            merged[k] = row
    counts = {s: sub[s].n for s in STRATA}
    return Arm(key=key, label=label, path=repo / rel_paths["pilot"], rows=merged), counts


def run(repo: Path, out_dir: Path, n_resamples: int, seed: int, alpha: float) -> dict:
    benches = {s: load_jsonl(repo / BENCH_PATHS[s]) for s in STRATA}
    stratum_texts = {s: {r["text_id"] for r in benches[s]} for s in STRATA}
    overlap = stratum_texts["pilot"] & stratum_texts["robust"]
    if overlap:
        raise ValueError(f"pilot and robust benchmarks share text_ids: {sorted(overlap)[:5]}")

    # text_id -> prefixed cluster name; the prefix both prevents any name
    # collision across strata and makes sorted() put every pilot cluster first.
    text_to_cluster: dict = {}
    text_to_stratum: dict = {}
    robust_voice: dict = {}
    for r in benches["pilot"]:
        text_to_cluster[r["text_id"]] = STRATUM_PREFIX["pilot"] + r["root_id"]
        text_to_stratum[r["text_id"]] = "pilot"
    for r in benches["robust"]:
        text_to_cluster[r["text_id"]] = STRATUM_PREFIX["robust"] + r["voice_id"]
        text_to_stratum[r["text_id"]] = "robust"
        robust_voice[r["text_id"]] = r["voice_id"]

    arm_paths = {key: dict(rels) for key, _, rels in ARM_SPECS_POOLED}
    arms: dict = {}
    counts_ref: dict | None = None
    for key, label, rels in ARM_SPECS_POOLED:
        arms[key], counts = _load_pooled_arm(key, label, rels, repo, stratum_texts)
        if counts_ref is None:
            counts_ref = counts
        elif counts != counts_ref:
            raise ValueError(f"arm {key}: stratum sizes {counts} != {counts_ref}")

    ref_keys = sorted(arms[ARM_SPECS_POOLED[0][0]].rows)
    for key, arm in arms.items():
        if sorted(arm.rows) != ref_keys:
            raise ValueError(f"arm {key} does not cover the same items as "
                             f"{ARM_SPECS_POOLED[0][0]}")
    for k in ref_keys:  # E8 binding: a robust item is generated only by its own voice
        if text_to_stratum[k[0]] == "robust" and k[1] != robust_voice[k[0]]:
            raise ValueError(f"{k[0]}: run voice {k[1]!r} != benchmark voice "
                             f"{robust_voice[k[0]]!r} (E8 binding broken)")

    clusters_by_stratum = {
        s: sorted({text_to_cluster[k[0]] for k in ref_keys if text_to_stratum[k[0]] == s})
        for s in STRATA
    }

    idx_cache: dict[tuple[int, int], np.ndarray] = {}

    def idx_by_sizes(sizes: tuple[int, int]) -> np.ndarray:
        if sizes not in idx_cache:
            idx_cache[sizes] = stratified_resample_matrix(sizes, n_resamples, seed)
        return idx_cache[sizes]

    payload: dict = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "script": "src/stats/bootstrap_pooled.py",
            "estimator": "src/stats/bootstrap.py (imported, unchanged)",
            "plan": ("PLAN.md §12 machinery; pooled prereg reports/decisions.md "
                     "2026-08-31 (A8b-pooled)"),
            "n_resamples": n_resamples,
            "seed": seed,
            "stratum_seeds": {s: stratum_seed(seed, i) for i, s in enumerate(STRATA)},
            "alpha": alpha,
            "ci": f"{int(100 * (1 - alpha))}% percentile",
            "strata": {
                s: {
                    "cluster_unit": "root_id" if s == "pilot" else "voice_id",
                    "n_clusters": len(clusters_by_stratum[s]),
                    "clusters": [c[len(STRATUM_PREFIX[s]):] for c in clusters_by_stratum[s]],
                    "n_items": counts_ref[s],
                    "benchmark": BENCH_PATHS[s],
                }
                for s in STRATA
            },
            "weights": ("as the items fall: every drawn item counts once; strata enter "
                        "in proportion to their drawn item counts (~60:53); pooled point "
                        "delta = (60*delta_pilot + 53*delta_robust)/113; no reweighting"),
            "n_clusters_pilot": len(clusters_by_stratum["pilot"]),
            "n_clusters_robust": len(clusters_by_stratum["robust"]),
            "n_items_pilot": counts_ref["pilot"],
            "n_items_robust": counts_ref["robust"],
            "pairing_key": list(ITEM_KEY_FIELDS),
            "seeds": sorted({k[2] for k in ref_keys}),
            "n_items_per_arm": len(ref_keys),
            "metrics": [m.name for m in METRICS],
            "conditioning": "unconditional — all statuses included (PLAN §0 rule 4)",
            "interaction": "not pooled (pre-registered); see per-design files",
            "inputs": [
                {"path": str(p.relative_to(repo)), "rows": n, "sha256": sha256_of(p)}
                for p, n in [(repo / BENCH_PATHS[s], len(benches[s])) for s in STRATA]
                + [(repo / arm_paths[key][s], len(load_jsonl(repo / arm_paths[key][s])))
                   for key, _, _ in ARM_SPECS_POOLED for s in STRATA]
            ],
        },
        "comparisons": [],
    }
    for group, a_key, b_key in COMPARISONS_POOLED:
        comp = compare_pooled(arms[a_key], arms[b_key], arm_paths, text_to_cluster,
                              text_to_stratum, idx_by_sizes, alpha)
        comp["group"] = group
        payload["comparisons"].append(comp)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{NAME}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8")
    write_csv(out_dir / f"{NAME}.csv", payload)
    (out_dir / f"{NAME}.md").write_text(render_markdown_pooled(payload) + "\n",
                                        encoding="utf-8")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <repo>/results/v31_stats")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir is not None else repo / "results/v31_stats"
    payload = run(repo, Path(out_dir), args.resamples, args.seed, args.alpha)
    meta = payload["meta"]
    print(f"stratified pooled cluster bootstrap: {meta['n_resamples']} resamples, base "
          f"seed {meta['seed']}, {meta['n_items_per_arm']} paired items "
          f"({meta['n_items_pilot']}+{meta['n_items_robust']}), "
          f"{meta['n_clusters_pilot']}+{meta['n_clusters_robust']} clusters")
    for comp in payload["comparisons"]:
        r = comp["cells"]["ALL"]["metrics"]
        cr, w, cov = r["complete_rate"], r["wer"], r["source_coverage"]
        print(f"  [{comp['group']:9s}] {comp['comparison']:9s} ALL  "
              f"Complete {100 * cr['delta']:+7.2f} pp "
              f"[{100 * cr['ci_lo']:+.2f}, {100 * cr['ci_hi']:+.2f}]  "
              f"WER {100 * w['delta']:+6.2f} pp [{100 * w['ci_lo']:+.2f}, "
              f"{100 * w['ci_hi']:+.2f}]  "
              f"Cov {cov['delta']:+.3f} [{cov['ci_lo']:+.3f}, {cov['ci_hi']:+.3f}]")
    print(f"written: {out_dir}/{NAME}.{{md,csv,json}}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
