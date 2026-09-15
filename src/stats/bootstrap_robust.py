#!/usr/bin/env python
"""A2-robust / E8 «Robust-20»: paired cluster bootstrap with clusters = voice_id.

SAME machinery as src/stats/bootstrap.py — this file adds no estimator of its own.
`resample_matrix`, `paired_delta`, `percentile_ci`, `bootstrap_p_two_sided`, `METRICS`,
`compare` (incl. the H6-style interaction fit) and `write_csv` are all imported from
bootstrap.py and called unchanged.  The two E8-specific deviations, both pre-registered
(reports/decisions.md 2026-08-31, «ПРЕРЕГИСТРАЦИЯ E8»):

  * the benchmark is `data/benchmark/robust.jsonl` and the CLUSTER UNIT IS THE VOICE:
    every E8 item is generated only with its own voice, so `compare` receives a
    bench-like mapping whose `root_id` is the item's `voice_id` — the identical code
    path then resamples voices instead of documents;
  * contrasts are E7−E3 and E3−E1 on the three robust arms.

Every metric stays unconditional (PLAN §0 rule 4).  10 000 resamples, seed 0,
95 % percentile CI, pairing on (text_id, voice_id, seed).  One deterministic resample
matrix per cluster COUNT (a robust bucket cell covers only the voices that have that
bucket, unlike the pilot where every cell held all 6 roots), shared across metrics,
cells and comparisons — see `compare_robust`.

Usage::

    .venv-eval/bin/python src/stats/bootstrap_robust.py            # -> results/v31_robust/
    .venv-eval/bin/python src/stats/bootstrap_robust.py --resamples 1000 --out-dir tmp/x
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
    interaction_model,
    load_arm,
    load_jsonl,
    paired_delta,
    resample_matrix,
    sha256_of,
    write_csv,
)

REPO = Path(__file__).resolve().parents[2]

ARM_SPECS_ROBUST = [
    ("E7", "E7_robust Punct-SFT step 3001", "results/v31_robust/E7/per_item.jsonl"),
    ("E3", "E3_robust Long-SFT step 3001", "results/v31_robust/E3/per_item.jsonl"),
    ("E1", "E1_robust Base-Native", "results/v31_robust/E1/per_item.jsonl"),
]

COMPARISONS_ROBUST = [
    ("primary", "E7", "E3"),
    ("primary", "E3", "E1"),
]

NAME = "paired_deltas_robust"


def compare_robust(a: Arm, b: Arm, bench: dict, idx_by_n, alpha: float = DEFAULT_ALPHA) -> dict:
    """bootstrap.compare with ONE orchestration change, no estimator change.

    The pilot's cells always covered all 6 root clusters, so `compare` shares a single
    resample matrix.  A robust bucket cell covers only the voices that HAVE that bucket
    (B4 = 5 of 13), so the shared matrix is drawn per cluster COUNT via ``idx_by_n``
    (deterministic: seed and resample count fixed, one matrix per count, reused across
    metrics, cells and comparisons).  Every statistical call — ``build_clusters``,
    ``paired_delta``, ``interaction_model`` — is bootstrap.py's own, unchanged.
    """
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
        idx = idx_by_n(n_clusters)
        cell_out: dict = {"n_items": len(cell_keys), "n_clusters": n_clusters,
                          "roots": roots, "metrics": {}}
        for m in METRICS:
            va = np.array([m.getter(a.rows[k]) for k in cell_keys])
            vb = np.array([m.getter(b.rows[k]) for k in cell_keys])
            res = paired_delta(va, vb, cluster_index, idx, n_clusters, alpha)
            res.update({"metric": m.name, "label": m.label, "scale": m.scale,
                        "unit": m.unit, "better": m.better, "binary": m.binary})
            cell_out["metrics"][m.name] = res
        out["cells"][cell] = cell_out

    n_all = len({text_to_root[k[0]] for k in keys})
    out["interaction"] = interaction_model(a, b, keys, text_to_root, idx_by_n(n_all), alpha)
    return out


def render_markdown_robust(payload: dict) -> str:
    meta = payload["meta"]
    L: list[str] = []
    L.append("# E8 «Robust-20» — paired deltas with 95 % CI, clusters = voice "
             "(PLAN.md §12 machinery; prereg reports/decisions.md 2026-08-31)")
    L.append("")
    L.append(
        f"- generated: {meta['generated_utc']} by `src/stats/bootstrap_robust.py` "
        f"(estimator imported unchanged from `src/stats/bootstrap.py`)\n"
        f"- paired cluster bootstrap, **{meta['n_resamples']} resamples**, seed "
        f"**{meta['seed']}**, **{int(100 * (1 - meta['alpha']))} % percentile CI**\n"
        f"- pairing unit: `text_id × voice_id × seed` — {meta['n_items_per_arm']} items "
        f"per checkpoint\n"
        f"- **cluster unit: `voice_id` — {meta['n_clusters']} clusters** (every E8 item "
        f"is generated only with its own voice, so voice = the E8 analogue of the pilot's "
        f"root document)\n"
        f"- every metric is unconditional: failures stay in with the evaluator's numbers "
        f"(PLAN §0 rule 4, §9.1)"
    )
    L.append("")
    L.append("**Sign convention.** Δ is always A − B. The `better` arrow says which "
             "direction is good; a positive Δ on a `↓` row (WER-all %, Repeat %) means A "
             "is worse. Per-bucket cells here have ≈ 1 item per voice per bucket and a "
             "single seed — read the ALL rows and the continuous metrics first; the "
             "single-seed ±20 pp per-bucket Complete noise measured on the pilot "
             "(results/v31_seed1, 2026-08-30) applies here too.")
    L.append("")
    for comp in payload["comparisons"]:
        L.append(f"## {comp['arm_a_label']} − {comp['arm_b_label']}")
        L.append("")
        L.append(f"`{comp['arm_a_path']}` minus `{comp['arm_b_path']}`.")
        L.append("")
        L.append("| Metric | better | Bucket | n | A | B | Δ (A−B) | 95 % CI | "
                 "CI excludes 0 | p (boot) |")
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
        L.append(f"**Interaction (H6 form)** — `{inter['formula']}`, {inter['n_rows']} rows, "
                 f"{inter['n_clusters']} voice clusters, R² = {_fmt(inter['r2'], 3)}.")
        L.append("")
        L.append("| Coefficient | estimate | 95 % CI | CI excludes 0 | p (boot) |")
        L.append("|---|---:|---|---|---:|")
        for nm, c in inter["coefficients"].items():
            mark = "**" if nm == inter["interaction_name"] else ""
            L.append(f"| {mark}{nm}{mark} | {_fmt(c['estimate'], 4)} | "
                     f"[{_fmt(c['ci_lo'], 4)}, {_fmt(c['ci_hi'], 4)}] | "
                     f"{'yes' if c['excludes_zero'] else 'no'} | {_fmt(c['p_boot'], 4)} |")
        L.append("")
    L.append("## Caveats")
    L.append("")
    L.append(f"- **{meta['n_clusters']} voice clusters** give a finer resample grid than "
             "the pilot's 6 documents, but each voice contributes at most 5 items and "
             "buckets are unevenly covered (not every voice has a B4 unit), so bucket "
             "cells resample fewer clusters than ALL cells.")
    L.append("- One seed, one generation per item: inference variance is NOT inside these "
             "intervals (the bootstrap resamples voices, not re-generations).")
    L.append("- Texts differ per voice by construction (each voice reads its own units), "
             "so voice and text are deliberately confounded within an arm; the paired "
             "contrast across arms is unaffected (same item under both checkpoints).")
    L.append("- The interaction fit is the same descriptive unbounded OLS as the pilot's "
             "(bootstrap.py §4.7 caveats apply).")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```")
    L.append("cd .")
    L.append(f".venv-eval/bin/python src/stats/bootstrap_robust.py --resamples "
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


def run(repo: Path, out_dir: Path, n_resamples: int, seed: int, alpha: float) -> dict:
    bench_path = repo / "data/benchmark/robust.jsonl"
    bench_rows = load_jsonl(bench_path)
    # CLUSTER = VOICE: compare() reads only rec["root_id"] from the bench mapping, so a
    # mapping whose root_id is the item's voice_id makes the identical estimator resample
    # voices.  The real benchmark fields are untouched on disk.
    bench_voice = {r["text_id"]: {"root_id": r["voice_id"]} for r in bench_rows}

    arms = {}
    for key, label, rel in ARM_SPECS_ROBUST:
        arms[key] = load_arm(key, label, repo / rel)
    ref_keys = sorted(arms[ARM_SPECS_ROBUST[0][0]].rows)
    for key, arm in arms.items():
        if sorted(arm.rows) != ref_keys:
            raise ValueError(f"arm {key} does not cover the same items as "
                             f"{ARM_SPECS_ROBUST[0][0]}")
    for k in ref_keys:
        if k[0] not in bench_voice:
            raise ValueError(f"text_id {k[0]} is not in {bench_path}")
        if k[1] != bench_voice[k[0]]["root_id"]:
            raise ValueError(f"{k[0]}: run voice {k[1]!r} != benchmark voice "
                             f"{bench_voice[k[0]]['root_id']!r} (E8 binding broken)")

    voices = sorted({bench_voice[k[0]]["root_id"] for k in ref_keys})
    idx_cache: dict[int, "np.ndarray"] = {}

    def idx_by_n(n_clusters: int):
        if n_clusters not in idx_cache:
            idx_cache[n_clusters] = resample_matrix(n_clusters, n_resamples, seed)
        return idx_cache[n_clusters]

    payload: dict = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "script": "src/stats/bootstrap_robust.py",
            "estimator": "src/stats/bootstrap.py (imported, unchanged)",
            "plan": "PLAN.md §12 machinery; E8 prereg reports/decisions.md 2026-08-31",
            "n_resamples": n_resamples,
            "seed": seed,
            "alpha": alpha,
            "ci": f"{int(100 * (1 - alpha))}% percentile",
            "cluster_unit": "voice_id",
            "n_clusters": len(voices),
            "clusters": voices,
            "pairing_key": list(ITEM_KEY_FIELDS),
            "seeds": sorted({k[2] for k in ref_keys}),
            "n_items_per_arm": len(ref_keys),
            "metrics": [m.name for m in METRICS],
            "conditioning": "unconditional — all statuses included (PLAN §0 rule 4)",
            "inputs": [
                {"path": str(p.relative_to(repo)), "rows": n, "sha256": sha256_of(p)}
                for p, n in [(bench_path, len(bench_rows))]
                + [(arms[k].path, arms[k].n) for k, _, _ in ARM_SPECS_ROBUST]
            ],
        },
        "comparisons": [],
    }
    for group, a_key, b_key in COMPARISONS_ROBUST:
        comp = compare_robust(arms[a_key], arms[b_key], bench_voice, idx_by_n, alpha)
        comp["group"] = group
        payload["comparisons"].append(comp)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{NAME}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8")
    write_csv(out_dir / f"{NAME}.csv", payload)
    (out_dir / f"{NAME}.md").write_text(render_markdown_robust(payload) + "\n",
                                        encoding="utf-8")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <repo>/results/v31_robust")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir is not None else repo / "results/v31_robust"
    payload = run(repo, Path(out_dir), args.resamples, args.seed, args.alpha)
    meta = payload["meta"]
    print(f"paired cluster bootstrap (clusters = voice): {meta['n_resamples']} resamples, "
          f"seed {meta['seed']}, {meta['n_clusters']} clusters, "
          f"{meta['n_items_per_arm']} paired items per arm")
    for comp in payload["comparisons"]:
        r = comp["cells"]["ALL"]["metrics"]
        cr, w = r["complete_rate"], r["wer"]
        print(f"  [{comp['group']:8s}] {comp['comparison']:9s} ALL  "
              f"Complete {100 * cr['delta']:+7.2f} pp "
              f"[{100 * cr['ci_lo']:+.2f}, {100 * cr['ci_hi']:+.2f}]  "
              f"WER {100 * w['delta']:+6.2f} pp [{100 * w['ci_lo']:+.2f}, "
              f"{100 * w['ci_hi']:+.2f}]")
    print(f"written: {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
