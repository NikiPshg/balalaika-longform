"""A21-deconfound: paired cluster bootstrap for the E3L format-match arm.

Same estimator as src/stats/bootstrap.py (imported, not copied): 10 000 resamples,
seed 0, 95 % percentile CI, cluster = root_id, pairing on (text_id, voice_id, seed),
every metric unconditional. Pre-registered in reports/decisions.md 2026-09-01
(A21-deconfound): the two deltas that split A20 claim #1 are

    E7  - E3L   how much of E7's win survives once E3 also gets a train-format-matched
                input (if ~0 -> the win was mostly format match);
    E3L - E3    what input-format match ALONE buys the unchanged E3 checkpoint
                (if ~0 -> punctuated supervision carries the signal itself).

E7 - E3 is recomputed as a context row; it must reproduce
results/v31_stats/paired_deltas_e6e7.md.

Usage::

    .venv-eval/bin/python src/stats/bootstrap_lenient.py            # -> results/v31_lenient/stats/
    .venv-eval/bin/python src/stats/bootstrap_lenient.py --resamples 1000   # smoke
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stats.bootstrap import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    ITEM_KEY_FIELDS,
    METRICS,
    REPO,
    compare,
    load_arm,
    load_jsonl,
    render_markdown,
    resample_matrix,
    sha256_of,
    write_csv,
)

ARM_SPECS = [
    ("E7", "E7 Punct-SFT step 3001 (punctuated input, matched format)",
     "results/v31_punct/E7_punct_epoch_3_step_3001/per_item.jsonl"),
    ("E3L", "E3L Long-SFT step 3001 on lenient input (matched format)",
     "results/v31_lenient/E3L_pilot/per_item.jsonl"),
    ("E3", "E3 Long-SFT step 3001 on punctuated input (mismatched format)",
     "results/v31_sft/E3_epoch_3_step_3001/per_item.jsonl"),
]

COMPARISONS = [
    ("primary", "E7", "E3L"),
    ("primary", "E3L", "E3"),
    ("secondary", "E7", "E3"),
]

GROUP_TITLES = {
    "primary": "1. Deconfound contrasts — format match split out of the E7 win",
    "secondary": "2. Context — the original confounded contrast, recomputed",
}
GROUP_INTROS = {
    "primary": (
        "`E7 − E3L` compares the two training recipes when EACH reads an input in its own "
        "training format (E7: punctuated `text_tts`; E3L: the same texts lenient-normalized to "
        "E3's ROVER training format, `data/benchmark/pilot_lenient.jsonl`, ё kept). "
        "`E3L − E3` is the SAME checkpoint (`exp/long/pilot/epoch_3_step_3001.pt`) on the two "
        "input formats — pure input-format effect, zero training difference. The WER reference "
        "(`text_ref`) is identical in both benchmark files, so all deltas are content deltas, "
        "not reference artifacts."
    ),
    "secondary": (
        "`E7 − E3` recomputed with this file's resample matrix; it must agree with "
        "`results/v31_stats/paired_deltas_e6e7.md` (same estimator, seed, cluster rule)."
    ),
}

SIGN_NOTE = (
    "One caution here: on `Repeat %` and `dur ratio` an arm that stops early scores "
    "favourably without being better; read them together with `Coverage`."
)


def run(repo: Path, out_dir: Path, n_resamples: int, seed: int, alpha: float) -> dict:
    bench_path = repo / "data/benchmark/pilot.jsonl"
    bench_lenient_path = repo / "data/benchmark/pilot_lenient.jsonl"
    bench = {r["text_id"]: r for r in load_jsonl(bench_path)}
    bench_l = {r["text_id"]: r for r in load_jsonl(bench_lenient_path)}
    if sorted(bench) != sorted(bench_l):
        raise ValueError("pilot.jsonl and pilot_lenient.jsonl do not share text_ids")
    for t in bench:
        if bench[t]["text_ref"] != bench_l[t]["text_ref"]:
            raise ValueError(f"text_ref differs for {t}: the WER reference must be identical")

    arms = {}
    for key, label, rel in ARM_SPECS:
        arms[key] = load_arm(key, label, repo / rel)
    first = ARM_SPECS[0][0]
    ref_keys = sorted(arms[first].rows)
    for key, arm in arms.items():
        if sorted(arm.rows) != ref_keys:
            raise ValueError(f"arm {key} does not cover the same items as {first}")
    for k in ref_keys:
        if k[0] not in bench:
            raise ValueError(f"text_id {k[0]} is not in {bench_path}")

    roots = sorted({bench[k[0]]["root_id"] for k in ref_keys})
    idx = resample_matrix(len(roots), n_resamples, seed)

    payload: dict = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "script": "src/stats/bootstrap_lenient.py",
            "plan": "PLAN.md §12; pre-registered reports/decisions.md 2026-09-01 (A21-deconfound)",
            "preset": "lenient",
            "title": (
                "# Paired E3L format-match deltas with 95 % CI "
                "(A20 claim #1 deconfound; pre-registered 2026-09-01)"
            ),
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
            "sign_note": SIGN_NOTE,
            "group_titles": GROUP_TITLES,
            "group_intros": GROUP_INTROS,
            "inputs": [
                {"path": str(p.relative_to(repo)), "rows": n, "sha256": sha256_of(p)}
                for p, n in [(bench_path, len(bench)), (bench_lenient_path, len(bench_l))]
                + [(arms[k].path, arms[k].n) for k, _, _ in ARM_SPECS]
            ],
        },
        "comparisons": [],
    }

    for group, a_key, b_key in COMPARISONS:
        comp = compare(arms[a_key], arms[b_key], bench, idx, alpha)
        comp["group"] = group
        payload["comparisons"].append(comp)

    name = "paired_deltas_lenient"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / f"{name}.csv", payload)
    (out_dir / f"{name}.md").write_text(render_markdown(payload) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <repo>/results/v31_lenient/stats")
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args(argv)
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir is not None else repo / "results/v31_lenient/stats"
    payload = run(repo, out_dir, args.resamples, args.seed, args.alpha)
    for comp in payload["comparisons"]:
        cell = comp["cells"]["ALL"]["metrics"]
        cr, wer = cell["complete_rate"], cell["wer"]
        print(
            f"{comp['comparison']}: ALL Complete {cr['delta']*100:+.2f} pp "
            f"[{cr['ci_lo']*100:+.2f}, {cr['ci_hi']*100:+.2f}] | "
            f"WER-all {wer['delta']*100:+.2f} pp [{wer['ci_lo']*100:+.2f}, {wer['ci_hi']*100:+.2f}]"
        )
    print(f"written: {out_dir}/paired_deltas_lenient.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
