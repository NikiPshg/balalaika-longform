#!/usr/bin/env python
"""Generate the measured numbers of ``reports/training_budget_comparison.md`` from files.

Why this exists
---------------
Two of the three blockers the M4 critic raised against ``reports/pilot_sft_results.md``
(2026-08-30) were *stale numbers copied out of this report's prose*: the short-arm token
cap (17 000, superseded by 20 700 before a single short step ran) and the dev-packing
counts (114 units / 1804 windows / 10.68 h / 960 727 tokens, superseded when the owner's
v3.1 dataset replaced the arms on 2026-08-29).  Both survived because they lived only as
hand-typed markdown.  Every block this module prints is re-derived from the artefacts on
disk, so the next rebuild makes the report wrong *loudly* (``--check`` fails) instead of
quietly.

Sources, all read-only
    exp/{long,short}/pilot/budget_verify_step200.json   the ±3 % budget match (§4.1)
    logs/v31_sft/budget_match.log                       check_budget_match.py's verdict
    configs/train/cv3_{long,short}_sft.yaml             the frozen token caps
    data/train/budget/*_summary.json                    the context budget table (§2)
    data/train/{long,short,dev_long,dev_short}/         stats.json + token_manifest.jsonl
                                                        + parquet_summary.json   (§3)
    logs/v31_curr/train_s*.log                          duration_filter epoch-end lines (§8)

Usage
    python src/training/budget_report_numbers.py --list
    python src/training/budget_report_numbers.py --block dev_packing
    python src/training/budget_report_numbers.py --all
    python src/training/budget_report_numbers.py --json
    python src/training/budget_report_numbers.py --check     # exit 1 on any disagreement

CPU only, stdlib only, writes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]

ARMS = ("long", "short", "dev_long", "dev_short")
CURRICULUM_STAGES = (("S1", 90.0), ("S2", 180.0), ("S3", 900.0))

# The values this report carried before 2026-08-30, kept so the footnotes can name what
# was superseded instead of silently dropping it.
SUPERSEDED_V1 = {
    "dev_long": {"utts": 114, "hours": 10.68, "tokens": 960_727},
    "dev_short": {"utts": 1804, "hours": 10.68, "tokens": 960_727},
    "long": {"utts": 4434, "hours": 218.23, "tokens": 19_623_778},
    "short": {"utts": 36_884, "hours": 217.56, "tokens": 19_563_811},
    "budget_rows": {"train_long": 4442, "train_short": 37_011, "dev": 114},
    "short_cap": 17_000,
}


# --------------------------------------------------------------------------- helpers


def _read_json(path: Path) -> dict:
    with open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input
                raise SystemExit(f"{path}:{i}: bad json: {exc}") from exc
    return rows


def _thousands(n: float | int, digits: int = 0) -> str:
    """1234567 -> '1 234 567' (non-breaking-free spaces, the report's convention)."""
    return f"{n:,.{digits}f}".replace(",", " ")


def _gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


# --------------------------------------------------------------------------- §4.1


def budget_match() -> dict:
    """The ±3 % budget match the Lead required on the first 200 pilot steps of both arms."""
    out: dict[str, Any] = {"steps": None, "tolerance": None, "arms": {}}
    for arm in ("long", "short"):
        path = REPO / f"exp/{arm}/pilot/budget_verify_step200.json"
        payload = _read_json(path)
        run = payload["runs"][0]
        out["steps"] = run["steps_used"]
        out["tolerance"] = payload["tolerance"]
        out["arms"][arm] = {
            "source": str(path.relative_to(REPO)),
            "steps_used": run["steps_used"],
            "first_step": run["first_step"],
            "last_step": run["last_step"],
            "mean": run["target_tokens_mean"],
            "median": run["target_tokens_median"],
            "min": run["target_tokens_min"],
            "max": run["target_tokens_max"],
            "stdev": run["target_tokens_stdev"],
            "total": run["target_tokens_total"],
            "samples_mean": run["samples_mean"],
            "padded_positions_max": run["padded_positions_max"],
            "step_time_sec_median": run["step_time_sec_median"],
        }
    long_mean = out["arms"]["long"]["mean"]
    short_mean = out["arms"]["short"]["mean"]
    out["mean_ratio"] = long_mean / short_mean
    out["deviation_from_equal"] = abs(out["mean_ratio"] - 1.0)
    out["within_tolerance"] = out["deviation_from_equal"] <= out["tolerance"]
    out["caps"] = token_caps()
    out["packing"] = {
        arm: out["arms"][arm]["mean"] / out["caps"][arm] for arm in ("long", "short")
    }
    out["log"] = _budget_match_log()
    out["full_run"] = _full_run_tokens()
    return out


def pilot_runtime() -> dict:
    """VRAM / padded positions / step time actually reached by the two pilot arms.

    §5 and §6 were written before the pilot ran and say the VRAM peak under the lowered
    28 672 guard "has NOT been re-measured"; it has, on 3 000 steps of each arm.
    """
    out: dict[str, Any] = {}
    for arm in ("long", "short"):
        path = REPO / f"exp/{arm}/pilot/train_stats.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl(path)
        times = sorted(r["step_time_sec"] for r in rows if r.get("step_time_sec") is not None)
        out[arm] = {
            "source": str(path.relative_to(REPO)),
            "steps": len(rows),
            "peak_vram_alloc_gb": max(r["peak_vram_alloc_gb"] for r in rows),
            "peak_vram_reserved_gb": max(r["peak_vram_reserved_gb"] for r in rows),
            "padded_positions_max": max(r["padded_positions"] for r in rows),
            "rss_gb_max": max(r["rss_gb"] for r in rows),
            "step_time_median": times[len(times) // 2] if times else None,
            "step_time_mean": (sum(times) / len(times)) if times else None,
            "wall_sec_total": sum(times) if times else None,
        }
    return out


def block_pilot_runtime(n: dict) -> str:
    pr = n["pilot_runtime"]
    lines = [
        "| arm | steps | peak VRAM alloc / reserved | padded positions max (guard 28 672) | "
        "RSS max | step time median / mean | train wall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (("long", "long (E3)"), ("short", "short (E2)")):
        r = pr.get(arm)
        if not r:
            continue
        lines.append(
            f"| {label} | {_thousands(r['steps'])} | {r['peak_vram_alloc_gb']:.1f} / "
            f"**{r['peak_vram_reserved_gb']:.1f}** GB | {_thousands(r['padded_positions_max'])} | "
            f"{r['rss_gb_max']:.1f} GB | {r['step_time_median']:.2f} s / "
            f"{r['step_time_mean']:.2f} s | {r['wall_sec_total'] / 3600.0:.2f} h |"
        )
    return "\n".join(lines)


def _full_run_tokens() -> dict:
    """What each arm actually saw over ALL its logged optimizer steps, not just the first 200.

    The ±3 % gate is defined on the first 200 steps (that is when a correction is still
    possible), but the *budget* claim of the paper is about the whole 3 000-step run, so the
    two must not be conflated: the first-200 mean is a gate, this is the outcome.
    """
    out: dict[str, Any] = {}
    for arm in ("long", "short"):
        path = REPO / f"exp/{arm}/pilot/train_stats.jsonl"
        if not path.exists():
            continue
        toks = [r["speech_tokens"] for r in _read_jsonl(path) if "speech_tokens" in r]
        if not toks:
            continue
        out[arm] = {
            "source": str(path.relative_to(REPO)),
            "steps": len(toks),
            "mean": sum(toks) / len(toks),
            "total": sum(toks),
        }
    if "long" in out and "short" in out:
        out["mean_ratio"] = out["long"]["mean"] / out["short"]["mean"]
        out["total_gap"] = out["short"]["total"] - out["long"]["total"]
        out["total_gap_share_of_long"] = out["total_gap"] / out["long"]["total"]
    return out


def token_caps() -> dict[str, int]:
    """The frozen ``max_speech_tokens_in_batch`` of each arm, read out of the yaml."""
    caps: dict[str, int] = {}
    for arm in ("long", "short"):
        path = REPO / f"configs/train/cv3_{arm}_sft.yaml"
        text = path.read_text(encoding="utf-8")
        hits = re.findall(r"^\s*max_speech_tokens_in_batch:\s*(\d+)", text, flags=re.M)
        if len(hits) != 1:
            raise SystemExit(f"{path}: expected one max_speech_tokens_in_batch, found {hits}")
        caps[arm] = int(hits[0])
    return caps


def _budget_match_log() -> dict:
    """check_budget_match.py's own verdict line, so the report quotes the gate, not a guess."""
    path = REPO / "logs/v31_sft/budget_match.log"
    if not path.exists():
        return {"path": None, "verdict": None, "mean_ratio": None}
    text = path.read_text(encoding="utf-8")
    ratio = re.search(r"mean ratio\s+([0-9.]+)", text)
    verdict = re.search(r"^RESULT:\s*(.+)$", text, flags=re.M)
    return {
        "path": str(path.relative_to(REPO)),
        "mean_ratio": float(ratio.group(1)) if ratio else None,
        "verdict": verdict.group(1).strip() if verdict else None,
    }


def block_budget_match(n: dict) -> str:
    b = n["budget_match"]
    lo, sh = b["arms"]["long"], b["arms"]["short"]
    lines = [
        "| arm | cap | steps | tokens/step mean | median | min / max | sd | samples/step | "
        "padded max | step time median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, r in (("long (E3)", lo), ("short (E2)", sh)):
        cap = b["caps"]["long" if arm.startswith("long") else "short"]
        lines.append(
            f"| {arm} | {_thousands(cap)} | {r['steps_used']} | **{_thousands(r['mean'], 1)}** | "
            f"{_thousands(r['median'], 1)} | {_thousands(r['min'])} / {_thousands(r['max'])} | "
            f"{_thousands(r['stdev'], 1)} | {r['samples_mean']:.2f} | "
            f"{_thousands(r['padded_positions_max'])} | {r['step_time_sec_median']:.2f} s |"
        )
    lines.append("")
    lines.append(
        f"Mean ratio long / short = **{b['mean_ratio']:.4f}** "
        f"({b['deviation_from_equal'] * 100:.2f} % from equal, tolerance "
        f"{b['tolerance'] * 100:.2f} %) → **{b['log']['verdict'] or 'PASS'}**. "
        f"Packing against the cap: long {b['packing']['long'] * 100:.0f} %, "
        f"short {b['packing']['short'] * 100:.0f} %."
    )
    fr = b.get("full_run") or {}
    if "long" in fr and "short" in fr:
        lines.append("")
        lines.append(
            f"Over the WHOLE pilot (all logged steps, `exp/*/pilot/train_stats.jsonl`): long "
            f"{fr['long']['steps']} steps × {_thousands(fr['long']['mean'], 1)} = "
            f"**{_thousands(fr['long']['total'])}** target tokens, short {fr['short']['steps']} × "
            f"{_thousands(fr['short']['mean'], 1)} = **{_thousands(fr['short']['total'])}** — "
            f"ratio {fr['mean_ratio']:.4f}, the short arm ahead by "
            f"{_thousands(fr['total_gap'])} tokens "
            f"({fr['total_gap_share_of_long'] * 100:.2f} % of the long arm's total)."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- §2


BUDGET_MANIFESTS = (
    ("train_long (≤ 15 min)", "train_long", "long"),
    ("train_short (10–30 s windows)", "train_short", "short"),
    ("dev (long units)", "dev", "dev_long"),
)


def context_budget() -> dict:
    out: dict[str, Any] = {}
    for label, stem, arm in BUDGET_MANIFESTS:
        s = _read_json(REPO / f"data/train/budget/{stem}_summary.json")
        stats = _read_json(REPO / f"data/train/{arm}/stats.json")
        out[stem] = {
            "label": label,
            "source": f"data/train/budget/{stem}_summary.json",
            "rows": s["rows"],
            "fit": s["fit"],
            "not_fit": s["not_fit"],
            "empty_text": s["empty_text"],
            "empty_text_hours": stats["excluded_hours"],
            "seq_max": s["seq_max"]["max"],
            "seq_p99": s["seq_max"]["p99"],
            "context_occupancy_max": s["context_occupancy_max"],
            "budget": s["budget"],
            "context_limit": s["context_limit"],
            "margin": s["margin"],
        }
    return out


def block_context_budget(n: dict) -> str:
    cb = n["context_budget"]
    lines = [
        "| manifest | rows | fit | not fit | empty text (listed, excluded explicitly) | "
        "max seq | p99 seq | max occupancy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, stem, _arm in BUDGET_MANIFESTS:
        r = cb[stem]
        lines.append(
            f"| {r['label']} | {_thousands(r['rows'])} | {_thousands(r['fit'])} | "
            f"{r['not_fit']} | {r['empty_text']} ({r['empty_text_hours']:.3f} h) | "
            f"{_thousands(r['seq_max'])} | {_thousands(r['seq_p99'])} | "
            f"{r['context_occupancy_max']:.2f} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- §3


def arms() -> dict:
    """Every arm as its own files describe it: stats.json + token_manifest.jsonl + parquet."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        d = REPO / "data/train" / arm
        stats = _read_json(d / "stats.json")
        rows = _read_jsonl(d / "token_manifest.jsonl")
        if len(rows) != stats["utts"]:
            raise SystemExit(
                f"{arm}: token_manifest.jsonl has {len(rows)} rows but stats.json says "
                f"{stats['utts']} utts"
            )
        # the parquet builder writes one summary per PAIR of arms (it slices both at once)
        summary_dir = "long" if arm in ("long", "short") else "dev_long"
        psum = _read_json(REPO / "data/train" / summary_dir / "parquet_summary.json")
        shard_bytes = psum["long_audio_bytes" if arm.endswith("long") else "short_audio_bytes"]
        out[arm] = {
            "dir": f"data/train/{arm}",
            "kind": stats["kind"],
            "utts": len(rows),
            "speakers": stats["speakers"],
            "parents": stats["parents"],
            "hours": sum(r["duration"] for r in rows) / 3600.0,
            "hours_stats_json": stats["hours"],
            "target_speech_tokens": sum(r["n_speech_token"] for r in rows),
            "max_tokens_per_utt": max(r["n_speech_token"] for r in rows),
            "excluded": stats["excluded"],
            "excluded_hours": stats["excluded_hours"],
            "shards": psum["shards"],
            "parquet_bytes": shard_bytes,
        }
    out["dev_tokens_equal"] = (
        out["dev_long"]["target_speech_tokens"] == out["dev_short"]["target_speech_tokens"]
    )
    out["dev_hours_equal_to_1e6"] = (
        abs(out["dev_long"]["hours"] - out["dev_short"]["hours"]) < 1e-6
    )
    out["train_token_gap"] = (
        out["long"]["target_speech_tokens"] - out["short"]["target_speech_tokens"]
    )
    out["train_token_gap_share"] = (
        out["train_token_gap"] / out["long"]["target_speech_tokens"]
    )
    return out


def block_arms(n: dict) -> str:
    a = n["arms"]
    lines = [
        "| arm | utts | speakers | hours | target speech tokens (exact, tokenizer) | "
        "max tokens/utt | shards | parquet size |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, label in (
        ("long", "long (train)"),
        ("short", "short (train)"),
        ("dev_long", "dev_long"),
        ("dev_short", "dev_short"),
    ):
        r = a[arm]
        lines.append(
            f"| {label} | {_thousands(r['utts'])} | {r['speakers']} | {r['hours']:.3f} | "
            f"{_thousands(r['target_speech_tokens'])} | {_thousands(r['max_tokens_per_utt'])} | "
            f"{r['shards']} | {_gb(r['parquet_bytes'])} |"
        )
    return "\n".join(lines)


def block_dev_packing(n: dict) -> str:
    """The two dev rows on their own — the exact lines the Lead told A6 to replace."""
    a = n["arms"]
    lines = [
        "| arm | utts | speakers | parents | hours | target speech tokens | max tokens/utt | shards |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("dev_long", "dev_short"):
        r = a[arm]
        lines.append(
            f"| {arm} | {_thousands(r['utts'])} | {r['speakers']} | {r['parents']} | "
            f"{r['hours']:.3f} | {_thousands(r['target_speech_tokens'])} | "
            f"{_thousands(r['max_tokens_per_utt'])} | {r['shards']} |"
        )
    lines.append("")
    lines.append(
        f"Token equality on dev: {_thousands(a['dev_long']['target_speech_tokens'])} = "
        f"{_thousands(a['dev_short']['target_speech_tokens'])} "
        f"({'exact' if a['dev_tokens_equal'] else 'NOT EQUAL'}), same "
        f"{a['dev_long']['parents']} parents and {a['dev_long']['speakers']} speakers, "
        f"hours equal to 1e-6: {a['dev_hours_equal_to_1e6']}."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- §8


EPOCH_END_RE = re.compile(
    r"duration_filter\[pid (?P<pid>\d+)\] epoch end "
    r"\(epoch #(?P<epoch>\d+), mode=(?P<mode>\w+), active=(?P<active>\w+), "
    r"ceiling max=(?P<ceiling>[\w.]+) s min=\S+ s\): "
    r"kept (?P<kept>\d+)/(?P<seen>\d+) rows \((?P<kept_h>[\d.]+) h\), "
    r"dropped (?P<dropped>\d+)/(?P<seen2>\d+) rows \((?P<drop_h>[\d.]+) h"
)


def curriculum_expected() -> dict:
    """What each ceiling keeps, re-derived from data/train/long/token_manifest.jsonl."""
    rows = _read_jsonl(REPO / "data/train/long/token_manifest.jsonl")
    total_h = sum(r["duration"] for r in rows) / 3600.0
    total_tok = sum(r["n_speech_token"] for r in rows)
    out: dict[str, Any] = {
        "source": "data/train/long/token_manifest.jsonl",
        "arm_units": len(rows),
        "arm_hours": total_h,
        "arm_tokens": total_tok,
        "arm_max_duration_sec": max(r["duration"] for r in rows),
        "stages": {},
    }
    for stage, ceiling in CURRICULUM_STAGES:
        kept = [r for r in rows if r["duration"] <= ceiling]
        dropped = [r for r in rows if r["duration"] > ceiling]
        out["stages"][stage] = {
            "ceiling_sec": ceiling,
            "units": len(kept),
            "units_share": len(kept) / len(rows),
            "hours": sum(r["duration"] for r in kept) / 3600.0,
            "tokens": sum(r["n_speech_token"] for r in kept),
            "tokens_share": sum(r["n_speech_token"] for r in kept) / total_tok,
            "units_dropped": len(dropped),
            "hours_dropped": sum(r["duration"] for r in dropped) / 3600.0,
        }
    return out


def _group_epochs(lines: list[dict], expected_seen: int) -> list[dict]:
    """Sum the per-worker epoch-end lines back into whole epochs.

    ``duration_filter`` writes one epoch-end line per dataloader WORKER PROCESS, and each
    worker only sees its own half of the shards, so a stage epoch is the SUM over the
    workers.  The worker pids are new every epoch and every line says ``epoch #1`` (it is
    that process's first epoch), so neither can group the lines.  What can: each line
    reports how many rows that worker SAW, and the workers of one epoch partition the arm
    exactly once -- so a group closes when the running ``seen`` total reaches the arm size.
    A group that overshoots means the partition assumption broke and is reported, never
    silently folded into the next epoch.
    """
    groups: list[dict] = []
    bucket: list[dict] = []
    running = 0
    for rec in lines:
        bucket.append(rec)
        running += rec["seen"]
        if running >= expected_seen:
            g = _sum_epoch(bucket)
            g["partition_ok"] = running == expected_seen
            g["seen_expected"] = expected_seen
            groups.append(g)
            bucket, running = [], 0
    if bucket:  # a stage still running: the tail is an incomplete epoch, kept and marked
        g = _sum_epoch(bucket)
        g["partition_ok"] = False
        g["seen_expected"] = expected_seen
        g["incomplete"] = True
        groups.append(g)
    return groups


def curriculum_observed(log_dir: str = "logs/v31_curr") -> dict:
    """What the running E4 chain actually logged, summed over the dataloader workers.

    ``duration_filter`` writes one epoch-end line per worker process per epoch, so an epoch
    of a stage is the SUM over the workers of that epoch; the workers split the shards, and
    the split differs between epochs (shuffle), which is why only the sum is meaningful.
    """
    root = REPO / log_dir
    out: dict[str, Any] = {"log_dir": log_dir, "stages": {}, "dev": {}}
    if not root.is_dir():
        out["present"] = False
        return out
    out["present"] = True
    n_train_rows = len(_read_jsonl(REPO / "data/train/long/token_manifest.jsonl"))
    n_dev_rows = _read_json(REPO / "data/train/dev_long/stats.json")["utts"]
    out["expected_seen_train"] = n_train_rows
    out["expected_seen_dev"] = n_dev_rows
    for stage, ceiling in CURRICULUM_STAGES:
        path = root / f"train_s{stage[-1]}.log"
        if not path.exists():
            out["stages"][stage] = {"log": None, "epochs": []}
            continue
        train_lines: list[dict] = []
        dev_lines: list[dict] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = EPOCH_END_RE.search(line)
            if not m:
                continue
            rec = {
                "pid": int(m["pid"]),
                "epoch": int(m["epoch"]),
                "mode": m["mode"],
                "active": m["active"] == "True",
                "ceiling_sec": float(m["ceiling"]),
                "kept": int(m["kept"]),
                "seen": int(m["seen"]),
                "kept_hours": float(m["kept_h"]),
                "dropped": int(m["dropped"]),
                "dropped_hours": float(m["drop_h"]),
                "ts": line.split(" INFO")[0].strip(),
            }
            (train_lines if rec["mode"] == "train" else dev_lines).append(rec)
        epochs = _group_epochs(train_lines, n_train_rows)
        whole = [e for e in epochs if e["partition_ok"]]
        out["stages"][stage] = {
            "log": str(path.relative_to(REPO)),
            "ceiling_sec": ceiling,
            "epochs": epochs,
            "n_epochs_logged": len(whole),
            "n_partial_epochs": len(epochs) - len(whole),
            "workers_per_epoch": sorted({e["workers"] for e in whole}) if whole else [],
            "kept_set": sorted({e["kept"] for e in whole}),
            "dropped_set": sorted({e["dropped"] for e in whole}),
            "kept_hours_set": sorted({round(e["kept_hours"], 2) for e in whole}),
            "last_ts": train_lines[-1]["ts"] if train_lines else None,
        }
        if dev_lines:
            dev_epochs = _group_epochs(dev_lines, n_dev_rows)
            whole_dev = [e for e in dev_epochs if e["partition_ok"]]
            out["dev"][stage] = {
                "units_set": sorted({e["kept"] for e in whole_dev}),
                "dropped_set": sorted({e["dropped"] for e in whole_dev}),
                "hours_set": sorted({round(e["kept_hours"], 2) for e in whole_dev}),
                "active_set": sorted({e["active"] for e in whole_dev}),
                "n_epochs_logged": len(whole_dev),
                "n_partial_epochs": len(dev_epochs) - len(whole_dev),
            }
    return out


def _sum_epoch(bucket: Iterable[dict]) -> dict:
    b = list(bucket)
    return {
        "workers": len(b),
        "pids": sorted(r["pid"] for r in b),
        "kept": sum(r["kept"] for r in b),
        "seen": sum(r["seen"] for r in b),
        "dropped": sum(r["dropped"] for r in b),
        "kept_hours": sum(r["kept_hours"] for r in b),
        "dropped_hours": sum(r["dropped_hours"] for r in b),
        "active": all(r["active"] for r in b),
        "ts": b[-1]["ts"],
    }


def curriculum_check(expected: dict, observed: dict) -> dict:
    """Every logged epoch of every stage must reproduce the manifest-derived kept/dropped."""
    problems: list[str] = []
    checked = 0
    for stage, _ceiling in CURRICULUM_STAGES:
        exp = expected["stages"][stage]
        obs = observed.get("stages", {}).get(stage)
        if not obs or not obs.get("epochs"):
            problems.append(f"{stage}: no epoch-end lines logged yet")
            continue
        for e in obs["epochs"]:
            if not e["partition_ok"]:
                # a stage still running mid-epoch: not a mismatch, but never counted as agreement
                continue
            checked += 2
            if e["kept"] != exp["units"]:
                problems.append(
                    f"{stage} epoch@{e['ts']}: log kept {e['kept']} vs manifest {exp['units']}"
                )
            if e["dropped"] != exp["units_dropped"]:
                problems.append(
                    f"{stage} epoch@{e['ts']}: log dropped {e['dropped']} vs manifest "
                    f"{exp['units_dropped']}"
                )
    return {"checked": checked, "problems": problems, "ok": not problems}


def dev_check(arms_: dict, observed: dict) -> dict:
    """The E4 cv pass must see the whole (unfiltered) dev set: 120 units, active=False."""
    problems: list[str] = []
    checked = 0
    want = arms_["dev_long"]["utts"]
    for stage, _ceiling in CURRICULUM_STAGES:
        d = observed.get("dev", {}).get(stage)
        if not d:
            continue
        checked += 3
        if d["units_set"] != [want]:
            problems.append(f"{stage} dev: logged {d['units_set']} units, dev_long has {want}")
        if d["dropped_set"] != [0]:
            problems.append(f"{stage} dev: dropped {d['dropped_set']}, must be [0] (PLAN §11.5)")
        if d["active_set"] != [False]:
            problems.append(f"{stage} dev: active={d['active_set']}, the cv pass must not filter")
    return {"checked": checked, "problems": problems, "ok": not problems}


def block_curriculum(n: dict) -> str:
    exp = n["curriculum_expected"]
    obs = n["curriculum_observed"]
    lines = [
        "| stage | ceiling | units | of arm | hours | target speech tokens | of arm | "
        "units dropped | hours dropped | epochs logged | kept / dropped in the log |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for stage, _ceiling in CURRICULUM_STAGES:
        e = exp["stages"][stage]
        o = obs.get("stages", {}).get(stage) or {}
        if o.get("epochs"):
            kept = "/".join(str(x) for x in o["kept_set"])
            dropped = "/".join(str(x) for x in o["dropped_set"])
            logged = f"**{kept} / {dropped}**"
            n_ep = str(o["n_epochs_logged"])
        else:
            logged, n_ep = "not logged yet", "0"
        lines.append(
            f"| {stage} | ≤ {int(e['ceiling_sec'])} s | {_thousands(e['units'])} | "
            f"{e['units_share'] * 100:.1f} % | {e['hours']:.2f} | {_thousands(e['tokens'])} | "
            f"{e['tokens_share'] * 100:.1f} % | {_thousands(e['units_dropped'])} | "
            f"{e['hours_dropped']:.2f} | {n_ep} | {logged} |"
        )
    lines.append("")
    lines.append(
        f"Whole long arm: {_thousands(exp['arm_units'])} units / {exp['arm_hours']:.2f} h / "
        f"{_thousands(exp['arm_tokens'])} target tokens; longest unit "
        f"{exp['arm_max_duration_sec']:.1f} s."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- driver


BLOCKS = {
    "budget_match": ("§4.1 measured ±3 % budget match", block_budget_match),
    "context_budget": ("§2 context budget table", block_context_budget),
    "arms": ("§3 arms table", block_arms),
    "dev_packing": ("§3/§6/§8 dev packing", block_dev_packing),
    "curriculum": ("§8 E4 curriculum stage table", block_curriculum),
    "pilot_runtime": ("§5/§6 measured pilot VRAM, packing and wall time", block_pilot_runtime),
}


def numbers() -> dict:
    n: dict[str, Any] = {
        "budget_match": budget_match(),
        "context_budget": context_budget(),
        "arms": arms(),
        "pilot_runtime": pilot_runtime(),
        "curriculum_expected": curriculum_expected(),
        "curriculum_observed": curriculum_observed(),
        "superseded_v1": SUPERSEDED_V1,
    }
    n["checks"] = {
        "curriculum": curriculum_check(n["curriculum_expected"], n["curriculum_observed"]),
        "dev": dev_check(n["arms"], n["curriculum_observed"]),
    }
    return n


REPORT = "reports/training_budget_comparison.md"
# ``dev_packing`` is a convenience view of the two rows the Lead's 2026-08-30 ruling names;
# its numbers reach the report inside the §3 ``arms`` table and its prose, so it is the one
# block that does not carry its own marker.
REPORT_BLOCKS = ("context_budget", "arms", "budget_match", "pilot_runtime", "curriculum")
MARKER_RE = re.compile(r"<!--\s*GENERATED:\s*\S+\s+--block\s+(?P<block>\w+)\s*-->")


def verify_report(report: str = REPORT) -> dict:
    """Every ``<!-- GENERATED: ... --block X -->`` table in the report must still match X.

    Only the markdown TABLE is compared (the contiguous run of ``|`` lines after the marker):
    the prose under a block is written by hand and is allowed to differ.  This is the check
    that would have caught both M4 blockers the day the dataset was rebuilt.
    """
    path = REPO / report
    out: dict[str, Any] = {"report": report, "blocks": {}, "problems": [], "checked": 0}
    if not path.exists():
        out["problems"].append(f"{report}: not found")
        out["ok"] = False
        return out
    lines = path.read_text(encoding="utf-8").splitlines()
    n = numbers()
    for i, line in enumerate(lines):
        m = MARKER_RE.search(line)
        if not m:
            continue
        name = m["block"]
        if name not in BLOCKS:
            out["problems"].append(f"line {i + 1}: unknown block {name!r}")
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        found = []
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            found.append(lines[j].rstrip())
            j += 1
        want = [
            ln.rstrip() for ln in BLOCKS[name][1](n).splitlines() if ln.lstrip().startswith("|")
        ]
        out["blocks"][name] = {"line": i + 1, "rows_in_report": len(found)}
        out["checked"] += max(len(found), len(want))
        if found != want:
            for k in range(max(len(found), len(want))):
                a = found[k] if k < len(found) else "<missing in report>"
                b = want[k] if k < len(want) else "<not generated>"
                if a != b:
                    out["problems"].append(f"{name} (line {i + 1}) row {k}:\n    report: {a}\n    fresh:  {b}")
    for name in REPORT_BLOCKS:
        if name not in out["blocks"]:
            out["problems"].append(f"{name}: no GENERATED marker in {report}")
    out["ok"] = not out["problems"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--block", choices=sorted(BLOCKS), help="print one markdown block")
    ap.add_argument("--all", action="store_true", help="print every markdown block")
    ap.add_argument("--json", action="store_true", help="print every number as JSON")
    ap.add_argument("--list", action="store_true", help="list the block names")
    ap.add_argument("--check", action="store_true",
                    help="cross-check the E4 logs against the manifests; exit 1 on any gap")
    ap.add_argument("--verify-report", action="store_true",
                    help=f"check every GENERATED table in {REPORT} against a fresh run")
    args = ap.parse_args(argv)

    if args.list:
        for name, (desc, _fn) in sorted(BLOCKS.items()):
            print(f"{name:18s} {desc}")
        return 0

    n = numbers()

    if args.json:
        print(json.dumps(n, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.verify_report:
        res = verify_report()
        print(f"[{'OK' if res['ok'] else 'FAILED'}] {res['report']}: "
              f"{len(res['blocks'])} generated table(s), {res['checked']} row comparison(s), "
              f"{len(res['problems'])} problem(s)")
        for p in res["problems"]:
            print(f"    {p}")
        return 0 if res["ok"] else 1

    if args.check:
        rc = 0
        for name, res in n["checks"].items():
            status = "OK" if res["ok"] else "FAILED"
            print(f"[{status}] {name}: {res['checked']} comparisons, "
                  f"{len(res['problems'])} problem(s)")
            for p in res["problems"]:
                print(f"    {p}")
            if not res["ok"]:
                rc = 1
        return rc

    if args.block:
        print(BLOCKS[args.block][1](n))
        return 0

    if args.all:
        for name, (desc, fn) in sorted(BLOCKS.items()):
            print(f"<!-- {name}: {desc} -->")
            print(fn(n))
            print()
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
