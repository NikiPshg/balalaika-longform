#!/usr/bin/env python
"""Hidden-set feasibility measurement (reports/benchmark_composition.md §7.1).

The pilot's charset rule ("no digit and no unknown Latin token anywhere in the
selection window") is strict enough that it may not admit the 8 dataset roots in
8 distinct channels that the sealed hidden set needs.  This script *measures*
that instead of asserting it, over any manifest, for a frozen ladder of
relaxations:

    S1  the frozen pilot rule: no digits, Latin only from the config table
    S2  S1 + the extra brand names in `extra_latin_transliteration` below
    S3  S2 + single Latin letters treated as ASR junk and deleted
    S4  S3 + digits allowed and expanded by the frozen number rule
        (Roman numerals still disqualify: reading them needs case morphology)
    S5  no charset filter at all

Every variant also has to satisfy the non-charset part of the pilot recipe:
duration, asr_consistency, single speaker, word-group timestamps, a clean prefix
start (configs/benchmark_pilot.yaml: dataset_roots.prefix_start) and a sentence
boundary inside every bucket.  So the numbers are directly comparable with the
pilot build.

For S4 the script also measures the *cost* of allowing digits: the share of root
words that the frozen expansion produces, since those words come out in the
nominative masculine and therefore raise the human-audio ASR floor.

Run:

    python src/benchmark/feasibility.py                       # dev + test, stdout
    python src/benchmark/feasibility.py --json  reports/hidden_feasibility.json \
                                        --markdown reports/hidden_feasibility.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import disfluency as dis            # noqa: E402
from src.benchmark.build_pilot import (                # noqa: E402
    cut_points,
    find_prefix_start,
    latin_tokens,
    load_yaml,
    read_jsonl,
    word_groups,
)
from src.eval import normalize as norm                  # noqa: E402

# Frozen extra transliterations tested by S2 — brand and platform names that
# appear in this corpus and have an unambiguous Russian spelling.  They are NOT
# in configs/benchmark_pilot.yaml: only a variant that A0 adopts moves there.
EXTRA_LATIN: dict[str, str] = {
    "telegram": "телеграм",
    "whatsapp": "вотсап",
    "facebook": "фейсбук",
    "instagram": "инстаграм",
    "twitter": "твиттер",
    "tiktok": "тикток",
    "windows": "виндоус",
    "android": "андроид",
    "microsoft": "майкрософт",
    "apple": "эпл",
    "internet": "интернет",
    "wifi": "вайфай",
    "pdf": "пэдээф",
    "usb": "юэсби",
    "ok": "окей",
    "okay": "окей",
    "hello": "хэлло",
    "yandex": "яндекс",
    "vk": "вэка",
    "chatgpt": "чатджипити",
    "gpt": "джипити",
    "ai": "эйай",
}

ROMAN_RE = re.compile(r"\b[IVXLCDM]{2,}\b")
DIGIT_RUN_RE = re.compile(r"\d+")

VARIANTS = ["S1", "S2", "S3", "S4", "S5"]
VARIANT_LABEL = {
    "S1": "S1 — frozen pilot rule (no digits, Latin only from the config table)",
    "S2": "S2 — S1 + brand names in the transliteration table",
    "S3": "S3 — S2 + single Latin letters deleted as ASR junk",
    "S4": "S4 — S3 **and digits allowed**, expanded by the frozen number rule",
    "S5": "S5 — no charset filter at all",
}


def charset_verdict(text: str, table: set[str], variant: str) -> str:
    """"" when the window passes ``variant``, else the blocking reason."""
    if variant == "S5":
        return ""
    if ROMAN_RE.search(text):
        return "roman_numeral"
    latin = latin_tokens(text)
    if variant in ("S2", "S3", "S4"):
        latin = latin - set(EXTRA_LATIN)
    if variant in ("S3", "S4"):
        latin = {t for t in latin if len(t) > 1}
    latin = latin - table
    if latin:
        return "unknown_latin"
    if variant in ("S1", "S2", "S3") and DIGIT_RUN_RE.search(text):
        return "digits"
    return ""


def numeral_cost(text: str) -> tuple[int, int]:
    """(words produced by expanding every digit run, total words after expansion)."""
    produced = 0
    for m in DIGIT_RUN_RE.finditer(text):
        produced += len(norm.number_to_russian_words(int(m.group(0))).split())
    expanded = DIGIT_RUN_RE.sub(
        lambda m: " " + norm.number_to_russian_words(int(m.group(0))) + " ", text
    )
    return produced, len(dis.tokenize(expanded))


def scan_manifest(manifest: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Measure every variant over one manifest."""
    dcfg = cfg["dataset_roots"]
    buckets = cfg["buckets"]
    table = {k.lower() for k in (dcfg.get("latin_transliteration") or {})}
    window = float(dcfg["selection_window_sec"])

    rows = read_jsonl(manifest)
    stage: dict[str, int] = {"rows": len(rows)}
    per_variant: dict[str, dict[str, Any]] = {
        v: {"roots": [], "channels": set(), "reasons": {}} for v in VARIANTS
    }
    numeral_shares: list[float] = []

    for row in rows:
        if row["duration_sec"] < dcfg["min_duration_sec"]:
            continue
        stage["duration_ok"] = stage.get("duration_ok", 0) + 1
        if row["asr_consistency"] < dcfg["min_asr_consistency"]:
            continue
        if dcfg["require_single_speaker"] and not row["is_single_speaker"]:
            continue
        stage["quality_ok"] = stage.get("quality_ok", 0) + 1
        stage["quality_ok_channels"] = 0  # filled in below

        groups = word_groups(row["json_path"], dcfg["timestamp_source"])
        if not groups:
            continue
        stage["timestamps_ok"] = stage.get("timestamps_ok", 0) + 1
        started = find_prefix_start(groups, dcfg)
        if started is None:
            continue
        stage["clean_start_ok"] = stage.get("clean_start_ok", 0) + 1
        _start_index, start_time = started
        win = [g for g in groups if g[0] >= start_time and g[1] <= start_time + window]
        if not win:
            continue
        text = " ".join(g[2] for g in win)

        # a sentence boundary inside every bucket (charset-independent)
        cand = {
            "start_time": start_time,
            "groups": win,
            "pieces_clean": [g[2] for g in win],
        }
        if cut_points(cand, buckets) is None:
            continue
        stage["buckets_ok"] = stage.get("buckets_ok", 0) + 1

        # cost of S4, measured only on the roots S4 actually admits
        produced, total_words = numeral_cost(text)
        if produced and not charset_verdict(text, table, "S4"):
            numeral_shares.append(100.0 * produced / max(1, total_words))

        for v in VARIANTS:
            why = charset_verdict(text, table, v)
            slot = per_variant[v]
            if why:
                slot["reasons"][why] = slot["reasons"].get(why, 0) + 1
                continue
            slot["roots"].append(row["sample_id"])
            slot["channels"].add(row["channel_id"])

    # channels among the quality-filtered rows
    stage["quality_ok_channels"] = len(
        {
            r["channel_id"]
            for r in rows
            if r["duration_sec"] >= dcfg["min_duration_sec"]
            and r["asr_consistency"] >= dcfg["min_asr_consistency"]
            and (not dcfg["require_single_speaker"] or r["is_single_speaker"])
        }
    )

    numeral_shares.sort()
    n = len(numeral_shares)
    return {
        "manifest": str(manifest.relative_to(ROOT)) if manifest.is_relative_to(ROOT) else str(manifest),
        "stages": stage,
        "variants": {
            v: {
                "n_roots": len(per_variant[v]["roots"]),
                "n_channels": len(per_variant[v]["channels"]),
                "rejects": per_variant[v]["reasons"],
            }
            for v in VARIANTS
        },
        "numeral_expansion_cost_percent_of_root_words": {
            "n_roots_with_digits": n,
            "median": round(numeral_shares[n // 2], 3) if n else None,
            "mean": round(sum(numeral_shares) / n, 3) if n else None,
            "max": round(numeral_shares[-1], 3) if n else None,
        },
    }


def markdown(results: dict[str, dict[str, Any]], n_needed_channels: int) -> str:
    dev, test = results["dev"], results["test"]
    lines: list[str] = []
    ts, ds = test["stages"], dev["stages"]
    lines.append(
        f"`data/manifests/test.jsonl`: {ts['rows']} rows, **{ts.get('quality_ok', 0)}** are "
        f"≥ 480 s with `asr_consistency ≥ 90` and single-speaker, spread over "
        f"**{ts.get('quality_ok_channels', 0)}** channels; "
        f"**{ts.get('clean_start_ok', 0)}** of those also have a clean prefix start and "
        f"**{ts.get('buckets_ok', 0)}** a sentence boundary in every bucket. "
        f"(`dev.jsonl`: {ds['rows']} rows, {ds.get('quality_ok', 0)} quality-ok in "
        f"{ds.get('quality_ok_channels', 0)} channels, {ds.get('buckets_ok', 0)} bucket-ok.) "
        "Applying the charset filter to the selection window of each:"
    )
    lines.append("")
    lines.append("| variant of the rule | test roots | distinct test channels | dev roots | distinct dev channels |")
    lines.append("|---|---:|---:|---:|---:|")
    for v in VARIANTS:
        t, d = test["variants"][v], dev["variants"][v]
        tc = f"**{t['n_channels']}**" if v in ("S1", "S4") else str(t["n_channels"])
        lines.append(
            f"| {VARIANT_LABEL[v]} | {t['n_roots']} | {tc} | {d['n_roots']} | {d['n_channels']} |"
        )
    lines.append("")
    s1 = test["variants"]["S1"]
    s4 = test["variants"]["S4"]
    lines.append(
        f"The binding constraint is **digits, not Latin**: under S1 the test split loses "
        f"{s1['rejects'].get('digits', 0)} eligible roots to digits and "
        f"{s1['rejects'].get('unknown_latin', 0)} to unknown Latin, and Roman numerals "
        f"(`XIX`, `XX`, `XVIII` — centuries) block {s1['rejects'].get('roman_numeral', 0)} "
        "more; Roman numerals stay a disqualifier under every variant, because reading them "
        "correctly needs case morphology and deleting them would destroy meaning."
    )
    lines.append("")
    cost = test["numeral_expansion_cost_percent_of_root_words"]
    verdict = (
        "reaches" if s4["n_channels"] >= n_needed_channels else "still misses"
    )
    lines.append(
        f"**S4 {verdict} the {n_needed_channels} distinct test channels the hidden set needs "
        f"({s4['n_channels']} available).** Cost of allowing digits: the frozen expansion is "
        "nominative-masculine only, so `в 1990 году` becomes "
        "`в тысяча девятьсот "
        "девяносто году`. The same "
        "wrong form appears in `text_tts` *and* in `text_ref`, so the paired E1/E2/E3 comparison "
        "is untouched; only the human-audio ASR floor rises, exactly like the filler-removal "
        "mismatch of §4, and the floor is published next to every WER (PLAN §9.4). "
        f"Measured over the {cost['n_roots_with_digits']} test roots S4 admits that contain "
        f"digits, expanded numerals are **{cost['median']} % of root words at the median, "
        f"{cost['mean']} % on average, {cost['max']} % at the maximum**."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark_pilot.yaml"))
    ap.add_argument("--dev", default=str(ROOT / "data" / "manifests" / "dev.jsonl"))
    ap.add_argument("--test", default=str(ROOT / "data" / "manifests" / "test.jsonl"))
    ap.add_argument("--json", default=None, help="write the raw measurement here")
    ap.add_argument("--markdown", default=None, help="write the §7.1 fragment here")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config).expanduser().resolve())
    needed = int(cfg["hidden_set"]["n_roots"]) - 2 * int(cfg["pilot"]["n_external_roots"])
    results = {
        "dev": scan_manifest(Path(args.dev).expanduser().resolve(), cfg),
        "test": scan_manifest(Path(args.test).expanduser().resolve(), cfg),
    }
    results["n_dataset_roots_needed_for_hidden_set"] = needed
    md = markdown(results, needed)

    if args.json:
        Path(args.json).write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    if args.markdown:
        Path(args.markdown).write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
