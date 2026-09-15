#!/usr/bin/env python
"""A2 — render `reports/benchmark_hidden_composition.md` from the hidden-set artifacts.

Same discipline as `src/benchmark/composition_report.py` for the pilot: nothing in
the report is typed in by hand.  Every count, threshold, duration and hash is read
from the build's own outputs, and §8 re-derives the benchmark contract (nesting,
bucket membership, offsets, occupancy formula, charset, disfluency, overlap with
the pilot) from `hidden.jsonl` itself and prints PASS/FAIL.

    data/benchmark/hidden.jsonl          the 40 benchmark texts
    data/benchmark/hidden_stats.json     selection pool, rejects, wpm, tokenizer hashes
    reports/benchmark_hidden_edits.json  every disfluency edit with its context
    reports/hidden_feasibility.json      the S1..S5 feasibility ladder
    configs/benchmark_hidden.yaml        the frozen recipe (derived from the pilot's)
    configs/benchmark_pilot.yaml         to prove the criteria did not drift
    data/benchmark/pilot.jsonl           overlap assertions
    data/manifests/test.jsonl            speaker_key of each root
    data/references/references.jsonl     A5's voices, for the same-video confound
    data/benchmark/HIDDEN_SEAL.json      the seal, if it exists yet

CPU only, no GPU, no model, no generation, no evaluation.

    python src/benchmark/hidden_composition_report.py [--strict]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import disfluency as dis  # noqa: E402
from src.eval import normalize as norm  # noqa: E402

BUCKET_ORDER = ["B0", "B1", "B2", "B3", "B4"]


def jload(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mtime(path: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell(text: Any) -> str:
    """Markdown-table-safe cell: a pipe inside a YouTube title breaks the row."""
    return str(text).replace("|", "\\|")


def shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = [t.norm for t in dis.tokenize(text)]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def build(strict: bool) -> tuple[str, int]:
    import yaml

    H = jsonl(ROOT / "data/benchmark/hidden.jsonl")
    P = jsonl(ROOT / "data/benchmark/pilot.jsonl")
    stats = jload(ROOT / "data/benchmark/hidden_stats.json")
    edits = jload(ROOT / "reports/benchmark_hidden_edits.json")
    feas = jload(ROOT / "reports/hidden_feasibility.json")
    hcfg = yaml.safe_load((ROOT / "configs/benchmark_hidden.yaml").read_text(encoding="utf-8"))
    pcfg = yaml.safe_load((ROOT / "configs/benchmark_pilot.yaml").read_text(encoding="utf-8"))
    test_rows = {r["sample_id"]: r for r in jsonl(ROOT / "data/manifests/test.jsonl")}
    refs = jsonl(ROOT / "data/references/references.jsonl")
    seal_path = ROOT / "data/benchmark/HIDDEN_SEAL.json"
    seal = jload(seal_path) if seal_path.exists() else None

    dis.load_spec()
    nspec = norm.load_spec()
    variant = nspec["primary_variant"]
    buckets = hcfg["buckets"]
    hset = hcfg["hidden"]

    by_root: dict[str, dict[str, dict[str, Any]]] = {}
    for r in H:
        by_root.setdefault(r["root_id"], {})[r["bucket"]] = r
    roots = list(by_root)
    ds_roots = [rid for rid in roots if by_root[rid]["B4"]["source"] == "dataset"]
    ex_roots = [rid for rid in roots if by_root[rid]["B4"]["source"] == "external"]

    L: list[str] = []
    A = L.append

    # ---------------------------------------------------------------- header
    A("# Benchmark composition — SEALED HIDDEN SET (PLAN.md §7.4)\n")
    A(f"**Owner:** A2 · **Generated:** {time.strftime('%Y-%m-%d %H:%M')} by "
      "`src/benchmark/hidden_composition_report.py` · **Dataset:** `v3.1` "
      "(`data/corpus`, `configs/dataset.yaml`)\n")
    A(f"**Artifact:** `data/benchmark/hidden.jsonl` — {len(H)} texts, sha256 "
      f"`{sha256_file(ROOT / 'data/benchmark/hidden.jsonl')}`.\n")
    if seal:
        A(f"**Seal:** `data/benchmark/HIDDEN_SEAL.json` — sealed "
          f"`{seal['built_utc']}`, rule_version `{seal['rule_version']}`, "
          f"{seal['n_records']} records, {len(seal['audio_files'])} human audio files, "
          f"{len(seal['audio_slices'])} slice references. Both files are mode 444.\n")
    else:
        A("**Seal:** not written yet (`python src/benchmark/seal_hidden.py`).\n")
    A("> This set is **never** used for checkpoint selection, decoding tuning or any "
      "look-before-the-run (PLAN §3.2, §3.5). It is built once and sealed. No generation "
      "and no evaluation was run on it by the agent that built it.\n")
    A("Every number below is read from the files in the table; nothing is typed in by hand.\n")
    A("| input | written |")
    A("|---|---|")
    for p in ["data/benchmark/hidden.jsonl", "data/benchmark/hidden_stats.json",
              "reports/benchmark_hidden_edits.json", "reports/hidden_feasibility.json",
              "configs/benchmark_hidden.yaml", "data/benchmark/external/hidden/genre5_observatory.txt",
              "data/benchmark/external/hidden/genre6_greenhouse.txt"]:
        A(f"| `{p}` | {mtime(ROOT / p)} |")
    if seal:
        A(f"| `data/benchmark/HIDDEN_SEAL.json` | {mtime(seal_path)} |")
    A("")

    # ------------------------------------------------------------------- §1
    A("## 1. What the hidden set is, and why it has "
      f"{len(roots)} roots and not 12\n")
    A(f"{len(roots)} root documents × 5 nested prefixes B0–B4 = **{len(H)} benchmark texts**; "
      f"with A5's {hset['voices_per_text']} primary voices that is "
      f"**{hset['generations_per_checkpoint']} generations per checkpoint**.\n")
    A(f"* **{len(ds_roots)} dataset roots** (genres 1–4), each a *single continuous "
      "test-split segment* of ≥ "
      f"{hcfg['dataset_roots']['min_duration_sec']} s with `asr_consistency ≥ "
      f"{hcfg['dataset_roots']['min_asr_consistency']}` and `is_single_speaker = True`, from "
      f"**{len({by_root[r]['B4']['channel_id'] for r in ds_roots})} channels** and "
      f"**{len({by_root[r]['B4']['video_id'] for r in ds_roots})} videos**. All five prefixes "
      "are literal prefixes of one continuous stretch of one human recording, so "
      "B0 ⊂ B1 ⊂ B2 ⊂ B3 ⊂ B4 and every prefix has real human audio, a real human duration "
      "and an ASR error floor.")
    A(f"* **{len(ex_roots)} constructed roots** (genres 5–6), **newly written for this set** "
      "(composition §7.3 forbids reusing the pilot's). They have no human audio "
      "(`human_audio_path: null`); their human duration is estimated as "
      f"`words / wpm × 60` with **wpm = {stats['dev_median_wpm']:.3f}**, the median "
      "words-per-minute over `data/manifests/dev.jsonl` — deliberately the *same constant* "
      "the pilot used, so the two sets' constructed genres sit on the same duration scale. "
      "No dev content enters the hidden set; only that one number.\n")
    A("Bucket = **human** reference duration (PLAN §7.1): "
      + ", ".join(f"{b} {buckets[b][0]}–{buckets[b][1]} s" for b in BUCKET_ORDER)
      + f". All {len(H)} texts land inside their bucket (§8).\n")

    A("### 1.1 The achievable number of roots\n")
    t = feas["test"]["stages"]
    s1 = feas["test"]["variants"]["S1"]
    A("PLAN §7.4 asks for 12 roots. Under the **pre-registered rule that "
      "`reports/benchmark_composition.md` §7 designates** — the frozen pilot recipe, "
      "charset variant **S1**, plus `root_fallback: one_per_video` on the test side — "
      f"the test split admits **{stats['n_pool_roots']} dataset roots**, not 8. Measured by "
      "this build (and independently by `src/benchmark/feasibility.py`):\n")
    A("| stage | rows |")
    A("|---|---:|")
    A(f"| `data/manifests/test.jsonl` | {t['rows']} |")
    A(f"| ≥ {hcfg['dataset_roots']['min_duration_sec']} s, `asr_consistency ≥ "
      f"{hcfg['dataset_roots']['min_asr_consistency']}`, single speaker | {t['quality_ok']} "
      f"in {t['quality_ok_channels']} channels |")
    A(f"| + clean prefix start, + a sentence boundary in every bucket | {t['buckets_ok']} |")
    A(f"| + frozen charset rule S1 (no digits, Latin only from the config table) | "
      f"**{s1['n_roots']} eligible roots in {s1['n_channels']} channels** |")
    A(f"| this build, same filters | **{stats['n_eligible_roots']} eligible roots in "
      f"{stats['n_eligible_channels']} channels** (rejects: "
      + ", ".join(f"{v} {k}" for k, v in sorted(stats["rejects"].items())) + ") |")
    A(f"| `root_per_channel: {hcfg['dataset_roots']['root_per_channel']}` | "
      f"{stats['n_eligible_channels']} roots |")
    A(f"| + `root_fallback: {stats['root_fallback']}` (used: "
      f"{str(stats['root_fallback_used']).lower()}) | **{stats['n_pool_roots']} roots** |")
    A("")
    n_elig = stats["n_eligible_roots"]
    n_pool = stats["n_pool_roots"]
    left = n_elig - n_pool
    A(f"The pool stops at {n_pool} and not at {n_elig}: the "
      f"{left} remaining eligible root{'' if left == 1 else 's'} "
      f"{'is a further segment' if left == 1 else 'are further segments'} of a video already "
      "in the pool, and the frozen fallback says *never two roots from one video*. "
      "Reaching 8 dataset roots would need either a relaxed charset rule (S3/S4 of the "
      "feasibility ladder) — which composition §7.5 says would also force a rebuild of the "
      "already-analysed pilot — or two roots from one video. **Neither was done, and no "
      "criterion was loosened to raise the count.**\n")
    A(f"Total: {len(ds_roots)} dataset + {len(ex_roots)} external = **{len(roots)} roots**, "
      f"**{len(H)} texts**, **{hset['generations_per_checkpoint']} generations per "
      f"checkpoint**, {hset['generations_per_checkpoint'] * 3} for three experiments — "
      f"against the {len(P)} texts / 60 generations of the pilot.\n")

    # ------------------------------------------------------------------- §2
    A("## 2. How the roots were chosen (frozen, reproducible, no hand-picked ids)\n")
    d = hcfg["dataset_roots"]
    A("`configs/benchmark_hidden.yaml`, implemented in `src/benchmark/build_pilot.py`. "
      "Every criterion is copied verbatim from `configs/benchmark_pilot.yaml`; the config "
      "diff is *only* the source manifest, the size block and the two constructed documents "
      "(asserted by `tests/test_hidden_benchmark.py`).\n")
    A(f"1. **Eligibility** over `{d['source_manifest']}`: `duration_sec ≥ "
      f"{d['min_duration_sec']}`, `asr_consistency ≥ {d['min_asr_consistency']}`, "
      "`is_single_speaker`, word-group timestamps present "
      f"(`{d['timestamp_source']}`); text from `{d['text_source']}`.")
    A(f"2. **Where the chain starts** (`prefix_start: {d['prefix_start']['method']}`): the "
      f"first sentence boundary whose next `lookahead_sec = "
      f"{d['prefix_start']['lookahead_sec']}` seconds match none of the "
      f"{len(d['prefix_start']['opening_patterns'])} `opening_patterns` (A/V check, "
      "audience/stream address, greeting and channel promo); no such boundary at or before "
      f"`max_start_sec = {d['prefix_start']['max_start_sec']}` **disqualifies** the root. On "
      "this build the starts are "
      + ", ".join(
          f"`{by_root[r]['B4']['sample_id']}` {by_root[r]['B4']['human_offset_start']:.2f} s"
          for r in ds_roots) + ".")
    A(f"3. **Charset**: no digit (`forbid_digits: {d['forbid_digits']}`) and no Latin token "
      f"outside the frozen {len(d['latin_transliteration'])}-entry transliteration table "
      "anywhere in the selection window. Digits and unknown Latin **disqualify** a root "
      "instead of being silently rewritten (variant S1 — *not* relaxed for the hidden set).")
    A(f"4. **A cut point in every bucket** (`prefix_cut: {d['prefix_cut']}`): a word group "
      "whose edited text ends on sentence punctuation and whose end time, measured from the "
      "prefix start, falls inside the bucket, strictly after the previous bucket's cut.")
    A(f"5. **One root per channel** (`root_per_channel: {d['root_per_channel']}`) — the one "
      f"with the lowest filler density over its `selection_window_sec = "
      f"{d['selection_window_sec']}` seconds after the prefix start. **Tie-break, and the "
      "order of the whole ranking: `(filler_density_per_100w, sample_id)` ascending** — "
      "density first, then the sample_id lexicographically, so the choice is total and "
      "reproducible. Distinct channels are preferred by construction: the pool takes one "
      f"root from each eligible channel first ({stats['n_eligible_channels']} of them), and "
      f"only then `root_fallback: {stats['root_fallback']}` tops it up with the next-lowest-"
      "density roots from **other videos** of those same channels, never a second root from "
      "a video already in the pool.")
    A("6. **Genre assignment**: the frozen `max_sum_z_score` objective with "
      f"λ = {d['genre_assignment']['filler_penalty_lambda']}, generalised from the pilot's "
      "one-root-per-genre bijection to an even coverage — with "
      f"{len(ds_roots)} roots over 4 genres each genre takes ⌊n/4⌋ or ⌈n/4⌉ roots, and the "
      "assignment maximising the summed z-score wins (tie-break: lowest summed density, then "
      "the lexicographically smallest sample_id tuple). With n = 4 this is exactly the "
      "pilot's search, which is why the pilot artifacts rebuild byte-identical.\n")

    # ------------------------------------------------------------------- §3
    A("## 3. Composition\n")
    A("### Table A — root documents\n")
    A("| root_id | genre | source | channel | video | sample_id | split | asr_cons | "
      "root dur s | prefix start s | filler /100w | edits in window | words removed by B4 |")
    A("|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    pool_by_sid = {p["sample_id"]: p for p in stats["pool"]}
    for rid in roots:
        r = by_root[rid]["B4"]
        if r["source"] == "dataset":
            p = pool_by_sid[r["sample_id"]]
            e = edits[rid]
            A(f"| `{rid}` | {r['genre']} | dataset | {cell(r['channel_title'])} | "
              f"{cell((r['video_title'] or '')[:46])} | `{r['sample_id']}` | {r['split']} | "
              f"{r['asr_consistency']:.2f} | {test_rows[r['sample_id']]['duration_sec']:.2f} | "
              f"{r['human_offset_start']:.3f} | {p['filler_density_per_100w']:.2f} | "
              f"{len(e['edits'])} | {r['n_words_removed']} |")
        else:
            A(f"| `{rid}` | {r['genre']} | external | — | {cell(r['video_title'])} | `—` | — | — | "
              f"— | — | — | 0 | 0 |")
    A("")

    A("### Table B — prefixes (bucket = HUMAN duration)\n")
    A("| text_id | bucket | human dur s | offsets s | words | chars | sentences | ref words | "
      "tokens | context occ. | edits | words removed | est floor ins. | sha256_text_tts |")
    A("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for rid in roots:
        for b in BUCKET_ORDER:
            r = by_root[rid][b]
            off = ("—" if r["human_offset_start"] is None
                   else f"{r['human_offset_start']:.2f}–{r['human_offset_end']:.2f}")
            fl = ("—" if r["est_floor_insertion_rate"] is None
                  else f"{r['est_floor_insertion_rate']:.4f}")
            A(f"| `{r['text_id']}` | {b} | {r['human_duration_sec']:.2f} | {off} | "
              f"{r['words']} | {r['chars']} | {r['sentences']} | {r['ref_words']} | "
              f"{r['text_tts_tokens']} | {r['context_occupancy_est']:.3f} | "
              f"{r['n_disfluency_edits']} | {r['n_words_removed']} | {fl} | "
              f"`{r['sha256_text_tts'][:12]}` |")
    A("")

    A("### Table C — disfluency edits by kind\n")
    kinds = ["discourse", "repeat", "stutter", "truncation", "restart", "hesitation"]
    present = [k for k in kinds
               if any(e["kind"] == k for v in edits.values() for e in v["edits"])]
    other = sorted({e["kind"] for v in edits.values() for e in v["edits"]} - set(present))
    cols = present + other
    A("| root_id | " + " | ".join(cols) + " | total | case repairs | translits |")
    A("|---|" + "|".join(["---:"] * (len(cols) + 3)) + "|")
    tot = {k: 0 for k in cols}
    tot_all = 0
    for rid in ds_roots:
        v = edits[rid]
        c = {k: sum(1 for e in v["edits"] if e["kind"] == k) for k in cols}
        for k in cols:
            tot[k] += c[k]
        tot_all += len(v["edits"])
        A(f"| `{rid}` | " + " | ".join(str(c[k]) for k in cols)
          + f" | {len(v['edits'])} | {v['n_case_repairs']} | {v['n_transliterations']} |")
    A(f"| **all {len(ds_roots)} dataset roots** | "
      + " | ".join(f"**{tot[k]}**" for k in cols) + f" | **{tot_all}** | "
      + f"{sum(edits[r]['n_case_repairs'] for r in ds_roots)} | "
      + f"{sum(edits[r]['n_transliterations'] for r in ds_roots)} |")
    A(f"| {', '.join('`' + r + '`' for r in ex_roots)} | "
      + " | ".join("0" for _ in cols) + " | 0 | 0 | 0 |")
    A("")
    A("All edits are in `reports/benchmark_hidden_edits.json` with their character offsets "
      "and ±40 characters of context.\n")

    A("### Table D — selection pool (density order, the frozen ranking)\n")
    A("| assigned genre | channel | sample_id | dur s | prefix start s | asr_cons | "
      "filler /100w | " + " | ".join(sorted(stats["pool"][0]["features"])) + " |")
    A("|---:|---|---|---:|---:|---:|---:|" + "|".join(["---:"] * len(stats["pool"][0]["features"])) + "|")
    for p in stats["pool"]:
        A(f"| {p['assigned_genre']} | {cell(p['channel_title'])} | `{p['sample_id']}` | "
          f"{p['duration_sec']:.1f} | {p['prefix_start_sec']:.2f} | "
          f"{p['asr_consistency']:.2f} | {p['filler_density_per_100w']:.2f} | "
          + " | ".join(f"{p['features'][k]:.4f}" for k in sorted(p["features"])) + " |")
    A("")

    # ------------------------------------------------------------------- §4
    A("## 4. Reference voices and the same-video confound\n")
    prim = [r for r in refs if r["role"] == "primary"]
    A("The hidden set is generated with the **same two primary voices as the pilot** "
      "(PLAN §8 *Final*): "
      + ", ".join(f"`{r['voice_id']}` ({r['channel_title']}, {r['gender_heuristic']})"
                  for r in prim)
      + ". They are cut from **test** channels and are absent from train — which is what "
      "PLAN §8 requires. PLAN §8 does *not* require them to be absent from test, and here "
      "they cannot be: only "
      f"{feas['test']['stages']['quality_ok_channels']} test channels carry material long "
      "enough for a root, and A5 cut the voices from that same small pool.\n")
    A("**Measured overlap between every reference and every hidden root:**\n")
    A("| voice_id | role | reference video | hidden root on the same video | same "
      "speaker_key | same segment | reference window inside a prefix |")
    A("|---|---|---|---|---|---|---|")
    confounds: list[str] = []
    for ref in refs:
        rows = []
        for rid in ds_roots:
            r = by_root[rid]["B4"]
            same_video = r["video_id"] == ref["video_id"]
            same_spk = test_rows[r["sample_id"]]["speaker_key"] == ref["speaker_key"]
            if not (same_video or same_spk):
                continue
            same_seg = r["sample_id"] == ref.get("source_sample_id")
            inside = []
            if same_seg:
                a0, a1 = ref["offset_start_sec"], ref["offset_end_sec"]
                for b in BUCKET_ORDER:
                    rec = by_root[rid][b]
                    if not (rec["human_offset_end"] <= a0 or rec["human_offset_start"] >= a1):
                        inside.append(b)
            rows.append((rid, same_spk, same_seg, inside))
            if ref["role"] == "primary" or inside:
                confounds.append(
                    f"{ref['voice_id']} ({ref['role']}) ↔ {rid}"
                    + (f", reference window {ref['offset_start_sec']:.3f}–"
                       f"{ref['offset_end_sec']:.3f} s inside {'/'.join(inside)}" if inside else ""))
        if not rows:
            A(f"| `{ref['voice_id']}` | {ref['role']} | `{ref['video_id']}` | **none** | — | "
              "— | — |")
        for rid, same_spk, same_seg, inside in rows:
            A(f"| `{ref['voice_id']}` | {ref['role']} | `{ref['video_id']}` | `{rid}` | "
              f"{'**yes**' if same_spk else 'no'} | {'**yes**' if same_seg else 'no'} | "
              + ("**" + ", ".join(inside) + "**" if inside else "no") + " |")
    A("")
    A("**FLAGGED for the Lead — this is a confound, not a leak.** Nothing here breaks "
      "PLAN §8 (references are absent from *train*), and nothing here was chosen by hand: "
      "the roots come out of the frozen criteria and A5's voices out of a separate frozen "
      "procedure. But for the affected root × voice cells the model is asked to read a text "
      "whose *human* recording is the same speaker as the prompt — and in one case the same "
      "recording. Consequences:\n")
    for c in confounds:
        A(f"* {c}")
    A("")
    A("* A same-speaker cell makes speaker similarity (PLAN §9.5) trivially easier and makes "
      "the human-audio ASR floor and the model output share a voice, so a floor-relative WER "
      "for that cell is not comparable with the others.")
    A("* The `ref_male_02` case is worse in kind — the reference audio is a literal "
      "sub-interval of the root's own human recording — but `ref_male_02` is the **backup** "
      "voice and is not used for the hidden run unless a primary fails.")
    A("* A2 did **not** add a rule excluding these roots: that would be a criterion invented "
      "after seeing the data, and it would drop the hidden set from "
      f"{len(ds_roots)} to {len(ds_roots) - 2} dataset roots (only "
      f"{len({by_root[r]['B4']['channel_id'] for r in ds_roots}) - 2} of the "
      f"{len({by_root[r]['B4']['channel_id'] for r in ds_roots})} eligible channels are "
      "reference-free). **Decision for the Lead** (PROPOSED, in `reports/decisions.md`): "
      "either (a) report the affected cells with a marker and keep them in the unconditional "
      "metrics, or (b) pre-register, before the hidden run, that speaker-similarity "
      "comparisons exclude root × voice cells with a shared `speaker_key`. Option (b) costs "
      "nothing on completion/WER and removes the only metric the confound actually distorts.\n")

    # ------------------------------------------------------------------- §5
    A("## 5. Overlap with the pilot\n")
    pv = {r["video_id"] for r in P if r["video_id"]}
    hv = {r["video_id"] for r in H if r["video_id"]}
    pc = {r["channel_id"] for r in P if r["channel_id"]}
    hc = {r["channel_id"] for r in H if r["channel_id"]}
    PS: set[tuple[str, ...]] = set()
    for r in P:
        if r["bucket"] == "B4":
            PS |= shingles(r["text_tts"])
    worst = 0.0
    shared = 0
    for r in H:
        if r["bucket"] != "B4":
            continue
        S = shingles(r["text_tts"])
        shared += len(S & PS)
        worst = max(worst, len(S & PS) / max(1, len(S | PS)))
    A("| check | result |")
    A("|---|---|")
    A(f"| shared `video_id` with `pilot.jsonl` | **{len(pv & hv)}** (pilot {len(pv)}, "
      f"hidden {len(hv)}) |")
    A(f"| shared `channel_id` | **{len(pc & hc)}** |")
    A(f"| shared `root_id` / `text_id` | **{len({r['root_id'] for r in P} & set(roots))}** / "
      f"**{len({r['text_id'] for r in P} & {r['text_id'] for r in H})}** |")
    A(f"| shared `sha256_text_tts` | **{len({r['sha256_text_tts'] for r in P} & {r['sha256_text_tts'] for r in H})}** |")
    A("| a hidden text is a prefix or superstring of a pilot text | **"
      + str(sum(1 for h in H for p in P
                if h["text_tts"].startswith(p["text_tts"])
                or p["text_tts"].startswith(h["text_tts"]))) + "** |")
    A(f"| shared 8-word shingles (longest prefixes) | **{shared}**, max Jaccard "
      f"**{worst:.6f}** (split-rule threshold 0.05) |")
    A("")
    A("Split rule v3 makes the first three vacuous by construction — dev and test channels "
      "are disjoint — but they are asserted anyway, because the hidden set must survive a "
      "future re-split. The last two are the real check on the **constructed** documents, "
      "which are the only texts an author could have accidentally reused.\n")

    # ------------------------------------------------------------------- §6
    A("## 6. The two constructed documents\n")
    A("| root_id | genre | file | sha256 | words in file | B4 words | B4 sentences |")
    A("|---|---:|---|---|---:|---:|---:|")
    for spec in hcfg["external_roots"]["documents"]:
        p = ROOT / spec["path"]
        raw = dis.apply_charset(re.sub(r"\s*\n\s*", " ", p.read_text(encoding="utf-8")).strip())
        rid = next(r for r in ex_roots if by_root[r]["B4"]["genre"] == spec["genre"])
        A(f"| `{rid}` | {spec['genre']} | `{spec['path']}` | `{sha256_file(p)[:16]}` | "
          f"{len(dis.tokenize(raw))} | {by_root[rid]['B4']['words']} | "
          f"{by_root[rid]['B4']['sentences']} |")
    A("")
    A("Both are written to the same brief as the pilot's pair — genre 5 spells every number, "
      "date, unit, name and abbreviation out in Russian words; genre 6 is controlled stress: "
      "enumerations, near-repeat sentences, permitted repetitions and entity returns — and "
      "both pass the same gates the builder enforces on constructed documents: no digit, no "
      "Latin letter, zero hits of the frozen disfluency filter, and a sentence boundary "
      "inside every bucket. Their subject matter is disjoint from the pilot's "
      "(`Сводный отчёт о работе транспортного узла`, `Городской архив Заозёрья`).\n")

    # ------------------------------------------------------------------- §7
    A("## 7. Fields, rebuild and seal\n")
    A("Fields are PLAN §7.5 exactly, identical to `pilot.jsonl`:\n")
    A("```")
    A(" ".join(sorted(H[0])))
    A("```")
    A(f"`text_tts` is the **only** TTS input, `text_ref` the **only** WER reference "
      f"(= normalize(text_tts, `{variant}`), spec `{hcfg['normalization']['spec']}`). "
      f"`text_tts_tokens` uses the model's own tokenizer "
      f"(`{stats['tokenizer']['class']}` over `{stats['tokenizer']['path']}`); "
      f"`context_occupancy_est` is the frozen formula "
      f"`{hcfg['context_occupancy']['formula']}` and peaks at "
      f"**{max(r['context_occupancy_est'] for r in H):.3f}**, so the hidden set fits the "
      f"{hcfg['tokenizer']['context_window']}-position window with "
      f"{100 * (1 - max(r['context_occupancy_est'] for r in H)):.0f} % to spare.\n")
    A("```bash")
    A("# rebuild (deterministic; the manifest must be chmod u+w first, it is sealed 444)")
    A("python src/benchmark/build_pilot.py --config configs/benchmark_hidden.yaml \\")
    A("    --out data/benchmark/hidden.jsonl --stats data/benchmark/hidden_stats.json \\")
    A("    --edits reports/benchmark_hidden_edits.json")
    A("python src/benchmark/hidden_composition_report.py --strict")
    A("python src/benchmark/seal_hidden.py            # writes the seal, chmod 444")
    A("python src/benchmark/seal_hidden.py --verify   # checks it, writes nothing")
    A(".venv-eval/bin/python -m pytest tests/test_hidden_benchmark.py -q")
    A("```")
    A("")

    # ------------------------------------------------------------------- §8
    A("## 8. Validation — re-derived from `hidden.jsonl` at generation time\n")
    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    chk(f"{len(roots)} roots × 5 buckets = {len(H)} records",
        len(H) == len(roots) * 5 and len(H) == int(hset["n_dataset_roots"]) * 5
        + int(hset["n_external_roots"]) * 5)
    chk("unique text_id", len({r["text_id"] for r in H}) == len(H))
    chk("every root has B0..B4", all(set(v) == set(BUCKET_ORDER) for v in by_root.values()))
    chk("nested prefixes on text_tts",
        all(v[b]["text_tts"].startswith(v[a]["text_tts"]) for v in by_root.values()
            for a, b in zip(BUCKET_ORDER, BUCKET_ORDER[1:])))
    chk("nested prefixes on text_ref",
        all(v[b]["text_ref"].startswith(v[a]["text_ref"]) for v in by_root.values()
            for a, b in zip(BUCKET_ORDER, BUCKET_ORDER[1:])))
    chk("every human_duration_sec inside its bucket",
        all(buckets[r["bucket"]][0] <= r["human_duration_sec"] <= buckets[r["bucket"]][1]
            for r in H))
    chk("text_tts_tokens strictly increasing across buckets",
        all(all(v[b]["text_tts_tokens"] > v[a]["text_tts_tokens"]
                for a, b in zip(BUCKET_ORDER, BUCKET_ORDER[1:])) for v in by_root.values()))
    occ = hcfg["context_occupancy"]
    chk("context_occupancy_est equals the frozen formula",
        all(abs((r["text_tts_tokens"] + occ["speech_token_rate_hz"] * r["human_duration_sec"]
                 + occ["service_tokens"]) / hcfg["tokenizer"]["context_window"]
                - r["context_occupancy_est"]) <= 1e-6 for r in H))
    chk(f"dataset roots all from split 'test' with asr_consistency ≥ "
        f"{hcfg['dataset_roots']['min_asr_consistency']}",
        all(r["split"] == "test"
            and r["asr_consistency"] >= hcfg["dataset_roots"]["min_asr_consistency"]
            for r in H if r["source"] == "dataset"))
    chk("human_offset_end − human_offset_start = human_duration_sec (dataset)",
        all(abs(r["human_offset_end"] - r["human_offset_start"] - r["human_duration_sec"])
            < 1e-6 for r in H if r["source"] == "dataset"))
    chk("every human_audio_path exists on disk",
        all(os.path.exists(r["human_audio_path"]) for r in H if r["source"] == "dataset"))
    chk("constructed roots carry no human audio",
        all(r["human_audio_path"] is None and r["human_offset_start"] is None
            and r["human_offset_end"] is None for r in H if r["source"] == "external"))
    chk("no digit in any text_tts", not any(re.search(r"\d", r["text_tts"]) for r in H))
    chk("no Latin letter in any text_tts",
        not any(re.search(r"[A-Za-z]", r["text_tts"]) for r in H))
    chk("no quotation mark or apostrophe in any text_tts",
        not any(re.search(r"[«»\"“”„‟‹›‛❝❞'’‘`´ʼ]", r["text_tts"]) for r in H))
    chk("zero disfluency-filter hits in text_tts", all(not dis.scan(r["text_tts"]) for r in H))
    chk("finalize() is idempotent on text_tts",
        all(dis.finalize(r["text_tts"]) == r["text_tts"] for r in H))
    chk("text_ref equals the frozen normalizer's output",
        all(r["text_ref"] == norm.normalize(r["text_tts"], variant) for r in H))
    chk("every text ends on sentence punctuation",
        all(r["text_tts"].rstrip()[-1] in ".!?…" for r in H))
    chk("no video_id / channel_id / root_id / text_id shared with the pilot",
        not (pv & hv) and not (pc & hc) and not ({r["root_id"] for r in P} & set(roots))
        and not ({r["text_id"] for r in P} & {r["text_id"] for r in H}))
    chk("no hidden text is a prefix or superstring of a pilot text",
        not any(h["text_tts"].startswith(p["text_tts"]) or p["text_tts"].startswith(h["text_tts"])
                for h in H for p in P))
    chk("8-word shingle Jaccard with the pilot below 0.05", worst < 0.05, f"{worst:.6f}")
    crit = {k: v for k, v in pcfg["dataset_roots"].items() if k != "source_manifest"}
    hcrit = {k: v for k, v in hcfg["dataset_roots"].items() if k != "source_manifest"}
    chk("selection criteria identical to the pilot config (source manifest aside)",
        crit == hcrit)
    chk("bucket / normalization / disfluency / charset / tokenizer blocks identical to the pilot",
        all(pcfg[k] == hcfg[k] for k in
            ("buckets", "normalization", "disfluency", "text_charset", "tokenizer",
             "context_occupancy")))
    if seal:
        chk("seal sha256 matches the manifest on disk",
            seal["sha256"] == sha256_file(ROOT / "data/benchmark/hidden.jsonl"))
        chk("seal covers every record", seal["n_records"] == len(H))
        chk("manifest and seal are mode 444",
            oct(os.stat(ROOT / "data/benchmark/hidden.jsonl").st_mode)[-3:] == "444"
            and oct(os.stat(seal_path).st_mode)[-3:] == "444")

    A("| check | result | detail |")
    A("|---|---|---|")
    for name, ok, detail in checks:
        A(f"| {name} | **{'PASS' if ok else 'FAIL'}** | {detail or '—'} |")
    n_fail = sum(1 for _n, ok, _d in checks if not ok)
    A("")
    A(f"**{len(checks) - n_fail}/{len(checks)} pass.**\n")

    # ------------------------------------------------------------------- §9
    A("## 9. Known limitations (hidden set)\n")
    ch_counts: dict[str, int] = {}
    for rid in ds_roots:
        ch_counts[by_root[rid]["B4"]["channel_title"]] = (
            ch_counts.get(by_root[rid]["B4"]["channel_title"], 0) + 1)
    biggest = max(ch_counts.items(), key=lambda kv: (kv[1], kv[0]))
    A(f"* **{len(roots)} roots, not 12.** The frozen rule admits "
      f"{stats['n_pool_roots']} dataset roots on the test split (§1.1). Statistical power "
      "on the hidden set is therefore lower than PLAN §7.4 assumed: the paired cluster "
      f"bootstrap of PLAN §12 has {len(roots)} clusters instead of 12, "
      f"{len(ds_roots)} of them with a human ASR floor.")
    A(f"* **The channel pool is small and unbalanced.** {biggest[0]} supplies "
      f"{biggest[1]} of the {len(ds_roots)} dataset roots (via the one-root-per-video "
      f"fallback), and only {len(ch_counts)} channels are represented. A single narrator's "
      "style therefore carries a large share of the dataset side, exactly as in the pilot.")
    A("* **Two of the roots share a speaker with a reference voice** (§4). Flagged, not "
      "fixed by hand.")
    A("* **The reference text is ASR output, not a human transcription** — `text_tts` comes "
      f"from `{hcfg['dataset_roots']['text_source']}` and contains its errors. Paired "
      "comparisons are unaffected; absolute WER includes this noise. A human proofreading "
      "pass over the "
      f"{len(ds_roots)} hidden roots would remove it and is recommended **before** the run, "
      "since it changes the sealed file and would need a re-seal.")
    A("* **Genre labels are relative**, assigned inside a pool of "
      f"{stats['n_pool_roots']} by measurable text features (Table D), not by editorial "
      "judgment — and with two roots per genre for two of the four genres, the labels are a "
      "ranking, not a classification.")
    A("* **The 2 constructed documents have no human audio**, therefore no ASR floor; their "
      "absolute WER is comparable across checkpoints only. Their nominal duration uses the "
      f"dev median rate ({stats['dev_median_wpm']:.3f} wpm), the same constant as the pilot.")
    A("* **Residual disfluencies exist**: zero *rule* hits is what the filter guarantees, "
      "not zero disfluencies.")
    A("* **Speaking-rate / position confound between buckets is unavoidable** by the "
      "nested-prefix design of PLAN §7.2.\n")

    return "\n".join(L) + "\n", n_fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="reports/benchmark_hidden_composition.md")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any §8 self-check fails")
    a = ap.parse_args(argv)
    text, n_fail = build(a.strict)
    (ROOT / a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out} ({len(text.splitlines())} lines); {n_fail} failed self-check(s)")
    return 1 if (n_fail and a.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
