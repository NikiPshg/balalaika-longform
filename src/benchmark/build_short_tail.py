#!/usr/bin/env python
"""A11-short / E11 «Short-tail buckets» (reports/decisions.md 2026-08-31): S0/S1 sets.

Two NEW benchmark files extend the frozen grid downwards without touching it:

  data/benchmark/short_tail_pilot.jsonl   the SAME 6 roots as the frozen pilot
                                          (4 dataset + 2 external), <= 1 item per
                                          bucket per root, read by the 2 pilot voices;
  data/benchmark/short_tail_robust.jsonl  the 13 frozen Robust-20 voices, <= 1 item
                                          per bucket per voice from that voice's OWN
                                          units (--per-item-voices).

Buckets (bucket = HUMAN duration of the span): **S0 = 0-3 s** (one clause/sentence),
**S1 = 3-20 s** (1-3 sentences).

The text pipeline is the frozen pilot pipeline, reused BY IMPORT (prefix_start,
word-group timestamps `asr_ts.gigaam-v3-e2e-ctc`, transliteration, case repair,
disfluency cleaning, charset S1, lenient normalizer, tokenizer fields) -- the only
new geometry is the FLOATING span: a span starts at any sentence boundary at or
after the frozen prefix start and ends 1-3 sentence boundaries later, because the
E11 guard forbids re-using audio the frozen benchmarks already occupy:

  * no time-range intersection with any span of pilot.jsonl / robust.jsonl (per
    sample_id), nor with any reference window (references.jsonl +
    references_robust.jsonl);
  * the sealed hidden set's sample_ids are excluded ENTIRELY;
  * the two new files are disjoint from each other;
  * a root/voice whose material is exhausted by the guard is reported as a GAP,
    never filled by weakening a rule (the report quantifies what the guard cost).

External documents (no human audio, no time axis) keep the pilot's frozen rule:
first sentence(s) from the document start, duration estimated with the SAME
dev-median-wpm constant, `human_duration_source = "estimated: ..."`, no ASR floor.

Run (cosyvoice env, CPU only):

    python src/benchmark/build_short_tail.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src" / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.benchmark import build_pilot as bp  # noqa: E402
from src.benchmark import build_robust as br  # noqa: E402
from src.benchmark import disfluency as dis  # noqa: E402
from src.eval import normalize as norm  # noqa: E402

BUCKET_ORDER = ("S0", "S1")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ used-span index
class SpanIndex:
    """Per-sample_id time ranges that short-tail spans may not intersect."""

    def __init__(self) -> None:
        self.by_sid: dict[str, list[tuple[float, float, str]]] = defaultdict(list)

    def add(self, sid: str, start: float, end: float, why: str) -> None:
        self.by_sid[sid].append((float(start), float(end), why))

    def hit(self, sid: str, start: float, end: float) -> str | None:
        for a, b, why in self.by_sid.get(sid, ()):
            if not (end <= a or start >= b):   # half-open: touching is NOT overlap
                return why
        return None


def build_used_index(cfg: dict) -> tuple[SpanIndex, set[str]]:
    guard = cfg["overlap_guard"]
    idx = SpanIndex()
    for rel in guard["exclude_spans_of"]:
        for r in read_jsonl(ROOT / rel):
            if r.get("sample_id") and r.get("human_offset_start") is not None:
                idx.add(r["sample_id"], r["human_offset_start"], r["human_offset_end"],
                        f"{Path(rel).name}:{r['text_id']}")
    for rel in guard["exclude_reference_windows_of"]:
        for r in read_jsonl(ROOT / rel):
            if r.get("source_sample_id") and r.get("offset_start_sec") is not None:
                idx.add(r["source_sample_id"], r["offset_start_sec"], r["offset_end_sec"],
                        f"{Path(rel).name}:{r['voice_id']}")
    hidden_sids = {r["sample_id"] for r in read_jsonl(ROOT / guard["exclude_sample_ids_of"])
                   if r.get("sample_id")}
    return idx, hidden_sids


# ------------------------------------------------------------------ span candidates
def span_candidates(cand: dict, lo: float, hi: float, smin: int, smax: int,
                    lo_exclusive: bool, min_dur: float, sid: str,
                    used: SpanIndex) -> tuple[list[dict], int]:
    """All clean spans of `cand` whose HUMAN duration fits the bucket.

    A span starts at the chain start or right after a sentence-boundary group and
    ends at the 1st..`smax`-th boundary after it.  Clean = charset S1 on the
    finalized text, zero disfluency-filter hits (a cut can re-expose a dangler on
    EITHER side -- the robust builder saw it at cut ends, a floating span can also
    start right after one), and `smin <= n_sentences <= smax` on the §7.5 field.

    Returns (candidates, n_blocked_by_overlap_only): the second number counts spans
    that passed EVERY frozen rule and died only on the E11 no-overlap guard -- the
    honest price of the guard, reported per gap.
    """
    groups, pieces = cand["groups"], cand["pieces_clean"]
    ok = [i for i, p in enumerate(pieces) if p.strip() and p.strip()[-1] in bp.SENT_END]
    starts = [0] + [i + 1 for i in ok if i + 1 < len(groups)]
    out: list[dict] = []
    n_overlap_only = 0
    for a in starts:
        bnds = [j for j in ok if j >= a][:smax]
        for j in bnds:
            d = groups[j][1] - groups[a][0]
            if d > hi:
                break
            if d < min_dur:
                continue
            if (d <= lo) if lo_exclusive else (d < lo):
                continue
            text = dis.finalize(" ".join(pieces[a:j + 1]))
            if not text or re.search(r"[0-9A-Za-z]", text) or dis.scan(text):
                continue
            ns = bp.n_sentences(text)
            if not (smin <= ns <= smax):
                continue
            s, e = groups[a][0], groups[j][1]
            why = used.hit(sid, s, e)
            if why is not None:
                n_overlap_only += 1
                continue
            out.append({"a": a, "j": j, "dur": d, "text": text, "start": s, "end": e,
                        "words": len(dis.tokenize(text))})
    return out, n_overlap_only


def pick_span(cands: list[dict], centre: float,
              pref_words: tuple[int, int] | None = None) -> dict:
    """Nearest the bucket centre; with `pref_words` (S0), candidates inside the
    preferred word range win first (E11: «1 short clause, typically 3-8 words» --
    a preference, never a gate), the centre rule breaks ties as everywhere else."""
    def key(c: dict):
        pref = (0 if pref_words and pref_words[0] <= c["words"] <= pref_words[1]
                else 1) if pref_words else 0
        return (pref, abs(c["dur"] - centre), c["a"], c["j"])
    return min(cands, key=key)


def has_preferred(cands: list[dict], pref_words: tuple[int, int] | None) -> bool:
    return bool(pref_words) and any(
        pref_words[0] <= c["words"] <= pref_words[1] for c in cands)


def bucket_params(cfg: dict, bucket: str, has_audio: bool) -> dict:
    lo, hi = cfg["buckets"][bucket]
    cons = cfg["constraints"]
    if bucket == "S0":
        smin, smax = cons["s0_sentences"]
        lo_excl = False
    else:
        smin, smax = cons["s1_sentences"]
        lo_excl = bool(cons["s1_lo_exclusive"])
    min_dur = float(cons["min_human_slice_sec"]) if has_audio else 0.0
    pref = tuple(cons["s0_preferred_words"]) if bucket == "S0" else None
    return {"lo": float(lo), "hi": float(hi), "smin": int(smin), "smax": int(smax),
            "lo_exclusive": lo_excl, "min_dur": min_dur, "pref_words": pref,
            "centre": (float(lo) + float(hi)) / 2.0}


# ------------------------------------------------------------------------- records
def span_record(root_id: str, bucket: str, cand: dict, span: dict, row: dict,
                pilot_cfg: dict, variant: str, genre: int | None,
                voice_id: str | None = None) -> dict:
    """One §7.5 record for a floating span; mirrors bp.make_dataset_records /
    br.make_record field for field, plus the span geometry."""
    a, j = span["a"], span["j"]
    groups = cand["groups"]
    text_tts = span["text"]
    text_un = dis.finalize(" ".join(cand["pieces_source"][a:j + 1]))
    start, end = groups[a][0], groups[j][1]
    duration = end - start
    # char offsets of the raw (cleaned-input) pieces, single-space joined, to count
    # the disfluency edits that fall INSIDE the span (same accounting as the pilot's
    # prefix version, restricted to [span_start_char, span_end_char))
    offs: list[int] = []
    cursor = 0
    for i, piece in enumerate(cand["pieces_raw"]):
        if i:
            cursor += 1
        offs.append(cursor)
        cursor += len(piece)
    span_c0 = offs[a]
    span_c1 = offs[j] + len(cand["pieces_raw"][j])
    in_span = [e for e in cand["edits"] if span_c0 <= e.start < span_c1]
    n_edits = len(in_span)
    n_removed = sum(len(dis.tokenize(e.surface)) for e in in_span if e.kind != "stutter")
    ref = norm.normalize(text_tts, variant)
    rec = {
        "text_id": f"{root_id}__{bucket}",
        "root_id": root_id,
        "bucket": bucket,
        "genre": genre,
        "source": "dataset",
        "channel_id": row["channel_id"],
        "channel_title": row["channel_title"],
        "video_id": row["video_id"],
        "video_title": row["video_title"],
        "sample_id": row["sample_id"],
        "split": row["split"],
        "license": row["license"],
        "asr_consistency": row["asr_consistency"],
        "human_audio_path": row["audio_path"],
        "human_audio_sha256": row["sha256_audio"],
        "human_offset_start": round(start, 3),
        "human_offset_end": round(end, 3),
        "human_duration_sec": round(duration, 3),
        "human_duration_source": "asr_ts.gigaam-v3-e2e-ctc word-group times",
        "prefix_start_index": cand["start_index"],
        "prefix_group_index": j,
        "span_group_start": a,
        "span_group_end": j,
        "text_tts": text_tts,
        "text_tts_unedited": text_un,
        "text_ref": ref,
        "normalization_variant": variant,
        "words": len(dis.tokenize(text_tts)),
        "chars": len(text_tts),
        "sentences": bp.n_sentences(text_tts),
        "ref_words": len(ref.split()),
        "ref_chars": len(ref),
        "n_disfluency_edits": n_edits,
        "n_words_removed": n_removed,
        "est_floor_insertion_rate": round(n_removed / max(1, len(dis.tokenize(text_tts))), 5),
        "sha256_text_tts": sha256_text(text_tts),
        **bp.size_fields(text_tts, duration, pilot_cfg),
    }
    if voice_id is not None:
        rec["voice_id"] = voice_id
        rec["speaker_key"] = row["speaker_key"]
    return rec


def external_short_records(spec: dict, cfg: dict, pilot_cfg: dict, wpm: float,
                           variant: str, gaps: list[dict]) -> list[dict]:
    """Pilot's frozen external rule (bp.make_external_records) with S0/S1 bounds
    and the 1-/1-3-sentence caps; an unfillable bucket becomes a GAP, not an error."""
    path = ROOT / spec["path"]
    raw = path.read_text(encoding="utf-8")
    text = dis.apply_charset(re.sub(r"\s*\n\s*", " ", raw).strip())
    if re.search(r"\d", text) or re.search(r"[A-Za-z]", text):
        raise RuntimeError(f"{path} contains digits or Latin letters")
    if dis.scan(text):
        raise RuntimeError(f"{path} still matches the disfluency filter")
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
    cum: list[tuple[int, int]] = []
    total = 0
    for i, s in enumerate(sentences):
        total += len(dis.tokenize(s))
        cum.append((i, total))

    root_id = f"ex{spec['genre']}_{path.stem}"
    out: list[dict] = []
    for bucket in BUCKET_ORDER:
        p = bucket_params(cfg, bucket, has_audio=False)
        pool = []
        for i, w in cum:
            if i + 1 > p["smax"]:
                break
            d = w / wpm * 60.0
            if d > p["hi"]:
                break
            if (d <= p["lo"]) if p["lo_exclusive"] else (d < p["lo"]):
                continue
            txt = " ".join(sentences[: i + 1])
            ns = bp.n_sentences(txt)
            if not (p["smin"] <= ns <= p["smax"]):
                continue
            pool.append((i, w, d, txt))
        if not pool:
            first_w = cum[0][1] if cum else 0
            gaps.append({
                "set": "pilot", "root_id": root_id, "bucket": bucket,
                "reason": "no_sentence_prefix_fits_bucket",
                "detail": (f"first sentence is {first_w} words ~ "
                           f"{first_w / wpm * 60.0:.1f} s at dev-median wpm {wpm:.2f}; "
                           f"bucket needs <= {p['hi']:g} s"),
                "n_candidates": 0, "n_blocked_by_overlap_only": 0,
            })
            continue
        i, w, d, txt = min(pool, key=lambda t: (abs(t[2] - p["centre"]), t[0]))
        ref = norm.normalize(txt, variant)
        out.append({
            "text_id": f"{root_id}__{bucket}",
            "root_id": root_id,
            "bucket": bucket,
            "genre": spec["genre"],
            "source": "external",
            "channel_id": None,
            "channel_title": None,
            "video_id": None,
            "video_title": spec.get("title"),
            "sample_id": None,
            "split": None,
            "license": "CC-BY-4.0 (written for this benchmark by A2)",
            "asr_consistency": None,
            "human_audio_path": None,
            "human_audio_sha256": None,
            "human_offset_start": None,
            "human_offset_end": None,
            "human_duration_sec": round(d, 3),
            "human_duration_source": f"estimated: words / dev median wpm ({wpm:.3f})",
            "prefix_start_index": 0,
            "prefix_group_index": i,
            "span_group_start": 0,
            "span_group_end": i,
            "text_tts": txt,
            "text_tts_unedited": txt,
            "text_ref": ref,
            "normalization_variant": variant,
            "words": len(dis.tokenize(txt)),
            "chars": len(txt),
            "sentences": bp.n_sentences(txt),
            "ref_words": len(ref.split()),
            "ref_chars": len(ref),
            "n_disfluency_edits": 0,
            "n_words_removed": 0,
            "est_floor_insertion_rate": None,
            "sha256_text_tts": sha256_text(txt),
            **bp.size_fields(txt, d, pilot_cfg),
        })
    return out


# ------------------------------------------------------------------------ provenance
SOURCES = ("src/benchmark/build_short_tail.py", "src/benchmark/build_pilot.py",
           "src/benchmark/build_robust.py", "src/benchmark/disfluency.py",
           "src/eval/normalize.py", "configs/benchmark_short_tail.yaml",
           "configs/benchmark_pilot.yaml", "configs/benchmark_robust.yaml")


def provenance() -> dict:
    import subprocess

    h = hashlib.sha256()
    per_file = {}
    for rel in SOURCES:
        b = open(ROOT / rel, "rb").read()
        fh = hashlib.sha256(b).hexdigest()
        per_file[rel] = fh
        h.update(rel.encode() + b"\0" + fh.encode() + b"\0")

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                  text=True, timeout=20).stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    return {"code_sha256": h.hexdigest(), "code_files_sha256": per_file,
            "git_revision": git("rev-parse", "HEAD")}


# ------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the E11 short-tail benchmark sets")
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark_short_tail.yaml"))
    ap.add_argument("--out-pilot",
                    default=str(ROOT / "data" / "benchmark" / "short_tail_pilot.jsonl"))
    ap.add_argument("--out-robust",
                    default=str(ROOT / "data" / "benchmark" / "short_tail_robust.jsonl"))
    ap.add_argument("--stats",
                    default=str(ROOT / "data" / "benchmark" / "short_tail_stats.json"))
    ap.add_argument("--report",
                    default=str(ROOT / "reports" / "benchmark_short_tail_composition.md"))
    a = ap.parse_args(argv)
    t_start = time.time()

    cfg = bp.load_yaml(Path(a.config))
    pilot_cfg = bp.load_yaml(ROOT / cfg["pilot_rules"])
    robust_cfg = bp.load_yaml(ROOT / cfg["robust_rules"])
    dcfg = pilot_cfg["dataset_roots"]
    dis.load_spec()
    variant = norm.load_spec()["primary_variant"]

    used, hidden_sids = build_used_index(cfg)
    n_used_ranges = sum(len(v) for v in used.by_sid.values())
    print(f"[short_tail] guard: {n_used_ranges} frozen ranges over "
          f"{len(used.by_sid)} sample_ids; {len(hidden_sids)} hidden sample_ids "
          "excluded entirely", file=sys.stderr)

    gaps: list[dict] = []
    records_pilot: list[dict] = []
    records_robust: list[dict] = []

    # ---------------- pilot set: the SAME 6 roots ----------------
    frozen_pilot = read_jsonl(ROOT / cfg["pilot_set"]["source_benchmark"])
    dev_rows = {r["sample_id"]: r
                for r in read_jsonl(ROOT / cfg["pilot_set"]["source_manifest"])}
    dataset_roots: dict[str, dict] = {}   # root_id -> {genre, sample_id, by_bucket ends}
    for r in frozen_pilot:
        if r["source"] != "dataset":
            continue
        d = dataset_roots.setdefault(r["root_id"], {
            "genre": r["genre"], "sample_id": r["sample_id"], "ends": {},
            "start": r["human_offset_start"]})
        d["ends"][r["bucket"]] = r["human_offset_end"]

    for root_id in sorted(dataset_roots):
        info = dataset_roots[root_id]
        sid = info["sample_id"]
        if sid in hidden_sids:
            raise RuntimeError(f"pilot root {root_id} is a hidden sample_id?!")
        row = dev_rows.get(sid)
        if row is None:
            raise RuntimeError(f"pilot root {root_id}: {sid} not in dev manifest")
        cand, why = bp.prepare_root(row, pilot_cfg)
        if cand is None:
            raise RuntimeError(f"pilot root {root_id} no longer prepares: {why}")
        # determinism check: the frozen pilot cuts must reproduce exactly
        if abs(cand["start_time"] - info["start"]) > 0.0015:
            raise RuntimeError(f"{root_id}: prefix start {cand['start_time']} != "
                               f"frozen {info['start']}")
        cuts = bp.cut_points(cand, pilot_cfg["buckets"])
        if cuts is None:
            raise RuntimeError(f"{root_id}: frozen buckets no longer cut")
        for b, k in cuts.items():
            end = round(cand["groups"][k][1], 3)
            if abs(end - info["ends"][b]) > 0.0015:
                raise RuntimeError(f"{root_id} {b}: reproduced cut end {end} != "
                                   f"frozen {info['ends'][b]}")
        for bucket in BUCKET_ORDER:
            p = bucket_params(cfg, bucket, has_audio=True)
            cands, n_ov = span_candidates(
                cand, p["lo"], p["hi"], p["smin"], p["smax"], p["lo_exclusive"],
                p["min_dur"], sid, used)
            if not cands:
                gaps.append({
                    "set": "pilot", "root_id": root_id, "bucket": bucket,
                    "reason": ("all_clean_spans_blocked_by_overlap_guard" if n_ov
                               else "no_clean_span_fits_bucket"),
                    "detail": (f"{n_ov} span(s) passed every frozen rule and died "
                               "only on the no-overlap guard" if n_ov else
                               "no clean sentence span of this duration exists in "
                               "the unused part of the frozen 12-min window"),
                    "n_candidates": 0, "n_blocked_by_overlap_only": n_ov,
                })
                continue
            span = pick_span(cands, p["centre"], p["pref_words"])
            rec = span_record(root_id, bucket, cand, span, row, pilot_cfg, variant,
                              genre=info["genre"])
            records_pilot.append(rec)
            used.add(sid, span["start"], span["end"],
                     f"short_tail_pilot:{rec['text_id']}")

    # external documents: the pilot's frozen first-sentence(s) rule
    ext_wpm_manifest = (
        ((pilot_cfg.get("external_roots") or {}).get("human_duration") or {})
        .get("source_manifest") or cfg["pilot_set"]["source_manifest"])
    wpm_rows = [r for r in read_jsonl(ROOT / ext_wpm_manifest)
                if r.get("words") and r.get("duration_sec")]
    wpm = statistics.median(r["words"] / r["duration_sec"] * 60.0 for r in wpm_rows)
    for spec in pilot_cfg["external_roots"]["documents"]:
        records_pilot += external_short_records(spec, cfg, pilot_cfg, wpm, variant, gaps)

    records_pilot.sort(key=lambda r: (r["root_id"], r["bucket"]))
    print(f"[short_tail] pilot set: {len(records_pilot)} records, "
          f"{sum(1 for g in gaps if g['set'] == 'pilot')} gaps", file=sys.stderr)

    # ---------------- robust set: the 13 frozen voices ----------------
    rstats = json.load(open(ROOT / cfg["robust_set"]["source_stats"], encoding="utf-8"))
    voices = rstats["voices"]
    refs = {r["voice_id"]: r
            for r in read_jsonl(ROOT / cfg["robust_set"]["references"])}
    rows_all: list[dict] = []
    for mpath in cfg["robust_set"]["manifests"]:
        rows_all += read_jsonl(ROOT / mpath)
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows_all:
        by_speaker[r["speaker_key"]].append(r)
    tcfg = robust_cfg["texts"]

    unit_reject: Counter = Counter()
    for v in voices:
        vid, key = v["voice_id"], v["speaker_key"]
        rank = int(vid[-2:])
        ref_sid = v["ref_sample_id"]
        units: dict[str, dict] = {}
        for r in by_speaker[key]:
            sid = r["sample_id"]
            if sid in hidden_sids:
                unit_reject["hidden_excluded"] += 1
                continue
            if sid == ref_sid:
                unit_reject["reference_unit"] += 1
                continue
            if r["asr_consistency"] < float(tcfg["unit_min_asr_consistency"]):
                unit_reject["consistency"] += 1
                continue
            if tcfg["require_single_speaker"] and not r["is_single_speaker"]:
                unit_reject["multi_speaker"] += 1
                continue
            cand, why = br.prepare_unit(r, dcfg)
            if cand is None:
                unit_reject[why] += 1
                continue
            cand["row"] = r
            units[sid] = cand
        used_units: set[str] = set()
        for bucket in cfg["robust_set"]["bucket_order_for_assignment"]:
            p = bucket_params(cfg, bucket, has_audio=True)
            order = sorted(units, key=lambda s: (0 if s not in used_units else 1,
                                                 units[s]["density"], s))
            chosen = None
            n_ov_total = 0
            per_unit: dict[str, list[dict]] = {}
            # pass 1: robust's frozen unit rank, but a unit only wins outright when
            # it offers a PREFERRED-word-count span (S0; for S1 pref is None and any
            # candidate wins, i.e. the plain robust behaviour)
            for sid in order:
                cands, n_ov = span_candidates(
                    units[sid], p["lo"], p["hi"], p["smin"], p["smax"],
                    p["lo_exclusive"], p["min_dur"], sid, used)
                n_ov_total += n_ov
                per_unit[sid] = cands
                if cands and (p["pref_words"] is None
                              or has_preferred(cands, p["pref_words"])):
                    chosen = (sid, pick_span(cands, p["centre"], p["pref_words"]))
                    break
            if chosen is None:
                # pass 2 (S0 fallback): no unit offers a 3-8-word span; take the
                # first unit in rank order with ANY clean span, nearest the centre
                for sid in order:
                    cands = per_unit.get(sid, [])
                    if cands:
                        chosen = (sid, pick_span(cands, p["centre"], p["pref_words"]))
                        break
            if chosen is None:
                gaps.append({
                    "set": "robust", "root_id": vid, "bucket": bucket,
                    "reason": ("all_clean_spans_blocked_by_overlap_guard" if n_ov_total
                               else "no_clean_span_fits_bucket"),
                    "detail": (f"{len(units)} eligible unit(s); {n_ov_total} span(s) "
                               "died only on the no-overlap guard" if n_ov_total else
                               f"{len(units)} eligible unit(s), none has a clean "
                               "sentence span of this duration"),
                    "n_candidates": 0, "n_blocked_by_overlap_only": n_ov_total,
                })
                continue
            sid, span = chosen
            rec = span_record(f"rv{rank:02d}_{sid}", bucket, units[sid], span,
                              units[sid]["row"], pilot_cfg, variant, genre=None,
                              voice_id=vid)
            records_robust.append(rec)
            used.add(sid, span["start"], span["end"],
                     f"short_tail_robust:{rec['text_id']}")
            used_units.add(sid)

    records_robust.sort(key=lambda r: (r["voice_id"], r["bucket"]))
    print(f"[short_tail] robust set: {len(records_robust)} records, "
          f"{sum(1 for g in gaps if g['set'] == 'robust')} gaps; "
          f"unit rejects {dict(unit_reject)}", file=sys.stderr)

    all_records = records_pilot + records_robust

    # ---------------- leakage vs train (robust machinery, verbatim) ----------------
    leak = br.train_leakage(all_records, cfg["leakage_check"])
    print(f"[short_tail] train leakage: max Jaccard {leak['max_jaccard']:.6f} "
          f"({leak['n_items_above_threshold']} items above {leak['threshold']}); "
          f"{sum(1 for r in all_records if not len(r['text_ref'].split()) >= 8)} "
          "item(s) have < 8 words (0 shingles by construction)", file=sys.stderr)

    # ---------------- self-checks ----------------
    forbidden = set(dis.QUOTE_CHARS + dis.APOSTROPHE_CHARS + "()[]{}«»–‒―−")
    checks: list[dict] = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    frozen_pilot_roots = sorted(dataset_roots) + sorted(
        {r["root_id"] for r in frozen_pilot if r["source"] == "external"})
    check("unique text_id across BOTH files",
          len({r["text_id"] for r in all_records}) == len(all_records))
    check("pilot roots are a subset of the frozen pilot's 6 roots",
          {r["root_id"] for r in records_pilot} <= set(frozen_pilot_roots),
          sorted({r["root_id"] for r in records_pilot}))
    check("robust voices are a subset of the 13 frozen voices",
          {r["voice_id"] for r in records_robust} <= {v["voice_id"] for v in voices})
    check("<= 1 item per root/voice x bucket",
          len({(r["root_id"], r["bucket"]) for r in records_pilot}) == len(records_pilot)
          and len({(r["voice_id"], r["bucket"]) for r in records_robust})
          == len(records_robust))
    buckets = cfg["buckets"]
    min_slice = float(cfg["constraints"]["min_human_slice_sec"])
    check("bucket = human duration inside its edges (S1 lo-exclusive)",
          all((buckets[r["bucket"]][0] <= r["human_duration_sec"] <= buckets[r["bucket"]][1])
              and (r["bucket"] != "S1" or r["human_duration_sec"] > buckets["S1"][0])
              for r in all_records))
    check(f"dataset spans >= {min_slice:g} s (frozen evaluator MIN_HUMAN_SLICE_SEC)",
          all(r["human_duration_sec"] >= min_slice for r in all_records
              if r["human_audio_path"] is not None))
    check("sentence caps: S0 == 1, S1 in [1, 3]",
          all((r["sentences"] == 1) if r["bucket"] == "S0" else (1 <= r["sentences"] <= 3)
              for r in all_records))
    check("charset S1: no digit / Latin / quote / apostrophe, 0 disfluency hits, "
          "finalize idempotent",
          all(not re.search(r"[0-9A-Za-z]", r["text_tts"])
              and not (set(r["text_tts"]) & forbidden)
              and not dis.scan(r["text_tts"])
              and dis.finalize(r["text_tts"]) == r["text_tts"] for r in all_records))
    check("text_ref == normalize(text_tts, primary variant)",
          all(r["text_ref"] == norm.normalize(r["text_tts"], variant)
              for r in all_records))
    frozen_ranges = SpanIndex()
    fr_files = list(cfg["overlap_guard"]["exclude_spans_of"])
    for rel in fr_files:
        for r in read_jsonl(ROOT / rel):
            if r.get("sample_id") and r.get("human_offset_start") is not None:
                frozen_ranges.add(r["sample_id"], r["human_offset_start"],
                                  r["human_offset_end"], rel)
    for rel in cfg["overlap_guard"]["exclude_reference_windows_of"]:
        for r in read_jsonl(ROOT / rel):
            if r.get("source_sample_id") and r.get("offset_start_sec") is not None:
                frozen_ranges.add(r["source_sample_id"], r["offset_start_sec"],
                                  r["offset_end_sec"], rel)
    check("no time-range intersection with pilot/robust spans or reference windows",
          all(frozen_ranges.hit(r["sample_id"], r["human_offset_start"],
                                r["human_offset_end"]) is None
              for r in all_records if r["sample_id"]))
    check("hidden sample_ids excluded entirely",
          not ({r["sample_id"] for r in all_records if r["sample_id"]} & hidden_sids))
    check("the two new files are disjoint from each other",
          all(not (min(r1["human_offset_end"], r2["human_offset_end"])
                   > max(r1["human_offset_start"], r2["human_offset_start"]))
              for r1 in records_pilot if r1["sample_id"]
              for r2 in records_robust if r2["sample_id"] == r1["sample_id"]))
    check("no train-split material",
          all(r["split"] in ("dev", "test") for r in all_records if r["sample_id"]))
    check("robust: every item bound to its own voice (speaker_key matches the "
          "frozen reference)",
          all(r["speaker_key"] == refs[r["voice_id"]]["speaker_key"]
              for r in records_robust))
    check("robust: reference unit of a voice provides no short-tail text",
          all(r["sample_id"] != refs[r["voice_id"]]["source_sample_id"]
              for r in records_robust))
    check("external items: no audio fields, estimate-source duration",
          all(r["human_audio_path"] is None and r["human_offset_start"] is None
              and str(r["human_duration_source"]).startswith("estimated")
              for r in all_records if r["source"] == "external"))
    check("context occupancy < 1",
          all(0 < r["context_occupancy_est"] < 1 for r in all_records))
    check(f"train leakage max Jaccard < {leak['threshold']}",
          leak["max_jaccard"] < leak["threshold"], f"{leak['max_jaccard']:.6f}")
    check("dev median wpm equals the frozen pilot constant",
          abs(wpm - json.load(open(ROOT / "data/benchmark/pilot_stats.json",
                                   encoding="utf-8"))["dev_median_wpm"]) < 1e-9,
          f"{wpm:.5f}")
    # composition targets of the E11 pre-registration -- recorded honestly, like the
    # robust builder's B4-voices target: a FAIL here means the frozen rules + the
    # no-overlap guard left no material, NOT that a rule was weakened.
    check("prereg target: 12 pilot texts (6 roots x 2 buckets)",
          len(records_pilot) == 12,
          f"{len(records_pilot)} built, {sum(1 for g in gaps if g['set'] == 'pilot')} "
          "gap(s) -- see the gaps table")
    check("prereg target: 26 robust texts (13 voices x 2 buckets)",
          len(records_robust) == 26,
          f"{len(records_robust)} built, {sum(1 for g in gaps if g['set'] == 'robust')} "
          "gap(s) -- see the gaps table")

    # ---------------- write outputs ----------------
    for path, recs in ((Path(a.out_pilot), records_pilot),
                       (Path(a.out_robust), records_robust)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wt", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def dist(vals):
        vs = sorted(vals)
        if not vs:
            return None
        return {"min": vs[0], "median": vs[len(vs) // 2], "max": vs[-1],
                "mean": round(sum(vs) / len(vs), 3)}

    per_bucket = {}
    for name, recs in (("pilot", records_pilot), ("robust", records_robust)):
        per_bucket[name] = {}
        for b in BUCKET_ORDER:
            rs = [r for r in recs if r["bucket"] == b]
            per_bucket[name][b] = {
                "n": len(rs),
                "duration_sec": dist([r["human_duration_sec"] for r in rs]),
                "words": dist([r["words"] for r in rs]),
                "ref_words": dist([r["ref_words"] for r in rs]),
                "sentences": dist([r["sentences"] for r in rs]),
            }

    prov = provenance()
    stats = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(Path(a.config).relative_to(ROOT)),
        "decision_ref": cfg["decision_ref"],
        "elapsed_sec": round(time.time() - t_start, 1),
        "normalization_variant": variant,
        "dev_median_wpm": wpm,
        "buckets": {b: list(v) for b, v in cfg["buckets"].items()},
        "min_human_slice_sec": min_slice,
        "n_guard_ranges": n_used_ranges,
        "n_hidden_sample_ids_excluded": len(hidden_sids),
        "unit_rejects_robust": dict(unit_reject),
        "n_records_pilot": len(records_pilot),
        "n_records_robust": len(records_robust),
        "per_bucket": per_bucket,
        "gaps": gaps,
        "n_gaps_pilot": sum(1 for g in gaps if g["set"] == "pilot"),
        "n_gaps_robust": sum(1 for g in gaps if g["set"] == "robust"),
        "train_leakage": leak,
        "self_checks": checks,
        "build": prov,
    }
    Path(a.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    write_report(Path(a.report), stats, records_pilot, records_robust, cfg)

    n_pass = sum(1 for c in checks if c["pass"])
    print(f"[short_tail] wrote {len(records_pilot)} pilot + {len(records_robust)} "
          f"robust records; {len(gaps)} gaps; self-checks {n_pass}/{len(checks)} PASS; "
          f"elapsed {stats['elapsed_sec']}s")
    for c in checks:
        if not c["pass"]:
            print(f"[short_tail]   FAILED CHECK: {c['check']} ({c['detail']})",
                  file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------- report
def write_report(path: Path, stats: dict, records_pilot: list[dict],
                 records_robust: list[dict], cfg: dict) -> None:
    L: list[str] = []
    L.append("# E11 «Short-tail buckets» benchmark — composition report")
    L.append("")
    L.append(f"- generated: {stats['generated']} by `src/benchmark/build_short_tail.py` "
             f"({stats['elapsed_sec']} s); config `{stats['config']}`")
    L.append(f"- pre-registration: {stats['decision_ref']}")
    L.append(f"- buckets: **S0 = 0–3 s** (1 sentence, ≥ {stats['min_human_slice_sec']:g} s "
             "where human audio exists — frozen evaluator constraint), "
             "**S1 = 3–20 s** (1–3 sentences); bucket = HUMAN duration of the span")
    L.append(f"- **pilot set: {stats['n_records_pilot']} texts** (target 12: same 6 roots "
             f"as the frozen pilot × 2 buckets; {stats['n_gaps_pilot']} gaps), "
             f"**robust set: {stats['n_records_robust']} texts** (target 26: 13 frozen "
             f"voices × 2 buckets; {stats['n_gaps_robust']} gaps)")
    L.append(f"- no-overlap guard: {stats['n_guard_ranges']} frozen ranges "
             f"(pilot + robust spans + all reference windows), "
             f"{stats['n_hidden_sample_ids_excluded']} sealed hidden sample_ids "
             "excluded entirely; the two new files are also disjoint from each other")
    L.append(f"- train leakage (8-word shingles, both text fields, lenient-normalized): "
             f"max Jaccard {stats['train_leakage']['max_jaccard']:.6f} over "
             f"{stats['train_leakage']['n_train_docs']} train docs; "
             f"{stats['train_leakage']['n_items_above_threshold']} items above "
             f"{stats['train_leakage']['threshold']} (items with < 8 normalized words "
             "carry 0 shingles by construction and cannot register)")
    L.append("")
    L.append("## How to read WER on the short tail (pre-registered caveat)")
    L.append("")
    L.append("At 0–3 s a reference has ~3–8 words, so **one wrong word ≈ 20–30 pp WER**: "
             "rates on S0 are quantized and a single substitution moves a per-item WER "
             "from 0 % to ≥ 12–33 %. Every analysis table MUST therefore report the "
             "**absolute error counts (S+D+I) next to the summed reference words** and "
             "the ASR floor of the same slices; a bucket-level rate alone is not "
             "interpretable at this length (E11 pre-registration, NB).")
    L.append("")
    for name, recs in (("pilot", records_pilot), ("robust", records_robust)):
        L.append(f"## {name} set — items")
        L.append("")
        L.append("| text_id | bucket | dur s | words | sentences | source | "
                 "span (s in sample) |")
        L.append("|---|---|---:|---:|---:|---|---|")
        for r in recs:
            span = ("-" if r["human_offset_start"] is None
                    else f"{r['human_offset_start']:.2f}–{r['human_offset_end']:.2f}")
            L.append(f"| {r['text_id']} | {r['bucket']} | {r['human_duration_sec']:.2f} | "
                     f"{r['words']} | {r['sentences']} | {r['source']} | {span} |")
        L.append("")
        L.append("| bucket | n | dur s (min/med/max) | words (min/med/max) |")
        L.append("|---|---:|---|---|")
        for b in BUCKET_ORDER:
            pb = stats["per_bucket"][name][b]
            if not pb["n"]:
                L.append(f"| {b} | 0 | - | - |")
                continue
            d, w = pb["duration_sec"], pb["words"]
            L.append(f"| {b} | {pb['n']} | {d['min']:.2f}/{d['median']:.2f}/{d['max']:.2f} | "
                     f"{w['min']}/{w['median']}/{w['max']} |")
        L.append("")
    L.append("## Gaps (buckets that could not be filled under the frozen rules)")
    L.append("")
    if not stats["gaps"]:
        L.append("None.")
    else:
        L.append("| set | root / voice | bucket | reason | detail |")
        L.append("|---|---|---|---|---|")
        for g in stats["gaps"]:
            L.append(f"| {g['set']} | {g['root_id']} | {g['bucket']} | {g['reason']} | "
                     f"{g['detail']} |")
        L.append("")
        L.append("A gap marked `all_clean_spans_blocked_by_overlap_guard` means clean "
                 "material EXISTS but every candidate span intersects audio already "
                 "used by the frozen benchmarks (nested prefixes consume the whole "
                 "12-min window of 3 of the 4 pilot roots up to the B4 cut). The "
                 "guard was applied as pre-registered instead of being weakened.")
    L.append("")
    L.append("## Self-checks")
    L.append("")
    L.append("| check | result | detail |")
    L.append("|---|---|---|")
    for c in stats["self_checks"]:
        L.append(f"| {c['check']} | {'PASS' if c['pass'] else '**FAIL**'} | "
                 f"{c['detail'][:160]} |")
    n_pass = sum(1 for c in stats["self_checks"] if c["pass"])
    L.append("")
    L.append(f"**{n_pass}/{len(stats['self_checks'])} PASS.** (A FAIL on the two "
             "pre-registered volume targets records a data/guard limitation, not a "
             "weakened rule.)")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
