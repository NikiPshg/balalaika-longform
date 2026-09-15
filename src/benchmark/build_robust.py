#!/usr/bin/env python
"""A2-robust / E8 «Robust-20» (reports/decisions.md 2026-08-31): 20-voice benchmark.

One record = one text x bucket x OWN voice.  Each voice is a dev/test `speaker_key`
that passes the frozen A5 reference QC (src/eval/make_references.py, window 5-10 s per
the E8 prereg), each text comes from that voice's OWN units through the frozen pilot
text pipeline (configs/benchmark_pilot.yaml: prefix_start, cut nearest bucket centre,
disfluency cleaning, charset S1, lenient normalizer), and the reference clip is cut
from a segment that provides NO benchmark text of that voice.

Selection is criteria-based, nothing hand-picked:
  * speakers must satisfy make_references.GATE (>= 3 segments, >= 600 s material, all
    audio-QC gates on the winning window), must not be on A1's voice-leakage watch or
    flagged lists, and their channel must not be a train channel;
  * <= 2 voices per channel; pairwise campplus cosine < 0.70 between ALL chosen voices
    (speaker-level mean of the A1 embedding cache data/manifests/spk_emb.npz);
  * B4-capable voices first, then test before dev, then more buckets, then the better
    reference window (configs/benchmark_robust.yaml `selection_sort`);
  * segments of the SEALED hidden set never provide a text or a reference window.

Run (cosyvoice env, CPU only):

    python src/benchmark/build_robust.py
    ... --dry-run   # selection table only, no audio written
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src" / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import make_references as mr  # noqa: E402  (src/eval; imports audio_qc)
import spk_campplus as sc  # noqa: E402
from src.benchmark import build_pilot as bp  # noqa: E402
from src.benchmark import disfluency as dis  # noqa: E402
from src.data import leakage_check as lc  # noqa: E402
from src.eval import normalize as norm  # noqa: E402

BUCKET_ORDER = ("B0", "B1", "B2", "B3", "B4")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- text pipeline
def prepare_unit(row: dict, dcfg: dict) -> tuple[dict | None, str]:
    """Pilot `prepare_root` minus the WINDOW-level digit/Latin disqualification.

    The charset check moves to the emitted text (`robust_cuts`), so a digit at
    minute 9 disqualifies the B4 cut of the unit, not its B0-B3 cuts.  Everything
    else — prefix_start, transliteration table, case repair, disfluency cleaning,
    density — is the frozen pilot code, called by import.
    """
    groups = bp.word_groups(row["json_path"], dcfg["timestamp_source"])
    if not groups:
        return None, "no_timestamps"
    started = bp.find_prefix_start(groups, dcfg)
    if started is None:
        return None, "no_clean_prefix_start"
    start_index, start_time = started
    window = float(dcfg["selection_window_sec"])
    win_groups = [g for g in groups if g[0] >= start_time and g[1] <= start_time + window]
    if not win_groups:
        return None, "empty_selection_window"
    raw_window_text = " ".join(g[2] for g in win_groups)
    table = {k.lower(): v for k, v in (dcfg.get("latin_transliteration") or {}).items()}
    pieces_source = [g[2] for g in win_groups]
    pieces: list[str] = []
    n_translit = 0
    for _s, _e, txt in win_groups:
        txt, t = bp.transliterate_latin(txt, table)
        n_translit += t
        pieces.append(txt)
    joined = " ".join(pieces)
    repaired, n_case = dis.repair_casing(joined)
    assert len(repaired) == len(joined), "repair_casing must preserve length"
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        if i:
            cursor += 1
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    pieces = [repaired[a:b] for a, b in spans]
    cleaned, edits = dis.clean_pieces(pieces)
    dens, n_fill, n_words = dis.density(raw_window_text)
    return {
        "row": row,
        "start_index": start_index,
        "start_time": start_time,
        "groups": win_groups,
        "pieces_source": pieces_source,
        "pieces_raw": pieces,
        "pieces_clean": cleaned,
        "edits": edits,
        "density": dens,
        "n_filler_tokens": n_fill,
        "n_words_window": n_words,
        "n_case_repairs": n_case,
        "n_transliterations": n_translit,
    }, ""


def robust_cuts(cand: dict, buckets: dict[str, list[float]]) -> dict[str, int]:
    """Per-bucket cut index: sentence boundary nearest the bucket centre whose
    finalized prefix is clean.  Buckets independent.

    Clean = no digit, no Latin (charset S1) AND zero hits of the frozen disfluency
    filter.  The second condition matters only at a cut: `clean_pieces` resolves an
    edit using its continuation, so a prefix ending right between the two halves can
    re-expose a dangler (observed: a `truncated` 'пэер-' at one B3 boundary); such a
    boundary is skipped in favour of the next-nearest one, never hand-edited.
    """
    t0 = cand["start_time"]
    ends = [g[1] - t0 for g in cand["groups"]]
    ok_cut = [i for i, p in enumerate(cand["pieces_clean"])
              if p.strip() and p.strip()[-1] in bp.SENT_END]
    out: dict[str, int] = {}
    for name in sorted(buckets):
        lo, hi = buckets[name]
        centre = (lo + hi) / 2.0
        pool = [i for i in ok_cut if lo <= ends[i] <= hi]
        for i in sorted(pool, key=lambda i: (abs(ends[i] - centre), i)):
            text = dis.finalize(" ".join(cand["pieces_clean"][: i + 1]))
            if re.search(r"\d", text) or re.search(r"[A-Za-z]", text):
                continue
            if dis.scan(text):
                continue
            out[name] = i
            break
    return out


def make_record(voice_id: str, rank: int, cand: dict, bucket: str, k: int,
                pilot_cfg: dict, variant: str) -> dict:
    """One §7.5 benchmark record; mirrors bp.make_dataset_records + voice binding."""
    row = cand["row"]
    root_id = f"rv{rank:02d}_{row['sample_id']}"
    text_tts = dis.finalize(" ".join(cand["pieces_clean"][: k + 1]))
    text_un = dis.finalize(" ".join(cand["pieces_source"][: k + 1]))
    start = cand["start_time"]
    end = cand["groups"][k][1]
    duration = end - start
    prefix_len = len(" ".join(cand["pieces_raw"][: k + 1]))
    in_prefix = [e for e in cand["edits"] if e.start < prefix_len]
    n_edits = len(in_prefix)
    n_removed = sum(len(dis.tokenize(e.surface)) for e in in_prefix if e.kind != "stutter")
    ref = norm.normalize(text_tts, variant)
    return {
        "text_id": f"{root_id}__{bucket}",
        "root_id": root_id,
        "bucket": bucket,
        "genre": None,  # E8 does not stratify by genre (prereg: voice robustness)
        "source": "dataset",
        "voice_id": voice_id,
        "speaker_key": row["speaker_key"],
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
        "prefix_group_index": k,
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


# ------------------------------------------------------------------ voice assembly
def assemble_voice(units_by_sid: dict[str, dict], ref_best_by_sid: dict[str, dict]) -> dict | None:
    """Choose the reference unit + the bucket->unit assignment for one speaker.

    The reference unit is the one whose exclusion loses the fewest buckets, scarcest
    bucket first (B4 > B3 > ... > B0), tie-break by the higher window score.  Then each
    bucket (B4 first) takes a remaining unit, preferring an unused unit, then the
    lowest filler density, then the smallest sample_id (the pilot's frozen ordering).
    """
    buckets_by_sid = {sid: set(u["cuts"]) for sid, u in units_by_sid.items()}
    full = set().union(*buckets_by_sid.values()) if buckets_by_sid else set()

    def coverage(excl: str | None) -> set:
        cov: set = set()
        for sid, bs in buckets_by_sid.items():
            if sid != excl:
                cov |= bs
        return cov

    best = None
    for sid, win in ref_best_by_sid.items():
        cov = coverage(sid)
        lost = tuple(int(b in full and b not in cov) for b in ("B4", "B3", "B2", "B1", "B0"))
        key = (lost, -win["score"], sid)
        if best is None or key < best[0]:
            best = (key, sid, win, cov)
    if best is None:
        return None
    _, ref_sid, ref_win, _cov = best

    assign: dict[str, str] = {}
    used: set = set()
    for b in ("B4", "B3", "B2", "B1", "B0"):
        cands = [(0 if sid not in used else 1, u["density"], sid)
                 for sid, u in units_by_sid.items()
                 if sid != ref_sid and b in u["cuts"]]
        if not cands:
            continue
        cands.sort()
        sid = cands[0][2]
        assign[b] = sid
        used.add(sid)
    if not assign:
        return None
    return {"ref_sid": ref_sid, "ref_win": ref_win, "assign": assign}


def speaker_mean_embeddings(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    by_key: dict[str, list] = defaultdict(list)
    for k, e in zip([str(x) for x in z["speaker_keys"]], z["emb"]):
        n = np.linalg.norm(e)
        if n > 0:
            by_key[k].append(e / n)
    return {k: np.mean(v, axis=0) for k, v in by_key.items()}


# ------------------------------------------------------------------ leakage vs train
def train_leakage(records: list[dict], lcfg: dict) -> dict:
    """Max 8-word-shingle Jaccard of every robust text against the train manifests.

    Same shingle machinery as src/data/leakage_check.py; both sides go through the
    frozen lenient normalizer so ROVER (lowercase) and e2e (cased/punctuated) text
    compare on one token stream.  The held text is `text_ref` (already normalized).
    """
    n = int(lcfg["shingle_n"])
    inv: dict[int, list[int]] = defaultdict(list)
    train_size: list[int] = []
    train_name: list[str] = []
    for path in lcfg["against"]:
        for r in read_jsonl(ROOT / path):
            for field in lcfg["fields"]:
                t = norm.normalize(r.get(field) or "", lcfg["normalize"])
                S = lc.shingles(t, n)
                if not S:
                    continue
                idx = len(train_size)
                train_size.append(len(S))
                train_name.append(f"{r['sample_id']}|{field}")
                for h in S:
                    inv[h].append(idx)
    per_item = []
    for rec in records:
        S = lc.shingles(rec["text_ref"], n)
        hit: Counter = Counter()
        for h in S:
            for idx in inv.get(h, ()):
                hit[idx] += 1
        best, best_tr, best_shared = 0.0, None, 0
        for idx, c in hit.items():
            den = len(S) + train_size[idx] - c
            j = c / den if den else 0.0
            if j > best:
                best, best_tr, best_shared = j, train_name[idx], c
        per_item.append({"text_id": rec["text_id"], "n_shingles": len(S),
                         "max_jaccard": round(best, 6), "shared_shingles": best_shared,
                         "best_train_doc": best_tr})
    mx = max((p["max_jaccard"] for p in per_item), default=0.0)
    return {
        "shingle_n": n,
        "threshold": float(lcfg["max_jaccard"]),
        "against": list(lcfg["against"]),
        "fields": list(lcfg["fields"]),
        "normalize": lcfg["normalize"],
        "n_train_docs": len(train_size),
        "max_jaccard": mx,
        "n_items_above_threshold": sum(1 for p in per_item
                                       if p["max_jaccard"] > float(lcfg["max_jaccard"])),
        "n_items_with_any_shared_shingle": sum(1 for p in per_item if p["shared_shingles"]),
        "per_item_top": sorted(per_item, key=lambda p: -p["max_jaccard"])[:10],
    }


# ------------------------------------------------------------------------ provenance
SOURCES = ("src/benchmark/build_robust.py", "src/benchmark/build_pilot.py",
           "src/benchmark/disfluency.py", "src/eval/make_references.py",
           "src/eval/audio_qc.py", "src/eval/spk_campplus.py", "src/eval/normalize.py",
           "configs/benchmark_robust.yaml", "configs/benchmark_pilot.yaml")


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
            "git_revision": git("rev-parse", "HEAD"),
            "env": mr._pkg_versions(),
            "campplus_onnx": sc.CAMPPLUS_ONNX, "campplus_onnx_sha256": sc.model_sha256()}


# ------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the E8 Robust-20 benchmark")
    ap.add_argument("--config", default=str(ROOT / "configs" / "benchmark_robust.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "benchmark" / "robust.jsonl"))
    ap.add_argument("--stats", default=str(ROOT / "data" / "benchmark" / "robust_stats.json"))
    ap.add_argument("--references-out",
                    default=str(ROOT / "data" / "references" / "references_robust.jsonl"))
    ap.add_argument("--report",
                    default=str(ROOT / "reports" / "benchmark_robust_composition.md"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    t_start = time.time()

    cfg = bp.load_yaml(Path(a.config))
    pilot_cfg = bp.load_yaml(ROOT / cfg["pilot_rules"])
    rcfg, scfg = cfg["robust"], cfg["sources"]
    refcfg, tcfg, lcfg = cfg["reference_clip"], cfg["texts"], cfg["leakage_check"]
    buckets = pilot_cfg["buckets"]
    dcfg = pilot_cfg["dataset_roots"]
    dis.load_spec()
    variant = norm.load_spec()["primary_variant"]

    # E8 prereg fixes the reference window at 5-10 s; make_references scores and gates
    # duration through its module constants, so they are re-pointed HERE, visibly.
    mr.MIN_DUR = float(refcfg["min_dur_sec"])
    mr.MAX_DUR = float(refcfg["max_dur_sec"])
    mr.DUR_CENTRE = 0.5 * (mr.MIN_DUR + mr.MAX_DUR)

    # ---------------- manifests, exclusions, gates ----------------
    rows: list[dict] = []
    for mpath in scfg["manifests"]:
        rows += read_jsonl(ROOT / mpath)
    excluded_sids = {r["sample_id"] for r in read_jsonl(ROOT / scfg["exclude_sample_ids_of"])
                     if r.get("sample_id")}
    split_ch = json.load(open(ROOT / scfg["split_channels"], encoding="utf-8"))
    train_channels = {k for k, v in split_ch["channels"].items() if v["split"] == "train"}
    lk = json.load(open(ROOT / scfg["leakage"], encoding="utf-8"))
    watch = {e["speaker_key"] for e in lk.get("watch") or []}
    flagged = {e["speaker_key"] for e in lk.get("flagged") or []}
    max_cos_to_train = {e["speaker_key"]: {"max_cos": round(e["max_cos"], 6),
                                           "nearest_train_speaker_key": (e["top"][0][1] if e.get("top") else None)}
                        for e in lk.get("voice") or []}
    clusters = json.load(open(ROOT / scfg["voice_clusters"], encoding="utf-8"))
    cluster_of = clusters.get("cluster_of_speaker_key") or {}

    spk_seg: Counter = Counter()
    spk_sec: dict[str, float] = defaultdict(float)
    for r in rows:
        spk_seg[r["speaker_key"]] += 1
        spk_sec[r["speaker_key"]] += float(r["duration_sec"])
    for r in rows:
        r["speaker_n_segments"] = spk_seg[r["speaker_key"]]
        r["speaker_total_sec"] = round(spk_sec[r["speaker_key"]], 3)

    gate_reject: Counter = Counter()
    shortlist: set[str] = set()
    for k in spk_seg:
        ch = k.split("/")[0]
        if spk_seg[k] < mr.GATE["min_speaker_segments"]:
            gate_reject["speaker_segments"] += 1
        elif spk_sec[k] < mr.GATE["min_speaker_total_sec"]:
            gate_reject["speaker_material"] += 1
        elif k in flagged:
            gate_reject["voice_leakage_flagged"] += 1
        elif k in watch:
            gate_reject["voice_watchlist"] += 1
        elif ch in train_channels:
            gate_reject["channel_in_train"] += 1
        else:
            shortlist.add(k)
    print(f"[robust] {len(rows)} dev+test rows, {len(spk_seg)} speaker_keys, "
          f"{len(shortlist)} pass the speaker gates; rejects {dict(gate_reject)}; "
          f"{len(excluded_sids)} hidden segments excluded", file=sys.stderr)

    # ---------------- text units ----------------
    unit_reject: Counter = Counter()
    units: dict[str, dict[str, dict]] = defaultdict(dict)   # speaker -> sid -> cand
    for r in rows:
        if r["speaker_key"] not in shortlist:
            continue
        if r["sample_id"] in excluded_sids:
            unit_reject["hidden_excluded"] += 1
            continue
        if r["asr_consistency"] < float(tcfg["unit_min_asr_consistency"]):
            unit_reject["consistency"] += 1
            continue
        if tcfg["require_single_speaker"] and not r["is_single_speaker"]:
            unit_reject["multi_speaker"] += 1
            continue
        cand, why = prepare_unit(r, dcfg)
        if cand is None:
            unit_reject[why] += 1
            continue
        cuts = robust_cuts(cand, buckets)
        if not cuts:
            unit_reject["no_bucket_cut"] += 1
            continue
        cand["cuts"] = cuts
        units[r["speaker_key"]][r["sample_id"]] = cand
    print(f"[robust] text units: {sum(len(v) for v in units.values())} eligible across "
          f"{len(units)} speakers; unit rejects {dict(unit_reject)}", file=sys.stderr)

    # ---------------- reference windows ----------------
    ref_rows = [r for r in rows if r["speaker_key"] in shortlist
                and r["sample_id"] not in excluded_sids]
    cands = []
    for r in ref_rows:
        cands += mr.candidate_windows(r, mr.MIN_DUR, mr.MAX_DUR,
                                      int(refcfg["min_words"]), 0.8,
                                      min_agreement=float(refcfg["min_agreement"]))
    print(f"[robust] {len(cands)} reference candidate windows "
          f"({len({c['speaker_key'] for c in cands})} speakers)", file=sys.stderr)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        s1 = list(ex.map(mr.stage1, cands, chunksize=8))
    s1 = [c for c in s1 if "error" not in c]
    print(f"[robust] stage-1 QC in {time.time() - t0:.1f}s ({len(s1)} ok)", file=sys.stderr)
    keep = []
    by_spk: dict[str, list] = defaultdict(list)
    for c in s1:
        by_spk[c["speaker_key"]].append(c)
    for k in sorted(by_spk):
        ranked = sorted(by_spk[k], key=lambda c: (-c["score1"], c["sample_id"],
                                                  c["offset_start_sec"]))
        keep += ranked[: int(refcfg["top_per_speaker"])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        s2 = list(ex.map(mr.stage2, keep, chunksize=2))
    print(f"[robust] stage-2 QC on {len(keep)} windows in {time.time() - t0:.1f}s",
          file=sys.stderr)
    f0_by_spk: dict[str, list] = defaultdict(list)
    for c in s2:
        v = (c.get("qc") or {}).get("f0_median_hz")
        if v:
            f0_by_spk[c["speaker_key"]].append(float(v))
    for c in s2:
        vs = f0_by_spk.get(c["speaker_key"]) or []
        c["speaker_f0_median_hz"] = round(float(np.median(vs)), 3) if vs else None
        c["speaker_f0_n_windows"] = len(vs)
        c["speaker_f0_spread_hz"] = round(float(max(vs) - min(vs)), 3) if len(vs) > 1 else 0.0
    scored = [mr.gate_and_score(c) for c in s2]
    win_reject: Counter = Counter()
    passing = []
    for c in scored:
        if c.get("passed"):
            passing.append(c)
        else:
            for g in c["gates_failed"]:
                win_reject[g] += 1
    print(f"[robust] {len(passing)}/{len(scored)} stage-2 windows pass the gates "
          f"({len({c['speaker_key'] for c in passing})} speakers); "
          f"window rejects {dict(win_reject)}", file=sys.stderr)

    ref_best: dict[str, dict[str, dict]] = defaultdict(dict)  # speaker -> sid -> best window
    for c in passing:
        cur = ref_best[c["speaker_key"]].get(c["sample_id"])
        if cur is None or (c["score"], c["sample_id"]) > (cur["score"], cur["sample_id"]):
            ref_best[c["speaker_key"]][c["sample_id"]] = c

    # ---------------- assemble voices ----------------
    voice_cands = []
    for k in sorted(shortlist):
        if k not in ref_best or k not in units:
            continue
        asm = assemble_voice(units[k], ref_best[k])
        if asm is None:
            continue
        voice_cands.append({
            "speaker_key": k,
            "split": next(r["split"] for r in rows if r["speaker_key"] == k),
            "channel_id": k.split("/")[0],
            "has_b4": "B4" in asm["assign"],
            "n_buckets": len(asm["assign"]),
            "ref_score": asm["ref_win"]["score"],
            **asm,
        })
    order = {"test": 0, "dev": 1}
    voice_cands.sort(key=lambda v: (0 if v["has_b4"] else 1, order[v["split"]],
                                    -v["n_buckets"], -v["ref_score"], v["speaker_key"]))

    mean_emb = speaker_mean_embeddings(ROOT / scfg["spk_emb_cache"])
    missing_emb = [v["speaker_key"] for v in voice_cands if v["speaker_key"] not in mean_emb]
    if missing_emb:  # cache is complete on 2026-08-31; re-embed on CPU if it ever is not
        print(f"[robust] {len(missing_emb)} speakers missing from the embedding cache; "
              "re-embedding up to 3 longest segments each (CPU)", file=sys.stderr)
        by_key_rows: dict[str, list] = defaultdict(list)
        for r in rows:
            by_key_rows[r["speaker_key"]].append(r)
        for k in missing_emb:
            embs = []
            for r in sorted(by_key_rows[k], key=lambda r: -r["duration_sec"])[:3]:
                e = sc.embed_file(r["audio_path"], max_sec=30.0, center=True)
                embs.append(e / np.linalg.norm(e))
            mean_emb[k] = np.mean(embs, axis=0)

    cos_max = float(rcfg["pairwise_campplus_cosine_max"])
    chosen: list[dict] = []
    sel_reject: list[dict] = []
    chan_count: Counter = Counter()
    for v in voice_cands:
        if len(chosen) >= int(rcfg["n_voices_target"]):
            break
        if chan_count[v["channel_id"]] >= int(rcfg["max_voices_per_channel"]):
            sel_reject.append({"speaker_key": v["speaker_key"], "reason": "channel_cap"})
            continue
        e = mean_emb[v["speaker_key"]]
        worst = max((sc.cosine(e, mean_emb[c["speaker_key"]]) for c in chosen), default=-1.0)
        if worst >= cos_max:
            sel_reject.append({"speaker_key": v["speaker_key"], "reason": "cosine",
                               "max_cos_to_chosen": round(worst, 4)})
            continue
        v["max_cos_to_prior_chosen"] = round(worst, 6) if chosen else None
        chosen.append(v)
        chan_count[v["channel_id"]] += 1
    n_b4 = sum(1 for v in chosen if v["has_b4"])
    print(f"[robust] chose {len(chosen)} voices (target {rcfg['n_voices_target']}), "
          f"{n_b4} with B4 (prereg target {rcfg['min_b4_voices_target']}), "
          f"{len(chan_count)} channels; selection rejects: {sel_reject}", file=sys.stderr)

    if a.dry_run:
        for i, v in enumerate(chosen, 1):
            print(f"  ref_r{i:02d} {v['speaker_key']:55s} {v['split']:4s} "
                  f"buckets={''.join(b[1] for b in sorted(v['assign']))} "
                  f"ref={v['ref_sid']} score={v['ref_score']:.3f}")
        return 0
    if not chosen:
        raise SystemExit("no voice passed all gates")

    # ---------------- write reference wavs + benchmark records ----------------
    out_dir = ROOT / refcfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i, v in enumerate(chosen, 1):
        v["voice_id"] = rcfg["voice_id_format"].format(rank=i)
        v["wav_fields"] = mr.write_reference_wavs(v["ref_win"], v["voice_id"], str(out_dir))
        for b in sorted(v["assign"]):
            cand = units[v["speaker_key"]][v["assign"][b]]
            records.append(make_record(v["voice_id"], i, cand, b, cand["cuts"][b],
                                       pilot_cfg, variant))
    records.sort(key=lambda r: (r["voice_id"], r["bucket"]))

    # campplus of the WRITTEN reference wavs: pairwise matrix over the chosen set
    ref_emb = {v["voice_id"]: sc.embed_file(str(out_dir / f"{v['voice_id']}.wav"))
               for v in chosen}
    ids = [v["voice_id"] for v in chosen]
    pair_mean, pair_ref = {}, {}
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            kx, ky = ids[x], ids[y]
            key = f"{kx}|{ky}"
            pair_mean[key] = round(sc.cosine(mean_emb[chosen[x]["speaker_key"]],
                                             mean_emb[chosen[y]["speaker_key"]]), 6)
            pair_ref[key] = round(sc.cosine(ref_emb[kx], ref_emb[ky]), 6)

    prov = provenance()
    ref_records = []
    for v in chosen:
        c, q = v["ref_win"], v["ref_win"]["qc"]
        others_mean = [pair_mean[k] for k in pair_mean if v["voice_id"] in k.split("|")]
        others_ref = [pair_ref[k] for k in pair_ref if v["voice_id"] in k.split("|")]
        ref_records.append({
            "voice_id": v["voice_id"],
            "role": "primary",
            "gender_heuristic": c["gender"],
            "gender_method": ("A5 heuristic (speaker-level median pyin F0 vs 165 Hz, "
                              "±15 Hz ambiguity band); recorded, NOT a selection axis in E8"),
            "gender_confidence": c["gender_confidence"],
            "speaker_f0_median_hz": c["speaker_f0_median_hz"],
            "speaker_f0_n_windows": c["speaker_f0_n_windows"],
            "speaker_f0_spread_hz": c["speaker_f0_spread_hz"],
            "split": v["split"],
            "channel_id": c["channel_id"],
            "channel_title": c["channel_title"],
            "video_id": c["video_id"],
            "video_title": c["video_title"],
            "speaker_key": c["speaker_key"],
            "voice_cluster": cluster_of.get(c["speaker_key"]),
            "source_sample_id": c["sample_id"],
            "source_audio_path": c["audio_path"],
            "source_json_path": c["json_path"],
            "source_segment_duration_sec": c["segment_duration_sec"],
            "source_segment_start_in_video_sec": c["segment_start_in_video_sec"],
            "offset_start_sec": c["offset_start_sec"],
            "offset_end_sec": c["offset_end_sec"],
            "start_in_video_sec": round(c["segment_start_in_video_sec"]
                                        + c["offset_start_sec"], 3),
            "duration_sec": q["duration_sec"],
            "ref_text": c["ref_text"],
            "ref_text_source": "asr_ts['gigaam-v3-e2e-ctc'] word groups inside the window "
                               "(punctuated, verbatim)",
            "n_words": c["n_words"],
            "asr_consistency": c["asr_consistency"],
            "window_asr_agreement": c["window_asr_agreement"],
            "window_asr_agreement_per_system": c["window_asr_agreement_per_system"],
            "window_asr_agreement_min": c["window_asr_agreement_min"],
            **v["wav_fields"],
            "qc": q,
            "selection_score": c["score"],
            "selection_rank": int(v["voice_id"][-2:]),
            "buckets_provided": sorted(v["assign"]),
            "has_b4": v["has_b4"],
            "text_sample_ids": sorted(set(v["assign"].values())),
            "reference_unit_in_texts": v["ref_sid"] in set(v["assign"].values()),
            "campplus": {
                "spk_mean_max_cos_to_other_chosen": max(others_mean) if others_mean else None,
                "ref_wav_max_cos_to_other_chosen": max(others_ref) if others_ref else None,
                "pairwise_gate": cos_max,
                "model_sha256": prov["campplus_onnx_sha256"],
            },
            "leakage": {
                "channel_in_train": c["channel_id"] in train_channels,
                "on_voice_watchlist": c["speaker_key"] in watch,
                "voice_leakage_flagged": c["speaker_key"] in flagged,
                "leakage_report": scfg["leakage"],
                "speaker_max_cos_to_train": max_cos_to_train.get(c["speaker_key"]),
            },
            "build": {"config": str(Path(a.config).relative_to(ROOT)),
                      "decision_ref": cfg["decision_ref"], **prov},
        })

    # ---------------- leakage vs train ----------------
    leak = train_leakage(records, lcfg)
    print(f"[robust] train leakage: max Jaccard {leak['max_jaccard']:.6f} "
          f"({leak['n_items_above_threshold']} items above {leak['threshold']})",
          file=sys.stderr)

    # ---------------- self-checks ----------------
    ref_by_voice = {r["voice_id"]: r for r in ref_records}
    items_by_voice: dict[str, list] = defaultdict(list)
    for r in records:
        items_by_voice[r["voice_id"]].append(r)
    forbidden = set(dis.QUOTE_CHARS + dis.APOSTROPHE_CHARS + "()[]{}«»–‒―−")
    checks: list[dict] = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    check("n_voices <= target", len(chosen) <= int(rcfg["n_voices_target"]),
          f"{len(chosen)} of {rcfg['n_voices_target']}")
    check("<= 2 voices per channel", all(n <= 2 for n in chan_count.values()),
          dict(chan_count))
    check("pairwise speaker-mean campplus cosine < 0.70",
          all(vv < cos_max for vv in pair_mean.values()),
          f"max {max(pair_mean.values()) if pair_mean else 0}")
    check("reference durations in [5, 10] s",
          all(mr.MIN_DUR <= r["duration_sec"] <= mr.MAX_DUR for r in ref_records),
          sorted(round(r["duration_sec"], 2) for r in ref_records))
    check("every reference passed all QC gates",
          all(not v["ref_win"]["gates_failed"] for v in chosen))
    check("reference unit provides no text of its voice",
          all(not r["reference_unit_in_texts"] for r in ref_records))
    check("every item's voice exists and matches its speaker_key",
          all(r["voice_id"] in ref_by_voice
              and r["speaker_key"] == ref_by_voice[r["voice_id"]]["speaker_key"]
              for r in records))
    check("unique text_id", len({r["text_id"] for r in records}) == len(records))
    check("<= 1 text per voice x bucket",
          all(len({r["bucket"] for r in rs}) == len(rs) for rs in items_by_voice.values()))
    check("bucket = human duration inside its frozen edges",
          all(buckets[r["bucket"]][0] <= r["human_duration_sec"] <= buckets[r["bucket"]][1]
              for r in records))
    check("charset S1: no digit / Latin / quote / apostrophe, 0 disfluency hits, "
          "finalize idempotent",
          all(not re.search(r"[0-9A-Za-z]", r["text_tts"])
              and not (set(r["text_tts"]) & forbidden)
              and not dis.scan(r["text_tts"])
              and dis.finalize(r["text_tts"]) == r["text_tts"] for r in records))
    check("text_ref == normalize(text_tts, primary variant)",
          all(r["text_ref"] == norm.normalize(r["text_tts"], variant) for r in records))
    check("no hidden segment provides a text or a reference window",
          not ({r["sample_id"] for r in records} & excluded_sids)
          and not ({r["source_sample_id"] for r in ref_records} & excluded_sids))
    check("context occupancy < 1", all(0 < r["context_occupancy_est"] < 1 for r in records))
    check(f"train leakage max Jaccard < {leak['threshold']}",
          leak["max_jaccard"] < leak["threshold"], f"{leak['max_jaccard']:.6f}")
    check("every voice has >= 1 bucket", all(items_by_voice[v["voice_id"]] for v in chosen))
    b4_ok = n_b4 >= int(rcfg["min_b4_voices_target"])
    check(f"B4 voices >= {rcfg['min_b4_voices_target']} (prereg target)", b4_ok,
          f"{n_b4} — data-limited if failed; gates were NOT weakened")

    # ---------------- write outputs ----------------
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(a.references_out, "wt", encoding="utf-8") as f:
        for rec in ref_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    per_bucket = Counter(r["bucket"] for r in records)
    per_split = Counter(v["split"] for v in chosen)
    stats = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(Path(a.config).relative_to(ROOT)),
        "pilot_rules": cfg["pilot_rules"],
        "decision_ref": cfg["decision_ref"],
        "elapsed_sec": round(time.time() - t_start, 1),
        "normalization_variant": variant,
        "reference_window_sec": [mr.MIN_DUR, mr.MAX_DUR],
        "n_manifest_rows": len(rows),
        "n_speaker_keys": len(spk_seg),
        "n_speakers_pass_gates": len(shortlist),
        "speaker_gate_rejects": dict(gate_reject),
        "n_hidden_sample_ids_excluded": len(excluded_sids),
        "unit_rejects": dict(unit_reject),
        "reference_window_rejects": dict(win_reject),
        "n_voice_candidates": len(voice_cands),
        "selection_rejects": sel_reject,
        "n_voices": len(chosen),
        "n_voices_b4": n_b4,
        "n_channels": len(chan_count),
        "voices_per_split": dict(per_split),
        "n_records": len(records),
        "records_per_bucket": {b: per_bucket.get(b, 0) for b in BUCKET_ORDER},
        "voices": [{
            "voice_id": v["voice_id"], "speaker_key": v["speaker_key"], "split": v["split"],
            "channel_id": v["channel_id"], "buckets": sorted(v["assign"]),
            "ref_sample_id": v["ref_sid"], "ref_score": round(v["ref_score"], 6),
            "text_sample_ids": sorted(set(v["assign"].values())),
            "voice_cluster": cluster_of.get(v["speaker_key"]),
        } for v in chosen],
        "pairwise_cosine_speaker_mean": pair_mean,
        "pairwise_cosine_reference_wavs": pair_ref,
        "pairwise_cosine_speaker_mean_max": max(pair_mean.values()) if pair_mean else None,
        "pairwise_cosine_speaker_mean_min": min(pair_mean.values()) if pair_mean else None,
        "pilot_sample_id_overlap": sorted({r["sample_id"] for r in records}
                                          & {x["sample_id"] for x in
                                             read_jsonl(ROOT / "data/benchmark/pilot.jsonl")
                                             if x.get("sample_id")}),
        "train_leakage": leak,
        "self_checks": checks,
        "build": prov,
    }
    Path(a.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    write_report(Path(a.report), stats, records, ref_records, cfg)

    n_pass = sum(1 for c in checks if c["pass"])
    print(f"[robust] wrote {len(records)} records / {len(chosen)} voices -> {out_path}; "
          f"self-checks {n_pass}/{len(checks)} PASS; elapsed {stats['elapsed_sec']}s")
    for c in checks:
        if not c["pass"]:
            print(f"[robust]   FAILED CHECK: {c['check']} ({c['detail']})", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------- report
def write_report(path: Path, stats: dict, records: list[dict],
                 ref_records: list[dict], cfg: dict) -> None:
    per_bucket = stats["records_per_bucket"]
    L: list[str] = []
    L.append("# E8 «Robust-20» benchmark — composition report")
    L.append("")
    L.append(f"- generated: {stats['generated']} by `src/benchmark/build_robust.py` "
             f"({stats['elapsed_sec']} s); config `{stats['config']}` "
             f"(text rules: `{stats['pilot_rules']}`)")
    L.append(f"- pre-registration: {cfg['decision_ref']}")
    L.append(f"- **{stats['n_voices']} voices** ({stats['voices_per_split']}), "
             f"**{stats['n_records']} texts**, {stats['n_channels']} channels, "
             f"{stats['n_voices_b4']} voices with B4")
    L.append(f"- per bucket: " + ", ".join(f"{b} {per_bucket[b]}" for b in BUCKET_ORDER))
    L.append(f"- pairwise campplus cosine (speaker means): max "
             f"{stats['pairwise_cosine_speaker_mean_max']}, min "
             f"{stats['pairwise_cosine_speaker_mean_min']} (gate < "
             f"{cfg['robust']['pairwise_campplus_cosine_max']})")
    L.append(f"- train leakage (8-word shingles, both text fields, lenient-normalized): "
             f"max Jaccard {stats['train_leakage']['max_jaccard']:.6f} over "
             f"{stats['train_leakage']['n_train_docs']} train docs; "
             f"{stats['train_leakage']['n_items_above_threshold']} items above "
             f"{stats['train_leakage']['threshold']}")
    L.append(f"- pilot unit overlap (dev side, informational): "
             f"{len(stats['pilot_sample_id_overlap'])} shared sample_id(s): "
             f"{stats['pilot_sample_id_overlap']}")
    L.append(f"- hidden set: {stats['n_hidden_sample_ids_excluded']} sealed segments "
             "excluded from texts AND reference windows; zero overlap asserted below")
    L.append("")
    L.append("## Voices")
    L.append("")
    L.append("| voice | speaker_key | split | buckets | ref segment | ref s | SNR dB | "
             "gender* | words B0-B4 |")
    L.append("|---|---|---|---|---|---:|---:|---|---|")
    by_voice: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_voice[r["voice_id"]][r["bucket"]] = r
    for rr in ref_records:
        vb = by_voice[rr["voice_id"]]
        words = "/".join(str(vb[b]["words"]) if b in vb else "-" for b in BUCKET_ORDER)
        L.append(f"| {rr['voice_id']} | `{rr['speaker_key']}` | {rr['split']} | "
                 f"{''.join(b[1] for b in rr['buckets_provided'])} | "
                 f"`{rr['source_sample_id']}` | {rr['duration_sec']:.2f} | "
                 f"{rr['qc']['snr_db']:.1f} | {rr['gender_heuristic']} | {words} |")
    L.append("")
    L.append("\\* gender is the A5 F0 heuristic, recorded only (not a selection axis).")
    L.append("")
    L.append("## Text stats per bucket")
    L.append("")
    L.append("| bucket | texts | human min | words (min/med/max) | tokens max | occ max |")
    L.append("|---|---:|---:|---|---:|---:|")
    for b in BUCKET_ORDER:
        rs = [r for r in records if r["bucket"] == b]
        if not rs:
            L.append(f"| {b} | 0 | - | - | - | - |")
            continue
        ws = sorted(r["words"] for r in rs)
        L.append(f"| {b} | {len(rs)} | {sum(r['human_duration_sec'] for r in rs) / 60:.1f} | "
                 f"{ws[0]}/{ws[len(ws) // 2]}/{ws[-1]} | "
                 f"{max(r['text_tts_tokens'] for r in rs)} | "
                 f"{max(r['context_occupancy_est'] for r in rs):.3f} |")
    L.append("")
    L.append("## Funnel")
    L.append("")
    L.append(f"- {stats['n_manifest_rows']} dev+test rows, {stats['n_speaker_keys']} "
             f"speaker_keys -> {stats['n_speakers_pass_gates']} past the speaker gates "
             f"(rejects: {stats['speaker_gate_rejects']})")
    L.append(f"- unit rejects: {stats['unit_rejects']}")
    L.append(f"- reference-window gate rejects: {stats['reference_window_rejects']}")
    L.append(f"- {stats['n_voice_candidates']} assembled voice candidates -> "
             f"{stats['n_voices']} chosen; selection rejects: {stats['selection_rejects']}")
    L.append("")
    L.append("## Self-checks")
    L.append("")
    L.append("| check | result | detail |")
    L.append("|---|---|---|")
    for c in stats["self_checks"]:
        L.append(f"| {c['check']} | {'PASS' if c['pass'] else '**FAIL**'} | {c['detail'][:120]} |")
    n_pass = sum(1 for c in stats["self_checks"] if c["pass"])
    L.append("")
    L.append(f"**{n_pass}/{len(stats['self_checks'])} PASS.**")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
