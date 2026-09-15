#!/usr/bin/env python3
"""A1-ext / E6 «Mixed-SFT recipe» — deterministic selection of the `ext_short` corpus
(~100 h of <= 20 s clips) from external/youtube_data_incoming/balalaika.parquet.

Pre-registration (reports/decisions.md, 2026-08-30, Lead):
    pool  : is_single_speaker, asr_consistency_percent == 100, rover non-empty,
            2.0 <= total_duration <= 20.0, music_prob <= 0.5,
            exclude EVERY video_id of the v3.1 dataset (313 ids)
    rank  : DistillMOS descending
    cap   : 6.0 min per video
    stop  : cumulative duration >= 100.0 h
    text  : rover (configs/dataset.yaml text.rover_key), .strip()

The source parquet and the audio under the source-data disk are READ-ONLY.

No silent filtering (PLAN §4.2): every stage reports rows/hours in, out and removed, and
the greedy pass reports how many rows it skipped because of the per-video cap and how many
it never reached (below the MOS cut).

Outputs (data/train/ext_short/):
    selection.jsonl        one row per clip
    selection_stats.json   counts/hours of every stage + distributions

Usage:
    python src/data/select_ext_short.py                       # first pass, no voice gate
    python src/data/select_ext_short.py --exclude_videos v.json   # after the voice gate
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

SRC_PARQUET = "external/youtube_data_incoming/balalaika.parquet"
V31_PLAN = "data/corpus/metadata/selection_plan.parquet"
V31_MANIFEST = "data/manifests/all.jsonl"
DES = "data/train/ext_short"

TARGET_HOURS = 100.0
PER_VIDEO_CAP_SEC = 360.0          # 6.0 min
MIN_DUR, MAX_DUR = 2.0, 20.0
MAX_MUSIC_PROB = 0.5
MIN_CONSISTENCY = 100.0

_WS = re.compile(r"\s+")
_BASE = re.compile(r"^(?P<start>[^_]+)_(?P<end>[^_]+)_(?P<date>[^_]+)_(?P<vid>.+)\.flac$")


def v31_video_ids():
    """Every video_id that the v3.1 long-form dataset uses (union of both records)."""
    import pyarrow.parquet as pq
    ids = set(pq.read_table(V31_PLAN, columns=["video_id"]).column("video_id").to_pylist())
    if os.path.exists(V31_MANIFEST):
        with open(V31_MANIFEST, encoding="utf-8") as f:
            for line in f:
                ids.add(json.loads(line)["video_id"])
    return {str(v) for v in ids}


def _stage(log, name, keep, dur, note=""):
    """Record one filter stage. `keep` is the boolean mask AFTER this stage."""
    prev = log[-1] if log else None
    rows_in = prev["rows_out"] if prev else int(keep.size)
    hours_in = prev["hours_out"] if prev else float(dur.sum() / 3600.0)
    rows_out = int(keep.sum())
    hours_out = float(dur[keep].sum() / 3600.0)
    log.append({"stage": name, "rows_in": rows_in, "rows_out": rows_out,
                "rows_removed": rows_in - rows_out, "hours_in": round(hours_in, 4),
                "hours_out": round(hours_out, 4),
                "hours_removed": round(hours_in - hours_out, 4), "note": note})
    return log


def load_pool(verbose=True):
    """Apply the pre-registered pool filters. Returns (DataFrame, stage_log, totals)."""
    import pyarrow.parquet as pq
    cols = ["filepath", "podcast_id", "speaker_id", "start", "end", "total_duration",
            "is_single_speaker", "DistillMOS", "music_prob", "asr_consistency_percent",
            "rover"]
    df = pq.read_table(SRC_PARQUET, columns=cols).to_pandas()
    dur = df["total_duration"].to_numpy(dtype=np.float64)
    log = []

    keep = np.ones(len(df), dtype=bool)
    _stage(log, "source", keep, dur, note=SRC_PARQUET)

    sing = df["is_single_speaker"].map(lambda v: v is True).to_numpy(dtype=bool)
    keep &= sing
    _stage(log, "is_single_speaker", keep, dur)

    cons = df["asr_consistency_percent"].to_numpy(dtype=np.float64)
    keep &= np.nan_to_num(cons, nan=-1.0) == MIN_CONSISTENCY
    _stage(log, "asr_consistency_percent==100", keep, dur)

    rover_ok = df["rover"].map(lambda s: bool(s) and bool(str(s).strip())).to_numpy(dtype=bool)
    keep &= rover_ok
    _stage(log, "rover_non_empty", keep, dur)

    keep &= (dur >= MIN_DUR) & (dur <= MAX_DUR)
    _stage(log, f"duration_in[{MIN_DUR},{MAX_DUR}]", keep, dur)

    music = df["music_prob"].to_numpy(dtype=np.float64)
    keep &= np.nan_to_num(music, nan=1.0) <= MAX_MUSIC_PROB
    _stage(log, f"music_prob<={MAX_MUSIC_PROB}", keep, dur, note="NaN treated as 1.0 (excluded)")

    v31 = v31_video_ids()
    vid = df["podcast_id"].astype(str).to_numpy()
    in_v31 = np.array([v in v31 for v in vid], dtype=bool)
    overlap_videos = sorted(set(vid[keep & in_v31]))
    keep &= ~in_v31
    _stage(log, "exclude_v3.1_video_ids", keep, dur,
           note=f"{len(v31)} v3.1 video_ids; {len(overlap_videos)} of them present in the pool")

    mos = df["DistillMOS"].to_numpy(dtype=np.float64)
    spk = df["speaker_id"].to_numpy(dtype=np.float64)
    keep &= ~np.isnan(mos) & ~np.isnan(spk)
    _stage(log, "DistillMOS_and_speaker_id_not_null", keep, dur)

    pool = df.loc[keep].copy()
    if verbose:
        for s in log:
            print(f"[sel] {s['stage']:<32s} rows {s['rows_in']:>7d} -> {s['rows_out']:>7d} "
                  f"(-{s['rows_removed']:>6d})  hours {s['hours_in']:>9.2f} -> {s['hours_out']:>9.2f}",
                  flush=True)
    totals = {"v31_video_ids": len(v31), "v31_overlap_videos_in_pool": overlap_videos}
    return pool, log, totals


def utt_of(filepath):
    """`ext_<video_id>_<start>_<end>` built from the on-disk basename (exact, unique, no spaces)."""
    m = _BASE.match(os.path.basename(filepath))
    if not m:
        raise ValueError(f"unexpected clip filename: {filepath}")
    return f"ext_{m.group('vid')}_{m.group('start')}_{m.group('end')}", m.group("vid")


def greedy_select(pool, banned_videos=(), target_hours=TARGET_HOURS,
                  cap_sec=PER_VIDEO_CAP_SEC):
    """DistillMOS-descending greedy with a per-video cap. Deterministic (ties broken by utt).

    Returns (selected_rows, info). `selected_rows` are dicts ready for selection.jsonl.
    """
    banned = set(banned_videos)
    rows = []
    for fp, vid_col, spk, st, en, d, mos, mp, cons, rover in zip(
            pool["filepath"], pool["podcast_id"], pool["speaker_id"], pool["start"],
            pool["end"], pool["total_duration"], pool["DistillMOS"], pool["music_prob"],
            pool["asr_consistency_percent"], pool["rover"]):
        utt, vid_from_name = utt_of(fp)
        assert vid_from_name == str(vid_col), (fp, vid_col)
        rows.append((-float(mos), utt, fp, str(vid_col), int(spk), float(st), float(en),
                     float(d), float(mos), float(mp), float(cons), str(rover)))
    rows.sort()                       # (-MOS, utt): deterministic, ties by utt

    per_video, out = {}, []
    total = 0.0
    target = target_hours * 3600.0
    skipped_cap, skipped_cap_sec, skipped_banned, n_ws_norm = 0, 0.0, 0, 0
    cut_index = -1
    for i, r in enumerate(rows):
        if total >= target:
            cut_index = i - 1
            break
        (_negmos, utt, fp, vid, spk, st, en, d, mos, mp, cons, rover) = r
        if vid in banned:
            skipped_banned += 1
            continue
        used = per_video.get(vid, 0.0)
        if used + d > cap_sec:
            skipped_cap += 1
            skipped_cap_sec += d
            continue
        per_video[vid] = used + d
        total += d
        text = _WS.sub(" ", rover.strip())
        n_ws_norm += int(text != rover.strip())
        out.append({"utt": utt, "video_id": vid, "path": fp, "start": st, "end": en,
                    "duration": round(d, 3), "text": text, "DistillMOS": mos,
                    "music_prob": mp, "asr_consistency_percent": cons,
                    "speaker_id": spk, "spk": f"ext/{vid}/{spk}"})
    else:
        cut_index = len(rows) - 1

    below_cut = rows[cut_index + 1:]
    info = {
        "target_hours": target_hours, "per_video_cap_sec": cap_sec,
        "pool_rows": len(rows), "banned_videos": sorted(banned),
        "selected_clips": len(out), "selected_hours": total / 3600.0,
        "selected_videos": len(per_video),
        "reached_target": total >= target,
        "cut_index": cut_index,
        "mos_min_selected": min(r["DistillMOS"] for r in out) if out else None,
        "mos_max_selected": max(r["DistillMOS"] for r in out) if out else None,
        "mos_max_below_cut": float(-below_cut[0][0]) if below_cut else None,
        "rows_below_cut": len(below_cut),
        "skipped_by_video_cap": skipped_cap,
        "skipped_by_video_cap_hours": skipped_cap_sec / 3600.0,
        "skipped_banned_videos_rows": skipped_banned,
        "text_whitespace_normalised": n_ws_norm,
        "per_video_sec": per_video,
    }
    return out, info


def _pct(a, qs=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray(a, dtype=np.float64)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean()),
                                                                "n": int(a.size)}


def _hist(a, edges):
    a = np.asarray(a, dtype=np.float64)
    c, _ = np.histogram(a, bins=edges)
    return {f"[{edges[i]:g},{edges[i+1]:g})": int(c[i]) for i in range(len(c))}


def build_stats(stage_log, totals, sel, info, elapsed):
    dur = [r["duration"] for r in sel]
    mos = [r["DistillMOS"] for r in sel]
    words = [len(r["text"].split()) for r in sel]
    chars = [len(r["text"]) for r in sel]
    pv = np.array(sorted(info["per_video_sec"].values())) / 3600.0
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_parquet": SRC_PARQUET,
        "rule": {"target_hours": TARGET_HOURS, "per_video_cap_sec": PER_VIDEO_CAP_SEC,
                 "min_duration_sec": MIN_DUR, "max_duration_sec": MAX_DUR,
                 "max_music_prob": MAX_MUSIC_PROB, "min_consistency": MIN_CONSISTENCY,
                 "rank": "DistillMOS desc, ties by utt", "text": "rover.strip()"},
        "stages": stage_log,
        "v31": totals,
        "selection": {k: v for k, v in info.items() if k != "per_video_sec"},
        "mos_threshold_reached": info["mos_min_selected"],
        "n_videos": info["selected_videos"],
        "hours": info["selected_hours"],
        "clips": info["selected_clips"],
        "per_video_hours": {"max": float(pv.max()) if pv.size else None,
                            "median": float(np.median(pv)) if pv.size else None,
                            "min": float(pv.min()) if pv.size else None,
                            "mean": float(pv.mean()) if pv.size else None,
                            "hist": _hist(pv, [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1])},
        "duration_sec": _pct(dur) | {"hist": _hist(dur, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20.001])},
        "DistillMOS": _pct(mos) | {"hist": _hist(mos, [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.9])},
        "text_words": _pct(words) | {"hist": _hist(words, [0, 5, 10, 20, 30, 40, 60, 200])},
        "text_chars": _pct(chars),
        "text_whitespace_normalised": None,   # filled by caller
        "elapsed_sec": elapsed,
    }


def write_selection(sel, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--des", default=DES)
    ap.add_argument("--target_hours", type=float, default=TARGET_HOURS)
    ap.add_argument("--cap_sec", type=float, default=PER_VIDEO_CAP_SEC)
    ap.add_argument("--exclude_videos", default=None,
                    help="json file: list of video_ids (or {excluded_videos: [...]}) banned by the voice gate")
    ap.add_argument("--stats_only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    banned = []
    if args.exclude_videos:
        obj = json.load(open(args.exclude_videos, encoding="utf-8"))
        banned = obj if isinstance(obj, list) else obj["excluded_videos"]
    pool, log, totals = load_pool()
    sel, info = greedy_select(pool, banned, args.target_hours, args.cap_sec)
    stats = build_stats(log, totals, sel, info, time.time() - t0)
    stats["voice_gate_excluded_videos"] = sorted(banned)
    stats["text_whitespace_normalised"] = info["text_whitespace_normalised"]
    if not args.stats_only:
        write_selection(sel, os.path.join(args.des, "selection.jsonl"))
    with open(os.path.join(args.des, "selection_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: stats[k] for k in ("clips", "hours", "n_videos",
                                            "mos_threshold_reached")}, indent=2))
    print(f"[sel] {len(sel)} clips, {info['selected_hours']:.3f} h, "
          f"{info['selected_videos']} videos, MOS >= {info['mos_min_selected']:.4f}, "
          f"{time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
