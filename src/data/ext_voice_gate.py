#!/usr/bin/env python3
"""A1-ext / E6 — voice-leakage gate for the `ext_short` corpus (PLAN §6.2 rule, reused).

For every video that the DistillMOS greedy (src/data/select_ext_short.py) selects, up to
`--clips_per_video` of its longest selected clips are embedded with campplus EXACTLY as
src/data/leakage_check.py::embed_one does (soundfile -> mono -> 16 kHz -> kaldi fbank 80,
dither 0 -> per-utterance mean subtraction -> campplus.onnx, CPUExecutionProvider). The
per-video embedding is the mean of those clip embeddings, L2-normalised — the same
"average, then normalise" rule that leakage_check.py::stage_report applies per speaker_key.

Decision statistic: max cosine to ANY dev/test speaker_key embedding in
data/manifests/spk_emb.npz (dev/test membership from data/manifests/split_channels.json,
cross-checked against data/manifests/{dev,test}.jsonl).

Threshold: tau = 0.740633738040924 — the frozen voice threshold of reports/leakage_report.md
(p95 of the train-internal null: max cosine of a train speaker_key to any speaker_key of a
different channel), read from reports/leakage_raw.json so the number is never re-typed.
A video with max cosine >= tau is EXCLUDED and the greedy is re-run with it banned, which
pulls in the next videos in MOS order until 100 h is restored. Iterate until no newly
selected video is flagged (cap --max_iter).

Outputs:
    data/train/ext_short/voice_gate.json   per-video max cosine, excluded list, iterations
    data/train/ext_short/video_emb.npz     cache of per-video mean embeddings
    data/train/ext_short/selection.jsonl   final (post-gate) selection
    data/train/ext_short/selection_stats.json

Usage:
    python src/data/ext_voice_gate.py --workers 4
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.select_ext_short import (DES, load_pool, greedy_select, build_stats,  # noqa: E402
                                       write_selection, TARGET_HOURS, PER_VIDEO_CAP_SEC)

LEAK_RAW = "reports/leakage_raw.json"
EMB_CACHE = "data/manifests/spk_emb.npz"
SPLIT_JSON = "data/manifests/split_channels.json"


def frozen_tau():
    raw = json.load(open(LEAK_RAW, encoding="utf-8"))
    return float(raw["calibration"]["tau"]), raw["calibration"]["tau_rule"]


def heldout_matrix():
    """(keys, M) — per dev/test speaker_key mean embedding, L2-normalised (rows of M)."""
    z = np.load(EMB_CACHE, allow_pickle=True)
    key_of = {str(s): str(k) for s, k in zip(z["sample_ids"], z["speaker_keys"])}
    emb = {str(s): e for s, e in zip(z["sample_ids"], z["emb"])}
    meta = json.load(open(SPLIT_JSON, encoding="utf-8"))
    chan_split = {c: m["split"] for c, m in meta["channels"].items()}
    by_key = {}
    for s, k in key_of.items():
        if chan_split.get(k.split("/")[0]) in ("dev", "test"):
            by_key.setdefault(k, []).append(emb[s])
    # cross-check: every speaker_key that dev.jsonl / test.jsonl mentions must be covered
    want = set()
    for p in ("data/manifests/dev.jsonl", "data/manifests/test.jsonl"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                want.add(json.loads(line)["speaker_key"])
    missing = sorted(want - set(by_key))
    if missing:
        raise SystemExit(f"[gate] {len(missing)} dev/test speaker_keys have no embedding "
                         f"(e.g. {missing[:3]}) — run src/data/leakage_check.py --embed")
    keys = sorted(by_key)
    M = np.stack([np.mean(by_key[k], axis=0) for k in keys]).astype(np.float64)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return keys, M, sorted(want)


def load_cache(path):
    if not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=True)
    return {str(v): (e, int(n)) for v, e, n in zip(z["videos"], z["emb"], z["n_clips"])}


def save_cache(path, cache):
    vids = sorted(cache)
    np.savez(path, videos=np.array(vids),
             emb=np.stack([cache[v][0] for v in vids]).astype(np.float32),
             n_clips=np.array([cache[v][1] for v in vids], dtype=np.int32))


def clips_for(sel, videos, max_clips):
    """video -> up to max_clips longest selected clips (deterministic: -duration, utt)."""
    by = {}
    for r in sel:
        if r["video_id"] in videos:
            by.setdefault(r["video_id"], []).append(r)
    out = {}
    for v, rs in by.items():
        rs.sort(key=lambda r: (-r["duration"], r["utt"]))
        out[v] = [(r["utt"], r["path"], r["duration"]) for r in rs[:max_clips]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--des", default=DES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clips_per_video", type=int, default=10)
    ap.add_argument("--max_iter", type=int, default=5)
    ap.add_argument("--target_hours", type=float, default=TARGET_HOURS)
    ap.add_argument("--cap_sec", type=float, default=PER_VIDEO_CAP_SEC)
    args = ap.parse_args()

    t0 = time.time()
    tau, tau_rule = frozen_tau()
    keys, M, heldout_keys = heldout_matrix()
    print(f"[gate] tau={tau:.6f} ({tau_rule}); dev/test speaker_keys={len(keys)}", flush=True)

    pool_df, stage_log, totals = load_pool()
    cache_path = os.path.join(args.des, "video_emb.npz")
    cache = load_cache(cache_path)
    maxcos, nearest, n_used = {}, {}, {}
    failed_clips = []

    def score(vids):
        for v in vids:
            e = cache[v][0].astype(np.float64)
            e = e / (np.linalg.norm(e) + 1e-12)
            s = M @ e
            j = int(np.argmax(s))
            maxcos[v] = float(s[j])
            nearest[v] = keys[j]
            n_used[v] = cache[v][1]

    banned, iterations = [], []
    sel, info = None, None
    for it in range(1, args.max_iter + 1):
        sel, info = greedy_select(pool_df, banned, args.target_hours, args.cap_sec)
        videos = sorted({r["video_id"] for r in sel})
        todo = [v for v in videos if v not in cache]
        n_clip_tasks = 0
        if todo:
            want = clips_for(sel, set(todo), args.clips_per_video)
            tasks = [(f"{v}##{u}", p, d) for v in todo for (u, p, d) in want[v]]
            n_clip_tasks = len(tasks)
            print(f"[gate] iter {it}: {len(videos)} videos selected, {len(todo)} new, "
                  f"{len(tasks)} clips to embed", flush=True)
            got = {}
            ctx = multiprocessing.get_context("spawn")     # never fork after torch/onnxruntime
            t1 = time.time()
            from src.data.leakage_check import embed_one
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
                for i, (sid, emb, err) in enumerate(ex.map(embed_one, tasks, chunksize=8)):
                    v = sid.split("##")[0]
                    if err:
                        failed_clips.append({"id": sid, "error": err})
                    else:
                        got.setdefault(v, []).append(emb)
                    if (i + 1) % 2000 == 0:
                        print(f"[gate]   embed {i+1}/{len(tasks)} "
                              f"{time.time()-t1:.0f}s", flush=True)
            for v in todo:
                if v not in got:
                    # cannot verify -> ban it (explicit, counted; never silently kept)
                    cache[v] = (np.zeros(M.shape[1], dtype=np.float32), 0)
                else:
                    cache[v] = (np.mean(got[v], axis=0).astype(np.float32), len(got[v]))
            save_cache(cache_path, cache)
        score(videos)
        unverifiable = [v for v in videos if n_used[v] == 0]
        flagged = sorted([v for v in videos if n_used[v] > 0 and maxcos[v] >= tau]
                         + unverifiable)
        new_flagged = [v for v in flagged if v not in banned]
        iterations.append({
            "iteration": it, "videos_selected": len(videos), "videos_embedded_this_iter": len(todo),
            "clips_embedded_this_iter": n_clip_tasks,
            "hours": info["selected_hours"], "clips": info["selected_clips"],
            "flagged_now": new_flagged, "n_flagged_now": len(new_flagged),
            "unverifiable": unverifiable,
            "max_cos_over_selected": max((maxcos[v] for v in videos), default=None),
            "banned_before": list(banned),
        })
        print(f"[gate] iter {it}: hours={info['selected_hours']:.3f} videos={len(videos)} "
              f"max_cos={max(maxcos[v] for v in videos):.4f} newly flagged={len(new_flagged)}",
              flush=True)
        if not new_flagged:
            converged = True
            break
        banned = sorted(set(banned) | set(new_flagged))
    else:
        converged = False

    videos_final = sorted({r["video_id"] for r in sel})
    cos_final = np.array([maxcos[v] for v in videos_final])
    gate = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tau": tau, "tau_rule": tau_rule, "tau_source": LEAK_RAW,
        "rule": "video excluded if max cosine to any dev/test speaker_key >= tau",
        "clips_per_video": args.clips_per_video, "workers": args.workers,
        "heldout_speaker_keys": len(keys),
        "iterations": iterations, "n_iterations": len(iterations), "converged": converged,
        "excluded_videos": banned, "n_excluded_videos": len(banned),
        "failed_clips": failed_clips[:50], "n_failed_clips": len(failed_clips),
        "videos_scored": len(maxcos),
        "max_cos_distribution_final": {
            "n": int(cos_final.size), "max": float(cos_final.max()),
            **{f"p{q}": float(np.percentile(cos_final, q)) for q in (50, 90, 95, 99, 99.9)},
            "n_above_watch_p90_null_0.7246": int((cos_final >= 0.7245622038841248).sum()),
        },
        "per_video_max_cos": {v: {"max_cos": round(maxcos[v], 6), "nearest": nearest[v],
                                  "n_clips": n_used[v]} for v in sorted(maxcos)},
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(args.des, "voice_gate.json"), "w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=2)

    write_selection(sel, os.path.join(args.des, "selection.jsonl"))
    stats = build_stats(stage_log, totals, sel, info, time.time() - t0)
    stats["voice_gate_excluded_videos"] = banned
    stats["text_whitespace_normalised"] = info["text_whitespace_normalised"]
    stats["voice_gate"] = {k: gate[k] for k in ("tau", "tau_rule", "n_excluded_videos",
                                                "n_iterations", "converged",
                                                "max_cos_distribution_final")}
    with open(os.path.join(args.des, "selection_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[gate] done: excluded {len(banned)} videos in {len(iterations)} iterations, "
          f"converged={converged}, final {info['selected_clips']} clips / "
          f"{info['selected_hours']:.3f} h / {info['selected_videos']} videos, "
          f"max_cos={cos_final.max():.4f}, {time.time()-t0:.0f} s", flush=True)
    if not converged:
        print("[gate] WARNING: gate did not converge within --max_iter", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
