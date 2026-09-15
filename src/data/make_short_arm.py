#!/usr/bin/env python
"""A1 / PLAN.md §6.3 — Short arm: cut every train_long segment into 10-30 s pieces.

No new audio is written. Each row of data/manifests/train_short.jsonl points at the
SAME parent .flac with (offset_start, offset_end) in seconds inside that file.

Boundaries come from the word-group timestamps stored in the per-segment json
(`asr_ts['gigaam-v3-e2e-ctc']`, lines "start\\tend\\ttext"). Cut points are scored by
inter-group silence + sentence punctuation of the preceding group, and a DP picks the
segmentation minimising (duration deviation from 20 s) + (cut cost) under the 10-30 s
constraint. Windows are contiguous and cover the whole parent file, so
  sum(short durations) == parent duration   and   sum(short words) == parent words.

Parent text is the ROVER transcript (PLAN §6.1); it is aligned to the e2e token stream
with difflib so every ROVER token is attached to exactly one timestamped group. Nothing
is dropped for being 'hard'.

Usage:
  python src/data/make_short_arm.py --in data/manifests/train_long.jsonl \
      --out data/manifests/train_short.jsonl
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

E2E_KEY = "gigaam-v3-e2e-ctc"
MIN_DUR, MAX_DUR, TARGET_DUR = 10.0, 30.0, 20.0
W_DUR, W_CUT = 1.0, 1.0
SENT_END = re.compile(r"[.!?…]\s*$")
SOFT_END = re.compile(r"[,;:—]\s*$")
PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)


def norm_tok(t):
    t = t.lower().replace("ё", "е")
    t = PUNCT.sub("", t)
    return t.replace("-", "")


def parse_ts(ts):
    lines = []
    for ln in ts.split("\n"):
        p = ln.split("\t")
        if len(p) != 3:
            continue
        try:
            a, b = float(p[0]), float(p[1])
        except ValueError:
            continue
        if p[2].strip():
            lines.append((a, b, p[2]))
    return lines


def align_tokens_to_lines(lines, rov_tokens):
    """Return line_of[j] for every ROVER token j (monotone, no gaps)."""
    e2e_words, e2e_line = [], []
    for i, (_, _, t) in enumerate(lines):
        for w in t.split():
            nw = norm_tok(w)
            if nw:
                e2e_words.append(nw)
                e2e_line.append(i)
    rov_norm = [norm_tok(w) for w in rov_tokens]
    line_of = [None] * len(rov_tokens)
    if e2e_words:
        sm = difflib.SequenceMatcher(None, e2e_words, rov_norm, autojunk=False)
        for i1, j1, size in sm.get_matching_blocks():
            for k in range(size):
                line_of[j1 + k] = e2e_line[i1 + k]
    # forward/backward fill + linear interpolation for unmatched ROVER tokens
    n = len(rov_tokens)
    known = [j for j in range(n) if line_of[j] is not None]
    if not known:
        return [0] * n if lines else [0] * n
    for j in range(known[0]):
        line_of[j] = line_of[known[0]]
    for j in range(known[-1] + 1, n):
        line_of[j] = line_of[known[-1]]
    for a, b in zip(known, known[1:]):
        if b - a > 1:
            la, lb = line_of[a], line_of[b]
            for k in range(a + 1, b):
                frac = (k - a) / (b - a)
                line_of[k] = int(round(la + frac * (lb - la)))
    # enforce monotonicity
    for j in range(1, n):
        if line_of[j] < line_of[j - 1]:
            line_of[j] = line_of[j - 1]
    return line_of


def cut_quality(lines, i):
    """Quality in [0,1] of cutting right after line i (higher = more natural)."""
    gap = lines[i + 1][0] - lines[i][1] if i + 1 < len(lines) else 0.0
    q = 0.5 * min(max(gap, 0.0), 1.0)
    txt = lines[i][2]
    if SENT_END.search(txt):
        q += 0.5
    elif SOFT_END.search(txt):
        q += 0.15
    return min(q, 1.0)


def segment_lines(lines, duration, min_dur=MIN_DUR, max_dur=MAX_DUR):
    """DP over line boundaries. Returns list of (start_line, end_line, t0, t1) or None."""
    n = len(lines)
    # cut_time[i] = audio time of the boundary AFTER line i ; cut_time[-1] -> 0.0
    cut_time = []
    for i in range(n):
        if i + 1 < n:
            cut_time.append(0.5 * (lines[i][1] + lines[i + 1][0]))
        else:
            cut_time.append(duration)
    start_time = [0.0] + cut_time[:-1]
    qual = [cut_quality(lines, i) for i in range(n)]

    INF = float("inf")
    best = [INF] * (n + 1)   # best[e+1] = cost of covering lines 0..e
    prev = [-1] * (n + 1)
    best[0] = 0.0
    for e in range(n):
        t1 = cut_time[e]
        for s in range(e, -1, -1):
            t0 = start_time[s]
            dur = t1 - t0
            if dur > max_dur and s != e:
                break
            if best[s] == INF:
                continue
            if dur < min_dur and e != n - 1:
                continue
            if dur > max_dur and s == e:
                pass  # single un-splittable line longer than max_dur: allowed
            elif dur > max_dur:
                continue
            c = W_DUR * ((dur - TARGET_DUR) / 10.0) ** 2
            if e != n - 1:
                c += W_CUT * (1.0 - qual[e])
            if best[s] + c < best[e + 1]:
                best[e + 1] = best[s] + c
                prev[e + 1] = s
    if best[n] == INF:
        return None
    out = []
    e = n
    while e > 0:
        s = prev[e]
        out.append((s, e - 1, start_time[s], cut_time[e - 1]))
        e = s
    out.reverse()
    return out


def uniform_windows(r, t0, t1, tokens, note, start_index=0):
    """Split [t0,t1) into equal windows of ~TARGET_DUR and spread tokens proportionally."""
    import hashlib as _h
    span = max(t1 - t0, 1e-6)
    k = max(1, int(round(span / TARGET_DUR)))
    while span / k > MAX_DUR:
        k += 1
    while k > 1 and span / k < MIN_DUR:
        k -= 1
    out = []
    n = len(tokens)
    for i in range(k):
        a = t0 + span * i / k
        b = t0 + span * (i + 1) / k
        ja = int(round(n * i / k))
        jb = int(round(n * (i + 1) / k))
        text = " ".join(tokens[ja:jb])
        out.append({
            "sample_id": f"{r['sample_id']}_w{start_index + i:03d}",
            "parent_sample_id": r["sample_id"],
            "window_index": start_index + i,
            "video_id": r["video_id"],
            "channel_id": r["channel_id"],
            "channel_title": r["channel_title"],
            "local_speaker_id": r["local_speaker_id"],
            "speaker_key": r["speaker_key"],
            "audio_path": r["audio_path"],
            "offset_start": round(a, 3),
            "offset_end": round(b, 3),
            "duration": round(round(b, 3) - round(a, 3), 3),
            "text": text,
            "words": len(text.split()),
            "chars": len(text),
            "asr_consistency": r["asr_consistency"],
            "is_single_speaker": r["is_single_speaker"],
            "license": r["license"],
            "split": r.get("split") or "train",
            "sha256_text": _h.sha256(text.encode("utf-8")).hexdigest(),
            "parent_duration_sec": float(r["duration_sec"]),
            "n_asr_groups": 0,
            "cut_note": note,
        })
    return out


def process_row(r):
    try:
        with open(r["json_path"], encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"sample_id": r["sample_id"], "error": f"json: {type(e).__name__}: {e}", "rows": []}
    ts = (d.get("asr_ts") or {}).get(E2E_KEY) or ""
    lines = parse_ts(ts)
    rov_tokens = (r["text"] or "").split()
    dur = float(r["duration_sec"])
    note = ""
    if not lines:
        # No ASR timestamps at all (silence / non-Russian / failed ASR). Nothing is
        # dropped: cover the parent with uniform windows and spread the ROVER tokens
        # proportionally to window duration.
        return {"sample_id": r["sample_id"], "error": "no asr_ts lines (uniform fallback)",
                "rows": uniform_windows(r, 0.0, dur, rov_tokens, "no_timestamps_uniform"),
                "parent_words": len(rov_tokens), "parent_dur": dur}

    seg = segment_lines(lines, dur)
    if seg is None:
        seg = segment_lines(lines, dur, 8.0, 40.0)
        note = "relaxed_8_40"
    if seg is None:
        seg = [(0, len(lines) - 1, 0.0, dur)]
        note = "single_window_fallback"

    line_of = align_tokens_to_lines(lines, rov_tokens)
    # ROVER token ranges per line
    tok_start = {}
    for j, li in enumerate(line_of):
        tok_start.setdefault(li, j)
    rows = []
    for wi, (s, e, t0, t1) in enumerate(seg):
        j0 = None
        for li in range(s, e + 1):
            if li in tok_start:
                j0 = tok_start[li]
                break
        rows.append({"s": s, "e": e, "t0": t0, "t1": t1, "j0": j0, "note": note})
    # token spans: window w gets tokens [j0_w, j0_{w+1})
    starts = []
    last = 0
    for w in rows:
        j0 = w["j0"]
        if j0 is None or j0 < last:
            j0 = last
        starts.append(j0)
        last = j0
    starts[0] = 0
    out_rows = []
    for wi, w in enumerate(rows):
        a = starts[wi]
        b = starts[wi + 1] if wi + 1 < len(starts) else len(rov_tokens)
        text = " ".join(rov_tokens[a:b])
        t0, t1 = max(0.0, w["t0"]), min(dur, w["t1"])
        out_rows.append({
            "sample_id": f"{r['sample_id']}_w{wi:03d}",
            "parent_sample_id": r["sample_id"],
            "window_index": wi,
            "video_id": r["video_id"],
            "channel_id": r["channel_id"],
            "channel_title": r["channel_title"],
            "local_speaker_id": r["local_speaker_id"],
            "speaker_key": r["speaker_key"],
            "audio_path": r["audio_path"],
            "offset_start": round(t0, 3),
            "offset_end": round(t1, 3),
            "duration": round(round(t1, 3) - round(t0, 3), 3),
            "text": text,
            "words": len(text.split()),
            "chars": len(text),
            "asr_consistency": r["asr_consistency"],
            "is_single_speaker": r["is_single_speaker"],
            "license": r["license"],
            "split": r.get("split") or "train",
            "sha256_text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parent_duration_sec": dur,
            "n_asr_groups": w["e"] - w["s"] + 1,
            "cut_note": w["note"],
        })
    # repair: any window still longer than MAX_DUR (ASR produced very few, very long
    # groups) is cut uniformly. Nothing is dropped.
    repaired = []
    for w in out_rows:
        if w["duration"] > MAX_DUR + 1e-6:
            repaired.extend(uniform_windows(r, w["offset_start"], w["offset_end"],
                                            w["text"].split(), "oversize_uniform_split",
                                            start_index=len(repaired)))
        else:
            w["sample_id"] = f"{r['sample_id']}_w{len(repaired):03d}"
            w["window_index"] = len(repaired)
            repaired.append(w)
    return {"sample_id": r["sample_id"], "error": None, "rows": repaired,
            "parent_words": len(rov_tokens), "parent_dur": dur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/manifests/train_long.jsonl")
    ap.add_argument("--out", default="data/manifests/train_short.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        parents = [json.loads(l) for l in f]
    print(f"[short] parents={len(parents)}", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(process_row, parents, chunksize=16)):
            results.append(res)
            if (i + 1) % 500 == 0:
                print(f"[short] {i+1}/{len(parents)} {time.time()-t0:.0f}s", flush=True)

    errors = [r for r in results if r["error"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_rows = 0
    sum_dur = 0.0
    sum_words = 0
    notes = {}
    durs = []
    with open(args.out, "w", encoding="utf-8") as f:
        for res in results:
            for row in res["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                sum_dur += row["duration"]
                sum_words += row["words"]
                durs.append(row["duration"])
                notes[row["cut_note"]] = notes.get(row["cut_note"], 0) + 1

    p_dur = sum(p["duration_sec"] for p in parents)
    p_words = sum(p["words"] for p in parents)
    durs.sort()
    stats = {
        "parents": len(parents),
        "parents_no_timestamps": len(errors),
        "parents_no_timestamps_ids": [e["sample_id"] for e in errors][:50],
        "parents_no_timestamps_reasons": sorted({e["error"] for e in errors}),
        "short_rows": n_rows,
        "short_hours": round(sum_dur / 3600, 4),
        "long_hours": round(p_dur / 3600, 4),
        "hours_ratio": round(sum_dur / p_dur, 6) if p_dur else None,
        "short_words": sum_words,
        "long_words": p_words,
        "words_ratio": round(sum_words / p_words, 6) if p_words else None,
        "dur_min": durs[0] if durs else None,
        "dur_p01": durs[int(0.01 * len(durs))] if durs else None,
        "dur_median": durs[len(durs) // 2] if durs else None,
        "dur_p99": durs[int(0.99 * len(durs))] if durs else None,
        "dur_max": durs[-1] if durs else None,
        "n_below_10s": sum(1 for d in durs if d < 10.0 - 1e-6),
        "n_above_30s": sum(1 for d in durs if d > 30.0 + 1e-6),
        "cut_notes": notes,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    # <out stem>_stats.json next to the output: train_short.jsonl -> train_short_stats.json,
    # dev_short.jsonl -> dev_short_stats.json (both arms use the same cutter)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    side = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{stem}_stats.json")
    with open(side, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
