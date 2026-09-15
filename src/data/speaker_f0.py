#!/usr/bin/env python
"""A1 — coarse gender proxy per speaker_key for the split (rule v3, constraint F7).

The corpus carries no gender label (PLAN §6.2 asks dev/test to cover both genders, and A5's
reference selection needs a male AND a female speaker with >= 600 s / >= 3 segments in the
TEST channels). This script measures a median F0 per speaker_key on the same centred 30 s
windows that `leakage_check.py --embed` uses (up to 3 segments per speaker_key), with
librosa.yin (fast; A5's reference QC uses pyin on the final references and is the label of
record). Boundary: < 165 Hz male, > 165 Hz female, +-15 Hz ambiguous — the same rule as
`src/eval/make_references.py`.

Output: data/manifests/spk_f0.json  {speaker_key: {f0_median_hz, gender, n_segments}}
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.leakage_check import WINDOW_SEC, load_jsonl, pick_segments  # noqa: E402

F0_GENDER_HZ = 165.0
F0_BAND_HZ = 15.0
FMIN, FMAX = 60.0, 400.0


def f0_one(task):
    sid, path = task
    try:
        import librosa
        import soundfile as sf
        info = sf.info(path)
        sr = info.samplerate
        want = int(WINDOW_SEC * sr)
        start = max(0, (info.frames - want) // 2)
        data, sr = sf.read(path, start=start, frames=min(want, info.frames), dtype="float32",
                           always_2d=True)
        y = data.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        frame, hop = 1024, 256
        f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=frame, hop_length=hop)
        rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
        n = min(len(f0), len(rms))
        f0, rms = f0[:n], rms[:n]
        thr = max(rms.max() * 0.1, 1e-4)
        voiced = (rms > thr) & (f0 > FMIN + 1) & (f0 < FMAX - 1)
        if voiced.sum() < 20:
            return sid, None, "too few voiced frames"
        return sid, float(np.median(f0[voiced])), None
    except Exception as e:  # noqa: BLE001
        return sid, None, f"{type(e).__name__}: {e}"


def gender_of(f0):
    if f0 is None:
        return "unknown"
    if f0 < F0_GENDER_HZ - F0_BAND_HZ:
        return "male"
    if f0 > F0_GENDER_HZ + F0_BAND_HZ:
        return "female"
    return "ambiguous"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/all.jsonl")
    ap.add_argument("--out", default="data/manifests/spk_f0.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rows = load_jsonl(args.manifest)
    picks = pick_segments(rows)
    tasks, owner = [], {}
    for k, segs in sorted(picks.items()):
        for sid, p, _d in segs:
            tasks.append((sid, p))
            owner[sid] = k
    have = {}
    if os.path.exists(args.out) and not args.force:
        old = json.load(open(args.out, encoding="utf-8"))
        for k, v in old.items():
            for sid, f0 in (v.get("per_segment") or {}).items():
                if sid in owner and owner[sid] == k:
                    have[sid] = f0
    todo = [t for t in tasks if t[0] not in have]
    print(f"[f0] speaker_keys={len(picks)} segments={len(tasks)} cached={len(have)} todo={len(todo)}",
          flush=True)
    t0 = time.time()
    errors = []
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (sid, f0, err) in enumerate(ex.map(f0_one, todo, chunksize=4)):
                if err:
                    errors.append((sid, err))
                    have[sid] = None
                else:
                    have[sid] = f0
                if (i + 1) % 100 == 0:
                    print(f"[f0] {i+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    per_key = defaultdict(dict)
    for sid, f0 in have.items():
        per_key[owner[sid]][sid] = f0
    out = {}
    for k in sorted(picks):
        vals = [v for v in per_key.get(k, {}).values() if v is not None]
        med = float(np.median(vals)) if vals else None
        out[k] = {"f0_median_hz": (round(med, 2) if med is not None else None),
                  "gender": gender_of(med), "n_segments": len(vals),
                  "per_segment": {s: (round(v, 2) if v is not None else None)
                                  for s, v in per_key.get(k, {}).items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"[f0] wrote {args.out}: {dict(Counter(v['gender'] for v in out.values()))}, "
          f"errors={len(errors)}, elapsed {time.time()-t0:.0f}s")
    if errors:
        print(errors[:5])


if __name__ == "__main__":
    main()
