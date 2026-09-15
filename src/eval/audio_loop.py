#!/usr/bin/env python
"""A5 / PLAN.md §3.4 — audio-loop detector (thin re-export of the implementation in
`src/eval/audio_qc.py`, plus a standalone CLI).

The detector lives in `audio_qc.py` because it shares that module's VAD (a repeated
stretch of silence is not a loop).  This file exists so the artifact list of PLAN §14 A5
(`src/eval/audio_loop.py`) resolves and so callers that only need loop detection do not
have to import the whole QC surface.

    python src/eval/audio_loop.py FILE.wav
    python src/eval/audio_loop.py FILE.wav --threshold 0.92 --min-repeat 3.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:  # imported as a package member (`from eval.audio_loop import ...`)
    from .audio_qc import (  # noqa: F401  (re-export)
        LOOP_MAX_LAG_S, LOOP_MIN_LAG_S, LOOP_MIN_REPEAT_S, LOOP_SIM_THRESHOLD,
        detect_audio_loop, load_audio, vad_mask,
    )
except ImportError:  # run as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from audio_qc import (  # noqa: F401  (re-export)
        LOOP_MAX_LAG_S, LOOP_MIN_LAG_S, LOOP_MIN_REPEAT_S, LOOP_SIM_THRESHOLD,
        detect_audio_loop, load_audio, vad_mask,
    )

__all__ = ["detect_audio_loop", "loop_check_file", "acoustic_recurrence",
           "LOOP_SIM_THRESHOLD", "LOOP_MIN_REPEAT_S", "LOOP_MIN_LAG_S", "LOOP_MAX_LAG_S",
           "RECUR_HOP_MS", "RECUR_EXCLUDE_SEC", "RECUR_THRESHOLDS"]

# --------------------------------------------------------------- recurrence (diagnostic)
# `detect_audio_loop` scans FIXED lags: it asks whether frame t and frame t+L match for a
# sustained stretch, which is true of a copy-pasted buffer and of a decoder stuck in a
# cycle, but not of a TTS model that re-synthesises the same words with different timing.
# A5 measured this on 240 v3.1 outputs (results/v31_loop/): the fixed-lag detector fires
# on none of them, including items whose ASR transcript repeats one phrase 37 times.
# `acoustic_recurrence` is the time-warp-tolerant companion: it makes no assumption about
# a constant lag, only that a repeated stretch of speech has frames that look like frames
# somewhere else in the file. It is a DIAGNOSTIC, not a status criterion -- natural speech
# recurs too, so the number only means something next to the natural-speech distribution
# in reports/speaker_thresholds.md.
RECUR_HOP_MS = 100.0
RECUR_N_MFCC = 20
RECUR_EXCLUDE_SEC = 2.0     # frames this close in voiced time are neighbours, not repeats
RECUR_THRESHOLDS = (0.90, 0.95)
RECUR_MAX_FRAMES = 20000    # guard: the hop is doubled until the matrix fits


def acoustic_recurrence(y, sr, hop_ms=RECUR_HOP_MS, n_mfcc=RECUR_N_MFCC,
                        exclude_sec=RECUR_EXCLUDE_SEC, thresholds=RECUR_THRESHOLDS,
                        voiced_mask=None, voiced_hop=None, max_frames=RECUR_MAX_FRAMES,
                        chunk=512):
    """Fraction of voiced frames whose content recurs elsewhere in the same file.

    MFCCs c1..c19 at `hop_ms`, silent frames dropped (`audio_qc.vad_mask`), mean/variance
    normalised over the file and L2-normalised per frame, so a dot product is a cosine.
    For every kept frame i the best match over all j with |i - j| > `exclude_sec` of
    voiced time is taken; the report is the share of frames whose best match reaches each
    threshold, plus the distribution of the best-match values themselves.

    Returns a JSON-serialisable dict; `recurrence_rate` maps "0.90"/"0.95" to that share.
    """
    import numpy as np
    import librosa

    out = {"params": {"hop_ms": hop_ms, "n_mfcc": n_mfcc, "exclude_sec": exclude_sec,
                      "thresholds": list(thresholds)}}
    dur = len(y) / sr
    if dur < 2 * exclude_sec + 1.0:
        out["skipped_reason"] = "audio shorter than 2*exclude + 1 s"
        return out
    if voiced_mask is None:
        voiced_mask, voiced_hop, _ = vad_mask(y, sr)
    while len(y) / (sr * hop_ms / 1000.0) > max_frames and hop_ms < 1000.0:
        hop_ms *= 2.0
    out["params"]["hop_ms"] = hop_ms
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    mfcc = librosa.feature.mfcc(y=np.ascontiguousarray(y), sr=sr, n_mfcc=n_mfcc,
                                hop_length=hop, n_fft=max(512, 2 * hop))
    f = mfcc[1:, :].T.astype(np.float32)
    n_all = f.shape[0]
    if len(voiced_mask):
        v_times = np.arange(len(voiced_mask)) * (voiced_hop / sr)
        f_times = np.arange(n_all) * (hop / sr)
        keep = np.interp(f_times, v_times, voiced_mask.astype(np.float64)) >= 0.5
    else:
        keep = np.ones(n_all, dtype=bool)
    f = f[keep]
    n = f.shape[0]
    out["n_frames_total"] = int(n_all)
    out["n_frames_voiced"] = int(n)
    if n < 20:
        out["skipped_reason"] = "fewer than 20 voiced frames"
        return out
    f = (f - f.mean(axis=0, keepdims=True)) / (f.std(axis=0, keepdims=True) + 1e-9)
    f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    excl = max(1, int(round(exclude_sec / (hop / sr))))
    best = np.full(n, -1.0, dtype=np.float32)
    idx = np.arange(n)
    for a0 in range(0, n, chunk):
        b0 = min(n, a0 + chunk)
        sim = f[a0:b0] @ f.T
        far = np.abs(idx[a0:b0, None] - idx[None, :]) > excl
        sim = np.where(far, sim, -1.0)
        best[a0:b0] = sim.max(axis=1)
    out["recurrence_rate"] = {f"{t:.2f}": round(float((best >= t).mean()), 6)
                              for t in thresholds}
    out["best_match_median"] = round(float(np.median(best)), 6)
    out["best_match_p95"] = round(float(np.percentile(best, 95)), 6)
    out["best_match_max"] = round(float(best.max()), 6)
    out["voiced_sec_analysed"] = round(n * hop / sr, 4)
    return out


def loop_check_file(path, sr=16000, **kw):
    """Load `path` (mono, resampled to `sr`) and run the loop detector on it."""
    y, file_sr = load_audio(path, sr=sr)
    mask, hop, _ = vad_mask(y, file_sr)
    rep = detect_audio_loop(y, file_sr, voiced_mask=mask, voiced_hop=hop, **kw)
    rep["path"] = path
    rep["sample_rate"] = int(file_sr)
    rep["duration_sec"] = round(len(y) / file_sr, 6)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="MFCC self-similarity loop detector -> JSON")
    ap.add_argument("wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--threshold", type=float, default=LOOP_SIM_THRESHOLD)
    ap.add_argument("--min-repeat", type=float, default=LOOP_MIN_REPEAT_S)
    ap.add_argument("--min-lag", type=float, default=LOOP_MIN_LAG_S)
    ap.add_argument("--max-lag", type=float, default=LOOP_MAX_LAG_S)
    ap.add_argument("--compact", action="store_true")
    a = ap.parse_args(argv)
    rep = loop_check_file(a.wav, sr=a.sr, sim_threshold=a.threshold,
                          min_repeat_s=a.min_repeat, min_lag_s=a.min_lag,
                          max_lag_s=a.max_lag)
    print(json.dumps(rep, ensure_ascii=False,
                     indent=None if a.compact else 2, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
