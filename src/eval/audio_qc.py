#!/usr/bin/env python
"""A5 / PLAN.md §9.2, §9.6 — audio quality control for reference and generated audio.

CPU-only, deterministic, no model is loaded; every entry point works on a waveform array
as well as on a file. Dependencies, all imported lazily so that importing this module is
cheap: numpy and soundfile always; **librosa** for `timbre_metrics` (spectral flatness),
`f0_metrics` (pyin) and `detect_audio_loop` (MFCC); **torch/torchaudio** for the default
resampler (`resample(..., res_type="torchaudio")`, the resampler CosyVoice itself uses).
That set exists in `external/miniconda3/envs/cosyvoice`. `.venv-eval` has numpy,
soundfile and **soxr** but no librosa/scipy/torch: there the duration / VAD / level /
clipping / validity / SNR path works, `load_audio(path, sr=16000)` works through the soxr
fallback in `resample` (with a warning, because soxr is not bit-identical to CosyVoice's
resampler), and anything touching librosa raises ImportError (see tests/test_audio_qc.py).

Measured quantities
-------------------
* raw duration, voiced duration, silence ratio, longest silence  (PLAN §9.2)
* clipping ratio, peak/RMS levels, crest factor, DC offset, NaN/Inf validity (PLAN §9.6)
* SNR estimate: two blind estimators, see `snr_energy_percentile` (primary) and
  `snr_wada` (secondary)
* audio-loop detection: MFCC self-similarity over lags 0.5-10 s (PLAN §3.4 `loop`)

VAD (documented, no external dependency)
----------------------------------------
Frame-energy VAD, not a neural VAD: the signal is framed at 25 ms / 10 ms hop, the RMS
of every frame is converted to dBFS, and the speech/silence threshold is

    thr_db = max(p95(frame_db) - VAD_REL_DB, VAD_ABS_DB)

i.e. 30 dB below the loud-frame level of *this* file, floored at an absolute -55 dBFS so
that a digitally silent file is not "all speech".  The binary mask is then median-
filtered over 5 frames (50 ms) and silence runs shorter than 200 ms are absorbed into
speech (they are intra-word stops, not silence).  The same mask is reused by the loop
detector to reject matches that are only silence-against-silence.

SNR
---
`snr_energy_percentile` (primary, reported as `snr_db`): frame powers are sorted; the
noise floor is the mean power of the frames at or below the 10th percentile, the speech
level is the mean power of the frames at or above the 90th percentile, and

    SNR = 10*log10( max(P_speech - P_noise, eps) / P_noise )

This assumes the quietest 10 % of frames contain no speech - true for the 8-15 s
sentence windows used for references, and roughly true for long-form recordings.

`snr_wada` (secondary): WADA-SNR (Kim & Stern, Interspeech 2008).  The statistic
G = log(E|x|) - E(log|x|) is monotone in SNR for speech-plus-Gaussian-noise; the
published implementations ship a hard-coded G->SNR table.  Rather than copying numbers
that cannot be verified here, the table is *regenerated* by Monte-Carlo from the model
the method assumes (clean speech amplitude ~ Gamma(shape=0.4) with random sign, additive
Gaussian noise), with a fixed seed, and its two analytic endpoints are asserted:
G -> log(0.4) - psi(0.4) = 1.6449 for noise-free Gamma(0.4), G -> 0.4094 for pure
Gaussian.  See `wada_table()`.

Audio-loop detection
--------------------
MFCCs (c1..c19, hop 20 ms) are mean/variance normalised over the file and L2-normalised
per frame, so a dot product between two frames is a cosine similarity.  For every lag L
in [0.5 s, 10 s] the similarity track s_L[t] = <f[t], f[t+L]> is smoothed with a 200 ms
moving average; a maximal run of frames with s_L >= 0.90 that lasts >= 2 s and is at
least 60 % voiced on *both* sides is reported as a repeated stretch.  The voiced
requirement is what stops a long pause (which is self-similar at every lag) from being
reported as a loop.

CLI
---
    python src/eval/audio_qc.py FILE.wav
        Default `--mode pipeline`: the exact report `make_references.py` stage 2 stores
        and reports/reference_qc.md tabulates - SNR, F0, WADA-SNR and the loop detector
        at 16 kHz (what the model and the evaluator consume), duration / VAD / levels /
        clipping / flatness / reverb at the file's native rate, and the native-rate SNR
        kept beside the 16 kHz one as `snr_db_native`.

    python src/eval/audio_qc.py FILE.wav --mode plain [--sr 24000 | --native-sr] \
                                         [--f0] [--wada] [--no-loop] [--no-timbre]
        One rate for everything (default 16 kHz), with the individual blocks opt-in.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import sys

import numpy as np

# ---------------------------------------------------------------------------- constants
FRAME_MS = 25.0
HOP_MS = 10.0
VAD_REL_DB = 30.0          # frames this far below p95 of the frame levels are silence
VAD_ABS_DB = -55.0         # absolute silence floor (dBFS)
VAD_MEDIAN_FRAMES = 5      # 50 ms median filter on the binary mask
VAD_MIN_SILENCE_MS = 200.0 # shorter silence runs are absorbed into speech
CLIP_LEVEL = 0.99          # |x| >= this counts as a clipped sample
CLIP_MIN_RUN = 3           # consecutive clipped samples that make a clipping *run*
EPS = 1e-12

# loop detector
LOOP_HOP_MS = 20.0
LOOP_N_MFCC = 20           # c0 is dropped -> 19 coefficients are used
LOOP_MIN_LAG_S = 0.5
LOOP_MAX_LAG_S = 10.0
LOOP_SIM_THRESHOLD = 0.90
LOOP_MIN_REPEAT_S = 2.0
LOOP_SMOOTH_MS = 200.0
LOOP_MIN_VOICED_FRAC = 0.60
LOOP_MAX_PAIR_OPS = 6e8    # guard: increase the hop instead of exploding on long files

# pitch (used for the male/female heuristic in make_references.py)
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
F0_FRAME_LENGTH = 2048

# The rate the model and the evaluator actually consume, and the rate the reference QC
# table's SNR / F0 / loop columns are measured at (see `audio_qc_pipeline`).
PIPELINE_SR = 16000

WADA_SEED = 20260827
WADA_N = 1_000_000
WADA_SNR_GRID = np.arange(-20.0, 101.0, 1.0)


# ------------------------------------------------------------------------------- io
_WARNED = set()


def resample(y, orig_sr, target_sr, res_type="torchaudio"):
    """Resample a 1-D waveform.

    `res_type`:

    * ``"torchaudio"`` (default) — `torchaudio.transforms.Resample`, the resampler
      CosyVoice itself applies to a prompt wav (`cosyvoice/utils/file_utils.py::load_wav`),
      so analysis and synthesis see the same signal.  **If torchaudio is not importable**
      (`.venv-eval` has no torch) this falls back to ``"soxr"`` after printing a loud
      one-time warning naming the substitution — the fallback exists so that the §9.5
      drift calibration can load 16 kHz audio in the eval venv at all, and it is never
      silent, because the two resamplers are not bit-identical.
    * ``"torchaudio_strict"`` — the same, but raises `ImportError` instead of falling back.
      Use it where a number must be comparable with the reference pipeline.
    * ``"soxr"`` — `soxr.resample(..., quality="VHQ")`; numpy + soxr only, so it works in
      both interpreters.
    * anything else — handed to `librosa.resample` as its own `res_type`.
    """
    if orig_sr == target_sr:
        return np.ascontiguousarray(y)
    if res_type in ("torchaudio", "torchaudio_strict"):
        try:
            import torch
            import torchaudio
        except ImportError as e:
            if res_type == "torchaudio_strict":
                raise
            if "torchaudio" not in _WARNED:
                _WARNED.add("torchaudio")
                print(f"[audio_qc] WARNING: torchaudio is not importable under "
                      f"{sys.executable} ({e}); resampling {orig_sr} -> {target_sr} Hz "
                      "with soxr VHQ instead. soxr is NOT the resampler CosyVoice uses, "
                      "so numbers produced here are not bit-comparable with the "
                      "reference pipeline (reports/reference_qc.md quotes the cosyvoice "
                      "env). Pass res_type='torchaudio_strict' to make this an error.",
                      file=sys.stderr)
            res_type = "soxr"
        else:
            w = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
            w = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)(w)
            return np.ascontiguousarray(w.squeeze(0).numpy().astype(np.float64))
    if res_type == "soxr":
        import soxr

        out = soxr.resample(np.ascontiguousarray(y, dtype=np.float64),
                            float(orig_sr), float(target_sr), quality="VHQ")
        return np.ascontiguousarray(np.asarray(out, dtype=np.float64))
    import librosa

    return np.ascontiguousarray(
        librosa.resample(np.ascontiguousarray(y), orig_sr=orig_sr, target_sr=target_sr,
                         res_type=res_type))


def load_audio(path, sr=None, mono=True, start_sec=None, end_sec=None,
               res_type="torchaudio"):
    """Read a file with soundfile. Returns (float64 mono waveform, sample_rate).

    `sr` resamples after mixing down (see `resample`); `sr=None` keeps the native rate.
    start/end are in seconds relative to the file.
    """
    import soundfile as sf

    info = sf.info(path)
    native_sr = info.samplerate
    kw = {}
    if start_sec is not None:
        kw["start"] = max(0, int(round(start_sec * native_sr)))
    if end_sec is not None:
        kw["stop"] = min(info.frames, int(round(end_sec * native_sr)))
    data, file_sr = sf.read(path, dtype="float64", always_2d=True, **kw)
    y = data.mean(axis=1) if mono else data
    if sr is not None and sr != file_sr:
        y = resample(y, file_sr, sr, res_type=res_type)
        file_sr = sr
    return np.ascontiguousarray(y), file_sr


# ---------------------------------------------------------------------------- framing
def frame_signal(y, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Return (n_frames, frame_len) view-like array; the tail is zero-padded."""
    flen = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if len(y) < flen:
        y = np.pad(y, (0, flen - len(y)))
    n = 1 + (len(y) - flen) // hop
    idx = np.arange(flen)[None, :] + hop * np.arange(n)[:, None]
    return y[idx], flen, hop


def frame_db(y, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Per-frame RMS in dBFS (full scale = 1.0)."""
    fr, _, hop = frame_signal(y, sr, frame_ms, hop_ms)
    rms = np.sqrt(np.mean(fr * fr, axis=1) + EPS)
    return 20.0 * np.log10(np.maximum(rms, 1e-12)), hop


def _median_filter_bool(mask, k):
    if k <= 1 or len(mask) < k:
        return mask
    pad = k // 2
    m = np.pad(mask.astype(np.float64), (pad, pad), mode="edge")
    idx = np.arange(k)[None, :] + np.arange(len(mask))[:, None]
    return np.median(m[idx], axis=1) >= 0.5


def _runs(mask):
    """Yield (start_idx, end_idx_exclusive, value) for maximal constant runs."""
    if len(mask) == 0:
        return []
    changes = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], changes, [len(mask)]))
    return [(int(bounds[i]), int(bounds[i + 1]), bool(mask[bounds[i]]))
            for i in range(len(bounds) - 1)]


def vad_mask(y, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS, rel_db=VAD_REL_DB,
             abs_db=VAD_ABS_DB, median_frames=VAD_MEDIAN_FRAMES,
             min_silence_ms=VAD_MIN_SILENCE_MS):
    """Energy VAD. Returns (bool mask per frame, hop_samples, threshold_db)."""
    db, hop = frame_db(y, sr, frame_ms, hop_ms)
    if len(db) == 0:
        return np.zeros(0, dtype=bool), hop, float(abs_db)
    thr = max(float(np.percentile(db, 95.0)) - rel_db, abs_db)
    mask = db > thr
    mask = _median_filter_bool(mask, median_frames)
    min_sil_frames = int(round(min_silence_ms / hop_ms))
    for a, b, v in _runs(mask):
        if (not v) and (b - a) < min_sil_frames and a > 0 and b < len(mask):
            mask[a:b] = True
    return mask, hop, thr


# ------------------------------------------------------------------------- measurements
def duration_metrics(y, sr, **vad_kw):
    """raw / voiced duration, silence ratio, longest silence, leading+trailing silence."""
    mask, hop, thr = vad_mask(y, sr, **vad_kw)
    hop_s = hop / sr
    raw = len(y) / sr
    voiced = float(mask.sum()) * hop_s
    voiced = min(voiced, raw)
    runs = _runs(mask)
    sil = [(a, b) for a, b, v in runs if not v]
    longest = max((b - a for a, b in sil), default=0) * hop_s
    lead = (runs[0][1] - runs[0][0]) * hop_s if runs and not runs[0][2] else 0.0
    trail = (runs[-1][1] - runs[-1][0]) * hop_s if runs and not runs[-1][2] else 0.0
    internal = [(b - a) * hop_s for a, b, v in runs
                if not v and a > 0 and b < len(mask)]
    return {
        "duration_sec": round(raw, 6),
        "voiced_sec": round(voiced, 6),
        "voiced_ratio": round(voiced / raw, 6) if raw > 0 else 0.0,
        "silence_ratio": round(1.0 - voiced / raw, 6) if raw > 0 else 1.0,
        "longest_silence_sec": round(longest, 6),
        "longest_internal_silence_sec": round(max(internal, default=0.0), 6),
        "leading_silence_sec": round(lead, 6),
        "trailing_silence_sec": round(trail, 6),
        "n_silence_runs": len(sil),
        "vad_threshold_dbfs": round(thr, 3),
        "vad_frame_ms": FRAME_MS,
        "vad_hop_ms": HOP_MS,
    }, mask, hop


def level_metrics(y, mask=None, hop=None):
    """Peak, RMS, crest factor, DC offset, clipping."""
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y * y))) if len(y) else 0.0
    clipped = np.abs(y) >= CLIP_LEVEL
    n_clip = int(clipped.sum())
    runs = [(a, b) for a, b, v in _runs(clipped) if v and (b - a) >= CLIP_MIN_RUN]
    voiced_rms = None
    if mask is not None and hop is not None and mask.any():
        sel = np.zeros(len(y), dtype=bool)
        for a, b, v in _runs(mask):
            if v:
                sel[a * hop:min(len(y), b * hop)] = True
        if sel.any():
            voiced_rms = float(np.sqrt(np.mean(y[sel] ** 2)))
    return {
        "peak": round(peak, 8),
        "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 3),
        "rms": round(rms, 8),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 3),
        "rms_voiced_dbfs": (round(20 * math.log10(max(voiced_rms, 1e-12)), 3)
                            if voiced_rms is not None else None),
        "crest_factor": round(peak / max(rms, 1e-12), 4),
        "crest_factor_db": round(20 * math.log10(peak / max(rms, 1e-12) + 1e-12), 3),
        "dc_offset": round(float(np.mean(y)) if len(y) else 0.0, 8),
        "clip_level": CLIP_LEVEL,
        "n_clipped_samples": n_clip,
        "clipping_ratio": round(n_clip / len(y), 9) if len(y) else 0.0,
        "n_clipping_runs": len(runs),
        "longest_clipping_run_samples": max((b - a for a, b in runs), default=0),
        "clipped": bool(runs),
    }


def validity_metrics(y):
    n_nan = int(np.isnan(y).sum())
    n_inf = int(np.isinf(y).sum())
    finite = y[np.isfinite(y)]
    return {
        "n_samples": int(len(y)),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_out_of_range": int(np.sum(np.abs(finite) > 1.0)),
        "all_zero": bool(len(finite) and not np.any(finite != 0.0)),
        "invalid": bool(n_nan or n_inf or (len(y) == 0)),
    }


def snr_energy_percentile(y, sr, low_pct=10.0, high_pct=90.0, **vad_kw):
    """Blind SNR from the frame-power distribution (primary estimator).

    noise = mean power of the frames at or below `low_pct`,
    speech = mean power of the frames at or above `high_pct`,
    SNR = 10*log10((P_speech - P_noise)/P_noise).
    """
    fr, _, _ = frame_signal(y, sr, vad_kw.get("frame_ms", FRAME_MS),
                            vad_kw.get("hop_ms", HOP_MS))
    p = np.mean(fr * fr, axis=1)
    if len(p) < 4:
        return float("nan"), float("nan"), float("nan")
    lo = np.percentile(p, low_pct)
    hi = np.percentile(p, high_pct)
    p_noise = float(np.mean(p[p <= lo])) if np.any(p <= lo) else float(lo)
    p_speech = float(np.mean(p[p >= hi])) if np.any(p >= hi) else float(hi)
    p_noise = max(p_noise, 1e-20)
    snr = 10.0 * math.log10(max(p_speech - p_noise, 1e-20) / p_noise)
    return snr, p_speech, p_noise


@functools.lru_cache(maxsize=1)
def wada_table(n=WADA_N, seed=WADA_SEED):
    """G-statistic vs SNR table for WADA-SNR, regenerated by Monte-Carlo.

    Model assumed by the method: clean speech amplitude ~ Gamma(shape=0.4) with a random
    sign, corrupted by additive white Gaussian noise.  Returns (snr_grid_db, g_vals),
    both 1-D and g_vals monotonically increasing.
    """
    rng = np.random.default_rng(seed)
    s = rng.gamma(shape=0.4, scale=1.0, size=n) * rng.choice((-1.0, 1.0), size=n)
    z = rng.standard_normal(n)
    s_pow = float(np.mean(s * s))
    z_pow = float(np.mean(z * z))
    g = np.empty(len(WADA_SNR_GRID), dtype=np.float64)
    for i, snr_db in enumerate(WADA_SNR_GRID):
        alpha = math.sqrt(s_pow / (z_pow * 10.0 ** (snr_db / 10.0)))
        x = np.abs(s + alpha * z)
        np.maximum(x, 1e-10, out=x)
        g[i] = math.log(float(np.mean(x))) - float(np.mean(np.log(x)))
    g = np.maximum.accumulate(g)  # enforce monotonicity against MC noise
    return WADA_SNR_GRID.copy(), g


def snr_wada(y):
    """WADA-SNR in dB (secondary estimator). Clamped to the table range."""
    y = np.asarray(y, dtype=np.float64)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak <= 0:
        return float("nan")
    a = np.abs(y / peak)
    np.maximum(a, 1e-10, out=a)
    v = math.log(float(np.mean(a))) - float(np.mean(np.log(a)))
    grid, g = wada_table()
    if v <= g[0]:
        return float(grid[0])
    if v >= g[-1]:
        return float(grid[-1])
    return float(np.interp(v, g, grid))


def timbre_metrics(y, sr, mask=None, hop=None, hop_ms=HOP_MS, decay_db=20.0,
                   max_decay_s=0.5):
    """Two *proxies* that PLAN §8 asks about but that have no calibrated detector here.

    `spectral_flatness_mean` - librosa spectral flatness averaged over voiced frames.
    Speech sits low (roughly 0.01-0.15); broadband noise, hiss and dense music push it
    up.  It is a music/noise indicator, **not** a music detector.

    `reverb_proxy_sec` - median time for the frame level to fall `decay_db` dB below the
    level just before a speech->silence transition (an EDT-flavoured measure, hence
    "proxy": it is contaminated by the noise floor and by the analysis window).  Close-
    miked dry speech lands at a few tens of milliseconds; a reverberant room stretches
    it.  `None` when the file has no usable speech offset.
    """
    import librosa

    out = {"spectral_flatness_mean": None, "reverb_proxy_sec": None,
           "reverb_proxy_n_offsets": 0, "reverb_proxy_decay_db": decay_db}
    if len(y) < int(0.2 * sr):
        return out
    if mask is None or hop is None:
        mask, hop, _ = vad_mask(y, sr)
    n_fft = 1024 if sr <= 24000 else 2048
    flat = librosa.feature.spectral_flatness(
        y=np.ascontiguousarray(y), n_fft=n_fft,
        hop_length=max(1, int(round(sr * hop_ms / 1000.0))))[0]
    if len(mask) and mask.any():
        m = mask[:len(flat)] if len(mask) >= len(flat) else np.pad(
            mask, (0, len(flat) - len(mask)), constant_values=False)
        sel = flat[m] if m.any() else flat
    else:
        sel = flat
    out["spectral_flatness_mean"] = round(float(np.mean(sel)), 6)

    db, hop_samples = frame_db(y, sr, hop_ms=hop_ms)
    hop_s = hop_samples / sr
    max_frames = int(round(max_decay_s / hop_s))
    times = []
    for i in range(1, min(len(mask), len(db)) - 1):
        if not (mask[i] and not mask[i + 1]):
            continue
        lo = max(0, i - 10)
        l0 = float(np.max(db[lo:i + 1]))
        stop = min(len(db), i + 1 + max_frames)
        below = np.flatnonzero(db[i + 1:stop] <= l0 - decay_db)
        if len(below):
            times.append((below[0] + 1) * hop_s)
    if times:
        out["reverb_proxy_sec"] = round(float(np.median(times)), 5)
        out["reverb_proxy_n_offsets"] = len(times)
    return out


def f0_metrics(y, sr, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, frame_length=F0_FRAME_LENGTH):
    """Median / quantiles of F0 over voiced frames, via librosa.pyin.

    Used only as the *gender heuristic* for reference selection: gender labels do not
    exist anywhere in this corpus (see reports/dataset_stats.md), so median F0 is the
    substitute.  pyin is deterministic for a fixed input.
    """
    import librosa

    if len(y) < frame_length:
        return {"f0_median_hz": None, "f0_voiced_ratio": 0.0}
    f0, voiced_flag, voiced_prob = librosa.pyin(
        np.ascontiguousarray(y), fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, center=True)
    ok = np.isfinite(f0) & voiced_flag
    if not ok.any():
        return {"f0_median_hz": None, "f0_voiced_ratio": 0.0}
    v = f0[ok]
    return {
        "f0_median_hz": round(float(np.median(v)), 3),
        "f0_mean_hz": round(float(np.mean(v)), 3),
        "f0_p10_hz": round(float(np.percentile(v, 10)), 3),
        "f0_p25_hz": round(float(np.percentile(v, 25)), 3),
        "f0_p75_hz": round(float(np.percentile(v, 75)), 3),
        "f0_p90_hz": round(float(np.percentile(v, 90)), 3),
        "f0_std_hz": round(float(np.std(v)), 3),
        "f0_voiced_ratio": round(float(ok.mean()), 6),
        "f0_n_voiced_frames": int(ok.sum()),
        "f0_fmin_hz": fmin,
        "f0_fmax_hz": fmax,
    }


# ------------------------------------------------------------------------ loop detection
def detect_audio_loop(y, sr, min_lag_s=LOOP_MIN_LAG_S, max_lag_s=LOOP_MAX_LAG_S,
                      hop_ms=LOOP_HOP_MS, n_mfcc=LOOP_N_MFCC,
                      sim_threshold=LOOP_SIM_THRESHOLD,
                      min_repeat_s=LOOP_MIN_REPEAT_S, smooth_ms=LOOP_SMOOTH_MS,
                      min_voiced_frac=LOOP_MIN_VOICED_FRAC, max_events=20,
                      voiced_mask=None, voiced_hop=None):
    """Detect repeated stretches by MFCC self-similarity over lags.

    Returns a dict with `loop_detected`, the list of `events`
    (`{lag_sec, start_sec, end_sec, duration_sec, mean_similarity, voiced_frac}`) and the
    per-lag maximum of the smoothed similarity track (`max_similarity_by_lag_summary`).
    """
    import librosa

    out = {
        "loop_detected": False, "n_events": 0, "events": [],
        "max_run_sec": 0.0, "total_looped_sec": 0.0,
        "params": {"min_lag_s": min_lag_s, "max_lag_s": max_lag_s, "hop_ms": hop_ms,
                   "n_mfcc": n_mfcc, "sim_threshold": sim_threshold,
                   "min_repeat_s": min_repeat_s, "smooth_ms": smooth_ms,
                   "min_voiced_frac": min_voiced_frac},
    }
    dur = len(y) / sr
    if dur < min_lag_s + min_repeat_s:
        out["skipped_reason"] = "audio shorter than min_lag + min_repeat"
        return out

    # guard against O(frames x lags) blow-up on very long files
    while True:
        hop = max(1, int(round(sr * hop_ms / 1000.0)))
        n_frames_est = len(y) // hop
        n_lags_est = int((min(max_lag_s, dur) - min_lag_s) * 1000.0 / hop_ms)
        if n_frames_est * max(n_lags_est, 1) <= LOOP_MAX_PAIR_OPS or hop_ms >= 100.0:
            break
        hop_ms *= 2.0
    out["params"]["hop_ms"] = hop_ms

    mfcc = librosa.feature.mfcc(y=np.ascontiguousarray(y), sr=sr, n_mfcc=n_mfcc,
                               hop_length=hop, n_fft=max(512, 2 * hop))
    f = mfcc[1:, :].T.astype(np.float64)              # drop c0 (loudness)
    if f.shape[0] < 4:
        out["skipped_reason"] = "too few frames"
        return out
    f = (f - f.mean(axis=0, keepdims=True)) / (f.std(axis=0, keepdims=True) + 1e-9)
    f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    n = f.shape[0]
    frame_s = hop / sr

    # voiced mask resampled onto the MFCC frame grid
    if voiced_mask is None:
        voiced_mask, voiced_hop, _ = vad_mask(y, sr)
    v_times = np.arange(len(voiced_mask)) * (voiced_hop / sr)
    f_times = np.arange(n) * frame_s
    if len(voiced_mask):
        voiced = np.interp(f_times, v_times, voiced_mask.astype(np.float64)) >= 0.5
    else:
        voiced = np.ones(n, dtype=bool)

    smooth_frames = max(1, int(round(smooth_ms / (frame_s * 1000.0))))
    min_run = max(1, int(round(min_repeat_s / frame_s)))
    lag_lo = max(1, int(round(min_lag_s / frame_s)))
    lag_hi = min(n - min_run, int(round(min(max_lag_s, dur) / frame_s)))
    if lag_hi <= lag_lo:
        out["skipped_reason"] = "no usable lag range"
        return out

    kernel = np.ones(smooth_frames) / smooth_frames
    events = []
    per_lag_max = []
    for lag in range(lag_lo, lag_hi + 1):
        sim = np.einsum("ij,ij->i", f[:-lag], f[lag:])
        if smooth_frames > 1:
            if len(sim) < smooth_frames:
                continue
            sim = np.convolve(sim, kernel, mode="same")
        per_lag_max.append(float(sim.max()))
        hits = sim >= sim_threshold
        if not hits.any():
            continue
        for a, b, v in _runs(hits):
            if not v or (b - a) < min_run:
                continue
            vf = 0.5 * (voiced[a:b].mean() + voiced[a + lag:b + lag].mean())
            if vf < min_voiced_frac:
                continue
            events.append({
                "lag_sec": round(lag * frame_s, 4),
                "start_sec": round(a * frame_s, 4),
                "end_sec": round(b * frame_s, 4),
                "duration_sec": round((b - a) * frame_s, 4),
                "mean_similarity": round(float(sim[a:b].mean()), 5),
                "voiced_frac": round(float(vf), 4),
            })

    events.sort(key=lambda e: (-e["duration_sec"], -e["mean_similarity"], e["lag_sec"]))
    # union of the covered source intervals, so overlapping lags are not double counted
    cov, cur = 0.0, None
    for iv in sorted([(e["start_sec"], e["end_sec"]) for e in events]):
        if cur is None or iv[0] > cur[1]:
            if cur:
                cov += cur[1] - cur[0]
            cur = list(iv)
        else:
            cur[1] = max(cur[1], iv[1])
    if cur:
        cov += cur[1] - cur[0]

    out["loop_detected"] = bool(events)
    out["n_events"] = len(events)
    out["events"] = events[:max_events]
    out["max_run_sec"] = round(max((e["duration_sec"] for e in events), default=0.0), 4)
    out["total_looped_sec"] = round(cov, 4)
    if per_lag_max:
        pl = np.asarray(per_lag_max)
        out["max_similarity_by_lag_summary"] = {
            "max": round(float(pl.max()), 5),
            "p99": round(float(np.percentile(pl, 99)), 5),
            "p50": round(float(np.percentile(pl, 50)), 5),
            "n_lags": int(len(pl)),
        }
    return out


# ------------------------------------------------------------------------------ report
def audio_qc(y, sr, with_f0=False, with_loop=True, with_wada=False, with_timbre=True,
             path=None):
    """Full QC report for one waveform. Returns a JSON-serialisable dict."""
    y = np.asarray(y, dtype=np.float64)
    rep = {"path": path, "sample_rate": int(sr)}
    rep.update(validity_metrics(y))
    if rep["invalid"]:
        return rep
    dm, mask, hop = duration_metrics(y, sr)
    rep.update(dm)
    rep.update(level_metrics(y, mask, hop))
    if with_timbre:
        rep.update(timbre_metrics(y, sr, mask, hop))
    snr, p_s, p_n = snr_energy_percentile(y, sr)
    rep["snr_db"] = round(snr, 3) if np.isfinite(snr) else None
    rep["snr_method"] = "energy_percentile_p10_p90"
    rep["snr_speech_power"] = float(p_s)
    rep["snr_noise_power"] = float(p_n)
    rep["noise_floor_dbfs"] = (round(10 * math.log10(max(p_n, 1e-20)), 3)
                               if np.isfinite(p_n) else None)
    if with_wada:
        rep["snr_wada_db"] = round(snr_wada(y), 3)
    if with_f0:
        rep.update(f0_metrics(y, sr))
    if with_loop:
        rep["loop"] = detect_audio_loop(y, sr, voiced_mask=mask, voiced_hop=hop)
    return rep


def audio_qc_file(path, sr=None, with_f0=False, with_loop=True, with_wada=False,
                  with_timbre=True, start_sec=None, end_sec=None):
    y, file_sr = load_audio(path, sr=sr, start_sec=start_sec, end_sec=end_sec)
    return audio_qc(y, file_sr, with_f0=with_f0, with_loop=with_loop,
                    with_wada=with_wada, with_timbre=with_timbre, path=path)


def audio_qc_pipeline(path, start_sec=None, end_sec=None, path_label=Ellipsis):
    """THE report the reference QC table is made of - one function, two call sites.

    Mixed-rate on purpose:

    * duration / VAD / levels / clipping / flatness / reverb are measured on the file at
      its **native** rate, because those are properties of the artifact that ships;
    * SNR, F0, WADA-SNR and the loop detector run on the **16 kHz mono** version
      (`PIPELINE_SR`), i.e. on what CosyVoice3's frontend and the evaluator actually
      consume, and are much cheaper there.  The native-rate SNR is kept alongside as
      `snr_db_native` so the two are never confused.

    `src/eval/make_references.py::stage2` and `python src/eval/audio_qc.py FILE.wav` both
    call this, so the documented command reproduces the numbers in
    reports/reference_qc.md exactly.

    `path_label` is what goes into the report's `path` key; the reference builder passes
    None so that an absolute dataset path is not duplicated inside every stored `qc`
    block (references.jsonl already carries `source_audio_path`).
    """
    label = path if path_label is Ellipsis else path_label
    y, sr = load_audio(path, sr=None, start_sec=start_sec, end_sec=end_sec)
    rep = audio_qc(y, sr, with_f0=False, with_loop=False, with_wada=False,
                   with_timbre=True, path=label)
    if rep.get("invalid"):
        return rep
    y16, _ = load_audio(path, sr=PIPELINE_SR, start_sec=start_sec, end_sec=end_sec)
    rep["snr_db_native"] = rep["snr_db"]
    rep["snr_db"] = round(snr_energy_percentile(y16, PIPELINE_SR)[0], 3)
    rep["snr_method"] = ("energy_percentile_p10_p90 @16 kHz "
                         "(native rate in snr_db_native)")
    rep["snr_wada_db"] = round(snr_wada(y16), 3)
    rep.update(f0_metrics(y16, PIPELINE_SR))
    m16, h16, _ = vad_mask(y16, PIPELINE_SR)
    rep["loop"] = detect_audio_loop(y16, PIPELINE_SR, voiced_mask=m16, voiced_hop=h16)
    rep["analysis_rates"] = {
        "native_sr": int(sr),
        "pipeline_sr": PIPELINE_SR,
        "native": ["duration", "vad", "levels", "clipping", "flatness", "reverb",
                   "snr_db_native"],
        "pipeline_sr_metrics": ["snr_db", "snr_wada_db", "f0_*", "loop"],
    }
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Audio QC (PLAN.md §9.2/§9.6) -> JSON. Default mode reproduces the "
                    "reference QC table: SNR/F0/loop at 16 kHz, the rest at the file's "
                    "native rate.")
    ap.add_argument("wav")
    ap.add_argument("--mode", choices=("pipeline", "plain"), default="pipeline",
                    help="pipeline (default): exactly what make_references.py stage 2 "
                         "measures, i.e. what reports/reference_qc.md shows. "
                         "plain: one single rate for everything, controlled by --sr.")
    ap.add_argument("--sr", type=int, default=PIPELINE_SR,
                    help=f"plain mode only: analysis rate (default {PIPELINE_SR}, the "
                         "rate the model and the evaluator consume)")
    ap.add_argument("--native-sr", action="store_true",
                    help="plain mode only: analyse at the file's own sample rate")
    ap.add_argument("--no-loop", action="store_true",
                    help="plain mode only: skip loop detection")
    ap.add_argument("--f0", action="store_true",
                    help="plain mode only: add pyin pitch statistics "
                         "(pipeline mode always computes them)")
    ap.add_argument("--wada", action="store_true",
                    help="plain mode only: add the WADA-SNR estimate "
                         "(pipeline mode always computes it)")
    ap.add_argument("--no-timbre", action="store_true",
                    help="plain mode only: skip the spectral-flatness / reverb proxies")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--compact", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "pipeline":
        # refuse rather than silently ignore: a flag that does nothing is a lie
        given = [n for n, v in (("--sr", a.sr != PIPELINE_SR), ("--native-sr", a.native_sr),
                                ("--no-loop", a.no_loop), ("--f0", a.f0),
                                ("--wada", a.wada), ("--no-timbre", a.no_timbre)) if v]
        if given:
            ap.error(f"these flags only apply to --mode plain: {', '.join(given)}; "
                     "pipeline mode fixes the rates and computes F0/WADA/loop always")
        rep = audio_qc_pipeline(a.wav, start_sec=a.start, end_sec=a.end)
    else:
        rep = audio_qc_file(a.wav, sr=None if a.native_sr else a.sr, with_f0=a.f0,
                            with_loop=not a.no_loop, with_wada=a.wada,
                            with_timbre=not a.no_timbre,
                            start_sec=a.start, end_sec=a.end)
    print(json.dumps(rep, ensure_ascii=False,
                     indent=None if a.compact else 2, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
