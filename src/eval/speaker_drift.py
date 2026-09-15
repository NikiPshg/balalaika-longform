#!/usr/bin/env python
"""A5 / PLAN.md §9.5 — speaker drift of a long generated output against its enrollment.

CPU only, deterministic, no torch. Everything runs inside `.venv-eval`
(numpy + soundfile + soxr + onnxruntime); `src/eval/audio_qc.py` supplies the VAD.

What is measured
----------------
The generated wav is split into **voiced windows of 10 s with a 5 s hop** (PLAN §9.5).
"Voiced" is literal: the energy VAD of `audio_qc.vad_mask` is applied first, the silent
samples are dropped, and the windows are cut on the *voiced* timeline, so a window always
carries 10 s of speech and never 10 s of a pause (A5 acceptance: "silent windows
excluded"). Every window is embedded and compared by cosine similarity with a single
enrollment embedding computed from the whole voiced part of `data/references/<voice>.wav`
— the same wav the synthesis was conditioned on.

Reported per item (PLAN §9.5): `first30_median`, `last30_median`, `delta`,
`slope_per_min` (Theil–Sen, robust), `p05`, `min`, and `t_voice_sec` — the wall-clock
start of the first run of `T_VOICE_MIN_RUN` consecutive windows below the frozen
threshold (PLAN §9.3: an event is only recorded when several windows agree). An item that
never crosses it is right-censored (`t_voice_sec = null`, `t_voice_censored = true`).

Speaker encoder (PLAN §9.5 "один speaker encoder, независимый от campplus")
--------------------------------------------------------------------------
**Primary: WeSpeaker ResNet34-TSTP-emb256 (voxceleb_resnet34_LM.onnx)**, HF repo
`Wespeaker/wespeaker-voxceleb-resnet34-LM`, revision `f0c48c298fd835726c27956a5d617bad7115627e`,
sha256 `7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068`, VoxCeleb2-dev,
CC-BY-4.0, run through onnxruntime on CPU. It is independent of the CAM++ encoder that
CosyVoice3 itself conditions on: different architecture (ResNet34 r-vector vs CAM++),
different training corpus, different embedding size (256 vs 192). `src/eval/spk_campplus.py`
stays available as the *secondary* (the model's own encoder).

The ONNX graph takes `feats` of shape (B, T, 80): 80-bin Kaldi fbank, 25 ms / 10 ms,
computed on the int16-scaled waveform, with cepstral mean normalisation over time — the
recipe of `wespeaker/cli/speaker.py::compute_fbank`. `.venv-eval` has no torchaudio, so
`kaldi_fbank()` below reimplements `torchaudio.compliance.kaldi.fbank` in numpy;
`tests/test_speaker_drift.py` checks it against the real torchaudio when torchaudio is
importable, and `scripts/a5_fbank_parity.py` prints the same check under the cosyvoice env.

CLI
---
    # one file against one reference
    python src/eval/speaker_drift.py --wav OUT.wav --reference data/references/ref_male_01_16k.wav

    # a whole experiment directory, joined to per_item.jsonl by output_path
    python src/eval/speaker_drift.py --per-item results/v31_sft/E3_.../per_item.jsonl \
        --references data/references/references.jsonl --out results/v31_drift/E3/per_item.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import audio_qc  # noqa: E402  (same package dir; keeps the module importable as a script)

# --------------------------------------------------------------------------- constants
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

WESPEAKER_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
WESPEAKER_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
WESPEAKER_ONNX = os.path.join(
    REPO_ROOT, ".cache/hf/models--Wespeaker--wespeaker-voxceleb-resnet34-LM/snapshots",
    WESPEAKER_REVISION, "voxceleb_resnet34_LM.onnx")
WESPEAKER_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
EMB_DIM = 256

TARGET_SR = 16000
WINDOW_SEC = 10.0          # PLAN §9.5: voiced windows 10-15 s
HOP_SEC = 5.0              # PLAN §9.5: hop 5 s
EDGE_SEC = 30.0            # first-30-s / last-30-s medians
TAIL_MIN_SEC = 1.0         # leftover voiced audio that earns an extra end-aligned window
T_VOICE_MIN_RUN = 3        # consecutive windows below threshold that make an event
RESAMPLER = "soxr"         # numpy-only; see `resampler_note()`

# fbank (torchaudio.compliance.kaldi defaults + wespeaker's num_mel_bins=80, dither=0)
FBANK_NUM_MEL_BINS = 80
FBANK_FRAME_LENGTH_MS = 25.0
FBANK_FRAME_SHIFT_MS = 10.0
FBANK_LOW_FREQ = 20.0
FBANK_HIGH_FREQ = 0.0      # <= 0 -> nyquist
FBANK_PREEMPH = 0.97
FBANK_EPSILON = float(np.finfo(np.float32).eps)   # torchaudio's EPSILON
INT16_SCALE = 32768.0      # wespeaker loads with normalize=False, i.e. int16 units

_SESSION = None
_MODEL_SHA = {}


def resampler_note():
    return (f"{RESAMPLER}; soxr VHQ, not torchaudio — see reports/speaker_thresholds.md §2 "
            f"(cosine between the two resamplers measured there)")


# ------------------------------------------------------------------------------- fbank
def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** int(np.ceil(np.log2(x)))


def _mel_scale(freq):
    return 1127.0 * np.log(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def mel_banks(num_bins, window_length_padded, sample_freq,
              low_freq=FBANK_LOW_FREQ, high_freq=FBANK_HIGH_FREQ):
    """Triangular mel filterbank, arithmetic of `torchaudio.compliance.kaldi.get_mel_banks`.

    Returns (num_bins, window_length_padded // 2) — the caller zero-pads the last
    (Nyquist) FFT bin, exactly as torchaudio does.
    """
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq = high_freq + nyquist
    fft_bin_width = sample_freq / window_length_padded
    mel_low = _mel_scale(low_freq)
    mel_high = _mel_scale(high_freq)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)
    b = np.arange(num_bins, dtype=np.float64)[:, None]
    left_mel = mel_low + b * mel_delta
    center_mel = mel_low + (b + 1.0) * mel_delta
    right_mel = mel_low + (b + 2.0) * mel_delta
    mel = _mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float64))[None, :]
    up = (mel - left_mel) / (center_mel - left_mel)
    down = (right_mel - mel) / (right_mel - center_mel)
    return np.maximum(0.0, np.minimum(up, down))


def kaldi_fbank(y, sr=TARGET_SR, num_mel_bins=FBANK_NUM_MEL_BINS,
                frame_length_ms=FBANK_FRAME_LENGTH_MS,
                frame_shift_ms=FBANK_FRAME_SHIFT_MS, low_freq=FBANK_LOW_FREQ,
                high_freq=FBANK_HIGH_FREQ, preemphasis=FBANK_PREEMPH):
    """`torchaudio.compliance.kaldi.fbank(..., dither=0)` in numpy. Returns (T, num_mel_bins).

    `y` is a 1-D waveform **in int16 units** (see INT16_SCALE): that is what
    wespeaker feeds the model (`torchaudio.load(..., normalize=False)`).
    Steps, in torchaudio's order: strided frames (snip_edges) -> remove DC ->
    pre-emphasis with a replicated left edge -> povey window -> zero-pad to the next
    power of two -> |rfft|^2 -> mel matrix -> log(max(., eps)).
    """
    y = np.asarray(y, dtype=np.float32).ravel()
    win = int(sr * frame_length_ms * 0.001)
    shift = int(sr * frame_shift_ms * 0.001)
    padded = _next_power_of_2(win)
    if len(y) < win:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    m = 1 + (len(y) - win) // shift
    idx = np.arange(win)[None, :] + shift * np.arange(m)[:, None]
    fr = y[idx].astype(np.float32)
    fr = fr - fr.mean(axis=1, keepdims=True, dtype=np.float32)          # remove_dc_offset
    if preemphasis != 0.0:
        left = np.concatenate([fr[:, :1], fr[:, :-1]], axis=1)          # replicate pad
        fr = fr - np.float32(preemphasis) * left
    n = np.arange(win, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (win - 1))              # periodic=False
    window = np.power(hann, 0.85).astype(np.float32)                    # povey
    fr = fr * window
    if padded > win:
        fr = np.concatenate([fr, np.zeros((m, padded - win), dtype=np.float32)], axis=1)
    spec = np.abs(np.fft.rfft(fr.astype(np.float64), axis=1)) ** 2      # use_power
    bank = mel_banks(num_mel_bins, padded, float(sr), low_freq, high_freq)
    bank = np.concatenate([bank, np.zeros((num_mel_bins, 1))], axis=1)  # Nyquist bin
    mel = spec.astype(np.float32) @ bank.astype(np.float32).T
    return np.log(np.maximum(mel, np.float32(FBANK_EPSILON))).astype(np.float32)


# -------------------------------------------------------------------------- the encoder
def model_sha256(model_path=WESPEAKER_ONNX):
    if model_path not in _MODEL_SHA:
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _MODEL_SHA[model_path] = h.hexdigest()
    return _MODEL_SHA[model_path]


def session(model_path=WESPEAKER_ONNX, threads=4):
    """Process-local onnxruntime session, CPU only."""
    global _SESSION
    if _SESSION is None:
        import onnxruntime

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"speaker encoder missing: {model_path}\n"
                f"download it with:  python src/eval/speaker_drift.py --download")
        got = model_sha256(model_path)
        if got != WESPEAKER_SHA256:
            raise RuntimeError(f"{model_path}: sha256 {got} != expected {WESPEAKER_SHA256}")
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = threads
        _SESSION = onnxruntime.InferenceSession(model_path, sess_options=opt,
                                                providers=["CPUExecutionProvider"])
    return _SESSION


def embed_waveform(y, sr=TARGET_SR):
    """`y`: 1-D mono waveform in [-1, 1] at 16 kHz. Returns a (256,) float32 embedding."""
    if sr != TARGET_SR:
        raise ValueError(f"embed_waveform expects {TARGET_SR} Hz, got {sr}")
    feat = kaldi_fbank(np.asarray(y, dtype=np.float64) * INT16_SCALE, sr)
    if feat.shape[0] == 0:
        raise ValueError("waveform shorter than one fbank frame")
    feat = feat - feat.mean(axis=0, keepdims=True)                      # CMN, wespeaker
    s = session()
    out = s.run(None, {s.get_inputs()[0].name: feat[None, :, :].astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


def embed_batch(windows, sr=TARGET_SR):
    """Embed a list of equal-or-unequal-length windows (one ORT call per window)."""
    return np.stack([embed_waveform(w, sr) for w in windows]) if windows else \
        np.zeros((0, EMB_DIM), dtype=np.float32)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


# ------------------------------------------------------------------------ voiced windows
def voiced_sample_mask(y, sr):
    """Frame-level VAD of `audio_qc.vad_mask`, expanded to a per-sample boolean mask."""
    mask, hop, thr = audio_qc.vad_mask(y, sr)
    out = np.zeros(len(y), dtype=bool)
    if len(mask) == 0:
        return out, thr
    starts = np.arange(len(mask)) * hop
    for a, b, v in audio_qc._runs(mask):
        if v:
            out[starts[a]:min(len(y), starts[b - 1] + hop)] = True
    return out, thr


def voiced_windows(y, sr, window_sec=WINDOW_SEC, hop_sec=HOP_SEC):
    """Cut `window_sec` of *voiced* audio every `hop_sec` of voiced audio.

    Returns (windows, meta). Each meta entry carries the wall-clock span the window's
    samples come from (`t_start_sec`, `t_end_sec`, `t_center_sec`) and its position on
    the voiced timeline (`voiced_start_sec`). A final end-aligned window is added when
    the regular grid would leave more than one hop of voiced audio unused, so the tail
    of a long output is always represented.
    """
    vmask, thr = voiced_sample_mask(y, sr)
    vidx = np.flatnonzero(vmask)
    win = int(round(window_sec * sr))
    hop = int(round(hop_sec * sr))
    n = len(vidx)
    meta = {"voiced_sec": round(n / sr, 4), "raw_sec": round(len(y) / sr, 4),
            "vad_threshold_dbfs": round(float(thr), 3), "window_sec": window_sec,
            "hop_sec": hop_sec}
    if n < win:
        return [], [], meta
    starts = list(range(0, n - win + 1, hop))
    # end-aligned tail window: without it the last seconds of a long output are never
    # measured, and the last-30-s median would ignore exactly the part §9.5 is about.
    # Added only when more than TAIL_MIN_SEC of voiced audio would otherwise be left out,
    # so it never duplicates the previous window.
    if (n - (starts[-1] + win)) > int(round(TAIL_MIN_SEC * sr)):
        starts.append(n - win)
    wins, info = [], []
    for st in starts:
        sel = vidx[st:st + win]
        wins.append(np.ascontiguousarray(y[sel]))
        info.append({
            "voiced_start_sec": round(st / sr, 4),
            "t_start_sec": round(float(sel[0]) / sr, 4),
            "t_end_sec": round(float(sel[-1] + 1) / sr, 4),
            "t_center_sec": round(0.5 * (float(sel[0]) + float(sel[-1] + 1)) / sr, 4),
        })
    return wins, info, meta


def load_16k(path, res_type=RESAMPLER):
    y, sr = audio_qc.load_audio(path, sr=TARGET_SR, res_type=res_type)
    return np.asarray(y, dtype=np.float64), sr


# ----------------------------------------------------------------------------- statistics
def theil_sen_slope(x, y):
    """Median of pairwise slopes. Robust to the outlier windows a loop produces."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return None
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-12
    if not ok.any():
        return None
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def _first_run_below(sims, threshold, min_run):
    """Index of the first window that starts a run of >= min_run windows below threshold."""
    below = np.asarray(sims, dtype=np.float64) < threshold
    run = 0
    for k, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= min_run:
            return k - min_run + 1
    return None


def drift_metrics(sims, info, threshold, min_run=T_VOICE_MIN_RUN, edge_sec=EDGE_SEC):
    """Turn a similarity track into the PLAN §9.5 numbers."""
    sims = np.asarray(sims, dtype=np.float64)
    n = len(sims)
    out = {"n_windows": int(n)}
    if n == 0:
        out.update({k: None for k in
                    ("first30_median", "last30_median", "delta", "slope_per_min",
                     "sim_p05", "sim_min", "sim_median", "sim_max", "t_voice_sec",
                     "t_voice_window", "n_windows_below_threshold")})
        out["t_voice_censored"] = None
        return out
    t_mid = np.array([w["t_center_sec"] for w in info], dtype=np.float64)
    t_end = max(w["t_end_sec"] for w in info)
    head = sims[t_mid <= edge_sec]
    tail = sims[t_mid >= max(t_end - edge_sec, 0.0)]
    out["first30_median"] = round(float(np.median(head)), 6) if len(head) else None
    out["last30_median"] = round(float(np.median(tail)), 6) if len(tail) else None
    out["first_last_windows_overlap"] = bool(t_end <= 2 * edge_sec)
    if out["first30_median"] is not None and out["last30_median"] is not None:
        out["delta"] = round(out["last30_median"] - out["first30_median"], 6)
    else:
        out["delta"] = None
    slope = theil_sen_slope(t_mid / 60.0, sims)
    out["slope_per_min"] = round(slope, 6) if slope is not None else None
    out["sim_median"] = round(float(np.median(sims)), 6)
    out["sim_p05"] = round(float(np.percentile(sims, 5)), 6)
    out["sim_min"] = round(float(sims.min()), 6)
    out["sim_max"] = round(float(sims.max()), 6)
    out["n_windows_below_threshold"] = int((sims < threshold).sum())
    k = _first_run_below(sims, threshold, min_run)
    out["t_voice_window"] = k
    out["t_voice_sec"] = info[k]["t_start_sec"] if k is not None else None
    out["t_voice_censored"] = k is None
    return out


# ------------------------------------------------------------------------------- per file
def drift_for_wav(wav_path, enroll_emb, threshold, window_sec=WINDOW_SEC,
                  hop_sec=HOP_SEC, res_type=RESAMPLER, keep_windows=True):
    """Full §9.5 record for one wav against one enrollment embedding."""
    y, sr = load_16k(wav_path, res_type=res_type)
    wins, info, meta = voiced_windows(y, sr, window_sec, hop_sec)
    rec = {"wav_path": wav_path, "sr": sr, **meta}
    if wins:
        embs = embed_batch(wins, sr)
        sims = [cosine(e, enroll_emb) for e in embs]
    else:
        sims = []
    rec.update(drift_metrics(sims, info, threshold))
    # whole-file similarity on all voiced audio: the only number available for outputs
    # too short for a single 10 s window, and a sanity check for the rest.
    vmask, _ = voiced_sample_mask(y, sr)
    vy = y[vmask]
    if len(vy) >= int(0.4 * sr):
        rec["sim_whole_voiced"] = round(cosine(embed_waveform(vy, sr), enroll_emb), 6)
    else:
        rec["sim_whole_voiced"] = None
    rec["too_short_for_windows"] = not wins
    if keep_windows:
        rec["windows"] = [dict(w, sim=round(float(s), 6)) for w, s in zip(info, sims)]
    return rec


def enrollment_embedding(ref_wav, res_type=RESAMPLER):
    """Enrollment = the whole *voiced* part of the reference wav, one embedding."""
    y, sr = load_16k(ref_wav, res_type=res_type)
    vmask, _ = voiced_sample_mask(y, sr)
    vy = y[vmask]
    if len(vy) < int(0.5 * sr):
        raise ValueError(f"{ref_wav}: less than 0.5 s of voiced audio")
    return embed_waveform(vy, sr), {"ref_wav": ref_wav, "ref_voiced_sec": round(len(vy) / sr, 4)}


# ------------------------------------------------------------------------------- config
def load_threshold(config_path=None):
    """Read the frozen T_voice threshold from configs/speaker_drift.yaml."""
    config_path = config_path or os.path.join(REPO_ROOT, "configs/speaker_drift.yaml")
    txt = open(config_path, encoding="utf-8").read()
    cfg = {}
    for line in txt.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line or line.startswith("-"):
            continue
        k, v = line.split(":", 1)
        v = v.strip().strip('"')
        if not v:
            continue
        try:
            cfg[k.strip()] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            cfg[k.strip()] = v
    for key in ("t_voice_similarity_min", "version", "frozen_at"):
        if key not in cfg:
            raise KeyError(f"{config_path}: missing {key}")
    return cfg


# ---------------------------------------------------------------------------------- CLI
def _download(dest_root=None):
    from huggingface_hub import hf_hub_download

    dest_root = dest_root or os.path.join(REPO_ROOT, ".cache/hf")
    p = hf_hub_download(WESPEAKER_REPO, "voxceleb_resnet34_LM.onnx",
                        revision=WESPEAKER_REVISION, cache_dir=dest_root)
    print(json.dumps({"path": p, "sha256": model_sha256(p),
                      "expected": WESPEAKER_SHA256,
                      "match": model_sha256(p) == WESPEAKER_SHA256}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="PLAN §9.5 speaker drift (CPU, onnxruntime)")
    ap.add_argument("--download", action="store_true", help="fetch the pinned encoder")
    ap.add_argument("--wav", help="one generated wav")
    ap.add_argument("--reference", help="enrollment wav for --wav")
    ap.add_argument("--per-item", help="results/<exp>/per_item.jsonl to walk")
    ap.add_argument("--references", default="data/references/references.jsonl")
    ap.add_argument("--out", help="output jsonl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    ap.add_argument("--hop-sec", type=float, default=HOP_SEC)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the frozen threshold (for calibration only)")
    ap.add_argument("--res-type", default=RESAMPLER,
                    help="audio_qc.resample res_type; 'soxr' (default, works in "
                         ".venv-eval) or 'torchaudio_strict' (needs the cosyvoice env)")
    a = ap.parse_args(argv)

    if a.download:
        return _download()

    thr = a.threshold
    cfg = None
    if thr is None:
        cfg = load_threshold(a.config)
        thr = cfg["t_voice_similarity_min"]

    if a.wav:
        emb, meta = enrollment_embedding(a.reference, res_type=a.res_type)
        rec = drift_for_wav(a.wav, emb, thr, a.window_sec, a.hop_sec, res_type=a.res_type)
        rec.update(meta)
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return 0

    if not a.per_item:
        ap.error("need --wav or --per-item")

    refs = {}
    refs_path = a.references if os.path.isabs(a.references) else os.path.join(REPO_ROOT, a.references)
    for line in open(refs_path, encoding="utf-8"):
        r = json.loads(line)
        refs[r["voice_id"]] = r
    enroll = {}
    per_item_path = a.per_item if os.path.isabs(a.per_item) else os.path.join(REPO_ROOT, a.per_item)
    rows = [json.loads(l) for l in open(per_item_path, encoding="utf-8")]
    out_rows = []
    for i, r in enumerate(rows):
        vid = r["voice_id"]
        if vid not in enroll:
            # The MASTER wav, not the stored 16 kHz copy: `make_references.py` built
            # that copy with torchaudio's resampler, and mixing it with soxr-resampled
            # windows costs up to 0.09 of cosine on the male reference (measured,
            # reports/speaker_thresholds.md §2). Both sides of every cosine must go
            # through the same resampler.
            ref_wav = os.path.join(REPO_ROOT, refs[vid]["wav_path"])
            enroll[vid] = enrollment_embedding(ref_wav, res_type=a.res_type)
        emb, emeta = enroll[vid]
        wav = r["output_path"]
        rec = {"text_id": r["text_id"], "voice_id": vid, "bucket": r["bucket"],
               "experiment_id": r.get("experiment_id"), "checkpoint": r.get("checkpoint"),
               "status": r.get("status"), "output_path": r["output_path"]}
        # output_path is None for items whose generation produced no audio at all
        # (final status empty_or_invalid_audio) - e.g. E6-M11 item 60/60, 2026-08-30.
        # There is nothing to embed: record it explicitly instead of crashing.
        if wav is None:
            rec.update({"error": "no_audio", "n_windows": 0})
            out_rows.append(rec)
            print(f"[{i + 1}/{len(rows)}] {r['text_id']} {vid} no audio (skipped)", file=sys.stderr)
            continue
        wav = wav if os.path.isabs(wav) else os.path.join(REPO_ROOT, wav)
        if not os.path.exists(wav):
            rec.update({"error": "missing_wav", "n_windows": 0})
        else:
            rec.update(drift_for_wav(wav, emb, thr, a.window_sec, a.hop_sec,
                                     res_type=a.res_type))
            rec.update(emeta)
        out_rows.append(rec)
        print(f"[{i + 1}/{len(rows)}] {r['text_id']} {vid} "
              f"n_win={rec.get('n_windows')} med={rec.get('sim_median')}", file=sys.stderr)
    header = {"_meta": True, "encoder": WESPEAKER_REPO, "revision": WESPEAKER_REVISION,
              "model_sha256": model_sha256(), "emb_dim": EMB_DIM,
              "window_sec": a.window_sec, "hop_sec": a.hop_sec,
              "t_voice_similarity_min": thr, "t_voice_min_run": T_VOICE_MIN_RUN,
              "resampler": a.res_type, "resampler_note": resampler_note(),
              "enrollment_source": "references.jsonl wav_path (master), resampled by this "
                                   "run with the same res_type as the outputs",
              "config": cfg}
    header["source_per_item"] = a.per_item
    header["references"] = a.references
    dest = a.out or "-"
    if dest != "-" and not os.path.isabs(dest):
        dest = os.path.join(REPO_ROOT, dest)
    if dest == "-":
        print(json.dumps(header, ensure_ascii=False))
        for r in out_rows:
            print(json.dumps(r, ensure_ascii=False))
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(out_rows)} rows -> {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
