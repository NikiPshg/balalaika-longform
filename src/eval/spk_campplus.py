#!/usr/bin/env python
"""A5 — campplus speaker embeddings, byte-identical to the recipe CosyVoice3 itself uses.

Reproduces `CosyVoiceFrontEnd._extract_spk_embedding`
(`third_party/CosyVoice/cosyvoice/cli/frontend.py:108-118`) together with
`cosyvoice.utils.file_utils.load_wav`:

    torchaudio.load(backend='soundfile') -> mean over channels -> resample to 16 kHz
    -> kaldi.fbank(num_mel_bins=80, dither=0, sample_frequency=16000)
    -> feat -= feat.mean(dim=0, keepdim=True)
    -> campplus.onnx  (CPUExecutionProvider, intra_op_num_threads=1, ORT_ENABLE_ALL)

campplus is the encoder the *model* conditions on, so PLAN §9.5 allows it only as a
**secondary** speaker measure; the primary drift encoder must be independent of it and is
added later (`src/eval/speaker_drift.py`), once the eval venv can host one.  This module
exists so the reference QC can answer "is the chosen reference closer to its own speaker
than to any other test speaker?" with the same numbers the model would see.

CPU only - no CUDA is touched anywhere in this file.

    python src/eval/spk_campplus.py A.wav B.wav      # cosine similarity of two files
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CAMPPLUS_ONNX = "models/cosyvoice3/campplus.onnx"
TARGET_SR = 16000
EMB_DIM = 192

_SESSION = None
_RESAMPLERS = {}
_MODEL_SHA = {}


def model_sha256(model_path=CAMPPLUS_ONNX):
    """sha256 of the onnx weights, memoised per path.

    Anything that caches embeddings must key on this: the same wav through a different
    campplus.onnx is a different vector, and a stale cache would silently mix the two.
    """
    if model_path not in _MODEL_SHA:
        import hashlib

        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _MODEL_SHA[model_path] = h.hexdigest()
    return _MODEL_SHA[model_path]


def session(model_path=CAMPPLUS_ONNX):
    """Lazily create the (process-local) onnxruntime session on CPU."""
    global _SESSION
    if _SESSION is None:
        import onnxruntime

        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = 1
        _SESSION = onnxruntime.InferenceSession(model_path, sess_options=opt,
                                                providers=["CPUExecutionProvider"])
    return _SESSION


def _resampler(orig, new):
    import torchaudio

    key = (orig, new)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(orig_freq=orig, new_freq=new)
    return _RESAMPLERS[key]


def embed_tensor(speech_1xN):
    """`speech_1xN`: torch float tensor (1, N) already at 16 kHz. Returns (192,) float32."""
    import torchaudio.compliance.kaldi as kaldi

    feat = kaldi.fbank(speech_1xN, num_mel_bins=80, dither=0,
                       sample_frequency=TARGET_SR)
    feat = feat - feat.mean(dim=0, keepdim=True)
    s = session()
    emb = s.run(None, {s.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
    return emb.astype(np.float32)


def embed_waveform(y, sr):
    """`y`: 1-D numpy mono waveform in [-1, 1] at `sr`. Returns (192,) float32."""
    import torch

    w = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
    if sr != TARGET_SR:
        w = _resampler(sr, TARGET_SR)(w)
    return embed_tensor(w)


def embed_file(path, start_sec=None, end_sec=None, max_sec=None, center=True):
    """Embed (a slice of) an audio file.

    `max_sec` keeps a centred (`center=True`) or leading window of that many seconds,
    which is how A1's leakage check samples long segments (30 s, centred).
    """
    import soundfile as sf
    import torch

    info = sf.info(path)
    sr = info.samplerate
    a = 0 if start_sec is None else max(0, int(round(start_sec * sr)))
    b = info.frames if end_sec is None else min(info.frames, int(round(end_sec * sr)))
    if max_sec is not None and (b - a) > int(max_sec * sr):
        want = int(max_sec * sr)
        a = a + ((b - a - want) // 2 if center else 0)
        b = a + want
    data, file_sr = sf.read(path, start=a, frames=b - a, dtype="float32", always_2d=True)
    w = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
    if file_sr != TARGET_SR:
        w = _resampler(file_sr, TARGET_SR)(w)
    return embed_tensor(w)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="campplus embedding / cosine similarity")
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--max-sec", type=float, default=None)
    a = ap.parse_args(argv)
    embs = {p: embed_file(p, max_sec=a.max_sec) for p in a.wav}
    out = {"model": CAMPPLUS_ONNX, "model_sha256": model_sha256(), "dim": EMB_DIM,
           "files": {p: {"norm": round(float(np.linalg.norm(e)), 6)} for p, e in embs.items()}}
    if len(a.wav) >= 2:
        out["cosine"] = {
            f"{os.path.basename(p)}|{os.path.basename(q)}": round(cosine(embs[p], embs[q]), 6)
            for i, p in enumerate(a.wav) for q in a.wav[i + 1:]
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
