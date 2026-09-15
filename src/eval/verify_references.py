#!/usr/bin/env python
"""A5 — regenerate every load-bearing verification of the voice references.

Wave 1 produced this evidence from ad-hoc scratch scripts under `tmp/a5/`, which is
gitignored: the proofs shipped as prose while the code that made them did not ship at
all.  This module is that code, tracked, with its output written to
`reports/reference_verification.{md,json}` so `reports/reference_qc.md` can cite a file
that actually exists in the repository.

Four independent checks, none of which trusts `make_references.py`:

1. **campplus byte-identity** — runs CosyVoice's own extractor
   (`frontend.py:108-118` + `file_utils.py:44-50`: load_wav -> mean over channels ->
   Resample -> kaldi.fbank(80, dither=0) -> mean-subtract -> campplus.onnx, CPU,
   intra_op=1, ORT_ENABLE_ALL) against `src/eval/spk_campplus.py` on every reference wav.
   Expected: max|diff| = 0.
2. **master fidelity** — re-reads the source FLAC at the recorded offsets with soundfile,
   mixes the channels down itself, and compares with the shipped PCM_24 master; then
   compares the shipped 16 kHz copy with `torchaudio.transforms.Resample` of that master.
   Expected: quantisation only (1 LSB24 = 5.96e-08, 1 LSB16 = 3.05e-05).
3. **F0 cross-check** — the gender label rests on pyin, so pyin is cross-checked against
   `librosa.yin` and an independent cepstral peak estimator implemented here.  Expected:
   all three on the same side of the 165 Hz boundary.
4. **loop detector on real audio** — no event on the four natural references; an event at
   the right lag when a 3 s stretch of each is repeated.  `--long-audio N` adds N of the
   longest test recordings (slow: ~10 s of decode each).

CPU only, no GPU, no writes outside the working directory.

    python src/eval/verify_references.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_qc as aq  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COSYVOICE = "third_party/CosyVoice"
LSB24 = 2.0 ** -23
LSB16 = 2.0 ** -15
F0_GENDER_HZ = 165.0


# ------------------------------------------------------------------ 1. campplus identity
def repo_embedding(path, campplus_onnx):
    """Verbatim `CosyVoiceFrontEnd._extract_spk_embedding` (frontend.py:108-118)."""
    import onnxruntime
    import torchaudio.compliance.kaldi as kaldi

    sys.path.insert(0, COSYVOICE)
    sys.path.insert(0, os.path.join(COSYVOICE, "third_party", "Matcha-TTS"))
    from cosyvoice.utils.file_utils import load_wav  # noqa: PLC0415

    speech = load_wav(path, 16000)
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    sess = onnxruntime.InferenceSession(campplus_onnx, sess_options=option,
                                        providers=["CPUExecutionProvider"])
    emb = sess.run(None, {sess.get_inputs()[0].name:
                          feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten()
    return np.asarray(emb, dtype=np.float32)


def check_campplus(refs):
    import spk_campplus as sc

    rows = []
    for r in refs:
        for key in ("wav_path", "wav16k_path"):
            p = os.path.join(ROOT, r[key])
            a = repo_embedding(p, sc.CAMPPLUS_ONNX)
            b = sc.embed_file(p)
            rows.append({"voice_id": r["voice_id"], "which": key, "path": r[key],
                         "dim": int(a.shape[0]),
                         "maxabsdiff": float(np.max(np.abs(a - b))),
                         "cosine": round(sc.cosine(a, b), 9)})
    worst = max(r["maxabsdiff"] for r in rows)
    return {"model": sc.CAMPPLUS_ONNX, "model_sha256": sc.model_sha256(),
            "rows": rows, "worst_maxabsdiff": worst, "pass": worst == 0.0}


# ------------------------------------------------------------------- 2. master fidelity
def check_master(refs):
    import soundfile as sf

    rows = []
    for r in refs:
        src = r["source_audio_path"]
        info = sf.info(src)
        # offsets in references.jsonl are relative to the segment FLAC itself
        i0 = int(round(r["offset_start_sec"] * info.samplerate))
        i1 = int(round(r["offset_end_sec"] * info.samplerate))
        data, sr_src = sf.read(src, dtype="float64", always_2d=True, start=i0, stop=i1)
        mix = data.mean(axis=1)
        master, sr_m = sf.read(os.path.join(ROOT, r["wav_path"]), dtype="float64",
                               always_2d=True)
        master = master[:, 0]
        n = min(len(mix), len(master))
        err = float(np.max(np.abs(mix[:n] - master[:n])))
        k16 = os.path.join(ROOT, r["wav16k_path"])
        y16, sr16 = sf.read(k16, dtype="float64", always_2d=True)
        y16 = y16[:, 0]
        ref16 = aq.resample(master, sr_m, sr16, res_type="torchaudio_strict")
        n16 = min(len(y16), len(ref16))
        err16 = float(np.max(np.abs(y16[:n16] - ref16[:n16])))
        rows.append({
            "voice_id": r["voice_id"], "source": src,
            "n_source_samples": int(len(mix)), "n_master_samples": int(len(master)),
            "samples_equal": len(mix) == len(master),
            "master_sr": int(sr_m), "wav16k_sr": int(sr16),
            "master_vs_exact_mix_maxerr": err, "master_vs_exact_mix_lsb24": err / LSB24,
            "wav16k_vs_torchaudio_maxerr": err16,
            "wav16k_vs_torchaudio_lsb16": err16 / LSB16,
        })
    ok = all(r["samples_equal"] and r["master_vs_exact_mix_lsb24"] <= 1.5
             and r["wav16k_vs_torchaudio_lsb16"] <= 1.5 for r in rows)
    return {"rows": rows, "pass": ok,
            "note": "1 LSB24 = 5.96e-08, 1 LSB16 = 3.05e-05; anything at or below that is "
                    "quantisation, not a processing difference"}


# --------------------------------------------------------------------- 3. F0 cross-check
def cepstral_f0(y, sr, fmin=aq.F0_MIN_HZ, fmax=aq.F0_MAX_HZ, frame_ms=64.0, hop_ms=20.0):
    """Median cepstral-peak F0 over voiced frames — independent of librosa's estimators.

    Real cepstrum of a Hann-windowed frame; the peak is searched in the quefrency band
    [1/fmax, 1/fmin] s and only frames whose peak stands clearly above the band's median
    contribute.
    """
    mask, hop, _ = aq.vad_mask(y, sr)
    flen = int(round(sr * frame_ms / 1000.0))
    step = int(round(sr * hop_ms / 1000.0))
    q0, q1 = int(sr / fmax), int(sr / fmin)
    vals = []
    for i in range(0, max(0, len(y) - flen), step):
        mi = min(int(i / hop), len(mask) - 1)
        if not mask[mi]:
            continue
        fr = y[i:i + flen] * np.hanning(flen)
        sp = np.abs(np.fft.rfft(fr)) + 1e-12
        cep = np.fft.irfft(np.log(sp))
        band = cep[q0:q1 + 1]
        if len(band) < 3:
            continue
        k = int(np.argmax(band))
        if band[k] <= 3.0 * np.median(np.abs(band)):
            continue
        vals.append(sr / float(q0 + k))
    return float(np.median(vals)) if vals else None


def check_f0(refs):
    import librosa

    rows = []
    for r in refs:
        y, sr = aq.load_audio(os.path.join(ROOT, r["wav_path"]), sr=aq.PIPELINE_SR)
        pyin = aq.f0_metrics(y, sr)["f0_median_hz"]
        yin = librosa.yin(np.ascontiguousarray(y), fmin=aq.F0_MIN_HZ, fmax=aq.F0_MAX_HZ,
                          sr=sr, frame_length=aq.F0_FRAME_LENGTH)
        yin_med = float(np.median(yin[np.isfinite(yin)]))
        cep = cepstral_f0(y, sr)
        est = [e for e in (pyin, yin_med, cep) if e is not None]
        rows.append({
            "voice_id": r["voice_id"], "stored_label": r["gender_heuristic"],
            "pyin_hz": round(float(pyin), 1) if pyin else None,
            "yin_hz": round(yin_med, 1), "cepstrum_hz": round(cep, 1) if cep else None,
            "all_same_side_of_165hz": len({e > F0_GENDER_HZ for e in est}) == 1,
            "label_from_pyin": "female" if pyin and pyin > F0_GENDER_HZ else "male",
        })
    ok = all(r["all_same_side_of_165hz"] and r["label_from_pyin"] == r["stored_label"]
             for r in rows)
    return {"rows": rows, "pass": ok, "boundary_hz": F0_GENDER_HZ,
            "note": "F0 is a proxy for gender and nothing more; no one has listened"}


# ----------------------------------------------------------------- 4. loop on real audio
def check_loop(refs, long_audio=0, manifest="data/manifests/test.jsonl"):
    nat, rep = [], []
    for r in refs:
        y, sr = aq.load_audio(os.path.join(ROOT, r["wav_path"]), sr=aq.PIPELINE_SR)
        d = aq.detect_audio_loop(y, sr)
        nat.append({"voice_id": r["voice_id"], "loop_detected": d["loop_detected"],
                    "max_smoothed_similarity": (d.get("max_similarity_by_lag_summary")
                                                or {}).get("max"),
                    "n_events": d["n_events"]})
        piece = y[int(2.0 * sr):int(5.0 * sr)]
        looped = np.concatenate([y[:int(2.0 * sr)], piece, piece, y[int(5.0 * sr):]])
        d2 = aq.detect_audio_loop(looped, sr)
        ev = (d2.get("events") or [{}])[0]
        rep.append({"voice_id": r["voice_id"], "loop_detected": d2["loop_detected"],
                    "n_events": d2["n_events"],
                    "best_lag_sec": ev.get("lag_sec"),
                    "duration_sec": ev.get("duration_sec"),
                    "similarity": ev.get("mean_similarity")})
    out = {"natural_references": nat, "with_3s_repeat": rep,
           "pass": (not any(r["loop_detected"] for r in nat)
                    and all(r["loop_detected"] for r in rep)
                    and all(abs((r["best_lag_sec"] or 0) - 3.0) < 0.15 for r in rep))}
    if long_audio:
        rows = json.load(open(os.path.join(ROOT, manifest), encoding="utf-8")) \
            if manifest.endswith(".json") else \
            [json.loads(l) for l in open(os.path.join(ROOT, manifest), encoding="utf-8")]
        rows = sorted(rows, key=lambda r: -r["duration_sec"])[:long_audio]
        longs = []
        for r in rows:
            t0 = time.time()
            y, sr = aq.load_audio(r["audio_path"], sr=aq.PIPELINE_SR)
            tl = time.time() - t0
            t0 = time.time()
            d = aq.detect_audio_loop(y, sr)
            longs.append({"sample_id": r["sample_id"],
                          "duration_sec": round(len(y) / sr, 1),
                          "loop_detected": d["loop_detected"],
                          "n_events": d["n_events"],
                          "max_smoothed_similarity": (
                              d.get("max_similarity_by_lag_summary") or {}).get("max"),
                          "load_sec": round(tl, 1), "detect_sec": round(time.time() - t0, 1)})
        out["long_natural_recordings"] = longs
        out["pass"] = out["pass"] and not any(r["loop_detected"] for r in longs)
    return out


# ---------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate the A5 reference verification")
    ap.add_argument("--refs", default="data/references/references.jsonl")
    ap.add_argument("--out", default="reports/reference_verification")
    ap.add_argument("--long-audio", type=int, default=0,
                    help="also run the loop detector on the N longest test recordings "
                         "(~12 s each)")
    ap.add_argument("--skip", default="", help="comma-separated: campplus,master,f0,loop")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    refs = sorted([json.loads(l) for l in open(a.refs, encoding="utf-8")],
                  key=lambda r: r["voice_id"])

    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_references": len(refs), "skipped": sorted(skip)}
    order = [("master", check_master, (refs,)), ("f0", check_f0, (refs,)),
             ("loop", check_loop, (refs, a.long_audio)),
             ("campplus", check_campplus, (refs,))]  # campplus last: it initialises onnx
    for name, fn, args in order:
        if name in skip:
            continue
        t0 = time.time()
        res[name] = fn(*args)
        res[name]["wall_sec"] = round(time.time() - t0, 1)
        print(f"[verify] {name}: pass={res[name]['pass']} "
              f"({res[name]['wall_sec']}s)", file=sys.stderr)
    res["all_pass"] = all(v["pass"] for k, v in res.items()
                          if isinstance(v, dict) and "pass" in v)

    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    L = []
    L.append("# A5 — reference verification (regenerated)")
    L.append("")
    L.append(f"`src/eval/verify_references.py`, run {res['generated']}. "
             f"Overall: **{'PASS' if res['all_pass'] else 'FAIL'}**. Every number below "
             "is recomputed from the shipped wavs and the source dataset, without "
             "trusting `make_references.py`.")
    L.append("")
    if "campplus" in res:
        c = res["campplus"]
        L.append("## 1. campplus is byte-identical to the model's own extractor")
        L.append("")
        L.append(f"`{c['model']}` (sha256 `{c['model_sha256']}`), "
                 f"worst max|diff| over {len(c['rows'])} files: "
                 f"**{c['worst_maxabsdiff']:.3e}** — {'PASS' if c['pass'] else 'FAIL'}.")
        L.append("")
        L.append("| file | dim | max abs diff | cosine |")
        L.append("|---|---:|---:|---:|")
        for r in c["rows"]:
            L.append(f"| `{r['path']}` | {r['dim']} | {r['maxabsdiff']:.3e} | "
                     f"{r['cosine']} |")
        L.append("")
    if "master" in res:
        m = res["master"]
        L.append("## 2. Master fidelity")
        L.append("")
        L.append(f"{m['note']} — {'PASS' if m['pass'] else 'FAIL'}.")
        L.append("")
        L.append("| voice_id | samples | master vs exact mix | in LSB24 | 16k vs "
                 "torchaudio | in LSB16 |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for r in m["rows"]:
            L.append(f"| `{r['voice_id']}` | {r['n_master_samples']} | "
                     f"{r['master_vs_exact_mix_maxerr']:.3e} | "
                     f"{r['master_vs_exact_mix_lsb24']:.2f} | "
                     f"{r['wav16k_vs_torchaudio_maxerr']:.3e} | "
                     f"{r['wav16k_vs_torchaudio_lsb16']:.2f} |")
        L.append("")
    if "f0" in res:
        f = res["f0"]
        L.append("## 3. F0 cross-check (the gender label)")
        L.append("")
        L.append(f"Boundary {f['boundary_hz']} Hz. {f['note']} — "
                 f"{'PASS' if f['pass'] else 'FAIL'}.")
        L.append("")
        L.append("| voice_id | pyin Hz | librosa.yin Hz | cepstral peak Hz | same side | "
                 "stored label |")
        L.append("|---|---:|---:|---:|---|---|")
        for r in f["rows"]:
            L.append(f"| `{r['voice_id']}` | {r['pyin_hz']} | {r['yin_hz']} | "
                     f"{r['cepstrum_hz']} | {r['all_same_side_of_165hz']} | "
                     f"{r['stored_label']} |")
        L.append("")
    if "loop" in res:
        lp = res["loop"]
        L.append("## 4. Loop detector on real audio")
        L.append("")
        L.append(f"{'PASS' if lp['pass'] else 'FAIL'} — no event on the natural "
                 "references, an event at lag ≈ 3.0 s when a 3 s stretch of each is "
                 "repeated once.")
        L.append("")
        L.append("| voice_id | natural: detected / max sim | with 3 s repeat: detected / "
                 "lag s / dur s / sim |")
        L.append("|---|---|---|")
        for n, r in zip(lp["natural_references"], lp["with_3s_repeat"]):
            L.append(f"| `{n['voice_id']}` | {n['loop_detected']} / "
                     f"{n['max_smoothed_similarity']} | {r['loop_detected']} / "
                     f"{r['best_lag_sec']} / {r['duration_sec']} / {r['similarity']} |")
        L.append("")
        for r in lp.get("long_natural_recordings", []):
            L.append(f"- long natural recording `{r['sample_id']}` ({r['duration_sec']} s): "
                     f"detected={r['loop_detected']}, {r['n_events']} events, max sim "
                     f"{r['max_smoothed_similarity']}, load {r['load_sec']} s, detect "
                     f"{r['detect_sec']} s")
        L.append("")
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps({k: v.get("pass") for k, v in res.items()
                      if isinstance(v, dict) and "pass" in v}, indent=1))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
