#!/usr/bin/env python
"""A5 / PLAN.md §8 — build the unseen voice references from the TEST split.

Fully reproducible from the manifests: no sample_id, video_id or channel_id is hard
coded anywhere.  Everything is a criterion applied to whatever `data/manifests/test.jsonl`
currently contains, so the whole thing regenerates after the owner finishes filtering the
parquet (`bash scripts/build_dataset.sh && python src/eval/make_references.py`).

Pipeline
--------
1. **Candidate windows.**  Every test segment is cut at *sentence* boundaries using the
   word-group timestamps in the per-segment json (`asr_ts['gigaam-v3-e2e-ctc']`, lines
   `start\\tend\\ttext`).  A window may start only after a group whose text ends in
   `.!?…` (or at the segment start) and must end on such a group.  Its transcript
   (`ref_text`) is exactly the concatenation of those punctuated e2e groups - the same
   words the audio contains, with punctuation, which is what PLAN §8 means by "exact
   ref_text".  Windows must last 8-15 s, hold >= `--min-words` words, contain no digits
   and no Latin letters (A3 froze the repo text frontend to `''`, so digits would be
   read out character-wise), carry no hesitation filler (PLAN §6.5 wants clean fragments
   for benchmark material), have no internal group gap above `--max-gap` s, and reach
   `--min-agreement` word-level agreement with the *other* timestamped recognisers
   (gigaam-v3-ctc, gigaam-v3-rnnt, t-one) over the same time range - the only way to
   check that `ref_text` is verbatim without listening.
2. **Stage-1 QC** (cheap: VAD, levels, clipping, energy-percentile SNR) on every
   candidate; the best `--top-per-speaker` per `speaker_key` survive.
3. **Stage-2 QC** (adds pyin F0, the loop detector, WADA-SNR, flatness/reverb proxies).
4. **Gates + score + gender.**  Gender is a *heuristic*: this corpus carries no gender
   label anywhere (reports/dataset_stats.md), so the label is the **speaker-level** median
   F0 (median over that speaker_key's stage-2 windows of their per-window median pyin F0)
   against 165 Hz - `< 165` male, `> 165` female - with a +-15 Hz band around the boundary
   declared `ambiguous` and barred from selection.  Speaker level, not window level: a
   single 11 s window of a near-boundary speaker flips the label.
5. **Selection.**  Per gender, one candidate per `speaker_key`, ranked by score; the best
   is `role="primary"`, the runner-up `role="backup"`; a different channel is preferred
   for the runner-up and required between the two primaries.
6. **Write** `<voice_id>.wav` (mono, source sample rate, PCM_24) and
   `<voice_id>_16k.wav` (mono 16 kHz PCM_16), plus `references.jsonl` with sha256,
   provenance, offsets, QC numbers and the leakage note.
7. **campplus sanity check** (`--campplus`): cosine of each reference against the other
   segments of its own speaker_key and against every other test speaker.

    python src/eval/make_references.py            # full run
    python src/eval/make_references.py --dry-run  # candidate counts only, no audio write
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_qc as aq  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

E2E_KEY = "gigaam-v3-e2e-ctc"
# other ASR systems with *word-level* timestamps; vosk is excluded because its `asr_ts`
# is 5-second blocks, which cannot be cut at a window boundary.
AGREE_KEYS = ("gigaam-v3-ctc", "gigaam-v3-rnnt", "t-one")
SENT_END = re.compile(r"[.!?…]['\"»)\]]*\s*$")
HAS_DIGIT = re.compile(r"\d")
HAS_LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"\w+", re.UNICODE)
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
# Russian hesitation fillers as the ASR writes them: "э-э-э", "а-а", "м-мм", "ммм", "ээ".
# PLAN §6.5 keeps disfluencies in the *training* text but wants clean fragments for
# benchmark material; a reference prompt is benchmark material.
FILLER = re.compile(r"(?<![\w-])(?:[аэмунывиАЭМУНЫВИ]{1,3}(?:-[аэмунывиАЭМУНЫВИ]{1,3})+"
                    r"|([аэмунАЭМУН])\1{1,}"
                    r"|[аэмунывиАЭМУНЫВИ]{1,3}-)(?![\w-])", re.UNICODE)


def norm_words(text):
    """Word list for ASR-agreement: lowercase, e-umlaut folded, hyphen -> space, no
    punctuation.  Same convention as configs/text_normalization.yaml."""
    t = text.lower().replace("ё", "е").replace("-", " ")
    return PUNCT.sub(" ", t).split()

MIN_DUR, MAX_DUR = 8.0, 15.0
DUR_CENTRE = 0.5 * (MIN_DUR + MAX_DUR)
EDGE_PAD = 0.20          # seconds of head/tail kept around the first/last word group
F0_GENDER_HZ = 165.0     # classic male/female median-F0 boundary
F0_GENDER_MARGIN_HZ = 15.0

# hard gates (PLAN §8: clean, single speaker, no clipping)
GATE = {
    "max_silence_ratio": 0.25,
    "max_internal_silence_sec": 0.80,
    "min_snr_db": 15.0,
    "min_f0_voiced_ratio": 0.35,
    "max_peak": 0.99,
    # enrollment material for the same speaker.  Not a comfort criterion: deliverable 4
    # needs the reference compared against *other* segments of its own speaker_key, and
    # PLAN §9.5 calibrates the drift thresholds on natural same-speaker windows, so a
    # speaker_key seen in one short fragment cannot be validated at all.
    "min_speaker_segments": 3,
    "min_speaker_total_sec": 600.0,
}
SPK_MATERIAL_FULL_SEC = 6000.0   # score saturates at ~1.7 h of the same speaker


def clamp01(x):
    return float(min(1.0, max(0.0, x)))


# --------------------------------------------------------------------- candidate windows
def parse_ts(ts):
    out = []
    for ln in ts.split("\n"):
        p = ln.split("\t")
        if len(p) != 3:
            continue
        try:
            a, b = float(p[0]), float(p[1])
        except ValueError:
            continue
        if p[2].strip():
            out.append((a, b, p[2].strip()))
    return out


def window_asr_agreement(other_lines, t0, t1, e2e_text, names=None, detail=False):
    """Median word-level agreement between the window's e2e transcript and the other
    timestamped ASR systems restricted to the same time range.

    `ref_text` has to be *verbatim* (PLAN §8), and the only verification available without
    listening is that the independent recognisers heard the same words.  Agreement is
    `difflib.SequenceMatcher(...).ratio()` over normalised word lists; a system whose
    words never fall inside the window is skipped.  Returns None if no system applies.

    The headline number is the **median over systems**, not per-system agreement: a
    median of 1.000 is compatible with one recogniser disagreeing.  `detail=True`
    therefore returns `(median, {system_name: ratio})` so the per-system values can be
    stored and reported instead of only the summary; `names` supplies the labels (it must
    be parallel to `other_lines`, and defaults to positional `system_0…`).
    """
    ref = norm_words(e2e_text)
    if not ref:
        return (None, {}) if detail else None
    names = list(names) if names is not None else [f"system_{i}" for i in
                                                  range(len(other_lines))]
    scores, per = [], {}
    for name, lines in zip(names, other_lines):
        ws = []
        for a, b, txt in lines:
            if t0 <= 0.5 * (a + b) <= t1:
                ws += norm_words(txt)
        if not ws:
            continue
        r = difflib.SequenceMatcher(None, ref, ws, autojunk=False).ratio()
        scores.append(r)
        per[name] = round(float(r), 6)
    if not scores:
        return (None, {}) if detail else None
    med = float(np.median(scores))
    return (med, per) if detail else med


def candidate_windows(row, min_dur=MIN_DUR, max_dur=MAX_DUR, min_words=15,
                      max_gap=0.8, max_per_segment=60, min_agreement=0.95,
                      allow_fillers=False):
    """All sentence-aligned windows of `row` that pass the text/timing filters."""
    try:
        with open(row["json_path"], encoding="utf-8") as f:
            js = json.load(f)
    except OSError:
        return []
    lines = parse_ts(js.get("asr_ts", {}).get(E2E_KEY, "") or "")
    if len(lines) < 2:
        return []
    other = [(k, parse_ts(js.get("asr_ts", {}).get(k, "") or "")) for k in AGREE_KEYS]
    other = [(k, l) for k, l in other if l]
    other_names = [k for k, _ in other]
    other_lines = [l for _, l in other]
    seg_dur = float(row["duration_sec"])
    n = len(lines)
    starts = [i for i in range(n) if i == 0 or SENT_END.search(lines[i - 1][2])]
    out = []
    for i in starts:
        for j in range(i, n):
            dur_core = lines[j][1] - lines[i][0]
            if dur_core > max_dur:
                break
            if not SENT_END.search(lines[j][2]):
                continue
            if dur_core < min_dur - 2 * EDGE_PAD:
                continue
            gaps = [lines[k + 1][0] - lines[k][1] for k in range(i, j)]
            if gaps and max(gaps) > max_gap:
                break
            text = " ".join(lines[k][2] for k in range(i, j + 1))
            text = re.sub(r"\s+", " ", text).strip()
            if HAS_DIGIT.search(text) or HAS_LATIN.search(text):
                continue
            if not allow_fillers and FILLER.search(text):
                continue
            words = WORD.findall(text)
            if len(words) < min_words:
                continue
            prev_end = lines[i - 1][1] if i > 0 else 0.0
            next_start = lines[j + 1][0] if j + 1 < n else seg_dur
            t0 = max(0.0, lines[i][0] - min(EDGE_PAD, max(lines[i][0] - prev_end, 0.0) / 2
                                            if i > 0 else EDGE_PAD))
            t1 = min(seg_dur, lines[j][1] + min(EDGE_PAD,
                                                max(next_start - lines[j][1], 0.0) / 2
                                                if j + 1 < n else EDGE_PAD))
            dur = t1 - t0
            if not (min_dur <= dur <= max_dur):
                continue
            agree, agree_per = window_asr_agreement(other_lines, t0, t1, text,
                                                    names=other_names, detail=True)
            if agree is None or agree < min_agreement:
                continue
            out.append({
                "window_asr_agreement": round(agree, 6),
                "window_asr_agreement_per_system": agree_per,
                "window_asr_agreement_min": round(min(agree_per.values()), 6),
                "n_agreement_systems": len(agree_per),
                "sample_id": row["sample_id"],
                "speaker_key": row["speaker_key"],
                "channel_id": row["channel_id"],
                "channel_title": row["channel_title"],
                "video_id": row["video_id"],
                "video_title": row.get("video_title"),
                "audio_path": row["audio_path"],
                "json_path": row["json_path"],
                "asr_consistency": row["asr_consistency"],
                "is_single_speaker": bool(row.get("is_single_speaker", True)),
                "speaker_n_segments": row.get("speaker_n_segments"),
                "speaker_total_sec": row.get("speaker_total_sec"),
                "segment_duration_sec": seg_dur,
                "segment_start_in_video_sec": row["start"],
                "line_from": i, "line_to": j,
                "offset_start_sec": round(t0, 3),
                "offset_end_sec": round(t1, 3),
                "duration_sec": round(dur, 3),
                "max_internal_group_gap_sec": round(max(gaps), 3) if gaps else 0.0,
                "ref_text": text,
                "n_words": len(words),
            })
            if len(out) >= max_per_segment:
                return out
    return out


# ------------------------------------------------------------------------------ QC stages
def _load_window(c):
    y, sr = aq.load_audio(c["audio_path"], sr=None,
                          start_sec=c["offset_start_sec"], end_sec=c["offset_end_sec"])
    return y, sr


def stage1(c):
    """Cheap QC: no pyin, no loop detection."""
    try:
        y, sr = _load_window(c)
        rep = aq.audio_qc(y, sr, with_f0=False, with_loop=False, with_wada=False,
                          with_timbre=False)
    except Exception as e:  # noqa: BLE001
        return {**c, "error": f"{type(e).__name__}: {e}"}
    s = (0.45 * clamp01(((rep.get("snr_db") or -99) - 12.0) / 23.0)
         + 0.25 * clamp01((0.25 - rep["silence_ratio"]) / 0.25)
         + 0.15 * clamp01((0.80 - rep["longest_internal_silence_sec"]) / 0.80)
         + 0.15 * clamp01(1.0 - abs(rep["duration_sec"] - DUR_CENTRE) / (DUR_CENTRE - MIN_DUR)))
    return {**c, "qc1": rep, "score1": round(s, 6)}


def stage2(c):
    """Full QC - `audio_qc.audio_qc_pipeline`, the same code path as

        python src/eval/audio_qc.py FILE.wav --start ... --end ...

    so the documented command reproduces the numbers stored here (and tabulated in
    reports/reference_qc.md) exactly.  Duration / VAD / level / clipping / flatness /
    reverb are measured at the file's **native** rate (a property of the artifact we
    ship); F0, the loop detector, the WADA estimate and the primary SNR run on the 16 kHz
    mono version, i.e. on what the model and the evaluator actually consume.
    `path_label=None`: references.jsonl already carries `source_audio_path`, no need to
    repeat an absolute /mnt path inside every stored `qc` block.
    """
    try:
        rep = aq.audio_qc_pipeline(c["audio_path"], start_sec=c["offset_start_sec"],
                                   end_sec=c["offset_end_sec"], path_label=None)
    except Exception as e:  # noqa: BLE001
        return {**c, "error": f"{type(e).__name__}: {e}"}
    return {**c, "qc": rep}


def gate_and_score(c):
    """Apply the hard gates and compute the final score + gender. Mutates a copy."""
    q = c.get("qc")
    if q is None or "error" in c:
        return {**c, "gates_failed": ["qc_error"], "score": 0.0,
                "gender": "unknown", "gender_confidence": 0.0, "passed": False}
    failed = []
    if q["invalid"]:
        failed.append("invalid_samples")
    if not c.get("is_single_speaker", True):
        failed.append("not_single_speaker")
    if c.get("speaker_n_segments", 0) < GATE["min_speaker_segments"]:
        failed.append("speaker_segments")
    if c.get("speaker_total_sec", 0.0) < GATE["min_speaker_total_sec"]:
        failed.append("speaker_material")
    if q["clipped"] or q["peak"] > GATE["max_peak"]:
        failed.append("clipping")
    if q["silence_ratio"] > GATE["max_silence_ratio"]:
        failed.append("silence_ratio")
    if q["longest_internal_silence_sec"] > GATE["max_internal_silence_sec"]:
        failed.append("internal_silence")
    snr = q.get("snr_db")
    if snr is None or snr < GATE["min_snr_db"]:
        failed.append("snr")
    if q.get("f0_median_hz") is None or q.get("f0_voiced_ratio", 0.0) < GATE["min_f0_voiced_ratio"]:
        failed.append("f0_voiced_ratio")
    if q["loop"]["loop_detected"]:
        failed.append("loop")
    if not (MIN_DUR <= q["duration_sec"] <= MAX_DUR):
        failed.append("duration")

    # Gender is decided at *speaker* level: a single 11 s window of a near-boundary
    # speaker flips the label (observed here), so the label uses the median of the F0
    # medians over all of that speaker_key's stage-2 windows.
    f0 = c.get("speaker_f0_median_hz") or q.get("f0_median_hz")
    if f0 is None:
        gender, conf = "unknown", 0.0
    elif f0 < F0_GENDER_HZ - F0_GENDER_MARGIN_HZ:
        gender, conf = "male", (F0_GENDER_HZ - f0) / F0_GENDER_HZ
    elif f0 > F0_GENDER_HZ + F0_GENDER_MARGIN_HZ:
        gender, conf = "female", (f0 - F0_GENDER_HZ) / F0_GENDER_HZ
    else:
        gender, conf = "ambiguous", 0.0

    rv = q.get("reverb_proxy_sec")
    rv_term = 1.0 if rv is None else clamp01((0.30 - rv) / 0.25)
    mat = c.get("speaker_total_sec", 0.0)
    mat_term = clamp01(math.log10(max(mat, 1.0) / GATE["min_speaker_total_sec"])
                       / math.log10(SPK_MATERIAL_FULL_SEC / GATE["min_speaker_total_sec"]))
    agree = c.get("window_asr_agreement") or 0.0
    score = (0.24 * clamp01(((snr or -99) - 15.0) / 25.0)
             + 0.14 * clamp01((0.25 - q["silence_ratio"]) / 0.25)
             + 0.09 * clamp01((0.80 - q["longest_internal_silence_sec"]) / 0.80)
             + 0.09 * rv_term
             + 0.07 * clamp01(q.get("f0_voiced_ratio", 0.0) / 0.60)
             + 0.09 * clamp01((c["asr_consistency"] - 90.0) / 10.0)
             + 0.09 * clamp01((agree - 0.95) / 0.05)
             + 0.07 * clamp01(1.0 - abs(q["duration_sec"] - DUR_CENTRE) / (DUR_CENTRE - MIN_DUR))
             + 0.12 * mat_term)
    return {**c, "gates_failed": failed, "passed": not failed,
            "gender": gender, "gender_confidence": round(conf, 4),
            "gender_f0_hz": round(float(f0), 3) if f0 is not None else None,
            "score": round(score, 6)}


# ------------------------------------------------------------------------------- writing
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_reference_wavs(c, voice_id, out_dir):
    """Mono master at the source sample rate (PCM_24) + a 16 kHz PCM_16 copy.

    The mix-down is `mean over channels`, exactly what CosyVoice's `load_wav` does, and
    the 16 kHz copy uses `torchaudio.transforms.Resample`, the same resampler the model
    applies to a prompt wav.
    """
    import soundfile as sf
    import torch
    import torchaudio

    info = sf.info(c["audio_path"])
    sr = info.samplerate
    a = int(round(c["offset_start_sec"] * sr))
    b = int(round(c["offset_end_sec"] * sr))
    data, file_sr = sf.read(c["audio_path"], start=a, frames=b - a, dtype="float64",
                            always_2d=True)
    ch_corr = None
    if data.shape[1] > 1:
        with np.errstate(invalid="ignore"):
            ch_corr = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
    y = data.mean(axis=1)
    master = os.path.join(out_dir, f"{voice_id}.wav")
    sf.write(master, y, file_sr, subtype="PCM_24")
    w = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    w16 = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=16000)(w)
    copy16 = os.path.join(out_dir, f"{voice_id}_16k.wav")
    sf.write(copy16, w16.squeeze(0).numpy(), 16000, subtype="PCM_16")
    return {
        "wav_path": os.path.relpath(master, ROOT),
        "wav_sample_rate": int(file_sr),
        "wav_subtype": "PCM_24",
        "wav_channels": 1,
        "wav_sha256": sha256_file(master),
        "wav_bytes": os.path.getsize(master),
        "wav16k_path": os.path.relpath(copy16, ROOT),
        "wav16k_sha256": sha256_file(copy16),
        "wav16k_subtype": "PCM_16",
        "source_channels": int(info.channels),
        "source_subtype": info.subtype,
        "source_channel_correlation": (round(ch_corr, 6) if ch_corr is not None else None),
    }


# --------------------------------------------------------------------- campplus checking
# ------------------------------------------------------------------- build provenance
A5_SOURCES = ("src/eval/make_references.py", "src/eval/audio_qc.py",
              "src/eval/spk_campplus.py")


def _pkg_versions():
    import importlib

    out = {"python": sys.version.split()[0], "executable": sys.executable}
    for m in ("numpy", "soundfile", "soxr", "librosa", "scipy", "torch", "torchaudio",
              "onnxruntime"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def build_provenance(with_campplus=True):
    """What code and what environment produced these artifacts (PLAN §0 rule 6).

    `code_sha256` is a hash of the A5 sources themselves, not of a commit: the working
    tree is normally dirty while a wave is in progress, so a bare `git rev-parse HEAD`
    would tie an artifact to a commit it was not built from.  The git revision is recorded
    beside it, together with whether those files were modified relative to it.
    """
    import subprocess

    h = hashlib.sha256()
    per_file = {}
    for rel in A5_SOURCES:
        b = open(os.path.join(ROOT, rel), "rb").read()
        fh = hashlib.sha256(b).hexdigest()
        per_file[rel] = fh
        h.update(rel.encode() + b"\0" + fh.encode() + b"\0")

    def git(*args):
        try:
            return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                                  text=True, timeout=20).stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    dirty = git("status", "--porcelain", "--", *A5_SOURCES)
    prov = {
        "code_sha256": h.hexdigest(),
        "code_files_sha256": per_file,
        "git_revision": git("rev-parse", "HEAD"),
        "git_a5_sources_dirty": bool(dirty),
        "env": _pkg_versions(),
    }
    if with_campplus:
        try:
            import spk_campplus as sc

            prov["campplus_onnx"] = sc.CAMPPLUS_ONNX
            prov["campplus_onnx_sha256"] = sc.model_sha256()
        except Exception as e:  # noqa: BLE001
            prov["campplus_onnx_sha256"] = f"unavailable: {type(e).__name__}: {e}"
    return prov


def emb_cache_key(sample_id, emb_window_sec, center, model_sha):
    """Cache key for one campplus embedding.

    An embedding is a function of (audio, window, model), so the key has to carry all
    three: the same `sample_id` embedded from a 30 s centred window and from a 10 s
    leading window are different vectors, and so is the same window through a different
    campplus.onnx.  Keying on `sample_id` alone silently returns the wrong vector after
    any of those change.
    """
    return (f"{sample_id}|win={float(emb_window_sec):g}|center={bool(center)}"
            f"|model={model_sha[:16]}")


def campplus_check(chosen, rows, out_dir, emb_per_speaker=3, emb_window_sec=30.0,
                   emb_center=True):
    """Cosine of each reference wav against its own speaker vs every other test speaker.

    Deliberately **single process**: onnxruntime and torch have already initialised their
    OpenMP pools in this interpreter by the time this runs, and a fork-based
    ProcessPoolExecutor on top of that deadlocks in `futex_wait_queue` (observed here -
    three processes stuck at 0 % CPU).  The work is ~100 x 30 s of FLAC decode, so serial
    is fine, and the result is cached in `test_spk_emb.npz`.

    Per `speaker_key` the `emb_per_speaker` longest segments are embedded from a
    `emb_window_sec` window, centred when `emb_center` - the same sampling A1's leakage
    check uses.  The cache is keyed by `emb_cache_key`, i.e. by
    (sample_id, emb_window_sec, center, campplus.onnx sha256), so changing any of them
    recomputes instead of reusing a vector that no longer means the same thing.  A cache
    written by an older version (no `keys` array) is ignored, not reinterpreted.
    """
    import spk_campplus as sc

    model_sha = sc.model_sha256()
    cache = os.path.join(out_dir, "test_spk_emb.npz")
    picked = []
    by_key = defaultdict(list)
    for r in rows:
        by_key[r["speaker_key"]].append(r)
    for k in sorted(by_key):
        rs = sorted(by_key[k], key=lambda r: (-r["duration_sec"], r["sample_id"]))
        picked += rs[:emb_per_speaker]
    tasks = [(emb_cache_key(r["sample_id"], emb_window_sec, emb_center, model_sha),
              r["sample_id"], r["audio_path"]) for r in picked]

    cached, errs = {}, {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if "keys" in z.files:
            cached = {str(k): v for k, v in zip(z["keys"], z["emb"])}
        else:
            print(f"[campplus] {cache} has no `keys` array (written before the cache was "
                  "keyed by window/model): ignoring it and re-embedding", file=sys.stderr)
    todo = [t for t in tasks if t[0] not in cached]
    if todo:
        t0 = time.time()
        for i, (key, sid, path) in enumerate(todo, 1):
            try:
                cached[key] = sc.embed_file(path, max_sec=emb_window_sec,
                                            center=emb_center)
            except Exception as e:  # noqa: BLE001
                errs[sid] = f"{type(e).__name__}: {e}"
            if i % 20 == 0 or i == len(todo):
                print(f"[campplus] {i}/{len(todo)} segments "
                      f"({time.time() - t0:.0f}s)", file=sys.stderr)
        ks = sorted(cached)
        np.savez_compressed(cache, keys=np.array(ks),
                            emb=np.stack([cached[k] for k in ks]),
                            meta=np.array(json.dumps({
                                "emb_window_sec": emb_window_sec,
                                "center": bool(emb_center),
                                "model": sc.CAMPPLUS_ONNX,
                                "model_sha256": model_sha,
                                "key_format": "<sample_id>|win=<sec>|center=<bool>|"
                                              "model=<sha256[:16]>",
                            }, ensure_ascii=False)))
        print(f"[campplus] embedded {len(todo)} test segments in {time.time() - t0:.1f}s "
              f"({len(errs)} errors); cache now holds {len(cached)} keys "
              f"(model sha {model_sha[:16]}, window {emb_window_sec}s, "
              f"center={emb_center})", file=sys.stderr)
    else:
        print(f"[campplus] all {len(tasks)} embeddings served from {cache} "
              f"(model sha {model_sha[:16]}, window {emb_window_sec}s, "
              f"center={emb_center})", file=sys.stderr)
    embs = {sid: cached[key] for key, sid, _ in tasks if key in cached}

    spk_of = {r["sample_id"]: r["speaker_key"] for r in rows}
    ch_of = {r["sample_id"]: r["channel_id"] for r in rows}
    by_spk = defaultdict(list)
    for sid, e in embs.items():
        by_spk[spk_of[sid]].append((sid, e))

    out = {}
    for c in chosen:
        ref_emb = sc.embed_file(os.path.join(ROOT, c["wav_path"]))
        own_key = c["speaker_key"]
        own = [(sid, e) for sid, e in by_spk.get(own_key, []) if sid != c["sample_id"]]
        own_incl_src = by_spk.get(own_key, [])
        own_used, own_note = own, "other segments of the same speaker_key"
        if not own:
            own_used = own_incl_src
            own_note = ("speaker_key has a single segment: compared against the parent "
                        "segment itself (30 s centred window, overlaps the reference)")
        own_cos = sorted(((sc.cosine(ref_emb, e), sid) for sid, e in own_used),
                         reverse=True)
        # "other" speakers = every OTHER speaker_key of the test split, EXCEPT the speaker_keys
        # of the reference's own video: `local_speaker_id` is a per-video diarisation index,
        # and a second index in the same video is usually the same person split by the
        # diariser (v3.1: LookAudioBook 7avaKyyvslc/0 vs /1 at cosine 0.93), not a different
        # speaker. Those keys are reported separately in `same_video_keys` instead of
        # silently failing the own-vs-other verdict.
        own_video = own_key.split("/")[1] if own_key.count("/") == 2 else None
        other, same_video = [], []
        for sid, e in embs.items():
            if spk_of[sid] == own_key:
                continue
            rec = (sc.cosine(ref_emb, e), sid, spk_of[sid], ch_of[sid])
            if own_video is not None and spk_of[sid].split("/")[1] == own_video:
                same_video.append(rec)
            else:
                other.append(rec)
        other.sort(reverse=True)
        same_video.sort(reverse=True)
        other_cross_ch = [o for o in other if o[3] != c["channel_id"]]
        own_vals = [v for v, _ in own_cos]
        other_vals = [o[0] for o in other]
        out[c["voice_id"]] = {
            "model": sc.CAMPPLUS_ONNX,
            "model_sha256": model_sha,
            "emb_window_sec": emb_window_sec,
            "emb_center": bool(emb_center),
            "n_own_segments": len(own_cos),
            "own_note": own_note,
            "own_cos_min": round(min(own_vals), 6) if own_vals else None,
            "own_cos_mean": round(float(np.mean(own_vals)), 6) if own_vals else None,
            "own_cos_max": round(max(own_vals), 6) if own_vals else None,
            "own_cos_detail": [{"sample_id": s, "cos": round(v, 6)} for v, s in own_cos],
            "n_other_test_speakers": len({o[2] for o in other}),
            "n_other_test_segments": len(other),
            "other_cos_max": round(max(other_vals), 6) if other_vals else None,
            "other_cos_p95": round(float(np.percentile(other_vals, 95)), 6) if other_vals else None,
            "other_cos_median": round(float(np.median(other_vals)), 6) if other_vals else None,
            "other_cos_max_cross_channel": (round(other_cross_ch[0][0], 6)
                                            if other_cross_ch else None),
            "nearest_other": ([{"speaker_key": o[2], "sample_id": o[1],
                                "cos": round(o[0], 6),
                                "same_channel": o[3] == c["channel_id"]}
                               for o in other[:3]]),
            "same_video_keys": ([{"speaker_key": o[2], "sample_id": o[1],
                                  "cos": round(o[0], 6)} for o in same_video[:3]]),
            "n_same_video_segments_excluded_from_other": len(same_video),
            "verdict_mean_own_gt_max_other": (
                bool(own_vals and other_vals and np.mean(own_vals) > max(other_vals))),
            "verdict_min_own_gt_max_other": (
                bool(own_vals and other_vals and min(own_vals) > max(other_vals))),
            "margin_mean_own_minus_max_other": (
                round(float(np.mean(own_vals)) - max(other_vals), 6)
                if own_vals and other_vals else None),
        }
        if errs:
            out[c["voice_id"]]["embedding_errors"] = errs
    return out


# ---------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build unseen voice references (PLAN §8)")
    ap.add_argument("--manifest", default="data/manifests/test.jsonl")
    ap.add_argument("--split-channels", default="data/manifests/split_channels.json")
    ap.add_argument("--leakage", default="reports/leakage_raw.json")
    ap.add_argument("--out-dir", default="data/references")
    ap.add_argument("--raw-out", default="reports/reference_candidates.json",
                    help="full stage-2 candidate table (every QC number + gate verdict). "
                         "Under reports/, not tmp/, because reports/reference_qc.md "
                         "cites it as evidence and tmp/ is gitignored.")
    ap.add_argument("--min-dur", type=float, default=MIN_DUR)
    ap.add_argument("--max-dur", type=float, default=MAX_DUR)
    ap.add_argument("--min-words", type=int, default=15)
    ap.add_argument("--max-gap", type=float, default=0.8)
    ap.add_argument("--min-agreement", type=float, default=0.95,
                    help="min word-level agreement between the e2e transcript of the "
                         "window and the other timestamped ASR systems")
    ap.add_argument("--allow-fillers", action="store_true",
                    help="keep windows containing hesitation fillers (э-э, а-а, ммм)")
    ap.add_argument("--top-per-speaker", type=int, default=5,
                    help="stage-2 windows per speaker_key; also the sample the "
                         "speaker-level F0 (and therefore the gender label) is taken over")
    ap.add_argument("--per-gender", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--allow-watchlist", action="store_true",
                    help="do not exclude speaker_keys on the voice-leakage watch list")
    ap.add_argument("--allow-missing-leakage", action="store_true",
                    help="build references WITHOUT A1's voice-leakage gate. The result "
                         "is NOT PLAN-conformant; every record is stamped "
                         "leakage.gate_applied=false. Default is to abort.")
    ap.add_argument("--emb-per-speaker", type=int, default=3,
                    help="segments per speaker_key embedded for the campplus check")
    ap.add_argument("--emb-window-sec", type=float, default=30.0,
                    help="length of the window embedded per segment (part of the "
                         "embedding cache key)")
    ap.add_argument("--emb-leading", action="store_true",
                    help="take the leading window instead of a centred one (part of the "
                         "embedding cache key)")
    ap.add_argument("--no-campplus", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="stop after stage 1 and print counts (no audio is written)")
    a = ap.parse_args(argv)

    os.chdir(ROOT)
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    split = json.load(open(a.split_channels, encoding="utf-8"))
    train_channels = {k for k, v in split["channels"].items() if v["split"] == "train"}
    test_channels = {k for k, v in split["channels"].items() if v["split"] == "test"}

    # voice-leakage numbers from A1 (reports/leakage_raw.json).  PLAN §6.2 makes this a
    # gate, not a nice-to-have, so a missing or malformed file ABORTS the build; the only
    # way past it is the explicit --allow-missing-leakage escape hatch, which stamps every
    # emitted record with leakage.gate_applied=false.
    watch, flagged = set(), set()
    tau = watch_level = None
    max_cos_to_train = {}
    leakage_gate_applied = False
    if not os.path.exists(a.leakage):
        if not a.allow_missing_leakage:
            raise SystemExit(
                f"FATAL: A1 voice-leakage report not found: {a.leakage}\n"
                "PLAN §6.2 requires the train/test voice-leakage check before references "
                "are drawn from the test channels; building without it would silently ship "
                "references whose speaker may also be in train.\n"
                "Fix: run A1's src/data/leakage_check.py to produce it, point --leakage at "
                "it, or pass --allow-missing-leakage to build a NON-conformant set on "
                "purpose.")
        print(f"[refs] WARNING: {a.leakage} missing and --allow-missing-leakage given: "
              "the voice-leakage gate is NOT applied", file=sys.stderr)
    else:
        lk = json.load(open(a.leakage, encoding="utf-8"))
        cal = lk.get("calibration") or {}
        missing = [k for k in ("tau", "watch_level_p90_null") if cal.get(k) is None]
        for k in ("voice", "flagged", "watch"):
            if lk.get(k) is None:
                missing.append(k)
        if missing and not a.allow_missing_leakage:
            raise SystemExit(
                f"FATAL: {a.leakage} is missing required key(s) {missing}; it is not an "
                "A1 leakage report this gate can use. Regenerate it with "
                "src/data/leakage_check.py or pass --allow-missing-leakage.")
        tau = cal.get("tau")
        watch_level = cal.get("watch_level_p90_null")
        watch = {e["speaker_key"] for e in lk.get("watch") or []}
        # `flagged` == above tau, i.e. a speaker A1 believes may also be in train. Unlike
        # the watch list this has NO override: such a speaker can never be a reference.
        flagged = {e["speaker_key"] for e in lk.get("flagged") or []}
        for e in lk.get("voice") or []:
            max_cos_to_train[e["speaker_key"]] = {
                "max_cos": round(e["max_cos"], 6),
                "nearest_train_speaker_key": (e["top"][0][1] if e.get("top") else None),
            }
        leakage_gate_applied = not missing

    print(f"[refs] {len(rows)} test rows, {len(test_channels)} test channels; "
          f"leakage gate applied: {leakage_gate_applied} "
          f"(tau={tau}, watch_level={watch_level}); "
          f"watch-list speaker_keys corpus-wide {len(watch)}, of them in test "
          f"{len([w for w in watch if w.split('/')[0] in test_channels])}; "
          f"flagged (hard-excluded) corpus-wide {len(flagged)}, of them in test "
          f"{len([w for w in flagged if w.split('/')[0] in test_channels])}",
          file=sys.stderr)

    # per-speaker enrollment material, from the manifest (no hand-picked ids)
    spk_seg, spk_sec = defaultdict(int), defaultdict(float)
    for r in rows:
        spk_seg[r["speaker_key"]] += 1
        spk_sec[r["speaker_key"]] += float(r["duration_sec"])
    for r in rows:
        r["speaker_n_segments"] = spk_seg[r["speaker_key"]]
        r["speaker_total_sec"] = round(spk_sec[r["speaker_key"]], 3)

    cands = []
    for r in rows:
        cands += candidate_windows(r, a.min_dur, a.max_dur, a.min_words, a.max_gap,
                                   min_agreement=a.min_agreement,
                                   allow_fillers=a.allow_fillers)
    print(f"[refs] {len(cands)} sentence-aligned candidate windows from "
          f"{len({c['sample_id'] for c in cands})} segments / "
          f"{len({c['speaker_key'] for c in cands})} speaker_keys", file=sys.stderr)
    if not cands:
        raise SystemExit("no candidate windows - check asr_ts availability")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        s1 = list(ex.map(stage1, cands, chunksize=8))
    print(f"[refs] stage-1 QC on {len(s1)} windows in {time.time() - t0:.1f}s",
          file=sys.stderr)
    s1 = [c for c in s1 if "error" not in c]

    keep = []
    by_spk = defaultdict(list)
    for c in s1:
        by_spk[c["speaker_key"]].append(c)
    for k in sorted(by_spk):
        ranked = sorted(by_spk[k], key=lambda c: (-c["score1"], c["sample_id"],
                                                  c["offset_start_sec"]))
        keep += ranked[:a.top_per_speaker]
    print(f"[refs] stage-2 on {len(keep)} windows "
          f"({a.top_per_speaker} per speaker_key x {len(by_spk)} speakers)",
          file=sys.stderr)

    if a.dry_run:
        print(json.dumps({"n_candidates": len(cands), "n_stage1_ok": len(s1),
                          "n_speaker_keys": len(by_spk), "n_stage2": len(keep)}, indent=2))
        return 0

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        s2 = list(ex.map(stage2, keep, chunksize=2))
    print(f"[refs] stage-2 QC in {time.time() - t0:.1f}s", file=sys.stderr)

    # speaker-level F0 (median over that speaker's stage-2 windows) -> gender label
    f0_by_spk = defaultdict(list)
    for c in s2:
        v = (c.get("qc") or {}).get("f0_median_hz")
        if v:
            f0_by_spk[c["speaker_key"]].append(float(v))
    for c in s2:
        vs = f0_by_spk.get(c["speaker_key"]) or []
        c["speaker_f0_median_hz"] = round(float(np.median(vs)), 3) if vs else None
        c["speaker_f0_n_windows"] = len(vs)
        c["speaker_f0_spread_hz"] = (round(float(max(vs) - min(vs)), 3)
                                     if len(vs) > 1 else 0.0)

    scored = [gate_and_score(c) for c in s2]
    for c in scored:
        c["on_voice_watchlist"] = c["speaker_key"] in watch
        c["voice_leakage_flagged"] = c["speaker_key"] in flagged
        c["channel_in_train"] = c["channel_id"] in train_channels
    ok = [c for c in scored
          if c.get("passed") and not c["channel_in_train"]
          and not c["voice_leakage_flagged"]
          and (a.allow_watchlist or not c["on_voice_watchlist"])
          and c["gender"] in ("male", "female")]
    hist = defaultdict(int)
    for c in scored:
        for g in c["gates_failed"]:
            hist[g] += 1
        if c.get("passed") and c["gender"] not in ("male", "female"):
            hist[f"gender_{c['gender']}"] += 1
        if c.get("passed") and c["voice_leakage_flagged"]:
            hist["voice_leakage_flagged"] += 1
        if c.get("passed") and c["on_voice_watchlist"] and not a.allow_watchlist:
            hist["voice_watchlist"] += 1
    # how much the leakage gate actually did on THIS run (0 is a legitimate answer and
    # must be reported as 0, not implied to be an active exclusion)
    leakage_effect = {
        "gate_applied": leakage_gate_applied,
        "n_stage2_candidates": len(scored),
        "n_candidates_on_watchlist": sum(1 for c in scored if c["on_voice_watchlist"]),
        "n_candidates_flagged": sum(1 for c in scored if c["voice_leakage_flagged"]),
        "n_excluded_by_watchlist": sum(1 for c in scored if c.get("passed")
                                       and c["on_voice_watchlist"]
                                       and not a.allow_watchlist),
        "n_excluded_by_flagged": sum(1 for c in scored if c.get("passed")
                                     and c["voice_leakage_flagged"]),
        "n_watchlist_speaker_keys_corpus_wide": len(watch),
        "n_watchlist_speaker_keys_in_test": len(
            [w for w in watch if w.split("/")[0] in test_channels]),
        "n_watchlist_speaker_keys_among_stage2_candidates": len(
            {c["speaker_key"] for c in scored if c["on_voice_watchlist"]}),
        "n_flagged_speaker_keys_corpus_wide": len(flagged),
        "watchlist_override_used": bool(a.allow_watchlist),
    }
    print(f"[refs] leakage gate effect: {leakage_effect}", file=sys.stderr)
    print(f"[refs] {len(ok)}/{len(scored)} candidates pass all gates; "
          f"rejections: {dict(sorted(hist.items()))}", file=sys.stderr)
    print(f"[refs] gender split of the surviving candidates: "
          f"{ {g: sum(1 for c in ok if c['gender'] == g) for g in ('male', 'female')} }",
          file=sys.stderr)

    chosen = []
    for gender in ("female", "male"):
        pool = [c for c in ok if c["gender"] == gender]
        best_per_spk = {}
        for c in pool:
            k = c["speaker_key"]
            if k not in best_per_spk or (c["score"], c["sample_id"]) > \
                    (best_per_spk[k]["score"], best_per_spk[k]["sample_id"]):
                best_per_spk[k] = c
        ranked = sorted(best_per_spk.values(),
                        key=lambda c: (-c["score"], c["speaker_key"]))
        picked = []
        for c in ranked:
            if len(picked) >= a.per_gender:
                break
            # the runner-up should come from another channel when one is available
            if picked and c["channel_id"] == picked[0]["channel_id"] and \
                    any(x["channel_id"] != picked[0]["channel_id"] for x in ranked):
                continue
            picked.append(c)
        for rank, c in enumerate(picked, start=1):
            c = dict(c)
            c["voice_id"] = f"ref_{gender}_{rank:02d}"
            c["role"] = "primary" if rank == 1 else "backup"
            c["rank_in_gender"] = rank
            c["n_candidates_in_gender"] = len(ranked)
            chosen.append(c)

    if not chosen:
        raise SystemExit("no candidate passed the gates")
    prim = [c for c in chosen if c["role"] == "primary"]
    if len(prim) == 2 and prim[0]["channel_id"] == prim[1]["channel_id"]:
        print("[refs] WARNING: both primaries come from the same channel", file=sys.stderr)

    os.makedirs(a.out_dir, exist_ok=True)
    for c in chosen:
        c.update(write_reference_wavs(c, c["voice_id"], a.out_dir))

    cp = {}
    if not a.no_campplus:
        cp = campplus_check(chosen, rows, a.out_dir,
                            emb_per_speaker=a.emb_per_speaker,
                            emb_window_sec=a.emb_window_sec,
                            emb_center=not a.emb_leading)

    prov = build_provenance(with_campplus=not a.no_campplus)
    out_path = os.path.join(a.out_dir, "references.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in sorted(chosen, key=lambda c: c["voice_id"]):
            q = c["qc"]
            rec = {
                "voice_id": c["voice_id"],
                "role": c["role"],
                "gender_heuristic": c["gender"],
                "gender_method": (f"median over the speaker's stage-2 windows of the "
                                  f"per-window median pyin F0, vs {F0_GENDER_HZ} Hz "
                                  f"(+-{F0_GENDER_MARGIN_HZ} Hz ambiguity band); "
                                  "the corpus carries no gender label"),
                "gender_confidence": c["gender_confidence"],
                "speaker_f0_median_hz": c["speaker_f0_median_hz"],
                "speaker_f0_n_windows": c["speaker_f0_n_windows"],
                "speaker_f0_spread_hz": c["speaker_f0_spread_hz"],
                "split": "test",
                "channel_id": c["channel_id"],
                "channel_title": c["channel_title"],
                "video_id": c["video_id"],
                "video_title": c["video_title"],
                "speaker_key": c["speaker_key"],
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
                "ref_text_source": f"asr_ts['{E2E_KEY}'] word groups inside the window "
                                   "(punctuated, verbatim)",
                "n_words": c["n_words"],
                "asr_consistency": c["asr_consistency"],
                "window_asr_agreement": c["window_asr_agreement"],
                "window_asr_agreement_note": ("MEDIAN over the systems below, not "
                                              "per-system agreement; see "
                                              "window_asr_agreement_per_system"),
                "window_asr_agreement_per_system": c["window_asr_agreement_per_system"],
                "window_asr_agreement_min": c["window_asr_agreement_min"],
                "window_asr_agreement_systems": list(AGREE_KEYS),
                "wav_path": c["wav_path"],
                "wav_sample_rate": c["wav_sample_rate"],
                "wav_subtype": c["wav_subtype"],
                "wav_channels": c["wav_channels"],
                "wav_sha256": c["wav_sha256"],
                "wav_bytes": c["wav_bytes"],
                "wav16k_path": c["wav16k_path"],
                "wav16k_sha256": c["wav16k_sha256"],
                "source_channels": c["source_channels"],
                "source_subtype": c["source_subtype"],
                "source_channel_correlation": c["source_channel_correlation"],
                "qc": q,
                "selection_score": c["score"],
                "rank_in_gender": c["rank_in_gender"],
                "n_candidate_speakers_in_gender": c["n_candidates_in_gender"],
                "leakage": {
                    "channel_in_train": c["channel_in_train"],
                    "channel_in_test": c["channel_id"] in test_channels,
                    "verified_against": a.split_channels,
                    "n_train_channels": len(train_channels),
                    "gate_applied": leakage_gate_applied,
                    "leakage_report": a.leakage,
                    "on_voice_watchlist": c["on_voice_watchlist"],
                    "voice_leakage_flagged": c["voice_leakage_flagged"],
                    "voice_tau": tau,
                    "voice_watch_level": watch_level,
                    "speaker_max_cos_to_train": max_cos_to_train.get(c["speaker_key"]),
                    "note": ("channel absent from train (checked against "
                             "split_channels.json); the whole channel is a test channel, "
                             "so no train segment shares this channel_id, video_id or "
                             "speaker_key"),
                },
                "campplus": cp.get(c["voice_id"]),
                "build": prov,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(a.out_dir, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "argv": sys.argv[1:] if argv is None else list(argv),
                   "params": vars(a), "gates": GATE,
                   "leakage_gate": leakage_effect, **prov},
                  f, ensure_ascii=False, indent=1)

    os.makedirs(os.path.dirname(a.raw_out), exist_ok=True)
    with open(a.raw_out, "w", encoding="utf-8") as f:
        json.dump({
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": a.manifest,
            "params": vars(a),
            "gates": GATE,
            "n_candidate_windows": len(cands),
            "n_stage1_ok": len(s1),
            "n_stage2": len(keep),
            "n_passed_gates": len(ok),
            "leakage_gate": leakage_effect,
            "build": prov,
            "watchlist": sorted(watch),
            "flagged": sorted(flagged),
            "candidates": scored,
        }, f, ensure_ascii=False, indent=1)

    print(json.dumps([{"voice_id": c["voice_id"], "role": c["role"],
                       "gender": c["gender"], "f0": c["qc"]["f0_median_hz"],
                       "snr": c["qc"]["snr_db"], "dur": c["qc"]["duration_sec"],
                       "score": c["score"], "channel": c["channel_title"]}
                      for c in sorted(chosen, key=lambda c: c["voice_id"])],
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
