#!/usr/bin/env python
"""A18-simtime — speaker similarity over TIME on the A12 window grid (5 s / hop 2.5 s).

For every clip the wall-clock window grid is **identical to A12-mos**
(`src/eval/distillmos_windows.py`: 5.0 s windows, hop 2.5 s, on the 16 kHz
waveform, final truncated window kept iff >= 2.5 s) so the resulting
fig_sim_vs_time pairs 1:1 with fig_mos_vs_time.  Each window is embedded and
compared by cosine similarity with the enrollment embedding of the item's
reference voice.

Encoder / preprocessing — copied EXACTLY from src/eval/speaker_drift.py
------------------------------------------------------------------------
Everything audio- and embedding-related is *imported from* `speaker_drift`
(no reimplementation), so the numbers live on the same scale as the published
§9.5 drift results:

* encoder: WeSpeaker ResNet34-TSTP-emb256 ``voxceleb_resnet34_LM.onnx``
  (repo/revision/sha pinned in speaker_drift.py), onnxruntime CPU;
* decode: soundfile float64, multichannel -> mean downmix
  (``audio_qc.load_audio``), resample to 16 kHz with **soxr VHQ**
  (``res_type='soxr'``) — the drift resampler; enrollment from the MASTER
  reference wav resampled by this run with the same resampler (both sides of
  every cosine go through soxr, per the speaker_drift.py comment);
* VAD: ``audio_qc.vad_mask`` expanded per sample
  (``speaker_drift.voiced_sample_mask``);
* fbank + CMN + embedding: ``speaker_drift.embed_waveform``;
* enrollment: ``speaker_drift.enrollment_embedding`` (whole voiced part of
  the reference wav).

Deviation from speaker_drift's own windowing (documented, intentional): drift
cuts 10 s windows on the *voiced* timeline; here windows are cut on the
*wall-clock* timeline (the A12 grid) and the VAD is applied INSIDE each
window: the window's silent samples are dropped and the embedding is computed
on the window's voiced samples only.  A window with less than
``MIN_VOICED_SEC = 1.0`` s of voiced audio is NOT scored (too little speech
for a stable speaker embedding; drift itself refuses whole-file sims under
0.4 s and enrollments under 0.5 s).  Skipped windows are counted per item
(``n_windows_skipped_low_voiced``) and never enter the curves — the n(t)
band of the figure therefore counts items that are both *alive* and
*voiced* at t.

Human slices are cut from ``human_audio_path`` at ``offset_start/end``
seconds (native-rate slicing inside soundfile, before the resample), exactly
as in A12.  NOTE the asymmetry, inherited from the benchmarks: a robust
human slice and its item reference come from the SAME speaker (per-item
voices), so the robust human curve is a same-speaker ceiling; a pilot human
slice is a DIFFERENT real speaker than either pilot reference voice, so the
pilot human curve is a cross-speaker floor (scored once per pilot reference
voice, mirroring the item grid).

CLI
---
Single file::

    python src/eval/spk_sim_windows.py --wav OUT.wav --reference REF.wav

Batch (tasks jsonl -> per-window parquet + per-item jsonl; shardable)::

    python src/eval/spk_sim_windows.py --jsonl tasks.jsonl \
        --out-window pw.parquet --out-item pi.jsonl [--shard K/N] [--threads 4]

Merge shards (concatenate + reorder to tasks order)::

    python src/eval/spk_sim_windows.py --merge-dir DIR --jsonl tasks.jsonl \
        --out-window per_window.parquet --out-item per_item.jsonl

Each task line: ``{"set", "system", "text_id", "voice_id", "bucket",
"wav", "offset_start", "offset_end", "ref_wav"}`` (offsets null for TTS).
Runs in `.venv-eval` (numpy + soundfile + soxr + onnxruntime + pyarrow),
CPU only — the drift venv.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import audio_qc  # noqa: E402
import speaker_drift as sd  # noqa: E402
from distillmos_windows import (  # noqa: E402  (pure window math, torch-less)
    HOP_SAMPLES,
    MIN_TAIL_SAMPLES,
    WIN_SAMPLES,
    t_center,
    window_spans,
)

SR = 16_000
assert SR == sd.TARGET_SR, "speaker_drift target rate changed"

WIN_SEC = WIN_SAMPLES / SR    # 5.0  (A12 grid)
HOP_SEC = HOP_SAMPLES / SR    # 2.5
MIN_VOICED_SEC = 1.0
MIN_VOICED_SAMPLES = int(round(MIN_VOICED_SEC * SR))

META_FIELDS = ("set", "system", "text_id", "voice_id", "bucket")
PW_COLUMNS = ["set", "system", "text_id", "voice_id", "bucket",
              "t_center", "sim", "voiced_sec"]


# ------------------------------------------------------------------ pure math
def voiced_selection(vmask: np.ndarray, spans) -> list:
    """For each (start, end) span: (span, n_voiced_samples, scored?).

    ``vmask`` is the per-sample boolean voiced mask of the whole clip.
    A span is scored iff it holds >= MIN_VOICED_SAMPLES voiced samples.
    Pure numpy — testable without the ONNX model.
    """
    out = []
    for a, b in spans:
        n_voiced = int(np.count_nonzero(vmask[a:b]))
        out.append(((a, b), n_voiced, n_voiced >= MIN_VOICED_SAMPLES))
    return out


def aggregate_windows(t_centers, sims, first_sec: float = 30.0,
                      after_sec: float = 300.0) -> dict:
    """Per-item aggregates over the SCORED windows (sim is not None).

    Mirrors distillmos_windows.aggregate_windows; adds the skip counter.
    ``sims[i] is None`` marks a low-voiced window that was not scored.
    """
    if len(t_centers) != len(sims):
        raise ValueError("t_centers and sims length mismatch")
    scored = [(t, s) for t, s in zip(t_centers, sims) if s is not None]
    if any(math.isnan(s) for _, s in scored):
        raise ValueError("NaN sim value in windows")
    out = {
        "n_windows_total": len(sims),
        "n_windows": len(scored),
        "n_windows_skipped_low_voiced": len(sims) - len(scored),
    }
    if not scored:
        out.update({"mean_sim": None, "min_sim": None,
                    "sim_first_30s": None, "sim_after_300s": None})
        return out
    vals = [s for _, s in scored]
    first = [s for t, s in scored if t <= first_sec]
    after = [s for t, s in scored if t >= after_sec]
    out.update({
        "mean_sim": sum(vals) / len(vals),
        "min_sim": min(vals),
        "sim_first_30s": (sum(first) / len(first)) if first else None,
        "sim_after_300s": (sum(after) / len(after)) if after else None,
    })
    return out


# ------------------------------------------------------------------- scoring
def load_wav_16k(path: str, offset_start=None, offset_end=None) -> np.ndarray:
    """Drift-identical decode: soundfile float64 mono(mean) -> soxr -> 16 kHz.

    Offsets in seconds slice at the file's native rate inside soundfile,
    before the resample (A12 human-slice convention).
    """
    y, sr = audio_qc.load_audio(path, sr=SR, mono=True,
                                start_sec=offset_start, end_sec=offset_end,
                                res_type=sd.RESAMPLER)
    assert sr == SR
    return np.asarray(y, dtype=np.float64)


def sim_windows_for_waveform(y16k: np.ndarray, enroll_emb: np.ndarray) -> list:
    """[(t_center_sec, sim | None, voiced_sec)] on the A12 grid."""
    vmask, _thr = sd.voiced_sample_mask(y16k, SR)
    spans = window_spans(len(y16k))
    rows = []
    for (a, b), n_voiced, ok in voiced_selection(vmask, spans):
        tc = t_center((a, b), SR)
        if not ok:
            rows.append((tc, None, n_voiced / SR))
            continue
        sel = np.flatnonzero(vmask[a:b]) + a
        emb = sd.embed_waveform(np.ascontiguousarray(y16k[sel]), SR)
        rows.append((tc, sd.cosine(emb, enroll_emb), n_voiced / SR))
    return rows


class EnrollCache:
    def __init__(self):
        self._embs = {}

    def get(self, ref_wav: str):
        if ref_wav not in self._embs:
            emb, meta = sd.enrollment_embedding(ref_wav, res_type=sd.RESAMPLER)
            self._embs[ref_wav] = (emb, meta)
        return self._embs[ref_wav]


def _write_parquet(rows: list, out_path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {c: [r[c] for r in rows] for c in PW_COLUMNS}
    table = pa.table({
        "set": pa.array(cols["set"], pa.string()),
        "system": pa.array(cols["system"], pa.string()),
        "text_id": pa.array(cols["text_id"], pa.string()),
        "voice_id": pa.array(cols["voice_id"], pa.string()),
        "bucket": pa.array(cols["bucket"], pa.string()),
        "t_center": pa.array(cols["t_center"], pa.float64()),
        "sim": pa.array(cols["sim"], pa.float64()),
        "voiced_sec": pa.array(cols["voiced_sec"], pa.float64()),
    })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def run_batch(tasks_path, out_window, out_item, shard=None, threads=4):
    tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
    for i, t in enumerate(tasks):
        missing = [f for f in META_FIELDS + ("wav", "ref_wav") if f not in t]
        if missing:
            raise ValueError(f"task line {i}: missing fields {missing}")
    idx = list(range(len(tasks)))
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        idx = [i for i in idx if i % n == k]
    sd.session(threads=threads)  # build the ORT session once, with the cap
    cache = EnrollCache()
    window_rows, n_done = [], 0
    Path(out_item).parent.mkdir(parents=True, exist_ok=True)
    with open(out_item, "w", buffering=1) as f_item:
        for i in idx:
            task = tasks[i]
            y = load_wav_16k(task["wav"], task.get("offset_start"),
                             task.get("offset_end"))
            emb, emeta = cache.get(task["ref_wav"])
            rows = sim_windows_for_waveform(y, emb)
            meta = {f: task[f] for f in META_FIELDS}
            for tc, sim, vsec in rows:
                if sim is not None:
                    window_rows.append({**meta, "t_center": tc, "sim": sim,
                                        "voiced_sec": vsec})
            agg = aggregate_windows([r[0] for r in rows], [r[1] for r in rows])
            item = {
                **meta,
                "task_index": i,
                "wav": task["wav"],
                "offset_start": task.get("offset_start"),
                "offset_end": task.get("offset_end"),
                "ref_wav": task["ref_wav"],
                "ref_voiced_sec": emeta["ref_voiced_sec"],
                "scored_duration_sec": len(y) / SR,
                "voiced_sec": round(float(sum(r[2] for r in rows)), 4),
                **agg,
            }
            f_item.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 10 == 0 or n_done == len(idx):
                print(f"[{n_done}/{len(idx)}] {task['set']}/{task['system']}/"
                      f"{task['text_id']}", file=sys.stderr, flush=True)
    _write_parquet(window_rows, out_window)
    print(f"wrote {len(window_rows)} windows -> {out_window}; "
          f"{n_done} items -> {out_item}", file=sys.stderr)


def merge_shards(merge_dir, tasks_path, out_window, out_item):
    import pyarrow.parquet as pq

    tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
    items = []
    for p in sorted(Path(merge_dir).glob("per_item_shard*.jsonl")):
        items.extend(json.loads(l) for l in p.open() if l.strip())
    if len(items) != len(tasks):
        raise RuntimeError(f"{len(items)} items != {len(tasks)} tasks")
    items.sort(key=lambda r: r["task_index"])
    for i, r in enumerate(items):
        if r["task_index"] != i:
            raise RuntimeError(f"missing/duplicate task_index {i}")
    window_rows = []
    order = {}
    for p in sorted(Path(merge_dir).glob("per_window_shard*.parquet")):
        tbl = pq.read_table(p)
        window_rows.extend(tbl.to_pylist())
    for i, t in enumerate(tasks):
        order[(t["set"], t["system"], t["text_id"], t["voice_id"])] = i
    window_rows.sort(key=lambda r: (order[(r["set"], r["system"],
                                           r["text_id"], r["voice_id"])],
                                    r["t_center"]))
    _write_parquet(window_rows, out_window)
    with open(out_item, "w") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_scored = sum(r["n_windows"] for r in items)
    if n_scored != len(window_rows):
        raise RuntimeError(f"window count mismatch: parquet {len(window_rows)} "
                           f"vs per-item sum {n_scored}")
    print(f"merged {len(window_rows)} windows, {len(items)} items "
          f"-> {out_window}, {out_item}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wav", help="score one file, print t_center,sim CSV")
    ap.add_argument("--reference", help="enrollment wav for --wav")
    ap.add_argument("--offset-start", type=float, default=None)
    ap.add_argument("--offset-end", type=float, default=None)
    ap.add_argument("--jsonl", help="tasks jsonl (batch / merge)")
    ap.add_argument("--out-window", help="per-window parquet path")
    ap.add_argument("--out-item", help="per-item jsonl path")
    ap.add_argument("--shard", default=None, help="K/N stride sharding")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--merge-dir", default=None,
                    help="dir with per_*_shard files: merge instead of score")
    a = ap.parse_args(argv)

    if a.wav:
        emb, _ = sd.enrollment_embedding(a.reference, res_type=sd.RESAMPLER)
        y = load_wav_16k(a.wav, a.offset_start, a.offset_end)
        print("t_center,sim,voiced_sec")
        for tc, sim, vsec in sim_windows_for_waveform(y, emb):
            s = "" if sim is None else f"{sim:.6f}"
            print(f"{tc:.3f},{s},{vsec:.3f}")
        return 0
    if not (a.jsonl and a.out_window and a.out_item):
        ap.error("batch/merge mode needs --jsonl, --out-window, --out-item")
    if a.merge_dir:
        merge_shards(a.merge_dir, a.jsonl, a.out_window, a.out_item)
    else:
        run_batch(a.jsonl, a.out_window, a.out_item, a.shard, a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
