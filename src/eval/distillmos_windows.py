"""Objective MOS proxy over TIME: DistillMOS scored in sliding windows.

This is an *objective MOS proxy (DistillMOS), not a listening test*.

Windowing
---------
A clip is scored in windows of ``WIN_SEC = 5.0`` s with hop ``HOP_SEC = 2.5`` s
on the 16 kHz waveform (integer sample arithmetic: 80 000 / 40 000 samples).
Full windows start at 0, HOP, 2*HOP, ... while ``start + WIN <= n``.  After the
last full window one final *partial* window covering the remaining tail
``[K*HOP, n)`` is kept iff its length is >= ``MIN_TAIL_SEC = 2.5`` s
(40 000 samples); a shorter tail is dropped (documented contract).  This is
equivalent to the grid reading: a window starts at every multiple of HOP,
spans ``[s, min(s + WIN, n))``, and a truncated final window is kept iff it
is at least 2.5 s long.  Because ``WIN = 2 * HOP`` the truncated final window
of any clip >= 5 s always has length in ``[2.5 s, 5.0 s)`` and is always kept
(at an exact multiple, e.g. n = 5.0 s, it is the final 2.5 s, nested inside
the last full window); the drop rule therefore bites only for grid starts
with < 2.5 s remaining and for clips shorter than 2.5 s, which yield no
windows at all.  A clip with ``2.5 s <= n < 5 s`` yields the single partial
window ``[0, n)``.  Each window is reported at its centre time
``t_center = (start + end) / 2 / sr`` in seconds.

Preprocessing — copied EXACTLY from the balalaika pipeline DistillMOS stage
---------------------------------------------------------------------------
Model loading is copied from
``third_party/balalaika/src/separation/distillmos_process.py``
(``run_inference_worker``): ``distillmos.ConvTransformerSQAModel()`` ->
``.to(device)`` -> ``.eval()``, scored under ``torch.inference_mode()`` with a
``[B, T]`` float32 batch, MOS = ``sqa_model(batch).detach().flatten()``.

Audio preprocessing is copied from
``third_party/balalaika/src/utils/datasets/separation.py``
(``DistillMOSDataset.__getitem__`` / ``DISTILLMOS_SAMPLE_RATE = 16_000``):
load -> keep the FIRST channel if multichannel (``waveform[:1]``, not a
downmix) -> ``torchaudio.functional.resample(waveform, sr, 16_000)`` with
default parameters -> 1-D float32.

One deliberate deviation, justified by the pipeline's own code: the stage
decodes via ``torchaudio.load_with_torchcodec``, but torchcodec fails to load
in the frozen venv ``external/balalaika/.dev_venv`` (libtorchcodec_core4
OSError).  ``separation.py`` documents (``_RANGED_DECODE_FORMATS``) that a
libsndfile (soundfile) decode of WAV/FLAC PCM_16/24/32/FLOAT/DOUBLE is
bit-exact (``torch.equal``) to torchcodec's float32 full decode — every input
here is WAV PCM_16 (TTS outputs, 24 kHz mono) or FLAC (human references), so
we read with ``soundfile`` and obtain identical float32 samples.

Human reference slices: ``offset_start`` / ``offset_end`` (seconds) are
applied at the file's native sample rate (frame = round(offset * sr)) BEFORE
resampling, matching how the benchmark offsets were defined on the source
FLAC.

Batching: all full windows of a clip have identical length (80 000 samples)
and are scored in ``[B, 80000]`` batches; the final partial window is scored
alone at its true length, so no zero-padding ever touches the model input
(the pipeline pads within mixed-length batches; equal-length batching makes
padding a no-op here by construction).

CLI
---
Single file (prints ``t_center,mos`` CSV to stdout)::

    python -m src.eval.distillmos_windows --wav path.wav \
        [--offset-start S --offset-end S] [--device cuda:0]

Batch mode (tasks jsonl -> per-window parquet + per-item jsonl)::

    python -m src.eval.distillmos_windows --jsonl tasks.jsonl \
        --out-window per_window.parquet --out-item per_item.jsonl \
        [--device cuda:0] [--batch-size 64]

Each task line: ``{"wav": str, "set": str, "system": str, "text_id": str,
"voice_id": str, "bucket": str, "offset_start": float|null,
"offset_end": float|null}``.

Per-item aggregates (see :func:`aggregate_windows`): ``mean_mos``/``min_mos``
over all windows, ``mos_first_30s`` = mean over windows with
``t_center <= 30.0`` s, ``mos_after_300s`` = mean over windows with
``t_center >= 300.0`` s (null when no such window).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# NOTE: torch / torchaudio / distillmos / soundfile / pandas are imported
# lazily inside functions so that the pure window math stays importable from
# the torch-less evaluation venv (tests import this module there).

DISTILLMOS_SAMPLE_RATE = 16_000  # separation.py: DISTILLMOS_SAMPLE_RATE
WIN_SEC = 5.0
HOP_SEC = 2.5
MIN_TAIL_SEC = 2.5

WIN_SAMPLES = int(round(WIN_SEC * DISTILLMOS_SAMPLE_RATE))       # 80_000
HOP_SAMPLES = int(round(HOP_SEC * DISTILLMOS_SAMPLE_RATE))       # 40_000
MIN_TAIL_SAMPLES = int(round(MIN_TAIL_SEC * DISTILLMOS_SAMPLE_RATE))  # 40_000

META_FIELDS = ("set", "system", "text_id", "voice_id", "bucket")


def window_spans(
    n_samples: int,
    win: int = WIN_SAMPLES,
    hop: int = HOP_SAMPLES,
    min_tail: int = MIN_TAIL_SAMPLES,
) -> List[Tuple[int, int]]:
    """Return ``[(start, end), ...]`` sample spans for one clip.

    Full windows ``[k*hop, k*hop + win)`` for ``k = 0..K-1`` with
    ``K = (n - win)//hop + 1`` when ``n >= win``; then one partial tail window
    ``[K*hop, n)`` kept iff ``n - K*hop >= min_tail`` (dropped otherwise).
    A clip with ``min_tail <= n < win`` yields the single partial window
    ``[0, n)``; ``n < min_tail`` yields no windows.
    """
    if n_samples < min_tail:
        return []
    spans: List[Tuple[int, int]] = []
    if n_samples >= win:
        n_full = (n_samples - win) // hop + 1
    else:
        n_full = 0
    for k in range(n_full):
        spans.append((k * hop, k * hop + win))
    tail_start = n_full * hop
    tail_len = n_samples - tail_start
    if min_tail <= tail_len < win:
        spans.append((tail_start, n_samples))
    return spans


def t_center(span: Tuple[int, int], sr: int = DISTILLMOS_SAMPLE_RATE) -> float:
    """Window centre time in seconds."""
    return (span[0] + span[1]) / 2.0 / sr


def aggregate_windows(
    t_centers: Sequence[float],
    mos: Sequence[float],
    first_sec: float = 30.0,
    after_sec: float = 300.0,
) -> dict:
    """Per-item aggregates over (t_center, mos) windows.

    ``mos_first_30s``: mean over windows with ``t_center <= first_sec``.
    ``mos_after_300s``: mean over windows with ``t_center >= after_sec``
    (``None`` when the clip never reaches that far).  Empty input yields
    ``n_windows = 0`` and all-``None`` aggregates (never NaN in the output).
    """
    if len(t_centers) != len(mos):
        raise ValueError("t_centers and mos length mismatch")
    if not mos:
        return {
            "n_windows": 0,
            "mean_mos": None,
            "min_mos": None,
            "mos_first_30s": None,
            "mos_after_300s": None,
        }
    if any(math.isnan(v) for v in mos):
        raise ValueError("NaN MOS value in windows")
    first = [v for t, v in zip(t_centers, mos) if t <= first_sec]
    after = [v for t, v in zip(t_centers, mos) if t >= after_sec]
    return {
        "n_windows": len(mos),
        "mean_mos": sum(mos) / len(mos),
        "min_mos": min(mos),
        "mos_first_30s": (sum(first) / len(first)) if first else None,
        "mos_after_300s": (sum(after) / len(after)) if after else None,
    }


def load_waveform_16k(
    wav_path: str,
    offset_start: Optional[float] = None,
    offset_end: Optional[float] = None,
):
    """Load audio exactly like the balalaika DistillMOS stage (see module doc).

    Returns a 1-D float32 torch tensor at 16 kHz.  Offsets (seconds) slice at
    the native sample rate before resampling.
    """
    import soundfile as sf
    import torch
    import torchaudio

    info = sf.info(wav_path)
    sr = info.samplerate
    start_frame = 0 if offset_start is None else max(0, int(round(offset_start * sr)))
    stop_frame = info.frames if offset_end is None else min(
        info.frames, int(round(offset_end * sr))
    )
    if stop_frame <= start_frame:
        raise ValueError(
            f"empty slice [{offset_start}, {offset_end}] for {wav_path}"
        )
    data, sr = sf.read(
        wav_path,
        start=start_frame,
        stop=stop_frame,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(data.T)  # [C, T] float32, torchcodec layout
    if waveform.shape[0] > 1:
        waveform = waveform[:1]  # first channel, NOT a downmix (separation.py)
    if sr != DISTILLMOS_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            DISTILLMOS_SAMPLE_RATE,
        )
    return waveform.squeeze(0).contiguous()


def load_model(device):
    """Model loading copied from distillmos_process.run_inference_worker."""
    import distillmos

    sqa_model = distillmos.ConvTransformerSQAModel()
    sqa_model.to(device)
    sqa_model.eval()
    return sqa_model


def score_waveform(
    sqa_model,
    wave_16k,
    device,
    batch_size: int = 64,
) -> List[Tuple[float, float]]:
    """Score one 16 kHz 1-D waveform in windows; returns [(t_center, mos)].

    Full windows are batched together (equal length); the partial tail window
    is scored alone at its true length so nothing is zero-padded.
    """
    import torch

    n = int(wave_16k.shape[-1])
    spans = window_spans(n)
    if not spans:
        return []
    full = [s for s in spans if s[1] - s[0] == WIN_SAMPLES]
    partial = [s for s in spans if s[1] - s[0] != WIN_SAMPLES]
    results: List[Tuple[float, float]] = []
    with torch.inference_mode():
        for i in range(0, len(full), batch_size):
            chunk = full[i : i + batch_size]
            batch = torch.stack([wave_16k[a:b] for a, b in chunk]).to(device)
            mos = sqa_model(batch).detach().flatten().cpu()
            for span, val in zip(chunk, mos.tolist()):
                results.append((t_center(span), float(val)))
        for a, b in partial:
            batch = wave_16k[a:b].unsqueeze(0).to(device)
            mos = sqa_model(batch).detach().flatten().cpu()
            results.append((t_center((a, b)), float(mos[0])))
    results.sort(key=lambda r: r[0])
    if any(math.isnan(v) for _, v in results):
        raise RuntimeError("DistillMOS returned NaN")
    return results


def _resolve_device(arg: Optional[str]):
    import torch

    if arg:
        return torch.device(arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_batch(
    tasks_path: str,
    out_window: str,
    out_item: str,
    device_arg: Optional[str],
    batch_size: int,
) -> None:
    import pandas as pd

    device = _resolve_device(device_arg)
    tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
    for i, t in enumerate(tasks):
        missing = [f for f in META_FIELDS + ("wav",) if f not in t]
        if missing:
            raise ValueError(f"task line {i}: missing fields {missing}")
    sqa_model = load_model(device)
    window_rows: List[dict] = []
    Path(out_item).parent.mkdir(parents=True, exist_ok=True)
    n_total = len(tasks)
    with open(out_item, "w", buffering=1) as f_item:
        for i, task in enumerate(tasks):
            wave = load_waveform_16k(
                task["wav"], task.get("offset_start"), task.get("offset_end")
            )
            scored = score_waveform(sqa_model, wave, device, batch_size)
            meta = {f: task[f] for f in META_FIELDS}
            for tc, mos in scored:
                window_rows.append({**meta, "t_center": tc, "mos": mos})
            agg = aggregate_windows([t for t, _ in scored], [m for _, m in scored])
            item = {
                **meta,
                "wav": task["wav"],
                "offset_start": task.get("offset_start"),
                "offset_end": task.get("offset_end"),
                "scored_duration_sec": wave.shape[-1] / DISTILLMOS_SAMPLE_RATE,
                **agg,
            }
            f_item.write(json.dumps(item, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0 or i + 1 == n_total:
                print(f"[{i + 1}/{n_total}] {task['set']}/{task['system']}/"
                      f"{task['text_id']}", file=sys.stderr, flush=True)
    df = pd.DataFrame(
        window_rows,
        columns=list(META_FIELDS) + ["t_center", "mos"],
    )
    if df["mos"].isna().any():
        raise RuntimeError("NaN in per-window MOS")
    Path(out_window).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_window, index=False)
    print(f"wrote {len(df)} windows -> {out_window}; "
          f"{n_total} items -> {out_item}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wav", help="score one file, print t_center,mos CSV")
    p.add_argument("--offset-start", type=float, default=None)
    p.add_argument("--offset-end", type=float, default=None)
    p.add_argument("--jsonl", help="batch mode: tasks jsonl")
    p.add_argument("--out-window", help="batch mode: per-window parquet path")
    p.add_argument("--out-item", help="batch mode: per-item jsonl path")
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    if bool(args.wav) == bool(args.jsonl):
        p.error("exactly one of --wav / --jsonl is required")
    if args.wav:
        device = _resolve_device(args.device)
        sqa_model = load_model(device)
        wave = load_waveform_16k(args.wav, args.offset_start, args.offset_end)
        print("t_center,mos")
        for tc, mos in score_waveform(sqa_model, wave, device, args.batch_size):
            print(f"{tc:.3f},{mos:.4f}")
    else:
        if not (args.out_window and args.out_item):
            p.error("--jsonl requires --out-window and --out-item")
        run_batch(args.jsonl, args.out_window, args.out_item,
                  args.device, args.batch_size)


if __name__ == "__main__":
    main()
