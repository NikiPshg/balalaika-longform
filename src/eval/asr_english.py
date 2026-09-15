#!/usr/bin/env python
"""English ASR wrapper for the E9 "EN-probe" (EXPLORATORY probe, not a paper benchmark).

Pre-registered in reports/decisions.md, Lead 2026-08-31: English ASR goes through
the SAME ``onnx-asr`` library as the frozen Russian path, with an NVIDIA
Parakeet-family model; Whisper is forbidden in any form (PLAN.md §0.1) and
GigaAM-v3 is Russian-only.  This module MIRRORS ``src/eval/asr_gigaam.py``:

* same silero VAD, same FROZEN VAD options (imported, not copied);
* same GPU guard -- ``ALLOWED_CUDA_VISIBLE_DEVICES`` is imported from
  ``asr_gigaam`` (cards {0, 1, 2}; E9 itself runs on index 2 only);
* same in-repo HF cache (``<repo>/.cache/onnx-asr``), same CUDA preload;
* same ``TranscriptionResult`` rows, so the evaluation code that reads them
  does not care which language the transcriber speaks.

The frozen Russian module is imported read-only and NOT modified.

Available English (Parakeet-family) models in onnx-asr 0.12.0
(``onnx_asr/loader.py`` AsrNames):

    nemo-parakeet-tdt-0.6b-v2    TDT decoder, English, punctuation+caps  (DEFAULT)
    nemo-parakeet-tdt-0.6b-v3    TDT decoder, multilingual
    nemo-parakeet-ctc-0.6b       CTC decoder, English, lowercase
    nemo-parakeet-rnnt-0.6b      RNN-T decoder, English, lowercase

The default is ``nemo-parakeet-tdt-0.6b-v2`` -- the model the pre-registration
names, the strongest English model of the family on the Open ASR leaderboard,
and the one variant with punctuation (parallel to how the Russian benchmark was
built with ``gigaam-v3-e2e-ctc`` word groups).  WER never sees the punctuation:
``src/eval/normalize_english.py`` strips it from reference and hypothesis alike.

CLI (mirrors asr_gigaam.py)::

    CUDA_VISIBLE_DEVICES=2 .venv-eval/bin/python src/eval/asr_english.py \
        --input chapter.wav --output out.jsonl
    ... --input items.jsonl --audio-key audio_path --output out.jsonl --timestamps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Frozen Russian module: imported read-only, never modified (E9 contract).
from eval.asr_gigaam import (  # noqa: E402
    ALLOWED_CUDA_VISIBLE_DEVICES,  # noqa: F401  (re-exported: the E9 GPU guard set {0,1,2})
    DEFAULT_VAD,
    DEFAULT_VAD_BATCH_SIZE,
    FROZEN_VAD_OPTIONS,
    GpuGuardError,  # noqa: F401  (re-exported for callers)
    TranscriptionResult,
    _HF_CACHE_STATE,
    _providers_for,
    _session_providers,
    assert_gpu_allowed,
    ensure_hf_cache,
    preload_cuda_libraries,
    read_audio,
)

ENGLISH_VARIANTS = (
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
    "nemo-parakeet-ctc-0.6b",
    "nemo-parakeet-rnnt-0.6b",
)
DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v2"

# Word-initial subword markers used to fold the token-level timestamps of
# ``with_timestamps()`` into word-level times.  Parakeet's tokenizer (verified
# empirically on this host, onnx-asr 0.12.0) emits a LEADING SPACE on
# word-initial pieces (" Ch", "ap", "ter" -> "Chapter"); other NeMo BPE
# vocabularies use U+2581.  Both are handled.
_WORD_MARKS = ("▁", " ")


class EnglishTranscriber:
    """Parakeet + silero VAD transcriber, mirror of ``GigaAMTranscriber``.

    VAD is on by default and must stay on (PLAN.md §0.1 applies to the mirrored
    path too: multi-minute audio is cut into speech segments and the segment
    texts are concatenated in time order with a single space).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        use_vad: bool = True,
        device: str = "cuda",
        vad_model: str = DEFAULT_VAD,
        vad_options: dict | None = None,
        vad_batch_size: int = DEFAULT_VAD_BATCH_SIZE,
        cache_dir: str | Path | None = None,
        quantization: str | None = None,
    ) -> None:
        if model_id not in ENGLISH_VARIANTS:
            raise ValueError(
                f"{model_id!r} is not an English Parakeet-family model "
                f"{ENGLISH_VARIANTS}; the Russian path uses src/eval/asr_gigaam.py "
                "and Whisper is forbidden (PLAN.md §0.1)"
            )
        ensure_hf_cache(cache_dir)
        self.model_id = model_id
        self.use_vad = use_vad
        self.device = device
        self.vad_model_name = vad_model if use_vad else None
        self.vad_options = dict(FROZEN_VAD_OPTIONS if vad_options is None else vad_options)
        self.vad_batch_size = vad_batch_size

        # PLAN.md §0.1 + owner grants: never open a CUDA session on somebody
        # else's card.  Same guard object as the Russian path.
        self.gpu_guard = assert_gpu_allowed(device)

        if device != "cpu":
            self.preloaded_cuda_libs = preload_cuda_libraries()
        else:
            self.preloaded_cuda_libs = []

        import onnx_asr

        self.onnx_asr_version = onnx_asr.__version__
        providers = _providers_for(device)
        t0 = time.perf_counter()
        self._model = onnx_asr.load_model(model_id, providers=providers, quantization=quantization)
        self._vad = onnx_asr.load_vad(vad_model, providers=providers) if use_vad else None
        if self._vad is not None:
            self._runner = self._model.with_vad(
                self._vad, batch_size=vad_batch_size, **self.vad_options
            )
        else:
            self._runner = self._model
        self.load_time_sec = time.perf_counter() - t0

        provs = _session_providers(self._model)
        self.provider = provs[0] if provs else "unknown"
        self.all_providers = provs

    # -- API (mirrors GigaAMTranscriber) ------------------------------------

    def transcribe(self, path: str | Path) -> TranscriptionResult:
        path = str(path)
        try:
            wave, sr = read_audio(path)
        except Exception as exc:  # audio unreadable = empty hypothesis
            return TranscriptionResult(
                audio_path=path, text="", model_id=self.model_id,
                onnx_asr_version=self.onnx_asr_version, vad_model=self.vad_model_name,
                use_vad=self.use_vad, provider=self.provider,
                error=f"{type(exc).__name__}: {exc}",
            )
        return self.transcribe_waveform(wave, sr, label=path)

    def transcribe_slice(
        self, path: str | Path, start_sec: float | None = None, end_sec: float | None = None
    ) -> TranscriptionResult:
        """Transcribe ``[start_sec, end_sec)`` of an audio file (ASR floor slices)."""
        path = str(path)
        try:
            wave, sr = read_audio(path)
        except Exception as exc:
            return TranscriptionResult(
                audio_path=path, text="", model_id=self.model_id,
                onnx_asr_version=self.onnx_asr_version, vad_model=self.vad_model_name,
                use_vad=self.use_vad, provider=self.provider,
                error=f"{type(exc).__name__}: {exc}",
            )
        a = int(round((start_sec or 0.0) * sr))
        b = int(round(end_sec * sr)) if end_sec is not None else len(wave)
        a = max(0, min(a, len(wave)))
        b = max(a, min(b, len(wave)))
        label = f"{path}#{start_sec or 0.0}-{end_sec if end_sec is not None else ''}"
        return self.transcribe_waveform(wave[a:b], sr, label=label)

    def transcribe_waveform(self, wave, sr: int, label: str = "<array>") -> TranscriptionResult:
        path = label
        duration = len(wave) / sr if sr else 0.0
        t0 = time.perf_counter()
        segments: list[dict] = []
        error: str | None = None
        try:
            result = self._runner.recognize(wave, sample_rate=sr)
            if self.use_vad:
                segments = [
                    {"start": float(s.start), "end": float(s.end), "text": s.text}
                    for s in result
                ]
                segments.sort(key=lambda s: (s["start"], s["end"]))
                text = " ".join(s["text"].strip() for s in segments if s["text"].strip())
            else:
                text = str(result)
                segments = [{"start": 0.0, "end": duration, "text": text}]
        except Exception as exc:
            text = ""
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0

        return TranscriptionResult(
            audio_path=path, text=text, segments=segments, model_id=self.model_id,
            onnx_asr_version=self.onnx_asr_version, vad_model=self.vad_model_name,
            use_vad=self.use_vad, provider=self.provider, elapsed_sec=elapsed,
            audio_duration_sec=duration, sample_rate=sr,
            rtf=(elapsed / duration) if duration > 0 else None,
            speed_x_realtime=(duration / elapsed) if elapsed > 0 else None,
            error=error,
        )

    def transcribe_many(self, paths: Iterable[str | Path]) -> list[TranscriptionResult]:
        return [self.transcribe(p) for p in paths]

    # -- E9 addition: word-level timestamps for benchmark construction -------

    def transcribe_words(self, wave, sr: int) -> list[dict]:
        """VAD + token timestamps folded into word-level times (absolute seconds).

        Used ONLY to build the benchmark (locating where the human reading of a
        contiguous text fragment starts/ends); the metrics never read this.

        Returns ``[{"word", "t_start", "t_end", "seg_start", "seg_end", "seg_index"}]``
        in time order.  ``t_end`` of a word is the timestamp of the next token
        (or the segment end for the last word) -- an upper bound, good enough to
        cut at silence-padded sentence boundaries.

        onnx-asr emits token timestamps relative to the VAD segment; this is
        verified per segment (a final token time far beyond the segment length
        raises) and converted to absolute time by adding ``segment.start``.
        """
        if not self.use_vad:
            raise RuntimeError("transcribe_words requires VAD (segment offsets)")
        out: list[dict] = []
        results = self._runner.with_timestamps().recognize(wave, sample_rate=sr)
        segs = sorted(results, key=lambda s: (float(s.start), float(s.end)))
        for seg_i, seg in enumerate(segs):
            tokens = list(seg.tokens or [])
            stamps = [float(t) for t in (seg.timestamps or [])]
            if len(tokens) != len(stamps):
                raise RuntimeError(
                    f"segment {seg_i}: {len(tokens)} tokens vs {len(stamps)} timestamps"
                )
            if not tokens:
                continue
            seg_dur = float(seg.end) - float(seg.start)
            if stamps and max(stamps) > seg_dur + 5.0:
                raise RuntimeError(
                    f"segment {seg_i}: token timestamp {max(stamps):.2f}s exceeds the "
                    f"segment duration {seg_dur:.2f}s by more than 5 s -- timestamps do "
                    "not look segment-relative; refusing to guess"
                )
            words: list[dict] = []
            for tok, ts in zip(tokens, stamps):
                is_word_start = tok.startswith(_WORD_MARKS)
                piece = tok.lstrip("▁ ")
                if is_word_start or not words:
                    words.append({"word": piece, "t_start": float(seg.start) + ts})
                else:
                    words[-1]["word"] += piece
            for i, w in enumerate(words):
                w["t_end"] = (
                    words[i + 1]["t_start"] if i + 1 < len(words) else float(seg.end)
                )
                w["seg_start"] = float(seg.start)
                w["seg_end"] = float(seg.end)
                w["seg_index"] = seg_i
            out.extend(w for w in words if w["word"].strip())
        return out

    # -- identity ------------------------------------------------------------

    def vad_fingerprint(self) -> dict:
        """Identical construction to ``GigaAMTranscriber.vad_fingerprint``."""
        import hashlib

        payload = {
            "use_vad": bool(self.use_vad),
            "vad_model": self.vad_model_name,
            "vad_options": {k: float(v) for k, v in sorted(self.vad_options.items())},
            "vad_batch_size": int(self.vad_batch_size),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return {
            "vad_payload": payload,
            "vad_options_hash": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
        }

    def describe(self) -> dict:
        import os

        d = {
            "model_id": self.model_id,
            "language": "en",
            "onnx_asr_version": self.onnx_asr_version,
            "use_vad": self.use_vad,
            "vad_model": self.vad_model_name,
            "vad_options": self.vad_options,
            "vad_batch_size": self.vad_batch_size,
            "device": self.device,
            "provider": self.provider,
            "all_providers": self.all_providers,
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
            "hf_hub_cache_overridden": _HF_CACHE_STATE.get("previous"),
            "cuda_visible_devices": self.gpu_guard.get("cuda_visible_devices"),
            "physical_gpu_uuid": self.gpu_guard.get("physical_gpu_uuid"),
            "load_time_sec": self.load_time_sec,
        }
        d.update(self.vad_fingerprint())
        return d


# ---------------------------------------------------------------------------
# CLI (mirrors asr_gigaam.py)
# ---------------------------------------------------------------------------


def _iter_inputs(input_path: str, audio_key: str) -> list[dict]:
    p = Path(input_path)
    if p.suffix.lower() == ".jsonl":
        items = []
        with open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    return [{audio_key: str(p)}]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="English Parakeet transcription (onnx-asr, VAD on)")
    ap.add_argument("--input", required=True, help="one audio file, or a .jsonl with audio paths")
    ap.add_argument("--output", required=True, help="output .jsonl")
    ap.add_argument("--audio-key", default="audio_path")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ENGLISH_VARIANTS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-vad", action="store_true", help="DEBUG ONLY - the protocol requires VAD")
    ap.add_argument("--vad-model", default=DEFAULT_VAD)
    ap.add_argument("--vad-batch-size", type=int, default=DEFAULT_VAD_BATCH_SIZE)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timestamps", action="store_true",
                    help="also emit word-level timestamps (benchmark construction)")
    args = ap.parse_args(argv)

    items = _iter_inputs(args.input, args.audio_key)
    if args.limit:
        items = items[: args.limit]

    tr = EnglishTranscriber(
        model_id=args.model, use_vad=not args.no_vad, device=args.device,
        vad_model=args.vad_model, vad_batch_size=args.vad_batch_size,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(tr.describe(), ensure_ascii=False), file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wt", encoding="utf-8") as f:
        for item in items:
            audio = item.get(args.audio_key)
            if not audio:
                continue
            res = tr.transcribe(audio)
            row = {k: v for k, v in item.items() if k != args.audio_key}
            row.update(res.to_dict())
            if args.timestamps and res.error is None:
                wave, sr = read_audio(audio)
                row["words_ts"] = tr.transcribe_words(wave, sr)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[asr_english] {audio} dur={res.audio_duration_sec:.1f}s "
                f"elapsed={res.elapsed_sec:.1f}s x{(res.speed_x_realtime or 0):.1f} "
                f"segs={len(res.segments)} words={len(res.text.split())}"
                + (f" ERROR={res.error}" if res.error else ""),
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
