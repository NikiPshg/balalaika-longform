#!/usr/bin/env python
"""GigaAM-v3 ASR wrapper for RuLongTTS evaluation (PLAN.md §0.1, §9.4).

Hard constraints from the project owner:

* the only ASR used for metrics is **GigaAM-v3 through the ``onnx-asr`` library**;
* **VAD is always on** (``use_vad=True``) -- multi-minute audio must be cut into
  speech segments before recognition, and the segment texts are concatenated in
  time order with a single space;
* Whisper is forbidden in any form.

Available GigaAM-v3 variants in onnx-asr 0.12.0 (read from
``onnx_asr/loader.py`` + ``onnx_asr/resolver.py``), all from the HF repo
``istupakov/gigaam-v3-onnx``:

    gigaam-v3-ctc        CTC, lowercase, no punctuation
    gigaam-v3-rnnt       RNN-T, lowercase, no punctuation
    gigaam-v3-e2e-ctc    CTC, punctuation + capitalization
    gigaam-v3-e2e-rnnt   RNN-T, punctuation + capitalization

Model files are downloaded from the HF Hub into ``<repo>/.cache/onnx-asr``
(``HF_HUB_CACHE`` is set here, not ``~/.cache``).

onnxruntime-gpu 1.29 needs the CUDA 13 / cuDNN 9 shared objects.  They are
installed as pip packages inside ``.venv-eval`` but the dynamic loader does not
look there, so :func:`preload_cuda_libraries` ``dlopen``s them with
``RTLD_GLOBAL`` before the first InferenceSession is created.  This is why the
wrapper works without the caller exporting ``LD_LIBRARY_PATH``.

CLI::

    python src/eval/asr_gigaam.py --input a.wav --output out.jsonl
    python src/eval/asr_gigaam.py --input items.jsonl --audio-key audio_path \\
        --output out.jsonl --model gigaam-v3-rnnt --device cuda
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE = REPO_ROOT / ".cache" / "onnx-asr"

GIGAAM_V3_VARIANTS = (
    "gigaam-v3-ctc",
    "gigaam-v3-rnnt",
    "gigaam-v3-e2e-ctc",
    "gigaam-v3-e2e-rnnt",
)
DEFAULT_MODEL = "gigaam-v3-rnnt"
DEFAULT_VAD = "silero"

# Frozen VAD settings (onnx-asr BaseVad defaults, pinned here so a library
# default change cannot silently move the metrics).
FROZEN_VAD_OPTIONS: dict[str, float] = {
    "threshold": 0.5,
    "neg_threshold": 0.35,
    "min_speech_duration_ms": 250.0,
    "max_speech_duration_s": 20.0,
    "min_silence_duration_ms": 100.0,
    "speech_pad_ms": 30.0,
}
DEFAULT_VAD_BATCH_SIZE = 8

# onnx-asr accepts these sample rates directly (it resamples internally with an
# ONNX resampler); anything else we resample ourselves.
NATIVE_SAMPLE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)
TARGET_SAMPLE_RATE = 16000

_CUDA_PRELOADED = False
_HF_CACHE_STATE: dict[str, str | None] = {"previous": None, "path": None}

# Retain the historical environment field for log-schema compatibility.
GPU_GUARD_OVERRIDE_ENV = "RULONGTTS_ALLOW_ANY_GPU"


class GpuGuardError(RuntimeError):
    """A CUDA run was attempted on a GPU this project is not allowed to touch."""


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def _nvidia_smi_gpus() -> dict[str, str]:
    """``{index: uuid}`` for every physical GPU, or ``{}`` if nvidia-smi is absent.

    ``nvidia-smi --query-gpu`` ignores ``CUDA_VISIBLE_DEVICES`` (verified on this
    host), so its indices are the physical ones and can be used to resolve which
    card ``CUDA_VISIBLE_DEVICES`` actually selects.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except Exception:
        return {}
    gpus: dict[str, str] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            gpus[parts[0]] = parts[1]
    return gpus


def assert_gpu_allowed(device: str = "cuda") -> dict:
    """Record the caller's device selection; local lab GPU assignments do not apply."""
    if device == "cpu":
        return {"guard": "cpu", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    override = os.environ.get(GPU_GUARD_OVERRIDE_ENV, "").strip() not in ("", "0", "false")
    gpus = _nvidia_smi_gpus()
    record = {
        "guard": "cuda",
        "cuda_visible_devices": cvd,
        "allowed": "caller-selected visible devices",
        "override": override,
        "physical_gpu_uuid": gpus.get((cvd or "").split(",")[0].strip()),
        "nvidia_smi_gpus": gpus,
    }
    print(f"[asr_gigaam] GPU guard: CUDA_VISIBLE_DEVICES={cvd!r} -> "
          f"{record['physical_gpu_uuid']}", file=sys.stderr)
    return record


def ensure_hf_cache(cache_dir: str | Path | None = None) -> Path:
    """Force huggingface_hub at the in-repo cache (never ``~/.cache``).

    This is an **override, not a default** (``setdefault`` was a bug): the
    working directory is the only place this project is allowed to write
    (PLAN.md §0.1), so an ``HF_HUB_CACHE`` inherited from the caller's shell --
    pointing at ``~/.cache`` or at another agent's directory -- must not win.
    The previous value is returned in ``describe()['hf_hub_cache_overridden']``
    so the override is visible rather than silent.
    """
    path = Path(cache_dir) if cache_dir else DEFAULT_HF_CACHE
    path.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("HF_HUB_CACHE")
    os.environ["HF_HUB_CACHE"] = str(path)
    if previous is not None and previous != str(path):
        print(
            f"[asr_gigaam] HF_HUB_CACHE overridden: {previous!r} -> {str(path)!r} "
            "(PLAN.md §0.1: writes stay inside the working directory)",
            file=sys.stderr,
        )
    _HF_CACHE_STATE["previous"] = previous
    _HF_CACHE_STATE["path"] = str(path)
    return path


def preload_cuda_libraries() -> list[str]:
    """dlopen the pip-installed CUDA 13 / cuDNN 9 shared objects with RTLD_GLOBAL.

    Returns the list of successfully loaded sonames (empty when the packages are
    not installed, in which case onnxruntime will simply fall back to CPU).
    """
    global _CUDA_PRELOADED
    if _CUDA_PRELOADED:
        return []
    loaded: list[str] = []
    site_dirs = [Path(p) for p in sys.path if p.endswith("site-packages")]
    for site in site_dirs:
        for sub in ("nvidia/cu13/lib", "nvidia/cudnn/lib", "nvidia/cublas/lib"):
            d = site / sub
            if not d.is_dir():
                continue
            for so in sorted(glob.glob(str(d / "*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(so))
                except OSError:
                    pass
    _CUDA_PRELOADED = True
    return loaded


def _providers_for(device: str) -> list[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device in ("cuda", "gpu"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"unknown device {device!r} (use 'cuda' or 'cpu')")


def _session_providers(obj: Any) -> list[str]:
    """Collect the actual providers of every onnxruntime session inside ``obj``."""
    out: list[str] = []
    seen: set[int] = set()

    def walk(o: Any, depth: int = 0) -> None:
        if depth > 3 or id(o) in seen:
            return
        seen.add(id(o))
        if hasattr(o, "get_providers") and callable(o.get_providers):
            try:
                out.extend(o.get_providers())
            except Exception:  # pragma: no cover - defensive
                pass
            return
        for value in list(getattr(o, "__dict__", {}).values()):
            if isinstance(value, (list, tuple)):
                for v in value:
                    walk(v, depth + 1)
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v, depth + 1)
            elif hasattr(value, "__dict__") or hasattr(value, "get_providers"):
                walk(value, depth + 1)

    walk(obj)
    # dedupe, keep order
    result: list[str] = []
    for p in out:
        if p not in result:
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def read_audio(path: str | Path) -> tuple["Any", int]:
    """Read any libsndfile-supported file (wav/flac/...) as float32 mono.

    24 kHz CosyVoice output is returned as-is at 24 kHz; onnx-asr resamples it to
    16 kHz internally.  Sample rates outside onnx-asr's supported set are
    resampled here with ``soxr`` (falling back to linear interpolation).
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    wave = np.ascontiguousarray(wave, dtype=np.float32)

    if sr not in NATIVE_SAMPLE_RATES:
        wave = resample(wave, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE
    return wave, sr


def resample(wave, src_sr: int, dst_sr: int):
    import numpy as np

    if src_sr == dst_sr:
        return wave
    try:
        import soxr

        return np.ascontiguousarray(soxr.resample(wave, src_sr, dst_sr), dtype=np.float32)
    except ImportError:  # pragma: no cover - soxr is in requirements-eval.lock
        n_out = int(round(len(wave) * dst_sr / src_sr))
        x_old = np.arange(len(wave), dtype=np.float64)
        x_new = np.linspace(0, len(wave) - 1, n_out)
        return np.interp(x_new, x_old, wave).astype(np.float32)


# ---------------------------------------------------------------------------
# transcriber
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    audio_path: str
    text: str
    segments: list[dict] = field(default_factory=list)
    model_id: str = ""
    onnx_asr_version: str = ""
    vad_model: str | None = None
    use_vad: bool = True
    provider: str = ""
    elapsed_sec: float = 0.0
    audio_duration_sec: float = 0.0
    sample_rate: int = 0
    rtf: float | None = None
    speed_x_realtime: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


class GigaAMTranscriber:
    """GigaAM-v3 + VAD transcriber. VAD is on by default and must stay on."""

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
        if model_id not in GIGAAM_V3_VARIANTS:
            # allow other onnx-asr names only for the optional secondary/audit ASR
            print(
                f"[asr_gigaam] WARNING: {model_id!r} is not one of the GigaAM-v3 "
                f"variants {GIGAAM_V3_VARIANTS}",
                file=sys.stderr,
            )
        ensure_hf_cache(cache_dir)
        self.model_id = model_id
        self.use_vad = use_vad
        self.device = device
        self.vad_model_name = vad_model if use_vad else None
        self.vad_options = dict(FROZEN_VAD_OPTIONS if vad_options is None else vad_options)
        self.vad_batch_size = vad_batch_size

        # PLAN.md §0.1: never open a CUDA session on somebody else's card.
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

    # -- API ---------------------------------------------------------------

    def transcribe(self, path: str | Path) -> TranscriptionResult:
        """Transcribe one audio file (wav/flac/...), VAD segments in time order."""
        path = str(path)
        try:
            wave, sr = read_audio(path)
        except Exception as exc:  # audio unreadable = empty hypothesis
            return TranscriptionResult(
                audio_path=path,
                text="",
                model_id=self.model_id,
                onnx_asr_version=self.onnx_asr_version,
                vad_model=self.vad_model_name,
                use_vad=self.use_vad,
                provider=self.provider,
                error=f"{type(exc).__name__}: {exc}",
            )
        return self.transcribe_waveform(wave, sr, label=path)

    def transcribe_slice(
        self, path: str | Path, start_sec: float | None = None, end_sec: float | None = None
    ) -> TranscriptionResult:
        """Transcribe ``[start_sec, end_sec)`` of an audio file.

        Used for the ASR error floor: the human reference of a benchmark text is
        often an offset range inside a longer recording.
        """
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
        """Transcribe a float32 mono waveform already in memory."""
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
            audio_path=path,
            text=text,
            segments=segments,
            model_id=self.model_id,
            onnx_asr_version=self.onnx_asr_version,
            vad_model=self.vad_model_name,
            use_vad=self.use_vad,
            provider=self.provider,
            elapsed_sec=elapsed,
            audio_duration_sec=duration,
            sample_rate=sr,
            rtf=(elapsed / duration) if duration > 0 else None,
            speed_x_realtime=(duration / elapsed) if elapsed > 0 else None,
            error=error,
        )

    def transcribe_many(self, paths: Iterable[str | Path]) -> list[TranscriptionResult]:
        return [self.transcribe(p) for p in paths]

    def vad_fingerprint(self) -> dict:
        """Stable identity of the VAD configuration, for cache keys.

        Anything that can move a transcription -- VAD on/off, VAD model, every
        frozen VAD option, the batch size -- goes into the hash.  The ASR model
        id and the normalization variant are *not* here; the caller adds them,
        so a cache key reads ``<text_id>|<model_id>|<variant>|<vad_hash>``.
        """
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
        d = {
            "model_id": self.model_id,
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
# CLI
# ---------------------------------------------------------------------------


def _iter_inputs(input_path: str, audio_key: str) -> list[dict]:
    p = Path(input_path)
    if p.suffix.lower() == ".jsonl":
        items = []
        with open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                items.append(obj)
        return items
    return [{audio_key: str(p)}]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GigaAM-v3 transcription (onnx-asr, VAD on)")
    ap.add_argument("--input", required=True, help="one audio file, or a .jsonl with audio paths")
    ap.add_argument("--output", required=True, help="output .jsonl")
    ap.add_argument("--audio-key", default="audio_path", help="key holding the audio path in jsonl input")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(GIGAAM_V3_VARIANTS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-vad", action="store_true", help="DEBUG ONLY - the protocol requires VAD")
    ap.add_argument("--vad-model", default=DEFAULT_VAD)
    ap.add_argument("--vad-batch-size", type=int, default=DEFAULT_VAD_BATCH_SIZE)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    items = _iter_inputs(args.input, args.audio_key)
    if args.limit:
        items = items[: args.limit]

    tr = GigaAMTranscriber(
        model_id=args.model,
        use_vad=not args.no_vad,
        device=args.device,
        vad_model=args.vad_model,
        vad_batch_size=args.vad_batch_size,
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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[asr_gigaam] {audio} dur={res.audio_duration_sec:.1f}s "
                f"elapsed={res.elapsed_sec:.1f}s x{(res.speed_x_realtime or 0):.1f} "
                f"segs={len(res.segments)} words={len(res.text.split())}"
                + (f" ERROR={res.error}" if res.error else ""),
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
