#!/usr/bin/env python3
"""Generate reproducible Russian voice-clone listening probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download
from silero_stress import load_accentor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_finetuning.full_utterance_utils import apply_silero_stress  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--voices-repo", default="bitmanagerai/tts-voices")
    parser.add_argument("--voices-revision", required=True)
    parser.add_argument("--voice", action="append", required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2608)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--target-stress-mode",
        choices=("silero", "none"),
        default="silero",
    )
    parser.add_argument(
        "--reference-stress-mode",
        choices=("as-is", "strip"),
        default="as-is",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_accentor_safely() -> Any:
    grad_enabled = torch.is_grad_enabled()
    try:
        with torch.no_grad():
            return load_accentor()
    finally:
        torch.set_grad_enabled(grad_enabled)


def force_eager_tokenizer_decode(tts: Qwen3TTSModel) -> None:
    tokenizer_model = getattr(getattr(tts.model, "speech_tokenizer", None), "model", None)
    if tokenizer_model is None:
        return
    for module in tokenizer_model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def voice_directory(
    repo_id: str,
    revision: str,
    voice_id: str,
) -> str:
    suffix = f"/{voice_id}/voice.json"
    matches = [
        name
        for name in HfApi().list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one enabled voice path for {voice_id!r}, got {matches}")
    return str(Path(matches[0]).parent).replace("\\", "/")


def prepare_voices(args: argparse.Namespace) -> list[dict[str, Any]]:
    voices: list[dict[str, Any]] = []
    refs_dir = args.output_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for voice_id in args.voice:
        directory = voice_directory(
            args.voices_repo,
            args.voices_revision,
            voice_id,
        )
        json_name = f"{directory}/voice.json"
        wav_name = f"{directory}/reference.wav"
        json_path = Path(
            hf_hub_download(
                args.voices_repo,
                json_name,
                repo_type="dataset",
                revision=args.voices_revision,
            )
        )
        wav_path = Path(
            hf_hub_download(
                args.voices_repo,
                wav_name,
                repo_type="dataset",
                revision=args.voices_revision,
            )
        )
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if not metadata.get("enabled", False):
            raise ValueError(f"Voice is disabled: {voice_id}")
        if metadata.get("ref_text_stress") != "manual" or "+" not in metadata.get(
            "ref_text", ""
        ):
            raise ValueError(f"Expected manual stressed ref_text for {voice_id}")
        copied_wav = refs_dir / f"{voice_id}.wav"
        copied_json = refs_dir / f"{voice_id}.json"
        shutil.copy2(wav_path, copied_wav)
        shutil.copy2(json_path, copied_json)
        info = sf.info(copied_wav)
        source_ref_text = str(metadata["ref_text"])
        ref_text = (
            source_ref_text.replace("+", "").replace("\u0301", "")
            if args.reference_stress_mode == "strip"
            else source_ref_text
        )
        voices.append(
            {
                "voice_id": voice_id,
                "directory": directory,
                "audio_path": str(copied_wav.resolve()),
                "source_ref_text": source_ref_text,
                "ref_text": ref_text,
                "display_name": metadata["display_name"],
                "reference_sample_rate": info.samplerate,
                "reference_duration_sec": info.duration,
            }
        )
    return voices


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    voices = prepare_voices(args)
    accentor = load_accentor_safely()
    texts = []
    for index, source_text in enumerate(args.text, start=1):
        model_text = (
            apply_silero_stress(accentor, source_text)
            if args.target_stress_mode == "silero"
            else source_text.strip()
        )
        texts.append(
            {
                "index": index,
                "source": source_text,
                "model_text": model_text,
            }
        )
    model_kwargs: dict[str, Any] = {
        "device_map": args.device,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision

    print(f"loading {args.model_label}: {args.model_path}", flush=True)
    tts = Qwen3TTSModel.from_pretrained(args.model_path, **model_kwargs)
    tts.model.eval()
    force_eager_tokenizer_decode(tts)

    records: list[dict[str, Any]] = []
    for voice_index, voice in enumerate(voices):
        prompt = tts.create_voice_clone_prompt(
            ref_audio=voice["audio_path"],
            ref_text=voice["ref_text"],
            x_vector_only_mode=False,
        )
        for text_item in texts:
            sample_seed = args.seed + voice_index * 100 + text_item["index"]
            set_seed(sample_seed)
            started = time.perf_counter()
            wavs, sample_rate = tts.generate_voice_clone(
                text=text_item["model_text"],
                language="Russian",
                voice_clone_prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                subtalker_dosample=True,
                subtalker_top_k=args.top_k,
                subtalker_top_p=args.top_p,
                subtalker_temperature=args.temperature,
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            wav = np.asarray(wavs[0], dtype=np.float32)
            if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
                raise RuntimeError(
                    f"Invalid generated waveform for {voice['voice_id']} text {text_item['index']}"
                )
            filename = f"{voice['voice_id']}__text-{text_item['index']:02d}.wav"
            output_path = args.output_dir / filename
            sf.write(output_path, wav, sample_rate, subtype="PCM_16")
            record = {
                "model_label": args.model_label,
                "model_path": args.model_path,
                "model_revision": args.model_revision,
                "voice_id": voice["voice_id"],
                "display_name": voice["display_name"],
                "reference_audio": voice["audio_path"],
                "reference_sample_rate": voice["reference_sample_rate"],
                "reference_duration_sec": voice["reference_duration_sec"],
                "source_ref_text": voice["source_ref_text"],
                "ref_text": voice["ref_text"],
                "text_index": text_item["index"],
                "source_text": text_item["source"],
                "model_text": text_item["model_text"],
                "seed": sample_seed,
                "output_wav": str(output_path.resolve()),
                "sample_rate": sample_rate,
                "duration_sec": float(wav.size / sample_rate),
                "generation_seconds": elapsed,
            }
            records.append(record)
            print(
                f"generated {filename}: {record['duration_sec']:.2f}s "
                f"in {elapsed:.2f}s",
                flush=True,
            )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "voices_repo": args.voices_repo,
        "voices_revision": args.voices_revision,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "target_stress_mode": args.target_stress_mode,
            "reference_stress_mode": args.reference_stress_mode,
        },
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
