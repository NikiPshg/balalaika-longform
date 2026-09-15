# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import logging
import os
import random
import re
import sys
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_BASE_MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_ADAPTER_PATH = "/home/tachka5090-1/kirill_workspace/Qwen3-TTS-streaming-custom/lora_finetuning/DO_NOT_REMOVE/checkpoint-step-240000"
DEFAULT_REF_AUDIO = "/home/tachka5090-1/kirill_workspace/Qwen3-TTS-streaming-custom/examples/ref_ru.wav"
DEFAULT_REF_TEXT = (
    "Понимаю, что у вас возникла ошибка на этапе регистрации с вводом номера телефона, "
    "и связь нестабильна. Сейчас я проверю состояние сети и уточню детали вашего подключения "
    "для решения проблемы."
)
DEFAULT_TEXT_SOURCES = [
    # "/workspace/Qwen3-TTS-streaming-custom/scripts/tts_bench_phrases_ru.txt",
    # "/workspace/Qwen3-TTS-streaming-custom/scripts/validation_cases/quality_cases.json",
    # "/workspace/Qwen3-TTS-streaming-custom/scripts/validation_cases/collector_channel1_cases.json",
    # "/workspace/Qwen3-TTS-streaming-custom/scripts/validation_cases/yo_e_pairs.json"
    "/home/tachka5090-1/kirill_workspace/Qwen3-TTS-streaming-custom/lora_finetuning/tts_bench_phrases_ru.txt"
]
LOGGER = logging.getLogger("batch_infer_lora_voice_clone")


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm.auto import tqdm

            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            super().emit(record)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_logging(level: str) -> None:
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))


def _force_eager_tokenizer_decode(tts) -> None:
    tokenizer_model = getattr(getattr(tts.model, "speech_tokenizer", None), "model", None)
    if tokenizer_model is None:
        return
    for module in tokenizer_model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def _slugify(value: str, max_len: int = 64) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", "_", value, flags=re.IGNORECASE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:max_len].strip("_") or "sample"


def _read_txt_source(path: str) -> List[Dict[str, Any]]:
    items = []
    section = None
    index = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                section = line.lstrip("#").strip() or None
                continue
            items.append(
                {
                    "id": f"{os.path.splitext(os.path.basename(path))[0]}_{index:04d}",
                    "text": line,
                    "source": path,
                    "source_type": "txt",
                    "section": section,
                }
            )
            index += 1
    return items


def _iter_json_text_items(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        for key in ("items", "cases", "data", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item


def _read_json_source(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    items = []
    basename = os.path.splitext(os.path.basename(path))[0]
    for index, item in enumerate(_iter_json_text_items(obj)):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        items.append(
            {
                "id": str(item.get("id") or f"{basename}_{index:04d}"),
                "text": text,
                "source": path,
                "source_type": "json",
                "lang": item.get("lang"),
                "expected_transcript": item.get("expected_transcript"),
                "required_tokens": item.get("required_tokens"),
                "metadata": item.get("metadata"),
            }
        )
    return items


def load_generation_items(paths: List[str], dedupe: bool = True) -> List[Dict[str, Any]]:
    items = []
    for path in paths:
        LOGGER.info("Reading text source: %s", path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            loaded = _read_txt_source(path)
        elif ext == ".json":
            loaded = _read_json_source(path)
        else:
            raise ValueError(f"Unsupported source extension for {path!r}; expected .txt or .json")
        items.extend(loaded)
        LOGGER.info("Loaded %d items from %s", len(loaded), path)

    if not dedupe:
        return items

    seen = set()
    deduped = []
    for item in items:
        key = re.sub(r"\s+", " ", item["text"]).strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    LOGGER.info("Deduplicated texts: %d -> %d", len(items), len(deduped))
    return deduped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--ref_audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref_text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--text_source", action="append", default=None, help="Text source path. Can be passed multiple times.")
    parser.add_argument("--output_dir", default="/home/tachka5090-1/kirill_workspace/Qwen3-TTS-streaming-custom/generated_new_lora")
    parser.add_argument("--manifest", default=None, help="Defaults to <output_dir>/manifest.jsonl.")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--no_merge_lora", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_duplicates", action="store_true")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    _configure_logging(args.log_level)

    LOGGER.info("Starting batch LoRA voice-clone inference")
    LOGGER.info("Base model: %s", args.base_model_path)
    LOGGER.info("LoRA adapter: %s", args.adapter_path)
    LOGGER.info("Reference audio: %s", args.ref_audio)
    LOGGER.info("Output directory: %s", args.output_dir)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if args.seed is not None:
        LOGGER.info("Setting random seed: %d", args.seed)
        _set_seed(args.seed)

    LOGGER.info("Importing inference dependencies")
    try:
        import soundfile as sf
        from peft import PeftModel
        from tqdm.auto import tqdm
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise SystemExit(
            "Qwen3-TTS inference dependencies are required. "
            "Install the project requirements, including soundfile, librosa, transformers, torchaudio, and tqdm."
        ) from exc

    text_sources = args.text_source or DEFAULT_TEXT_SOURCES
    LOGGER.info("Loading generation texts from %d source(s)", len(text_sources))
    items = load_generation_items(text_sources, dedupe=not args.keep_duplicates)
    if args.start_index:
        LOGGER.info("Applying start index: %d", args.start_index)
        items = items[args.start_index :]
    if args.limit is not None:
        LOGGER.info("Applying limit: %d", args.limit)
        items = items[: args.limit]
    if not items:
        raise SystemExit("No generation texts were loaded.")
    LOGGER.info("Prepared %d text item(s) for generation", len(items))

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.output_dir, "manifest.jsonl")
    LOGGER.info("Manifest path: %s", manifest_path)
    LOGGER.info("Manifest mode: %s", "overwrite" if args.overwrite else "append")

    LOGGER.info(
        "Loading base model with device=%s dtype=%s attn_implementation=%s",
        args.device,
        args.dtype,
        args.attn_implementation,
    )
    tts = Qwen3TTSModel.from_pretrained(
        args.base_model_path,
        device_map=args.device,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_implementation,
    )
    LOGGER.info("Base model loaded")

    LOGGER.info("Loading LoRA adapter")
    peft_model = PeftModel.from_pretrained(tts.model, args.adapter_path)
    LOGGER.info("LoRA adapter loaded")
    if args.no_merge_lora:
        LOGGER.info("Keeping LoRA adapter as PEFT wrapper")
    else:
        LOGGER.info("Merging LoRA adapter into base model")
    tts.model = peft_model if args.no_merge_lora else peft_model.merge_and_unload()
    tts.model.eval()
    LOGGER.info("Model set to eval mode")
    LOGGER.info("Forcing speech tokenizer decode attention to eager where applicable")
    _force_eager_tokenizer_decode(tts)
    LOGGER.info("Model is ready for generation")

    written = 0
    with open(manifest_path, "a" if not args.overwrite else "w", encoding="utf-8") as manifest_f:
        progress = tqdm(
            enumerate(items),
            total=len(items),
            desc="Generating wavs",
            unit="file",
            dynamic_ncols=True,
        )
        for offset, item in progress:
            global_index = args.start_index + offset
            wav_name = f"{global_index:04d}_{_slugify(item['id'])}.wav"
            output_wav = os.path.join(args.output_dir, wav_name)
            progress.set_postfix_str(wav_name[:48])
            if os.path.exists(output_wav) and not args.overwrite:
                LOGGER.info("Skipping existing file: %s", output_wav)
                continue

            LOGGER.info(
                "[%d/%d] Generating %s from source=%s id=%s",
                offset + 1,
                len(items),
                wav_name,
                item["source"],
                item["id"],
            )
            LOGGER.debug("Generation text: %s", item["text"])
            wavs, sr = tts.generate_voice_clone(
                text=item["text"],
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                language=args.language,
                max_new_tokens=args.max_new_tokens,
            )
            LOGGER.info("[%d/%d] Writing wav: %s", offset + 1, len(items), output_wav)
            sf.write(output_wav, wavs[0], sr)
            manifest_f.write(
                json.dumps(
                    {
                        "index": global_index,
                        "output_wav": output_wav,
                        "sample_rate": sr,
                        "text": item["text"],
                        "id": item["id"],
                        "source": item["source"],
                        "source_type": item["source_type"],
                        "section": item.get("section"),
                        "lang": item.get("lang"),
                        "expected_transcript": item.get("expected_transcript"),
                        "required_tokens": item.get("required_tokens"),
                        "metadata": item.get("metadata"),
                        "adapter_path": args.adapter_path,
                        "base_model_path": args.base_model_path,
                        "ref_audio": args.ref_audio,
                        "ref_text": args.ref_text,
                        "language": args.language,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest_f.flush()
            LOGGER.info("[%d/%d] Finished %s at %d Hz", offset + 1, len(items), wav_name, sr)
            written += 1

    LOGGER.info("Done. Generated %d file(s). Manifest: %s", written, manifest_path)


if __name__ == "__main__":
    main()
