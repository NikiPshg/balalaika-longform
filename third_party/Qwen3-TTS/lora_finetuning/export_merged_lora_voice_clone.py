# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import logging
import os
import shutil
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LOGGER = logging.getLogger("export_merged_lora_voice_clone")
DEFAULT_BASE_MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_ADAPTER_PATH = "/home/tachka5090-1/kirill_workspace/Qwen3-TTS-streaming-custom/lora_finetuning/DO_NOT_REMOVE/checkpoint-step-240000"
DEFAULT_OUTPUT_DIR = "/home/tachka5090-1/kirill_workspace/Qwen3-TTS-12Hz-1.7B-RU-CallCenter"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_or_copy_missing_model_assets(model, base_model_path: str, output_dir: str) -> None:
    speech_tokenizer = getattr(model, "speech_tokenizer", None)
    speech_tokenizer_dir = os.path.join(output_dir, "speech_tokenizer")
    if speech_tokenizer is not None and hasattr(speech_tokenizer, "save_pretrained") and not os.path.exists(speech_tokenizer_dir):
        LOGGER.info("Saving loaded speech tokenizer to %s", speech_tokenizer_dir)
        speech_tokenizer.save_pretrained(speech_tokenizer_dir)

    if not os.path.isdir(base_model_path):
        try:
            from huggingface_hub import snapshot_download

            LOGGER.info("Downloading speech_tokenizer assets from base repo: %s", base_model_path)
            snapshot_dir = snapshot_download(base_model_path, allow_patterns=["config.json", "speech_tokenizer/*"])
            base_config = os.path.join(snapshot_dir, "config.json")
            if os.path.exists(base_config):
                LOGGER.info("Copying base config.json -> merged export")
                shutil.copy2(base_config, os.path.join(output_dir, "config.json"))
            src = os.path.join(snapshot_dir, "speech_tokenizer")
            dst = os.path.join(output_dir, "speech_tokenizer")
            if os.path.exists(src) and not os.path.exists(dst):
                LOGGER.info("Copying %s -> %s", src, dst)
                shutil.copytree(src, dst)
        except Exception as exc:
            LOGGER.warning("Could not copy speech_tokenizer assets from remote base model: %s", exc)
        return

    for name in ("speech_tokenizer",):
        src = os.path.join(base_model_path, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            LOGGER.info("Copying %s -> %s", src, dst)
            shutil.copytree(src, dst)

    for name in ("config.json", "generation_config.json"):
        src = os.path.join(base_model_path, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src) and (name == "config.json" or not os.path.exists(dst)):
            LOGGER.info("Copying %s -> %s", src, dst)
            shutil.copy2(src, dst)


def validate_export(output_dir: str) -> None:
    required = [
        "config.json",
        "generation_config.json",
        "speech_tokenizer/config.json",
    ]
    missing = [path for path in required if not os.path.exists(os.path.join(output_dir, path))]
    if missing:
        raise RuntimeError(f"Export is missing required file(s): {missing}")


def force_full_config_save(model) -> None:
    def strip_dtype_keys(value):
        if isinstance(value, dict):
            return {key: strip_dtype_keys(item) for key, item in value.items() if key != "dtype"}
        if isinstance(value, list):
            return [strip_dtype_keys(item) for item in value]
        return value

    def save_pretrained_full_config(self, save_directory, push_to_hub=False, **kwargs):
        if push_to_hub:
            raise ValueError("push_to_hub from patched config.save_pretrained is not supported during local export")
        os.makedirs(save_directory, exist_ok=True)
        config = strip_dtype_keys(self.to_dict())
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")

    type(model.config).save_pretrained = save_pretrained_full_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cpu", help="Use cpu to avoid GPU memory pressure while exporting.")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", default=None, help="Example: bitmanagerai/Qwen3-TTS-12Hz-1.7B-RU-VoiceClone")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit_message", default="Upload merged Qwen3-TTS LoRA checkpoint")
    args = parser.parse_args()
    configure_logging()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    LOGGER.info("Importing model dependencies")
    try:
        from peft import PeftModel
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise SystemExit("Install the Qwen3-TTS inference dependencies and peft before exporting.") from exc

    LOGGER.info("Loading base model: %s", args.base_model_path)
    tts = Qwen3TTSModel.from_pretrained(
        args.base_model_path,
        device_map=args.device,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_implementation,
    )

    LOGGER.info("Loading LoRA adapter: %s", args.adapter_path)
    peft_model = PeftModel.from_pretrained(tts.model, args.adapter_path)

    LOGGER.info("Merging LoRA adapter into base model")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    LOGGER.info("Saving merged model to %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    force_full_config_save(merged_model)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)

    LOGGER.info("Saving processor/tokenizer assets")
    tts.processor.save_pretrained(args.output_dir)
    save_or_copy_missing_model_assets(merged_model, args.base_model_path, args.output_dir)

    LOGGER.info("Validating exported files")
    validate_export(args.output_dir)

    if args.push_to_hub:
        if not args.repo_id:
            raise SystemExit("--repo_id is required when --push_to_hub is set")
        LOGGER.info("Uploading %s to Hugging Face repo %s", args.output_dir, args.repo_id)
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub or use `hf upload` manually.") from exc
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        HfApi().upload_folder(
            folder_path=args.output_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    LOGGER.info("Done. Ready-to-use merged checkpoint: %s", args.output_dir)


if __name__ == "__main__":
    main()
