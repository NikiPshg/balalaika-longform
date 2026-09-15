# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _force_eager_tokenizer_decode(tts) -> None:
    tokenizer_model = getattr(getattr(tts.model, "speech_tokenizer", None), "model", None)
    if tokenizer_model is None:
        return
    for module in tokenizer_model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--output_wav", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--no_merge_lora", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Cap generation length to avoid runaway decoding.",
    )
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if args.seed is not None:
        _set_seed(args.seed)

    try:
        import soundfile as sf
        from peft import PeftModel
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise SystemExit(
            "Qwen3-TTS inference dependencies are required. "
            "Install the project requirements, including soundfile, librosa, transformers, and torchaudio."
        ) from exc

    tts = Qwen3TTSModel.from_pretrained(
        args.base_model_path,
        device_map=args.device,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_implementation,
    )

    peft_model = PeftModel.from_pretrained(tts.model, args.adapter_path)
    tts.model = peft_model if args.no_merge_lora else peft_model.merge_and_unload()
    tts.model.eval()
    _force_eager_tokenizer_decode(tts)

    wavs, sr = tts.generate_voice_clone(
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output_wav))
    os.makedirs(output_dir, exist_ok=True)
    sf.write(args.output_wav, wavs[0], sr)


if __name__ == "__main__":
    main()
