# coding=utf-8
"""Shared teacher-forcing and validation helpers for the main Talker.

Prepared rows contain one full transcript and one full ``[T, 16]`` codec
sequence. A deterministic acoustic prefix acts as the ICL reference; the loss
is codec-0 continuation cross entropy only. The 15-codebook predictor, speech
tokenizer, speaker encoder and every non-Talker module stay frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_finetuning.full_utterance_utils import (  # noqa: E402
    apply_silero_stress,
    build_assistant_text,
    build_ref_text,
    dotenv_value,
    str2bool,
    word_error_rate,
)


DEFAULT_TALKER_TARGET_REGEX = (
    r"^(talker\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))|talker\.codec_head)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-utterance main-Talker LoRA SFT")
    parser.add_argument("--model_path", default="bitmanagerai/Qwen3TTS-RL-2")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tensorboard_dir", required=True)
    parser.add_argument(
        "--validation_texts_file",
        default=str(Path(__file__).with_name("validation_main_talker_ru.txt")),
    )
    parser.add_argument("--hf_token_env", default="HF_TOKEN")
    parser.add_argument("--env_file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_regex", default=DEFAULT_TALKER_TARGET_REGEX)
    parser.add_argument("--resume_adapter", default=None)
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--validation_every_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_every_steps", type=int, default=100)
    parser.add_argument("--generate_at_start", type=str2bool, default=True)
    parser.add_argument("--eval_max_batches", type=int, default=8)
    parser.add_argument("--validation_references", type=int, default=2)
    parser.add_argument("--validation_texts", type=int, default=4)
    parser.add_argument("--validation_generation_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=str2bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--subtalker_dosample", type=str2bool, default=True)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--asr_enabled", type=str2bool, default=True)
    parser.add_argument("--asr_model_name", default="gigaam-v3-e2e-ctc")
    parser.add_argument(
        "--asr_model_path",
        default=str(REPO_ROOT / "artifacts/models/gigaam-v3-e2e-ctc-onnx"),
    )
    parser.add_argument(
        "--asr_trt_cache_dir",
        default=str(REPO_ROOT / "artifacts/tensorrt/gigaam-v3-e2e-ctc"),
    )
    parser.add_argument("--asr_device_id", type=int, default=0)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[main-talker-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def jsonl_offsets(path: str | Path) -> list[int]:
    offsets: list[int] = []
    with Path(path).open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def first_jsonl_rows(path: str | Path, count: int) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in islice((line for line in source if line.strip()), count)]


def normalize_ids(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    while tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {tuple(tensor.shape)}")
    return tensor


def normalize_codes(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[1] != 16:
        raise ValueError(f"{name} must be [T,16], got {tuple(tensor.shape)}")
    return tensor


class FullUtteranceDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.offsets = jsonl_offsets(path)
        self._file = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.offsets)

    def row(self, index: int) -> dict[str, Any]:
        if self._file is None:
            self._file = open(self.path, "r", encoding="utf-8")
        self._file.seek(self.offsets[index])
        return json.loads(self._file.readline())

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.row(index)
        codes = normalize_codes(row["full_codes"], "full_codes")
        prefix_frames = int(row["prefix_frames"])
        if not 0 < prefix_frames < codes.shape[0]:
            raise ValueError(f"Invalid prefix_frames={prefix_frames} for T={codes.shape[0]}")
        return {
            "text_ids": normalize_ids(row["text_ids"], "text_ids"),
            "full_codes": codes,
            "speaker_embedding": torch.as_tensor(row["speaker_embedding"], dtype=torch.float32).reshape(-1),
            "prefix_frames": prefix_frames,
            "language": row.get("language", "Russian") or "Russian",
            "text": row.get("text", ""),
            "key": row.get("source_key", str(index)),
            "index": index,
        }

    @staticmethod
    def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "text_ids": pad_sequence([item["text_ids"] for item in batch], batch_first=True, padding_value=0),
            "text_lengths": torch.tensor([item["text_ids"].numel() for item in batch], dtype=torch.long),
            "full_codes": pad_sequence([item["full_codes"] for item in batch], batch_first=True, padding_value=0),
            "code_lengths": torch.tensor([item["full_codes"].shape[0] for item in batch], dtype=torch.long),
            "speaker_embeddings": torch.stack([item["speaker_embedding"] for item in batch]),
            "prefix_frames": torch.tensor([item["prefix_frames"] for item in batch], dtype=torch.long),
            "languages": [item["language"] for item in batch],
            "texts": [item["text"] for item in batch],
            "keys": [item["key"] for item in batch],
            "indices": [item["index"] for item in batch],
        }


def core_model(model: Any) -> Any:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def text_embeddings(talker: Any, ids: torch.Tensor) -> torch.Tensor:
    embeddings = talker.get_text_embeddings()(ids)
    if hasattr(talker, "text_projection"):
        embeddings = talker.text_projection(embeddings)
    return embeddings


def code_embeddings(talker: Any, codes: torch.Tensor) -> torch.Tensor:
    pieces = []
    for codebook in range(codes.shape[1]):
        code_slice = codes[:, codebook : codebook + 1]
        if codebook == 0:
            pieces.append(talker.get_input_embeddings()(code_slice))
        else:
            pieces.append(talker.code_predictor.get_input_embeddings()[codebook - 1](code_slice))
    return torch.cat(pieces, dim=1).sum(1).unsqueeze(0)


def language_id_for(config: Any, language: str) -> int | None:
    if not language or language.lower() == "auto":
        return None
    key = language.lower()
    if key not in config.talker_config.codec_language_id:
        raise ValueError(f"Unsupported language {language!r}")
    return int(config.talker_config.codec_language_id[key])


def build_main_talker_teacher_sample(
    model: Any,
    *,
    text_ids: torch.Tensor,
    full_codes: torch.Tensor,
    prefix_frames: int,
    speaker_embedding: torch.Tensor,
    language: str,
) -> dict[str, torch.Tensor]:
    """Mirror vLLM-Omni Base's streaming ICL prompt, with combined full text."""
    base = core_model(model)
    talker = base.talker
    device = next(talker.parameters()).device
    embed_dtype = talker.get_input_embeddings().weight.dtype
    input_ids = text_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    codes = full_codes.to(device=device, dtype=torch.long)
    ref_codes = codes[:prefix_frames]
    continuation_codes = codes[prefix_frames:]
    speaker_embedding = speaker_embedding.to(device=device, dtype=embed_dtype)

    tts_bos_embed, tts_eos_embed, tts_pad_embed = text_embeddings(
        talker,
        torch.tensor(
            [[base.config.tts_bos_token_id, base.config.tts_eos_token_id, base.config.tts_pad_token_id]],
            device=device,
            dtype=torch.long,
        ),
    ).chunk(3, dim=1)
    language_id = language_id_for(base.config, language)
    if language_id is None:
        prefill_ids = [
            base.config.talker_config.codec_nothink_id,
            base.config.talker_config.codec_think_bos_id,
            base.config.talker_config.codec_think_eos_id,
        ]
    else:
        prefill_ids = [
            base.config.talker_config.codec_think_id,
            base.config.talker_config.codec_think_bos_id,
            language_id,
            base.config.talker_config.codec_think_eos_id,
        ]
    codec_prefix_0 = talker.get_input_embeddings()(
        torch.tensor([prefill_ids], device=device, dtype=torch.long)
    )
    codec_prefix_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[base.config.talker_config.codec_pad_id, base.config.talker_config.codec_bos_id]],
            device=device,
            dtype=torch.long,
        )
    )
    codec_prefix = torch.cat(
        [codec_prefix_0, speaker_embedding.view(1, 1, -1), codec_prefix_1], dim=1
    )
    role_embed = text_embeddings(talker, input_ids[:, :3])
    prompt_codec_side = torch.cat(
        [tts_pad_embed.expand(-1, codec_prefix.shape[1] - 2, -1), tts_bos_embed], dim=1
    ) + codec_prefix[:, :-1]
    prompt = torch.cat([role_embed, prompt_codec_side], dim=1)

    # ref_id + text_id are concatenated by generate_icl_prompt. Passing the
    # complete transcript as text_id is equivalent to the ref+target text seen
    # at Base inference, without inventing a text boundary or using timestamps.
    icl_embed, trailing_text = base.generate_icl_prompt(
        text_id=input_ids[:, 3:-5],
        ref_id=torch.empty((1, 0), device=device, dtype=torch.long),
        ref_code=ref_codes,
        tts_pad_embed=tts_pad_embed,
        tts_eos_embed=tts_eos_embed,
        non_streaming_mode=False,
    )
    prompt = torch.cat([prompt, icl_embed], dim=1)
    prompt_length = int(prompt.shape[1])

    continuation_embed = code_embeddings(talker, continuation_codes)
    text_parts = [
        trailing_text[:, frame : frame + 1] if frame < trailing_text.shape[1] else tts_pad_embed
        for frame in range(continuation_codes.shape[0])
    ]
    continuation_embed = continuation_embed + torch.cat(text_parts, dim=1)
    eos_codes = torch.zeros((1, 16), device=device, dtype=torch.long)
    eos_codes[:, 0] = base.config.talker_config.codec_eos_token_id
    eos_embed = code_embeddings(talker, eos_codes) + tts_pad_embed
    inputs_embeds = torch.cat([prompt, continuation_embed, eos_embed], dim=1).squeeze(0)

    labels = torch.full((inputs_embeds.shape[0],), -100, device=device, dtype=torch.long)
    labels[prompt_length : prompt_length + continuation_codes.shape[0]] = continuation_codes[:, 0]
    labels[prompt_length + continuation_codes.shape[0]] = base.config.talker_config.codec_eos_token_id
    return {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "attention_mask": torch.ones(inputs_embeds.shape[0], device=device, dtype=torch.long),
    }


def compute_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    return_token_metrics: bool = False,
    return_loss_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    if return_token_metrics and return_loss_components:
        raise ValueError("Request either token metrics or loss components, not both")
    samples = []
    for index in range(batch["text_ids"].shape[0]):
        text_length = int(batch["text_lengths"][index])
        code_length = int(batch["code_lengths"][index])
        samples.append(
            build_main_talker_teacher_sample(
                model,
                text_ids=batch["text_ids"][index, :text_length],
                full_codes=batch["full_codes"][index, :code_length],
                prefix_frames=int(batch["prefix_frames"][index]),
                speaker_embedding=batch["speaker_embeddings"][index],
                language=batch["languages"][index],
            )
        )
    inputs_embeds = pad_sequence(
        [sample["inputs_embeds"] for sample in samples], batch_first=True, padding_value=0.0
    )
    labels = pad_sequence([sample["labels"] for sample in samples], batch_first=True, padding_value=-100)
    attention_mask = pad_sequence(
        [sample["attention_mask"] for sample in samples], batch_first=True, padding_value=0
    )
    outputs = core_model(model).talker(
        inputs_embeds=inputs_embeds[:, :-1],
        attention_mask=attention_mask[:, :-1],
        labels=None,
        output_hidden_states=False,
        use_cache=False,
    )
    targets = labels[:, 1:]
    flat_logits = outputs.logits.reshape(-1, outputs.logits.shape[-1])
    flat_targets = targets.reshape(-1)
    token_losses = F.cross_entropy(
        flat_logits, flat_targets, ignore_index=-100, reduction="none"
    ).reshape_as(targets)
    valid_by_sample = targets.ne(-100)
    valid_token_count = valid_by_sample.sum()
    loss_sum = token_losses.sum()
    if return_loss_components:
        return loss_sum, {"token_count": valid_token_count.detach()}
    loss = loss_sum / valid_token_count.clamp_min(1)
    if not return_token_metrics:
        return loss
    with torch.no_grad():
        valid = valid_by_sample.reshape(-1)
        valid_logits = flat_logits[valid]
        valid_targets = flat_targets[valid]
        topk = valid_logits.topk(k=min(5, valid_logits.shape[-1]), dim=-1).indices
        top1_correct = topk[:, 0].eq(valid_targets).sum()
        top5_correct = topk.eq(valid_targets.unsqueeze(-1)).any(dim=-1).sum()
        predictions = outputs.logits.topk(
            k=min(5, outputs.logits.shape[-1]), dim=-1
        ).indices
        top1_by_sample = (
            predictions[..., 0].eq(targets) & valid_by_sample
        ).sum(dim=1)
        top5_by_sample = (
            predictions.eq(targets.unsqueeze(-1)).any(dim=-1) & valid_by_sample
        ).sum(dim=1)
        per_sample_metrics = torch.stack(
            (
                token_losses.sum(dim=1),
                valid_by_sample.sum(dim=1),
                top1_by_sample,
                top5_by_sample,
            ),
            dim=1,
        ).float()
    return loss, {
        "token_count": int(valid_targets.numel()),
        "top1_correct": int(top1_correct.item()),
        "top5_correct": int(top5_correct.item()),
        "per_sample_metrics": per_sample_metrics,
    }


@torch.no_grad()
def evaluate_loss(model: Any, dataloader: DataLoader, accelerator: Any, max_batches: int) -> float | None:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(dataloader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        loss = compute_loss(model, batch)
        gathered = accelerator.gather_for_metrics(loss.detach().reshape(1))
        losses.append(gathered)
    model.train()
    return float(torch.cat(losses).mean().item()) if losses else None


def force_eager_tokenizer_decode(model: Any) -> None:
    tokenizer_model = getattr(getattr(core_model(model), "speech_tokenizer", None), "model", None)
    if tokenizer_model is None:
        return
    for module in tokenizer_model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def tokenize(processor: Any, text: str) -> torch.Tensor:
    ids = processor(text=text, return_tensors="pt", padding=False)["input_ids"]
    return ids if ids.ndim == 2 else ids.unsqueeze(0)


def validation_texts(path: str | Path, count: int, accentor: Any) -> list[str]:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"No validation texts in {path}")
    return [apply_silero_stress(accentor, text) for text in lines[:count]]


def select_diverse_validation_references(
    rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select acoustically diverse references without reading speaker IDs."""
    count = min(max(0, int(count)), len(rows))
    if count == 0:
        return []
    if count == len(rows):
        return list(rows)
    embeddings = torch.stack(
        [torch.as_tensor(row["speaker_embedding"], dtype=torch.float32).reshape(-1) for row in rows]
    )
    embeddings = F.normalize(embeddings, p=2, dim=1)
    distances = 1.0 - embeddings @ embeddings.T
    if count == 1:
        centroid = F.normalize(embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
        first = int((1.0 - embeddings @ centroid.T).squeeze(1).argmax().item())
        return [rows[first]]

    pair_distances = distances.clone()
    pair_distances.fill_diagonal_(-1.0)
    flat_pair = int(pair_distances.argmax().item())
    width = int(pair_distances.shape[1])
    selected = [flat_pair // width, flat_pair % width]
    while len(selected) < count:
        distance_to_selected = distances[:, selected].amin(dim=1)
        distance_to_selected[selected] = -1.0
        selected.append(int(distance_to_selected.argmax().item()))
    return [rows[index] for index in selected]


class GigaAMValidator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        import onnx_asr
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"ONNX Runtime CUDA provider unavailable: {available}")
        cuda_options = {
            "device_id": self.args.asr_device_id,
            "arena_extend_strategy": "kSameAsRequested",
        }
        providers = [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]
        log(f"loading {self.args.asr_model_name} with CUDA -> CPU")
        self.model = onnx_asr.load_model(
            self.args.asr_model_name,
            path=self.args.asr_model_path,
            providers=providers,
        )
        return self.model

    def recognize(self, wav: np.ndarray, sample_rate: int) -> str:
        return str(self.load().recognize(np.asarray(wav, dtype=np.float32), sample_rate=sample_rate))


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = re.sub(r"\s+", "", reference.lower().replace("+", "").replace("ё", "е"))
    hyp = re.sub(r"\s+", "", hypothesis.lower().replace("+", "").replace("ё", "е"))
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for ref_index, ref_char in enumerate(ref, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hyp, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return float(previous[-1] / len(ref))


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.no_grad()
def generate_validation(
    *,
    model: Any,
    processor: Any,
    val_rows: Sequence[dict[str, Any]],
    texts: Sequence[str],
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    writer: Any,
    asr_validator: GigaAMValidator | None,
) -> dict[str, float]:
    import soundfile as sf

    base = core_model(model)
    base.eval()
    force_eager_tokenizer_decode(base)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = select_diverse_validation_references(
        val_rows, args.validation_references
    )
    log(
        "acoustically diverse validation references: "
        + ", ".join(str(row.get("source_key")) for row in references)
    )
    pairs = [(ref_index, text_index, row, text) for ref_index, row in enumerate(references) for text_index, text in enumerate(texts)]
    manifest_path = output_dir / "manifest.jsonl"
    wers: list[float] = []
    cers: list[float] = []
    durations: list[float] = []
    question_wers: list[float] = []
    question_cers: list[float] = []
    question_durations: list[float] = []
    statement_wers: list[float] = []
    statement_cers: list[float] = []
    statement_durations: list[float] = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for validation_batch_index, pair_batch in enumerate(
            chunks(pairs, max(1, args.validation_generation_batch_size))
        ):
            ref_codes = [normalize_codes(pair[2]["full_codes"], "full_codes").to(base.talker.device) for pair in pair_batch]
            speaker_embeddings = [
                torch.as_tensor(pair[2]["speaker_embedding"], dtype=base.talker.dtype, device=base.talker.device)
                for pair in pair_batch
            ]
            input_ids = [tokenize(processor, build_assistant_text(pair[3])).to(base.talker.device) for pair in pair_batch]
            ref_ids = [tokenize(processor, build_ref_text(pair[2]["text"])).to(base.talker.device) for pair in pair_batch]
            prompt = {
                "ref_code": ref_codes,
                "ref_spk_embedding": speaker_embeddings,
                "x_vector_only_mode": [False] * len(pair_batch),
                "icl_mode": [True] * len(pair_batch),
            }
            rng_devices = []
            if base.talker.device.type == "cuda":
                rng_devices = [base.talker.device.index or 0]
            # Use the same sampling stream at every checkpoint while preserving
            # the training RNG state. Listening comparisons then reflect the
            # adapter update rather than unrelated validation sampling noise.
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(args.seed + validation_batch_index)
                generated, _ = base.generate(
                    input_ids=input_ids,
                    ref_ids=ref_ids,
                    voice_clone_prompt=prompt,
                    languages=[pair[2].get("language", "Russian") for pair in pair_batch],
                    non_streaming_mode=False,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    subtalker_dosample=args.subtalker_dosample,
                    subtalker_temperature=args.subtalker_temperature,
                    subtalker_top_k=args.subtalker_top_k,
                    subtalker_top_p=args.subtalker_top_p,
                )
            combined = [torch.cat([ref, continuation.to(ref.device)], dim=0) for ref, continuation in zip(ref_codes, generated, strict=True)]
            wavs, sample_rate = base.speech_tokenizer.decode([{"audio_codes": codes} for codes in combined])
            for pair, ref, continuation, combined_codes, full_wav in zip(
                pair_batch, ref_codes, generated, combined, wavs, strict=True
            ):
                ref_index, text_index, row, target_text = pair
                full_wav = np.asarray(full_wav, dtype=np.float32)
                cut = round(int(ref.shape[0]) / max(1, int(combined_codes.shape[0])) * len(full_wav))
                wav = full_wav[cut:]
                filename = f"ref_{ref_index:02d}_text_{text_index:02d}.wav"
                wav_path = output_dir / filename
                sf.write(wav_path, wav, sample_rate)
                duration = float(len(wav) / sample_rate)
                durations.append(duration)
                is_question = "?" in target_text
                (question_durations if is_question else statement_durations).append(duration)
                tag = f"validation/ref_{ref_index:02d}/text_{text_index:02d}"
                writer.add_audio(tag, torch.from_numpy(wav).unsqueeze(0), step, sample_rate=sample_rate)
                writer.add_text(f"{tag}/target", target_text, step)
                hypothesis = None
                wer_value = None
                cer_value = None
                if asr_validator is not None and wav.size:
                    try:
                        hypothesis = asr_validator.recognize(wav, sample_rate)
                        wer_value = word_error_rate(target_text, hypothesis, drop_fillers=False)
                        cer_value = character_error_rate(target_text, hypothesis)
                        wers.append(wer_value)
                        cers.append(cer_value)
                        (question_wers if is_question else statement_wers).append(wer_value)
                        (question_cers if is_question else statement_cers).append(cer_value)
                        writer.add_text(f"{tag}/gigaam_hypothesis", hypothesis, step)
                    except Exception as exc:
                        log(f"ASR validation failed for {filename}: {exc}")
                manifest.write(
                    json.dumps(
                        {
                            "step": step,
                            "audio": filename,
                            "reference_key": row.get("source_key"),
                            "reference_text": row.get("text"),
                            "reference_source_text": row.get("source_text"),
                            "reference_transcript_source": row.get("transcript_source"),
                            "reference_gigaam_e2e_rover_wer": row.get(
                                "gigaam_e2e_rover_wer"
                            ),
                            "target_text": target_text,
                            "is_question": is_question,
                            "gigaam_hypothesis": hypothesis,
                            "wer": wer_value,
                            "cer": cer_value,
                            "duration_sec": duration,
                            "generated_frames": int(continuation.shape[0]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    metrics = {
        "validation/audio_count": float(len(durations)),
        "validation/duration_mean": float(np.mean(durations)) if durations else 0.0,
    }
    if wers:
        metrics["validation/gigaam_wer"] = float(np.mean(wers))
    if cers:
        metrics["validation/gigaam_cer"] = float(np.mean(cers))
    for group, group_wers, group_cers, group_durations in (
        ("question", question_wers, question_cers, question_durations),
        ("statement", statement_wers, statement_cers, statement_durations),
    ):
        metrics[f"validation/{group}_audio_count"] = float(len(group_durations))
        metrics[f"validation/{group}_duration_mean"] = (
            float(np.mean(group_durations)) if group_durations else 0.0
        )
        if group_wers:
            metrics[f"validation/{group}_gigaam_wer"] = float(np.mean(group_wers))
        if group_cers:
            metrics[f"validation/{group}_gigaam_cer"] = float(np.mean(group_cers))
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)
    writer.flush()
    base.train()
    return metrics


def validate_trainable_scope(model: Any) -> tuple[int, int]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    forbidden = [
        name
        for name, _ in trainable
        if any(token in name for token in ("code_predictor", "speech_tokenizer", "speaker_encoder"))
    ]
    if forbidden:
        raise RuntimeError(f"Non-main-Talker parameters became trainable: {forbidden[:10]}")
    unexpected = [
        name
        for name, _ in trainable
        if "talker.model.layers" not in name and "talker.codec_head" not in name
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:10]}")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    return trainable_count, total_count


def save_checkpoint(accelerator: Any, model: Any, output_dir: Path, step: int, args: argparse.Namespace) -> Path:
    checkpoint = output_dir / f"checkpoint-step-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(checkpoint, safe_serialization=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return checkpoint


def scheduler_for(optimizer: Any, warmup_steps: int, total_steps: int) -> LambdaLR:
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return LambdaLR(optimizer, scale)


def main() -> None:
    args = parse_args()
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from silero_stress import load_accentor
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.auto import tqdm

    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    token = os.environ.get(args.hf_token_env) or dotenv_value(args.env_file, args.hf_token_env)
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    log(f"loading {args.model_path} as Base ({dtype}, {args.attn_implementation})")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=str(accelerator.device) if accelerator.device.type == "cuda" else None,
        attn_implementation=args.attn_implementation,
        token=token,
    )
    qwen3tts.model.config.tts_model_type = "base"
    qwen3tts.model.tts_model_type = "base"
    if args.resume_adapter:
        model = PeftModel.from_pretrained(qwen3tts.model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            qwen3tts.model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=args.lora_target_regex,
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    qwen3tts.model = model
    trainable_count, total_count = validate_trainable_scope(model)
    log(
        f"trainable main-Talker params={trainable_count:,}/{total_count:,} "
        f"({100 * trainable_count / total_count:.3f}%); code_predictor is frozen"
    )

    train_dataset = FullUtteranceDataset(args.train_jsonl)
    val_dataset = FullUtteranceDataset(args.val_jsonl)
    if not train_dataset or not val_dataset:
        raise ValueError("Training and validation JSONL files must be non-empty")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=FullUtteranceDataset.collate,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=FullUtteranceDataset.collate,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps or (steps_per_epoch * args.epochs)
    scheduler = scheduler_for(optimizer, args.warmup_steps, total_steps)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    output_dir = Path(args.output_dir)
    tensorboard_dir = Path(args.tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if accelerator.is_main_process else None
    # silero_stress.load_accentor() currently calls torch.set_grad_enabled(False)
    # globally. Keep that implementation detail contained so LoRA backward is
    # not silently disabled for the rest of the training process.
    if accelerator.is_main_process:
        with torch.no_grad():
            accentor = load_accentor()
    else:
        accentor = None
    if not torch.is_grad_enabled():
        raise RuntimeError("Gradient mode was disabled while initializing validation helpers")
    fixed_texts = validation_texts(args.validation_texts_file, args.validation_texts, accentor) if accentor else []
    val_rows = first_jsonl_rows(args.val_jsonl, args.validation_references)
    asr_validator = GigaAMValidator(args) if args.asr_enabled and accelerator.is_main_process else None
    global_step = int(args.resume_step)

    if args.generate_at_start and accelerator.is_main_process:
        log("generating TensorBoard audio at step 0")
        unwrapped = accelerator.unwrap_model(model)
        metrics = generate_validation(
            model=unwrapped,
            processor=qwen3tts.processor,
            val_rows=val_rows,
            texts=fixed_texts,
            output_dir=output_dir / f"checkpoint-step-{global_step}" / "validation_audio",
            step=global_step,
            args=args,
            writer=writer,
            asr_validator=asr_validator,
        )
        log(f"step {global_step} audio metrics: {metrics}")
        save_checkpoint(accelerator, model, output_dir, global_step, args)
    accelerator.wait_for_everyone()

    model.train()
    ema_loss = None
    stop = global_step >= total_steps
    for epoch in range(args.epochs):
        if stop:
            break
        progress = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch}")
        for batch in progress:
            with accelerator.accumulate(model):
                loss = compute_loss(model, batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            loss_value = float(loss.detach().item())
            ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value
            progress.set_postfix(loss=f"{loss_value:.4f}", ema=f"{ema_loss:.4f}", step=global_step)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if writer is not None:
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/ema_loss", ema_loss, global_step)
                writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)

            validate_now = args.validation_every_steps > 0 and global_step % args.validation_every_steps == 0
            checkpoint_now = args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0
            if validate_now:
                val_loss = evaluate_loss(model, val_loader, accelerator, args.eval_max_batches)
                if accelerator.is_main_process:
                    if val_loss is not None:
                        writer.add_scalar("validation/loss", val_loss, global_step)
                    unwrapped = accelerator.unwrap_model(model)
                    audio_metrics = generate_validation(
                        model=unwrapped,
                        processor=qwen3tts.processor,
                        val_rows=val_rows,
                        texts=fixed_texts,
                        output_dir=output_dir / f"checkpoint-step-{global_step}" / "validation_audio",
                        step=global_step,
                        args=args,
                        writer=writer,
                        asr_validator=asr_validator,
                    )
                    log(f"step={global_step} val_loss={val_loss} audio={audio_metrics}")
                model.train()
                accelerator.wait_for_everyone()
            if checkpoint_now and accelerator.is_main_process:
                checkpoint = save_checkpoint(accelerator, model, output_dir, global_step, args)
                log(f"saved {checkpoint}")
            accelerator.wait_for_everyone()
            if global_step >= total_steps:
                stop = True
                break

    if accelerator.is_main_process:
        if args.checkpoint_every_steps <= 0 or global_step % args.checkpoint_every_steps != 0:
            checkpoint = save_checkpoint(accelerator, model, output_dir, global_step, args)
            log(f"saved final {checkpoint}")
        if writer is not None:
            writer.flush()
            writer.close()
    accelerator.wait_for_everyone()
    log(f"training finished at optimizer step {global_step}")


if __name__ == "__main__":
    main()
