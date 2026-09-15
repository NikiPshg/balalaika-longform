# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""LoRA SFT for Base-model voice clone prompts prepared as JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import islice
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def log_stage(message: str, start_time: Optional[float] = None) -> None:
    elapsed = ""
    if start_time is not None:
        elapsed = f" (+{time.monotonic() - start_time:.1f}s)"
    print(f"[qwen3-tts-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}{elapsed}", flush=True)


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_n_jsonl(path: str, n: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in islice((line for line in f if line.strip()), n)]


def jsonl_offsets(path: str) -> List[int]:
    offsets = []
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def normalize_1d_long(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    while tensor.dim() > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be a 1D token-id list, got shape {tuple(tensor.shape)}")
    return tensor


def normalize_codes(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must have shape [T, num_codebooks], got {tuple(tensor.shape)}")
    return tensor


def normalize_embedding(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    while tensor.dim() > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be a 1D embedding, got shape {tuple(tensor.shape)}")
    return tensor


def get_first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


class VoiceClonePreparedDataset(Dataset):
    """Dataset for cached Base voice-clone ICL rows.

    Required prepared fields are text_ids, ref_ids, ref_audio_codes,
    ref_spk_embedding, and audio_codes. Text/ref_text fallback is kept only
    to make small ad-hoc debug rows easier to inspect.
    """

    def __init__(
        self,
        rows: Optional[List[Dict[str, Any]]] = None,
        processor=None,
        jsonl_path: Optional[str] = None,
        offsets: Optional[List[int]] = None,
    ):
        if rows is None and jsonl_path is None:
            raise ValueError("VoiceClonePreparedDataset requires rows or jsonl_path")
        self.rows = rows
        self.processor = processor
        self.jsonl_path = jsonl_path
        self.offsets = offsets
        self._jsonl_file = None

    @classmethod
    def from_jsonl(cls, path: str, processor=None) -> "VoiceClonePreparedDataset":
        return cls(rows=None, processor=processor, jsonl_path=path, offsets=jsonl_offsets(path))

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_jsonl_file"] = None
        return state

    def __len__(self) -> int:
        if self.rows is not None:
            return len(self.rows)
        return len(self.offsets or [])

    def _row_at(self, idx: int) -> Dict[str, Any]:
        if self.rows is not None:
            return self.rows[idx]
        if self.jsonl_path is None or self.offsets is None:
            raise ValueError("JSONL dataset is missing path or offsets")
        if self._jsonl_file is None:
            self._jsonl_file = open(self.jsonl_path, "r", encoding="utf-8")
        self._jsonl_file.seek(self.offsets[idx])
        return json.loads(self._jsonl_file.readline())

    @staticmethod
    def build_assistant_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def build_ref_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def tokenize(self, text: str, name: str) -> torch.Tensor:
        if self.processor is None:
            raise ValueError(f"Missing cached {name} and no processor is available for fallback tokenization")
        encoded = self.processor(text=text, return_tensors="pt", padding=True)["input_ids"]
        if encoded.dim() == 2:
            encoded = encoded[0]
        return encoded.to(dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._row_at(idx)

        text_ids_value = get_first_present(row, ["text_ids", "input_ids"])
        if text_ids_value is None:
            text = get_first_present(row, ["text", "target_text"])
            if text is None:
                raise ValueError("Prepared row is missing text_ids")
            text_ids = self.tokenize(self.build_assistant_text(str(text)), "text_ids")
        else:
            text_ids = normalize_1d_long(text_ids_value, "text_ids")

        ref_ids_value = get_first_present(row, ["ref_ids", "ref_text_ids"])
        if ref_ids_value is None:
            ref_text = get_first_present(row, ["ref_text", "reference_text"])
            if ref_text is None:
                raise ValueError("Prepared row is missing ref_ids")
            ref_ids = self.tokenize(self.build_ref_text(str(ref_text)), "ref_ids")
        else:
            ref_ids = normalize_1d_long(ref_ids_value, "ref_ids")

        ref_codes_value = get_first_present(row, ["ref_audio_codes", "ref_codes", "ref_code"])
        if ref_codes_value is None:
            raise ValueError("Prepared row is missing ref_audio_codes")

        spk_value = get_first_present(row, ["ref_spk_embedding", "ref_speaker_embedding"])
        if spk_value is None:
            raise ValueError("Prepared row is missing ref_spk_embedding")

        return {
            "text_ids": text_ids,
            "ref_ids": ref_ids,
            "ref_audio_codes": normalize_codes(ref_codes_value, "ref_audio_codes"),
            "ref_spk_embedding": normalize_embedding(spk_value, "ref_spk_embedding"),
            "audio_codes": normalize_codes(row["audio_codes"], "audio_codes"),
            "language": row.get("language", "Auto") or "Auto",
            "text": row.get("text", row.get("target_text", "")),
            "ref_text": row.get("ref_text", row.get("reference_text", "")),
            "row_index": idx,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_lens = torch.tensor([item["text_ids"].numel() for item in batch], dtype=torch.long)
        ref_lens = torch.tensor([item["ref_ids"].numel() for item in batch], dtype=torch.long)
        ref_code_lens = torch.tensor([item["ref_audio_codes"].shape[0] for item in batch], dtype=torch.long)
        audio_lens = torch.tensor([item["audio_codes"].shape[0] for item in batch], dtype=torch.long)

        text_ids = pad_sequence([item["text_ids"] for item in batch], batch_first=True, padding_value=0)
        ref_ids = pad_sequence([item["ref_ids"] for item in batch], batch_first=True, padding_value=0)
        ref_audio_codes = pad_sequence(
            [item["ref_audio_codes"] for item in batch], batch_first=True, padding_value=0
        )
        audio_codes = pad_sequence([item["audio_codes"] for item in batch], batch_first=True, padding_value=0)
        ref_spk_embedding = torch.stack([item["ref_spk_embedding"] for item in batch], dim=0)

        return {
            "text_ids": text_ids,
            "text_lens": text_lens,
            "ref_ids": ref_ids,
            "ref_lens": ref_lens,
            "ref_audio_codes": ref_audio_codes,
            "ref_code_lens": ref_code_lens,
            "ref_spk_embedding": ref_spk_embedding,
            "audio_codes": audio_codes,
            "audio_lens": audio_lens,
            "languages": [item["language"] for item in batch],
            "texts": [item["text"] for item in batch],
            "ref_texts": [item["ref_text"] for item in batch],
            "row_indices": [item["row_index"] for item in batch],
        }


def core_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def force_eager_tokenizer_decode(model) -> None:
    """Avoid a Transformers SDPA mask aliasing bug in the 12 Hz waveform decoder."""
    base = core_model(model)
    tokenizer = getattr(base, "speech_tokenizer", None)
    tokenizer_model = getattr(tokenizer, "model", None)
    if tokenizer_model is None:
        return

    for module in tokenizer_model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def text_embeddings(talker, ids: torch.Tensor) -> torch.Tensor:
    embeds = talker.get_text_embeddings()(ids)
    if hasattr(talker, "text_projection"):
        embeds = talker.text_projection(embeds)
    return embeds


def code_embeddings(talker, codes: torch.Tensor) -> torch.Tensor:
    pieces = []
    for idx in range(codes.shape[1]):
        code_slice = codes[:, idx : idx + 1]
        if idx == 0:
            pieces.append(talker.get_input_embeddings()(code_slice))
        else:
            pieces.append(talker.code_predictor.get_input_embeddings()[idx - 1](code_slice))
    return torch.cat(pieces, dim=1).sum(1).unsqueeze(0)


def language_id_for(config, language: str) -> Optional[int]:
    if language is None or language.lower() == "auto":
        return None
    key = language.lower()
    if key not in config.talker_config.codec_language_id:
        raise NotImplementedError(f"Language {language} not implemented")
    return config.talker_config.codec_language_id[key]


def build_voice_clone_teacher_sample(
    model,
    text_ids: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_audio_codes: torch.Tensor,
    ref_spk_embedding: torch.Tensor,
    target_audio_codes: torch.Tensor,
    language: str,
    non_streaming_mode: bool,
) -> Dict[str, torch.Tensor]:
    base = core_model(model)
    talker = base.talker
    device = next(talker.parameters()).device
    embed_dtype = talker.get_input_embeddings().weight.dtype

    input_id = text_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    ref_id = ref_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    ref_code = ref_audio_codes.to(device=device, dtype=torch.long)
    target_codes = target_audio_codes.to(device=device, dtype=torch.long)
    speaker_embed = ref_spk_embedding.to(device=device, dtype=embed_dtype)

    tts_bos_embed, tts_eos_embed, tts_pad_embed = text_embeddings(
        talker,
        torch.tensor(
            [[base.config.tts_bos_token_id, base.config.tts_eos_token_id, base.config.tts_pad_token_id]],
            device=device,
            dtype=torch.long,
        ),
    ).chunk(3, dim=1)

    lang_id = language_id_for(base.config, language)
    if lang_id is None:
        codec_prefill = [
            base.config.talker_config.codec_nothink_id,
            base.config.talker_config.codec_think_bos_id,
            base.config.talker_config.codec_think_eos_id,
        ]
    else:
        codec_prefill = [
            base.config.talker_config.codec_think_id,
            base.config.talker_config.codec_think_bos_id,
            lang_id,
            base.config.talker_config.codec_think_eos_id,
        ]

    codec_prefix_0 = talker.get_input_embeddings()(
        torch.tensor([codec_prefill], device=device, dtype=torch.long)
    )
    codec_prefix_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[base.config.talker_config.codec_pad_id, base.config.talker_config.codec_bos_id]],
            device=device,
            dtype=torch.long,
        )
    )
    codec_prefix = torch.cat([codec_prefix_0, speaker_embed.view(1, 1, -1), codec_prefix_1], dim=1)

    role_embed = text_embeddings(talker, input_id[:, :3])
    prompt_codec_side = torch.cat(
        [tts_pad_embed.expand(-1, codec_prefix.shape[1] - 2, -1), tts_bos_embed], dim=1
    ) + codec_prefix[:, :-1]
    prompt_embed = torch.cat([role_embed, prompt_codec_side], dim=1)

    icl_embed, trailing_text_hidden = base.generate_icl_prompt(
        text_id=input_id[:, 3:-5],
        ref_id=ref_id[:, 3:-2],
        ref_code=ref_code,
        tts_pad_embed=tts_pad_embed,
        tts_eos_embed=tts_eos_embed,
        non_streaming_mode=non_streaming_mode,
    )
    prompt_embed = torch.cat([prompt_embed, icl_embed], dim=1)
    prompt_len = prompt_embed.shape[1]

    eos_code = torch.zeros((1, target_codes.shape[1]), device=device, dtype=torch.long)
    eos_code[:, 0] = base.config.talker_config.codec_eos_token_id

    target_codec_embed = code_embeddings(talker, target_codes)
    target_text_parts = []
    for idx in range(target_codes.shape[0]):
        if idx < trailing_text_hidden.shape[1]:
            target_text_parts.append(trailing_text_hidden[:, idx : idx + 1])
        else:
            target_text_parts.append(tts_pad_embed)
    target_text_embed = torch.cat(target_text_parts, dim=1)
    target_embed = target_codec_embed + target_text_embed
    eos_embed = code_embeddings(talker, eos_code) + tts_pad_embed

    inputs_embeds = torch.cat([prompt_embed, target_embed, eos_embed], dim=1).squeeze(0)
    labels = torch.full((inputs_embeds.shape[0],), -100, device=device, dtype=torch.long)
    labels[prompt_len : prompt_len + target_codes.shape[0]] = target_codes[:, 0]
    labels[prompt_len + target_codes.shape[0]] = base.config.talker_config.codec_eos_token_id

    codec_ids = torch.zeros(
        (inputs_embeds.shape[0], target_codes.shape[1]), device=device, dtype=torch.long
    )
    codec_ids[prompt_len : prompt_len + target_codes.shape[0]] = target_codes

    codec_mask = torch.zeros((inputs_embeds.shape[0],), device=device, dtype=torch.bool)
    codec_mask[prompt_len : prompt_len + target_codes.shape[0]] = True

    return {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "codec_ids": codec_ids,
        "codec_mask": codec_mask,
        "attention_mask": torch.ones((inputs_embeds.shape[0],), device=device, dtype=torch.long),
    }


def compute_loss(model, batch: Dict[str, Any], non_streaming_mode: bool = False) -> torch.Tensor:
    samples = []
    batch_size = batch["text_ids"].shape[0]
    for idx in range(batch_size):
        text_len = int(batch["text_lens"][idx])
        ref_len = int(batch["ref_lens"][idx])
        ref_code_len = int(batch["ref_code_lens"][idx])
        audio_len = int(batch["audio_lens"][idx])
        samples.append(
            build_voice_clone_teacher_sample(
                model=model,
                text_ids=batch["text_ids"][idx, :text_len],
                ref_ids=batch["ref_ids"][idx, :ref_len],
                ref_audio_codes=batch["ref_audio_codes"][idx, :ref_code_len],
                ref_spk_embedding=batch["ref_spk_embedding"][idx],
                target_audio_codes=batch["audio_codes"][idx, :audio_len],
                language=batch["languages"][idx],
                non_streaming_mode=non_streaming_mode,
            )
        )

    inputs_embeds = pad_sequence(
        [item["inputs_embeds"] for item in samples], batch_first=True, padding_value=0.0
    )
    labels = pad_sequence([item["labels"] for item in samples], batch_first=True, padding_value=-100)
    codec_ids = pad_sequence([item["codec_ids"] for item in samples], batch_first=True, padding_value=0)
    codec_mask = pad_sequence([item["codec_mask"] for item in samples], batch_first=True, padding_value=False)
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in samples], batch_first=True, padding_value=0
    )

    base = core_model(model)
    outputs = base.talker(
        inputs_embeds=inputs_embeds[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=None,
        output_hidden_states=True,
    )

    logits = outputs.logits
    codec_0_targets = labels[:, 1:]
    codec_0_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        codec_0_targets.reshape(-1),
        ignore_index=-100,
    )

    hidden_states = outputs.hidden_states[0][-1]
    target_hidden_states = hidden_states[codec_mask[:, 1:]]
    target_codec_ids = codec_ids[codec_mask]
    if target_codec_ids.numel() == 0:
        return codec_0_loss

    sub_logits, _ = base.talker.forward_sub_talker_finetune(
        target_codec_ids,
        target_hidden_states,
    )
    sub_talker_loss = F.cross_entropy(
        sub_logits.reshape(-1, sub_logits.size(-1)),
        target_codec_ids[:, 1:].reshape(-1),
    )
    return codec_0_loss + sub_talker_loss


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, accelerator: Accelerator, non_streaming_mode: bool) -> Optional[float]:
    model.eval()
    losses = []
    for batch in dataloader:
        loss = compute_loss(model, batch, non_streaming_mode=non_streaming_mode)
        gathered = accelerator.gather_for_metrics(loss.detach())
        if gathered.ndim == 0:
            gathered = gathered.unsqueeze(0)
        losses.append(gathered)
    if not losses:
        return None
    return torch.cat(losses).mean().item()


@torch.no_grad()
def evaluate_codebook_metrics(
    model,
    dataloader: DataLoader,
    accelerator: Accelerator,
    non_streaming_mode: bool,
) -> Dict[str, float]:
    model.eval()
    base = core_model(model)
    counts = torch.zeros(4, device=accelerator.device, dtype=torch.float64)

    for batch in dataloader:
        samples = []
        batch_size = batch["text_ids"].shape[0]
        for idx in range(batch_size):
            text_len = int(batch["text_lens"][idx])
            ref_len = int(batch["ref_lens"][idx])
            ref_code_len = int(batch["ref_code_lens"][idx])
            audio_len = int(batch["audio_lens"][idx])
            samples.append(
                build_voice_clone_teacher_sample(
                    model=model,
                    text_ids=batch["text_ids"][idx, :text_len],
                    ref_ids=batch["ref_ids"][idx, :ref_len],
                    ref_audio_codes=batch["ref_audio_codes"][idx, :ref_code_len],
                    ref_spk_embedding=batch["ref_spk_embedding"][idx],
                    target_audio_codes=batch["audio_codes"][idx, :audio_len],
                    language=batch["languages"][idx],
                    non_streaming_mode=non_streaming_mode,
                )
            )

        inputs_embeds = pad_sequence(
            [item["inputs_embeds"] for item in samples], batch_first=True, padding_value=0.0
        )
        labels = pad_sequence([item["labels"] for item in samples], batch_first=True, padding_value=-100)
        codec_ids = pad_sequence([item["codec_ids"] for item in samples], batch_first=True, padding_value=0)
        codec_mask = pad_sequence([item["codec_mask"] for item in samples], batch_first=True, padding_value=False)
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in samples], batch_first=True, padding_value=0
        )

        outputs = base.talker(
            inputs_embeds=inputs_embeds[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=None,
            output_hidden_states=True,
        )

        codec_0_targets = labels[:, 1:]
        codec_0_mask = codec_0_targets != -100
        codec_0_pred = outputs.logits.argmax(dim=-1)
        counts[0] += (codec_0_pred[codec_0_mask] == codec_0_targets[codec_0_mask]).sum()
        counts[1] += codec_0_mask.sum()

        hidden_states = outputs.hidden_states[0][-1]
        target_hidden_states = hidden_states[codec_mask[:, 1:]]
        target_codec_ids = codec_ids[codec_mask]
        if target_codec_ids.numel() > 0:
            sub_logits, _ = base.talker.forward_sub_talker_finetune(
                target_codec_ids,
                target_hidden_states,
            )
            sub_pred = sub_logits.argmax(dim=-1)
            sub_targets = target_codec_ids[:, 1:]
            counts[2] += (sub_pred == sub_targets).sum()
            counts[3] += sub_targets.numel()

    gathered = accelerator.gather_for_metrics(counts)
    if gathered.ndim == 1:
        totals = gathered
    else:
        totals = gathered.sum(dim=0)
    codec_0_correct, codec_0_total, sub_correct, sub_total = [float(x.item()) for x in totals]
    metrics = {
        "val/codec0_correct": codec_0_correct,
        "val/codec0_total": codec_0_total,
        "val/subcode_correct": sub_correct,
        "val/subcode_total": sub_total,
        "val/codec0_acc": codec_0_correct / codec_0_total if codec_0_total else 0.0,
        "val/subcode_acc": sub_correct / sub_total if sub_total else 0.0,
    }
    return metrics


def generation_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_dosample": args.subtalker_dosample,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }


def audio_codebook_size(model, fallback_items: Optional[List[Dict[str, Any]]] = None) -> int:
    base = core_model(model)
    tokenizer_model = getattr(getattr(base, "speech_tokenizer", None), "model", None)
    tokenizer_config = getattr(tokenizer_model, "config", None)
    size = getattr(tokenizer_config, "codebook_size", None)
    if size is not None:
        return int(size)

    max_code = 0
    for item in fallback_items or []:
        for key in ("audio_codes", "ref_audio_codes"):
            value = item.get(key)
            if value is not None:
                tensor = torch.as_tensor(value)
                if tensor.numel():
                    max_code = max(max_code, int(tensor.max().item()))
    return max_code + 1


def validate_audio_codes(codes: torch.Tensor, codebook_size: int, name: str) -> torch.Tensor:
    codes = codes.to(dtype=torch.long)
    if codes.numel() == 0:
        return codes
    min_code = int(codes.min().item())
    max_code = int(codes.max().item())
    if min_code < 0 or max_code >= codebook_size:
        raise ValueError(
            f"{name} contains audio code ids outside [0, {codebook_size - 1}] "
            f"(min={min_code}, max={max_code}). Use validation_decode_mode=generate "
            "for subjective audio checks instead of silently clamping invalid codes."
        )
    return codes


@torch.no_grad()
def teacher_forced_codes(model, item: Dict[str, Any]) -> torch.Tensor:
    base = core_model(model)
    device = next(base.talker.parameters()).device
    codebook_size = audio_codebook_size(base, [item])
    sample = build_voice_clone_teacher_sample(
        model=base,
        text_ids=item["text_ids"],
        ref_ids=item["ref_ids"],
        ref_audio_codes=item["ref_audio_codes"],
        ref_spk_embedding=item["ref_spk_embedding"],
        target_audio_codes=item["audio_codes"],
        language=item["language"],
        non_streaming_mode=False,
    )
    inputs_embeds = sample["inputs_embeds"].unsqueeze(0)
    attention_mask = sample["attention_mask"].unsqueeze(0)
    outputs = base.talker(
        inputs_embeds=inputs_embeds[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=None,
        output_hidden_states=True,
    )
    shifted_target_mask = sample["codec_mask"][1:]
    pred_codec_0 = outputs.logits[0, shifted_target_mask, :codebook_size].argmax(dim=-1)

    hidden_states = outputs.hidden_states[0][-1][0]
    target_hidden_states = hidden_states[shifted_target_mask]
    target_codec_ids = sample["codec_ids"][sample["codec_mask"]]
    sub_logits, _ = base.talker.forward_sub_talker_finetune(
        target_codec_ids,
        target_hidden_states,
    )
    pred_sub_codes = sub_logits[..., :codebook_size].argmax(dim=-1)
    return validate_audio_codes(
        torch.cat([pred_codec_0.unsqueeze(1), pred_sub_codes], dim=1),
        codebook_size,
        "teacher_forced_codes",
    )


@torch.no_grad()
def generate_validation_audio(
    model,
    rows: List[Dict[str, Any]],
    processor,
    output_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("soundfile is required for validation WAV writing. Install it with: pip install soundfile") from exc

    os.makedirs(output_dir, exist_ok=True)
    dataset = VoiceClonePreparedDataset(rows, processor=processor)
    base = core_model(model)
    base.eval()
    force_eager_tokenizer_decode(base)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    waveform_mses = []
    durations = []
    rms_values = []
    peak_values = []
    generated_code_lengths = []
    target_durations = []
    audio_items = []

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        audio_sample_count = len(dataset) if args.validation_audio_samples <= 0 else args.validation_audio_samples
        items = [dataset[i % len(dataset)] for i in range(audio_sample_count)]
        codebook_size = audio_codebook_size(base, items)
        device = next(base.talker.parameters()).device
        embed_dtype = base.talker.get_input_embeddings().weight.dtype
        ref_codes_list = [item["ref_audio_codes"].to(device=device, dtype=torch.long) for item in items]

        if args.validation_decode_mode == "teacher_forced":
            generated_list = [teacher_forced_codes(base, item) for item in items]
        else:
            prompt = {
                "ref_code": ref_codes_list,
                "ref_spk_embedding": [
                    item["ref_spk_embedding"].to(device=device, dtype=embed_dtype) for item in items
                ],
                "x_vector_only_mode": [False] * len(items),
                "icl_mode": [True] * len(items),
            }
            generation_kwargs = generation_kwargs_from_args(args)
            target_cap = max(int(item["audio_codes"].shape[0]) for item in items) + 4
            generation_kwargs["max_new_tokens"] = min(int(args.max_new_tokens), target_cap)
            generated_list, _ = base.generate(
                input_ids=[item["text_ids"].to(device=device, dtype=torch.long).unsqueeze(0) for item in items],
                ref_ids=[item["ref_ids"].to(device=device, dtype=torch.long).unsqueeze(0) for item in items],
                voice_clone_prompt=prompt,
                languages=[item["language"] for item in items],
                non_streaming_mode=args.non_streaming_mode,
                **generation_kwargs,
            )

        codes_for_decode_list = []
        for sample_idx, (ref_codes, generated) in enumerate(zip(ref_codes_list, generated_list)):
            combined_codes = torch.cat([ref_codes.to(generated.device), generated], dim=0)
            codes_for_decode_list.append(
                {
                    "audio_codes": validate_audio_codes(
                        combined_codes,
                        codebook_size,
                        f"validation sample {sample_idx}",
                    )
                }
            )
        wavs_all, sample_rate = base.speech_tokenizer.decode(codes_for_decode_list)
        target_wavs_all, target_sample_rate = base.speech_tokenizer.decode(
            [{"audio_codes": item["audio_codes"].to(device=device, dtype=torch.long)} for item in items]
        )

        for sample_idx, (item, ref_codes, generated_codes, encoded_item, wav, target_wav) in enumerate(
            zip(items, ref_codes_list, generated_list, codes_for_decode_list, wavs_all, target_wavs_all)
        ):
            ref_len = int(ref_codes.shape[0])
            total_len = int(encoded_item["audio_codes"].shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wav = np.asarray(wav[cut:], dtype=np.float32)
            target_wav = np.asarray(target_wav, dtype=np.float32)
            compare_len = min(int(wav.shape[0]), int(target_wav.shape[0]))
            waveform_mse = None
            if compare_len > 0:
                diff = wav[:compare_len] - target_wav[:compare_len]
                waveform_mse = float(np.mean(diff * diff))
                waveform_mses.append(waveform_mse)

            duration = float(wav.shape[0] / sample_rate) if sample_rate else 0.0
            target_duration = float(target_wav.shape[0] / target_sample_rate) if target_sample_rate else 0.0
            rms = float(np.sqrt(np.mean(wav * wav))) if wav.size else 0.0
            peak = float(np.max(np.abs(wav))) if wav.size else 0.0
            durations.append(duration)
            target_durations.append(target_duration)
            rms_values.append(rms)
            peak_values.append(peak)
            generated_code_lengths.append(float(generated_codes.shape[0]))

            wav_name = f"sample_{sample_idx:03d}.wav"
            wav_path = os.path.join(output_dir, wav_name)
            sf.write(wav_path, wav, sample_rate)
            audio_items.append(
                {
                    "key": f"audio/sample{sample_idx + 1}.wav",
                    "path": wav_path,
                    "caption": item["text"],
                    "sample_rate": sample_rate,
                }
            )
            manifest.write(
                json.dumps(
                    {
                        "sample_index": sample_idx,
                        "source_row_index": item["row_index"],
                        "audio": wav_name,
                        "text": item["text"],
                        "ref_text": item["ref_text"],
                        "language": item["language"],
                        "sample_rate": sample_rate,
                        "target_sample_rate": target_sample_rate,
                        "waveform_mse": waveform_mse,
                        "mse_compare_samples": compare_len,
                        "generated_duration": duration,
                        "target_duration": target_duration,
                        "generated_rms": rms,
                        "generated_peak": peak,
                        "generated_code_len": int(generated_codes.shape[0]),
                        "validation_decode_mode": args.validation_decode_mode,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def stats(prefix: str, values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_min": float(arr.min()),
            f"{prefix}_max": float(arr.max()),
        }

    metrics = {}
    metrics.update(stats("val/audio_waveform_mse", waveform_mses))
    metrics.update(stats("val/audio_duration_sec", durations))
    metrics.update(stats("val/audio_target_duration_sec", target_durations))
    metrics.update(stats("val/audio_rms", rms_values))
    metrics.update(stats("val/audio_peak", peak_values))
    metrics.update(stats("val/generated_code_len", generated_code_lengths))
    metrics["val/audio_count"] = float(len(audio_items))
    metrics["audio_items"] = audio_items
    return metrics


def build_wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "output_model_path": args.output_model_path,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "start_epoch": args.start_epoch,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "attn_implementation": args.attn_implementation,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "single_batch_test": args.single_batch_test,
        "max_steps": args.max_steps,
        "validation_audio_every": args.validation_audio_every,
        "validation_at_step": args.validation_at_step,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_bias": args.lora_bias,
        "lora_target_modules": args.lora_target_modules,
        "non_streaming_mode": args.non_streaming_mode,
        "validation_decode_mode": args.validation_decode_mode,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "subtalker_dosample": args.subtalker_dosample,
    }


def init_wandb(args: argparse.Namespace, config: Dict[str, Any]):
    if not args.wandb_enabled:
        return None, None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb logging requested, but wandb is not installed. Install it with: pip install wandb") from exc

    tags = parse_list(args.wandb_tags) if args.wandb_tags else None
    run = wandb.init(
        project=args.wandb_project or None,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group or None,
        tags=tags,
        id=args.wandb_id or None,
        resume=args.wandb_resume or None,
        mode=args.wandb_mode or None,
        dir=args.wandb_dir or None,
        config=config,
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="train/global_step")
    wandb.define_metric("audio/*", step_metric="train/global_step")
    return wandb, run


def log_wandb(run, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is None:
        return
    if step is None:
        run.log(metrics)
    else:
        run.log(metrics, step=step)


def save_adapter_checkpoint(accelerator, model, checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(checkpoint_dir, safe_serialization=True)


def log_validation_audio(
    *,
    model,
    val_rows: List[Dict[str, Any]],
    processor,
    output_dir: str,
    args: argparse.Namespace,
    wandb_module,
    wandb_run,
    step: int,
    epoch: int,
    audio_key_prefix: Optional[str] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    audio_metrics = generate_validation_audio(
        model=model,
        rows=val_rows,
        processor=processor,
        output_dir=output_dir,
        args=args,
    )
    audio_items = audio_metrics.pop("audio_items", [])
    wandb_audio = {}
    if wandb_run is not None and wandb_module is not None:
        for item in audio_items:
            key = f"{audio_key_prefix}/{item['key']}" if audio_key_prefix else item["key"]
            wandb_audio[key] = wandb_module.Audio(
                item["path"],
                caption=item["caption"],
                sample_rate=item["sample_rate"],
            )
    logged_metrics = {
        "epoch": epoch,
        "train/global_step": step,
        **(extra_metrics or {}),
        **audio_metrics,
        **wandb_audio,
    }
    log_wandb(wandb_run, logged_metrics, step=step)
    return audio_metrics


def current_lr(optimizer) -> Optional[float]:
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups is None and hasattr(optimizer, "optimizer"):
        param_groups = getattr(optimizer.optimizer, "param_groups", None)
    if not param_groups:
        return None
    return float(param_groups[0]["lr"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "--init_model_path",
        dest="model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default=None)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--epochs", "--num_epochs", dest="epochs", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_batch_test", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--validation_audio_every",
        type=int,
        default=1,
        help="Generate 10 validation wavs every N epochs. Use 0 to generate only after the final epoch.",
    )
    parser.add_argument(
        "--validation_audio_samples",
        type=int,
        default=10,
        help="Number of validation wavs to generate. Use 0 to generate one wav for every validation row.",
    )
    parser.add_argument(
        "--validation_at_step",
        type=int,
        default=None,
        help="Run validation and generate validation audio once when global_step reaches this value.",
    )
    parser.add_argument("--resume_adapter", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--eval_every", type=int, default=1, help=argparse.SUPPRESS)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=(
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,"
            "codec_head,"
            "lm_head.0,lm_head.1,lm_head.2,lm_head.3,lm_head.4,"
            "lm_head.5,lm_head.6,lm_head.7,lm_head.8,lm_head.9,"
            "lm_head.10,lm_head.11,lm_head.12,lm_head.13,lm_head.14"
        ),
    )

    parser.add_argument("--non_streaming_mode", type=str2bool, default=True)
    parser.add_argument(
        "--validation_decode_mode",
        choices=["generate", "teacher_forced"],
        default="generate",
        help="Use free-running generation or fast teacher-forced code prediction for validation wavs.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--do_sample", type=str2bool, default=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--subtalker_dosample", type=str2bool, default=True)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)

    parser.add_argument("--wandb_enabled", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    return parser


def main() -> None:
    script_start_time = time.monotonic()
    log_stage("script started")
    args = build_arg_parser().parse_args()

    log_stage("importing training dependencies")
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
    except ImportError as exc:
        raise SystemExit(
            "accelerate is required for LoRA fine-tuning. Install it with: pip install accelerate"
        ) from exc

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "peft is required for LoRA fine-tuning. Install it with: pip install peft"
        ) from exc

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("tqdm is required for the training progress bar. Install it with: pip install tqdm") from exc

    set_seed(args.seed)
    log_stage(f"seed set to {args.seed}", script_start_time)

    log_stage("importing Qwen3-TTS runtime")
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except ImportError as exc:
        raise SystemExit(
            "Qwen3-TTS runtime dependencies are required for training. "
            "Install the project requirements, including librosa, soundfile, transformers, and torchaudio."
        ) from exc

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
    )
    if accelerator.is_main_process:
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | accelerator initialized "
            f"(device={accelerator.device}, processes={accelerator.num_processes}, "
            f"mixed_precision={args.mixed_precision}) (+{time.monotonic() - script_start_time:.1f}s)"
        )
    wandb_module = None
    wandb_run = None
    if accelerator.is_main_process:
        accelerator.print(f"[qwen3-tts-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | initializing wandb")
        wandb_module, wandb_run = init_wandb(args, build_wandb_config(args))
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | wandb initialized "
            f"(enabled={args.wandb_enabled}) (+{time.monotonic() - script_start_time:.1f}s)"
        )

    dtype = torch.float32
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16

    model_load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }
    if accelerator.device.type == "cuda":
        model_load_kwargs["device_map"] = str(accelerator.device)
    if accelerator.is_main_process:
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | loading base model from {args.model_path} "
            f"(dtype={dtype}, attn={args.attn_implementation})"
        )
    qwen3tts = Qwen3TTSModel.from_pretrained(args.model_path, **model_load_kwargs)
    qwen3tts.model.config.tts_model_type = "base"
    qwen3tts.model.tts_model_type = "base"
    if accelerator.is_main_process:
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | base model loaded "
            f"(+{time.monotonic() - script_start_time:.1f}s)"
        )

    if args.resume_adapter:
        if accelerator.is_main_process:
            accelerator.print(f"[qwen3-tts-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | loading LoRA adapter from {args.resume_adapter}")
        model = PeftModel.from_pretrained(qwen3tts.model, args.resume_adapter, is_trainable=True)
    else:
        if accelerator.is_main_process:
            accelerator.print(
                "[qwen3-tts-lora] "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | creating LoRA adapter "
                f"(rank={args.lora_rank}, alpha={args.lora_alpha})"
            )
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=parse_list(args.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(qwen3tts.model, lora_config)
    qwen3tts.model = model
    if accelerator.is_main_process:
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | LoRA model ready "
            f"(+{time.monotonic() - script_start_time:.1f}s)"
        )

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    if accelerator.is_main_process:
        mode = "single-batch test" if args.single_batch_test else "full dataset"
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | reading JSONL data ({mode}) "
            f"train={args.train_jsonl}, val={args.val_jsonl or 'first train rows'}"
        )
    if args.single_batch_test:
        train_rows = first_n_jsonl(args.train_jsonl, args.batch_size)
        val_rows = first_n_jsonl(args.val_jsonl, args.batch_size) if args.val_jsonl else list(train_rows)
        train_dataset = VoiceClonePreparedDataset(train_rows, processor=qwen3tts.processor)
        shuffle_train = False
    else:
        train_rows = None
        train_dataset = VoiceClonePreparedDataset.from_jsonl(args.train_jsonl, processor=qwen3tts.processor)
        val_rows = read_jsonl(args.val_jsonl) if args.val_jsonl else first_n_jsonl(args.train_jsonl, 10)
        shuffle_train = True

    if len(train_dataset) == 0:
        raise ValueError("Training JSONL is empty")
    if not val_rows:
        raise ValueError("Validation/generation rows are empty")
    if accelerator.is_main_process:
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | JSONL data loaded "
            f"(train_rows={len(train_dataset)}, val_rows={len(val_rows)}) "
            f"(+{time.monotonic() - script_start_time:.1f}s)"
        )

    if accelerator.is_main_process:
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | building datasets and dataloaders "
            f"(batch_size={args.batch_size}, eval_batch_size={args.eval_batch_size or args.batch_size}, "
            f"workers={args.dataloader_num_workers})"
        )
    val_dataset = VoiceClonePreparedDataset(val_rows, processor=qwen3tts.processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    eval_batch_size = len(val_rows) if args.single_batch_test else (args.eval_batch_size or args.batch_size)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if accelerator.is_main_process:
        accelerator.print(f"[qwen3-tts-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | preparing model, optimizer, and dataloaders with accelerate")
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    if accelerator.is_main_process:
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | accelerate preparation complete "
            f"(+{time.monotonic() - script_start_time:.1f}s)"
        )

    single_batch = None
    if args.single_batch_test:
        if accelerator.is_main_process:
            accelerator.print("[qwen3-tts-lora] collecting the single training batch")
        single_batches = list(train_dataloader)
        if len(single_batches) != 1:
            raise ValueError("single_batch_test expected exactly one prepared training batch")
        single_batch = single_batches[0]

    model.train()
    global_step = 0
    if accelerator.is_main_process:
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | training starts "
            f"(epochs={args.epochs}, start_epoch={args.start_epoch}, lr={args.lr}) "
            f"(+{time.monotonic() - script_start_time:.1f}s)"
        )
        unwrapped = accelerator.unwrap_model(model)
        initial_checkpoint_dir = os.path.join(args.output_model_path, "checkpoint-step-0")
        accelerator.print(
            "[qwen3-tts-lora] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | generating validation audio before training"
        )
        log_validation_audio(
            model=unwrapped,
            val_rows=val_rows,
            processor=qwen3tts.processor,
            output_dir=os.path.join(initial_checkpoint_dir, "validation_audio"),
            args=args,
            wandb_module=wandb_module,
            wandb_run=wandb_run,
            step=global_step,
            epoch=args.start_epoch,
            audio_key_prefix="step_0",
        )
        save_adapter_checkpoint(accelerator, model, initial_checkpoint_dir)
        accelerator.print(f"Saved initial adapter checkpoint and validation audio to {initial_checkpoint_dir}")
    accelerator.wait_for_everyone()
    for local_epoch in range(args.epochs):
        epoch = args.start_epoch + local_epoch
        model.train()
        ema_loss = None
        if args.single_batch_test and args.max_steps is not None:
            train_iterable = (single_batch for _ in range(args.max_steps))
            progress_total = args.max_steps
        else:
            train_iterable = train_dataloader
            progress_total = None
        progress = tqdm(
            train_iterable,
            total=progress_total,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
            leave=True,
        )
        for step, batch in enumerate(progress):
            with accelerator.accumulate(model):
                loss = compute_loss(model, batch, non_streaming_mode=args.non_streaming_mode)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            loss_value = float(loss.detach().item())
            ema_loss = loss_value if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}", ema=f"{ema_loss:.4f}")
            global_step += 1
            if accelerator.is_main_process:
                log_wandb(
                    wandb_run,
                    {
                        "train/global_step": global_step,
                        "train/loss": loss_value,
                        "train/ema_loss": ema_loss,
                        "train/epoch": epoch,
                        "train/lr": current_lr(optimizer),
                    },
                    step=global_step,
                )
            if args.validation_at_step is not None and (global_step % args.validation_at_step == 0):
                if accelerator.is_main_process:
                    accelerator.print(f"[qwen3-tts-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | running step validation at global_step={global_step}")
                step_val_loss = evaluate(model, val_dataloader, accelerator, non_streaming_mode=args.non_streaming_mode)
                step_codebook_metrics = evaluate_codebook_metrics(
                    model,
                    val_dataloader,
                    accelerator,
                    non_streaming_mode=args.non_streaming_mode,
                )
                if accelerator.is_main_process and step_val_loss is not None:
                    accelerator.print(f"Global step {global_step} | Val Loss: {step_val_loss:.4f}")
                    step_metrics = {
                        "epoch": epoch,
                        "train/global_step": global_step,
                        "val/loss": step_val_loss,
                        "val/validation_at_step": float(global_step),
                        **step_codebook_metrics,
                    }
                    unwrapped = accelerator.unwrap_model(model)
                    accelerator.print(f"[qwen3-tts-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | generating step validation audio at global_step={global_step}")
                    audio_output_dir = os.path.join(args.output_model_path, f"checkpoint-step-{global_step}", "validation_audio")
                    log_validation_audio(
                        model=unwrapped,
                        val_rows=val_rows,
                        processor=qwen3tts.processor,
                        output_dir=audio_output_dir,
                        args=args,
                        wandb_module=wandb_module,
                        wandb_run=wandb_run,
                        step=global_step,
                        epoch=epoch,
                        audio_key_prefix=f"step_{global_step}",
                        extra_metrics=step_metrics,
                    )
                    step_checkpoint_dir = os.path.join(args.output_model_path, f"checkpoint-step-{global_step}")
                    save_adapter_checkpoint(accelerator, model, step_checkpoint_dir)
                    accelerator.print(
                        "Saved step adapter checkpoint and validation audio to "
                        f"{step_checkpoint_dir}"
                    )
                model.train()
                accelerator.wait_for_everyone()
            if args.max_steps is not None and step + 1 >= args.max_steps:
                break

        if accelerator.is_main_process:
            accelerator.print(f"[qwen3-tts-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | epoch {epoch} training complete; running validation")
        val_loss = evaluate(model, val_dataloader, accelerator, non_streaming_mode=args.non_streaming_mode)
        codebook_metrics = evaluate_codebook_metrics(
            model,
            val_dataloader,
            accelerator,
            non_streaming_mode=args.non_streaming_mode,
        )
        if accelerator.is_main_process and val_loss is not None:
            accelerator.print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
            val_metrics = {
                "epoch": epoch,
                "train/global_step": global_step,
                "val/loss": val_loss,
                **codebook_metrics,
            }
            log_wandb(wandb_run, val_metrics, step=global_step)
        model.train()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            unwrapped = accelerator.unwrap_model(model)
            final_epoch = local_epoch + 1 == args.epochs
            generate_audio = final_epoch if args.validation_audio_every == 0 else (
                final_epoch or (local_epoch + 1) % args.validation_audio_every == 0
            )
            if generate_audio:
                accelerator.print(f"[qwen3-tts-lora] {time.strftime('%Y-%m-%d %H:%M:%S')} | generating validation audio for epoch {epoch}")
                log_validation_audio(
                    model=unwrapped,
                    val_rows=val_rows,
                    processor=qwen3tts.processor,
                    output_dir=os.path.join(checkpoint_dir, "validation_audio"),
                    args=args,
                    wandb_module=wandb_module,
                    wandb_run=wandb_run,
                    step=global_step,
                    epoch=epoch,
                    audio_key_prefix=f"epoch_{epoch}",
                )
                save_adapter_checkpoint(accelerator, model, checkpoint_dir)
                accelerator.print(f"Saved adapter checkpoint and validation audio to {checkpoint_dir}")
            else:
                save_adapter_checkpoint(accelerator, model, checkpoint_dir)
                accelerator.print(f"Saved adapter checkpoint to {checkpoint_dir}")
        accelerator.wait_for_everyone()

    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
