# coding=utf-8
"""Qwen3-TTS helpers for TRL's full main-Talker SFT infrastructure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainerCallback
from trl import SFTTrainer

FINETUNING_ROOT = Path(__file__).resolve().parents[2]
if str(FINETUNING_ROOT) not in sys.path:
    sys.path.insert(0, str(FINETUNING_ROOT))
TRL_ROOT = Path(__file__).resolve().parents[1]
if str(TRL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRL_ROOT))

from sft_streaming_main_talker_full import (  # noqa: E402
    AudioValidationCallback,
    MainTalkerDataCollator,
    MainTalkerTrainingModel as _LegacyMainTalkerTrainingModel,
    SovaCandidateMapper,
)
from main_talker_training import (  # noqa: E402
    build_main_talker_teacher_sample,
    core_model,
)
from data_stream import load_untyped_streaming_dataset  # noqa: E402
from full_utterance_utils import (  # noqa: E402
    apply_silero_stress,
    build_assistant_text,
    deterministic_prefix_frames,
)
from sova_streaming import TARGET_SR  # noqa: E402
from sova_streaming import tokenize as tokenize_tts_text  # noqa: E402


class MainTalkerTrainingModel(_LegacyMainTalkerTrainingModel):
    """TRL-local wrapper for main-only or joint slow+fast Talker SFT."""

    _supports_sdpa = True

    def train(self, mode: bool = True) -> "MainTalkerTrainingModel":
        # The legacy wrapper deliberately puts the code predictor in eval mode.
        # Joint SFT opts it back into train mode while leaving the speech
        # tokenizer and speaker encoder frozen/eval.
        super().train(mode)
        if getattr(self.pipeline_args, "training_scope", "main_talker") == "all_talker":
            self.qwen_model.talker.code_predictor.train(mode)
        return self

    def forward(
        self,
        candidates: list[dict[str, Any]] | None = None,
        text_ids: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
        full_codes: torch.Tensor | None = None,
        code_lengths: torch.Tensor | None = None,
        speaker_embeddings: torch.Tensor | None = None,
        prefix_frames: torch.Tensor | None = None,
        languages: list[str] | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if getattr(self.pipeline_args, "training_scope", "main_talker") != "all_talker":
            return super().forward(
                candidates=candidates,
                text_ids=text_ids,
                text_lengths=text_lengths,
                full_codes=full_codes,
                code_lengths=code_lengths,
                speaker_embeddings=speaker_embeddings,
                prefix_frames=prefix_frames,
                languages=languages,
                labels=labels,
            )
        if candidates is not None:
            batch = self._encode_candidates(candidates)
            batch_size = len(candidates)
        else:
            required = {
                "text_ids": text_ids,
                "text_lengths": text_lengths,
                "full_codes": full_codes,
                "code_lengths": code_lengths,
                "speaker_embeddings": speaker_embeddings,
                "prefix_frames": prefix_frames,
                "languages": languages,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"Missing prepared batch fields: {missing}")
            batch = required
            batch_size = int(text_ids.shape[0])

        loss, metrics = compute_all_talker_loss(
            self.qwen_model,
            batch,
            code_predictor_loss_weight=float(
                getattr(self.pipeline_args, "code_predictor_loss_weight", 1.0)
            ),
            distributed=self.training,
            return_token_metrics=not self.training,
        )
        metric_logits = (
            loss.new_zeros((batch_size, 8))
            if self.training
            else metrics["per_sample_metrics"]
        )
        return {"loss": loss, "logits": metric_logits}


def configure_all_talker(model: Any) -> tuple[int, int, dict[str, int]]:
    """Train both generative transformers and freeze acoustic auxiliaries."""

    base = core_model(model)
    base.requires_grad_(False)
    base.talker.requires_grad_(True)
    speech_model = getattr(getattr(base, "speech_tokenizer", None), "model", None)
    if speech_model is not None:
        speech_model.requires_grad_(False)
        speech_model.eval()
    speaker_encoder = getattr(base, "speaker_encoder", None)
    if speaker_encoder is not None:
        speaker_encoder.requires_grad_(False)
        speaker_encoder.eval()
    base.talker.config.use_cache = False
    base.talker.code_predictor.config.use_cache = False

    trainable = [(name, value) for name, value in base.named_parameters() if value.requires_grad]
    forbidden = [name for name, _ in trainable if not name.startswith("talker.")]
    if forbidden:
        raise RuntimeError(f"Unexpected trainable parameters: {forbidden[:10]}")
    groups = {
        "main_talker": sum(
            value.numel()
            for name, value in trainable
            if not name.startswith("talker.code_predictor.")
        ),
        "code_predictor": sum(
            value.numel()
            for name, value in trainable
            if name.startswith("talker.code_predictor.")
        ),
    }
    if not all(groups.values()):
        raise RuntimeError(f"Both Talker transformers must be trainable, got {groups}")
    total = sum(value.numel() for value in base.parameters())
    return sum(groups.values()), total, groups


def _global_token_mean(loss_sum: torch.Tensor, token_count: torch.Tensor) -> torch.Tensor:
    """Globally token-normalize while accounting for DDP gradient averaging."""

    denominator = token_count.detach().to(device=loss_sum.device, dtype=torch.float32)
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        denominator = denominator.clone()
        torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    return loss_sum * world_size / denominator.clamp_min(1.0)


def compute_all_talker_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    code_predictor_loss_weight: float,
    distributed: bool,
    return_token_metrics: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher-force codebook 0 and the fast predictor's codebooks 1..15."""

    if not math.isfinite(code_predictor_loss_weight) or code_predictor_loss_weight <= 0:
        raise ValueError("code_predictor_loss_weight must be finite and positive")
    base = core_model(model)
    talker = base.talker
    samples: list[dict[str, torch.Tensor]] = []
    codec_labels: list[torch.Tensor] = []
    for index in range(batch["text_ids"].shape[0]):
        text_length = int(batch["text_lengths"][index])
        code_length = int(batch["code_lengths"][index])
        prefix_length = int(batch["prefix_frames"][index])
        full_codes = batch["full_codes"][index, :code_length]
        sample = build_main_talker_teacher_sample(
            model,
            text_ids=batch["text_ids"][index, :text_length],
            full_codes=full_codes,
            prefix_frames=prefix_length,
            speaker_embedding=batch["speaker_embeddings"][index],
            language=batch["languages"][index],
        )
        frame_positions = sample["labels"].ne(-100).nonzero(as_tuple=False).flatten()[:-1]
        continuation = full_codes[prefix_length:].to(
            device=sample["labels"].device, dtype=torch.long
        )
        if frame_positions.numel() != continuation.shape[0]:
            raise RuntimeError("main and fast Talker teacher targets are misaligned")
        all_codes = torch.full(
            (sample["labels"].shape[0], 16),
            -100,
            device=sample["labels"].device,
            dtype=torch.long,
        )
        all_codes[frame_positions] = continuation
        samples.append(sample)
        codec_labels.append(all_codes)

    inputs_embeds = pad_sequence(
        [sample["inputs_embeds"] for sample in samples], batch_first=True, padding_value=0.0
    )
    main_labels = pad_sequence(
        [sample["labels"] for sample in samples], batch_first=True, padding_value=-100
    )
    attention_mask = pad_sequence(
        [sample["attention_mask"] for sample in samples], batch_first=True, padding_value=0
    )
    all_codec_labels = pad_sequence(codec_labels, batch_first=True, padding_value=-100)

    outputs = talker(
        inputs_embeds=inputs_embeds[:, :-1],
        attention_mask=attention_mask[:, :-1],
        labels=None,
        output_hidden_states=True,
        use_cache=False,
    )
    main_targets = main_labels[:, 1:]
    main_losses = F.cross_entropy(
        outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
        main_targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(main_targets)
    main_mask = main_targets.ne(-100)

    hidden_stack = outputs.hidden_states[0]
    if not hidden_stack:
        raise RuntimeError("main Talker did not return hidden states for fast-Talker SFT")
    main_hidden = hidden_stack[-1]
    fast_targets_full = all_codec_labels[:, 1:]
    fast_frame_mask = fast_targets_full[..., 0].ne(-100)
    selected_hidden = main_hidden[fast_frame_mask]
    selected_codes = fast_targets_full[fast_frame_mask]
    if selected_hidden.numel() == 0:
        raise RuntimeError("joint Talker batch contains no continuation frames")
    # --- A10-qwen local edit (E10 P3, 2026-08-31): the single bulk fast-predictor
    # forward over ALL continuation frames OOMs on 48 GB for ~15-min units (probe,
    # reports/qwen_p3_plan.md §6 in .). On the training
    # path (grads on, no token metrics) run it in CHECKPOINTED chunks: loss-identical
    # (token-sum before normalization), activation memory ~ one chunk.
    _FAST_CHUNK = 2048
    if (torch.is_grad_enabled() and not return_token_metrics
            and selected_hidden.shape[0] > _FAST_CHUNK):
        from torch.utils.checkpoint import checkpoint as _ckpt

        def _fast_fwd(codes_c, hidden_c):
            logits_c, _ = talker.forward_sub_talker_finetune(codes_c, hidden_c)
            return logits_c

        losses_parts = []
        for _i in range(0, selected_hidden.shape[0], _FAST_CHUNK):
            _codes_c = selected_codes[_i:_i + _FAST_CHUNK]
            _hidden_c = selected_hidden[_i:_i + _FAST_CHUNK]
            _logits_c = _ckpt(_fast_fwd, _codes_c, _hidden_c, use_reentrant=False)
            losses_parts.append(F.cross_entropy(
                _logits_c.float().reshape(-1, _logits_c.shape[-1]),
                _codes_c[:, 1:].reshape(-1), reduction="none",
            ).reshape(_codes_c.shape[0], -1))
        fast_losses = torch.cat(losses_parts, dim=0)
        fast_targets = selected_codes[:, 1:]
        fast_logits = None  # metrics path never taken on this branch
    else:
        fast_logits, _ = talker.forward_sub_talker_finetune(selected_codes, selected_hidden)
        fast_targets = selected_codes[:, 1:]
        fast_losses = F.cross_entropy(
            fast_logits.float().reshape(-1, fast_logits.shape[-1]),
            fast_targets.reshape(-1),
            reduction="none",
        ).reshape_as(fast_targets)

    main_sum = main_losses.sum()
    main_count = main_mask.sum()
    fast_sum = fast_losses.sum()
    fast_count = torch.as_tensor(fast_targets.numel(), device=fast_sum.device)
    if distributed:
        main_mean = _global_token_mean(main_sum, main_count)
        fast_mean = _global_token_mean(fast_sum, fast_count)
    else:
        main_mean = main_sum / main_count.clamp_min(1)
        fast_mean = fast_sum / fast_count.clamp_min(1)
    loss = main_mean + code_predictor_loss_weight * fast_mean
    metrics: dict[str, torch.Tensor] = {
        "main_loss": main_mean.detach(),
        "fast_loss": fast_mean.detach(),
    }
    if not return_token_metrics:
        return loss, metrics

    with torch.no_grad():
        main_top = outputs.logits.topk(k=min(5, outputs.logits.shape[-1]), dim=-1).indices
        main_top1 = (main_top[..., 0].eq(main_targets) & main_mask).sum(dim=1)
        main_top5 = (
            main_top.eq(main_targets.unsqueeze(-1)).any(dim=-1) & main_mask
        ).sum(dim=1)
        fast_top = fast_logits.topk(k=min(5, fast_logits.shape[-1]), dim=-1).indices
        fast_top1_frames = fast_top[..., 0].eq(fast_targets).sum(dim=1)
        fast_top5_frames = fast_top.eq(fast_targets.unsqueeze(-1)).any(dim=-1).sum(dim=1)
        per_sample_fast_loss = fast_losses.new_zeros((len(samples),))
        per_sample_fast_count = fast_losses.new_zeros((len(samples),))
        per_sample_fast_top1 = fast_losses.new_zeros((len(samples),))
        per_sample_fast_top5 = fast_losses.new_zeros((len(samples),))
        owners = fast_frame_mask.nonzero(as_tuple=False)[:, 0]
        per_frame_loss = fast_losses.sum(dim=1)
        per_frame_count = torch.full_like(per_frame_loss, fast_targets.shape[1])
        per_sample_fast_loss.scatter_add_(0, owners, per_frame_loss)
        per_sample_fast_count.scatter_add_(0, owners, per_frame_count)
        per_sample_fast_top1.scatter_add_(0, owners, fast_top1_frames.to(per_frame_loss))
        per_sample_fast_top5.scatter_add_(0, owners, fast_top5_frames.to(per_frame_loss))
        metrics["per_sample_metrics"] = torch.stack(
            (
                (main_losses * main_mask).sum(dim=1),
                main_mask.sum(dim=1),
                main_top1,
                main_top5,
                per_sample_fast_loss,
                per_sample_fast_count,
                per_sample_fast_top1,
                per_sample_fast_top5,
            ),
            dim=1,
        ).float()
    return loss, metrics


def compute_all_talker_validation_metrics(prediction: Any) -> dict[str, float]:
    """Aggregate main and fast Talker token metrics emitted by the wrapper."""

    predictions = prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    values = np.asarray(predictions, dtype=np.float64).reshape(-1, 8).sum(axis=0)
    main_loss, main_count, main_top1, main_top5 = values[:4]
    fast_loss, fast_count, fast_top1, fast_top5 = values[4:]
    main_denominator = max(main_count, 1.0)
    fast_denominator = max(fast_count, 1.0)
    return {
        "codec_token_loss": main_loss / main_denominator,
        "token_accuracy_top1": main_top1 / main_denominator,
        "token_accuracy_top5": main_top5 / main_denominator,
        "target_tokens": main_count,
        "fast_codec_token_loss": fast_loss / fast_denominator,
        "fast_token_accuracy_top1": fast_top1 / fast_denominator,
        "fast_token_accuracy_top5": fast_top5 / fast_denominator,
        "fast_target_tokens": fast_count,
    }


class TalkerGradientMetricsCallback(TrainerCallback):
    """Expose separate slow/fast gradient norms at the normal log cadence."""

    def __init__(self) -> None:
        self.pending: dict[str, float] = {}

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        next_step = int(state.global_step) + 1
        interval = max(1, int(args.logging_steps))
        if next_step != 1 and next_step % interval:
            return
        model = core_model(kwargs["model"])
        device = next(model.talker.parameters()).device
        squares = {
            "grad_norm/main_talker": torch.zeros((), device=device, dtype=torch.float32),
            "grad_norm/code_predictor": torch.zeros((), device=device, dtype=torch.float32),
            "grad_norm/code_predictor_transformer": torch.zeros((), device=device, dtype=torch.float32),
            "grad_norm/code_predictor_heads": torch.zeros((), device=device, dtype=torch.float32),
            "grad_norm/code_predictor_embeddings": torch.zeros((), device=device, dtype=torch.float32),
        }
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not name.startswith("talker."):
                continue
            key = (
                "grad_norm/code_predictor"
                if name.startswith("talker.code_predictor.")
                else "grad_norm/main_talker"
            )
            squares[key] += parameter.grad.detach().float().square().sum()
            if key == "grad_norm/code_predictor":
                if ".lm_head." in name:
                    detail = "grad_norm/code_predictor_heads"
                elif ".codec_embedding." in name:
                    detail = "grad_norm/code_predictor_embeddings"
                else:
                    detail = "grad_norm/code_predictor_transformer"
                squares[detail] += parameter.grad.detach().float().square().sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for value in squares.values():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        self.pending = {name: float(value.sqrt().item()) for name, value in squares.items()}

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> None:
        if logs is not None and self.pending:
            logs.update(self.pending)
            self.pending = {}


def _wav_bytes(value: Any) -> bytes | None:
    """Normalize the binary representation returned by datasets>=5."""

    if isinstance(value, bytes):
        return value or None
    if isinstance(value, (bytearray, memoryview)):
        payload = bytes(value)
        return payload or None
    if not isinstance(value, Mapping):
        return None
    payload = value.get("bytes")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        payload = bytes(payload)
        return payload or None
    path = value.get("path")
    if isinstance(path, str) and path and Path(path).is_file():
        payload = Path(path).read_bytes()
        return payload or None
    return None


def _finite_float(value: Any, default: float) -> float:
    """Coerce heterogeneous JSON numbers and fail closed on invalid values."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


class BinarySafeSovaCandidateMapper(SovaCandidateMapper):
    """Adapt YouTube Balalaika rows to the existing stressed-text pipeline."""

    def _decode_audio(self, payload: bytes) -> np.ndarray:
        """Decode and resample FLAC in one TorchCodec call on a CPU worker."""

        from torchcodec.decoders import AudioDecoder

        samples = AudioDecoder(
            payload,
            sample_rate=TARGET_SR,
            num_channels=1,
        ).get_all_samples()
        wav = samples.data.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        return np.clip(wav, -1.0, 1.0).astype(np.float32, copy=False)

    def _candidate_from_normalized_row(
        self,
        row: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Select one transcript directly from the dataset confidence score.

        Unlike the inherited SOVA path, YouTube Balalaika already provides an
        aggregate ``asr_consistency`` percentage.  Using that score avoids a
        second Python WER gate over four hypotheses for every streamed row.
        """

        metadata = row.get("json") or {}
        duration = _finite_float(metadata.get("total_duration"), 0.0)
        if not self.args.min_duration_sec <= duration <= self.args.max_duration_sec:
            self.stats["skip_duration"] += 1
            return None
        if self.args.require_single_speaker and not bool(metadata.get("is_single_speaker")):
            self.stats["skip_multispeaker"] += 1
            return None
        if _finite_float(metadata.get("DistillMOS"), 0.0) < self.args.min_distill_mos:
            self.stats["skip_mos"] += 1
            return None
        if _finite_float(metadata.get("music_prob"), 1.0) > self.args.max_music_prob:
            self.stats["skip_music"] += 1
            return None

        consistency = _finite_float(metadata.get("asr_consistency"), 0.0)
        threshold = float(getattr(self.args, "asr_consistency_threshold", 75.0))
        if consistency >= threshold:
            source_field = "gigaam-v3-e2e-ctc.txt"
            self.stats["transcript_gigaam_e2e"] += 1
        else:
            source_field = "punct.txt"
            self.stats["transcript_punct_low_consistency"] += 1
        source_text = str(metadata.get(source_field) or "").strip()
        if not source_text:
            self.stats[f"skip_missing_{source_field}"] += 1
            return None
        if (
            len(source_text.split()) < self.args.min_words
            or len(source_text) > self.args.max_text_chars
        ):
            self.stats["skip_text_length"] += 1
            return None

        try:
            stressed_text = apply_silero_stress(self.accentor, source_text)
        except Exception:
            self.stats["skip_stress_error"] += 1
            return None

        wav_bytes = row.get("wav")
        if not isinstance(wav_bytes, bytes) or not wav_bytes:
            self.stats["skip_missing_wav"] += 1
            return None
        try:
            wav = self._decode_audio(wav_bytes)
        except Exception:
            self.stats["skip_audio_decode"] += 1
            return None
        decoded_duration = float(wav.shape[0] / TARGET_SR)
        if not self.args.min_duration_sec <= decoded_duration <= self.args.max_duration_sec:
            self.stats["skip_decoded_duration"] += 1
            return None

        source_url = str(row.get("__url__") or "")
        source_key = str(row.get("__key__") or "")
        return {
            "key": f"{source_url}#{source_key}",
            "source_key": source_key,
            "source_url": source_url,
            "wav": wav,
            "duration": decoded_duration,
            "text": stressed_text,
            "source_text": source_text,
            "transcript_source": source_field,
            "gigaam_e2e_rover_wer": None,
            "metadata_accent_text": metadata.get("accent.txt"),
            "asr_consensus_wer": {"asr_consistency_percent": consistency},
            "quality": {
                "distill_mos": _finite_float(metadata.get("DistillMOS"), 0.0),
                "music_prob": _finite_float(metadata.get("music_prob"), 1.0),
                "silence_percent": _finite_float(metadata.get("silence_percent"), 0.0),
                "score_bonafide": _finite_float(metadata.get("score_bonafide"), 0.0),
            },
        }

    def __call__(self, source_row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(source_row)
        audio_column = str(getattr(self.args, "audio_column", "flac"))
        payload = _wav_bytes(normalized.get(audio_column))
        if payload is not None:
            normalized["wav"] = payload

        metadata = normalized.get("json")
        if isinstance(metadata, Mapping):
            metadata = dict(metadata)
            # A few source rows contain ``""`` in members inferred as floats.
            # Normalize every number consumed by the strict inherited filter.
            for name in (
                "start",
                "end",
                "total_duration",
                "silence_percent",
                "max_silence_duration",
                "crest_factor",
                "asr_consistency",
                "DistillMOS",
                "score_bonafide",
                "score_spoof",
            ):
                metadata[name] = _finite_float(metadata.get(name), 0.0)
            metadata["music_prob"] = _finite_float(metadata.get("music_prob"), 1.0)
            asr = metadata.get("asr")
            asr = asr if isinstance(asr, Mapping) else {}
            metadata.update(
                {
                    # YouTube rows keep hypotheses in nested ``asr`` fields,
                    # while SOVA exposes the canonical ``*.txt`` columns
                    # directly. Preserve either representation instead of
                    # replacing valid SOVA text with a missing nested value.
                    "giga_ctc.txt": asr.get("gigaam-v3-ctc")
                    or metadata.get("giga_ctc.txt"),
                    "giga_rnnt.txt": asr.get("gigaam-v3-rnnt")
                    or metadata.get("giga_rnnt.txt"),
                    "vosk.txt": asr.get("vosk-model-ru")
                    or metadata.get("vosk.txt"),
                    "gigaam-v3-e2e-ctc.txt": asr.get("gigaam-v3-e2e-ctc")
                    or metadata.get("gigaam-v3-e2e-ctc.txt"),
                    "rover.txt": metadata.get("rover") or metadata.get("rover.txt"),
                    "punct.txt": metadata.get("punct") or metadata.get("punct.txt"),
                    "accent.txt": metadata.get("accent") or metadata.get("accent.txt"),
                }
            )
            normalized["json"] = metadata

        candidate = self._candidate_from_normalized_row(normalized)
        if candidate is None:
            return {"_accepted": False, "_candidate": {}}
        if not str(candidate.get("text") or "").strip():
            self.stats["skip_empty_stressed_text"] += 1
            return {"_accepted": False, "_candidate": {}}
        candidate["_silero_stress_applied"] = True
        self.stats["accepted_candidates"] += 1
        return {"_accepted": True, "_candidate": candidate}


class PreparedQwenFullUtteranceMapper:
    """Reconstruct full utterances from the high-agreement Qwen cache.

    The published cache stores a reference and target produced by one
    timestamp split.  We concatenate both codec sequences and both texts back
    into the original utterance, then choose our own deterministic acoustic
    prefix.  Consequently no timestamp or text boundary defines the SFT loss.
    Audio codec extraction and speaker encoding are already complete, so the
    hot path never decodes MP3 or runs the frozen speech tokenizer.
    """

    prepared_rows = True

    def __init__(
        self,
        args: SimpleNamespace,
        processor: Any,
        excluded_keys: set[str] | None = None,
    ) -> None:
        if processor is None:
            raise ValueError("The prepared Qwen adapter requires a TTS processor")
        self.args = args
        self.processor = processor
        self.excluded_keys = frozenset(excluded_keys or ())
        self.stats: defaultdict[str, int] = defaultdict(int)
        self._accentor: Any = None
        self._last_source_key: str | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["stats"] = defaultdict(int)
        state["_accentor"] = None
        state["_last_source_key"] = None
        return state

    @property
    def accentor(self) -> Any:
        if self._accentor is None:
            from silero_stress import load_accentor

            grad_enabled = torch.is_grad_enabled()
            try:
                with torch.no_grad():
                    self._accentor = load_accentor()
            finally:
                torch.set_grad_enabled(grad_enabled)
        return self._accentor

    def _reject(self, reason: str) -> dict[str, Any]:
        self.stats[reason] += 1
        return {"_accepted": False}

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        source_key = str(row.get("source_record_id") or "").strip()
        stable_key = f"qwen-high-agreement#{source_key}"
        if not source_key:
            return self._reject("skip_missing_source_record_id")
        if stable_key in self.excluded_keys:
            return self._reject("skip_validation_key")

        # The publication writes all split variants of one source record next
        # to each other. Keep exactly the first row in each group before the
        # accepted stream is shuffled. This avoids an unbounded in-memory set,
        # removes repeated full utterances, and accepts roughly one row per
        # source rather than one rare profile/tier combination.
        if bool(getattr(self.args, "prepared_deduplicate_consecutive", True)):
            if source_key == self._last_source_key:
                return self._reject("skip_duplicate_source_record_id")
            self._last_source_key = source_key

        expected = {
            "agreement_bucket": "agreement_ge_0_95",
            "text_source": str(
                getattr(self.args, "prepared_text_source", "rover_punctuated_accented")
            ),
        }
        optional_expected = {
            "orientation": getattr(self.args, "prepared_orientation", None),
            "profile": getattr(self.args, "prepared_profile", None),
            "boundary_type": getattr(self.args, "prepared_boundary_type", None),
            "boundary_tier": getattr(self.args, "prepared_boundary_tier", None),
        }
        expected.update(
            (field, str(value))
            for field, value in optional_expected.items()
            if value is not None and str(value).strip()
        )
        for field, value in expected.items():
            if str(row.get(field) or "") != value:
                return self._reject(f"skip_{field}")

        agreement = _finite_float(row.get("asr_agreement_mean"), 0.0)
        if agreement < float(getattr(self.args, "min_asr_agreement", 0.97)):
            return self._reject("skip_asr_agreement")
        duration = _finite_float(row.get("ref_duration"), 0.0) + _finite_float(
            row.get("duration"), 0.0
        )
        if not self.args.min_duration_sec <= duration <= self.args.max_duration_sec:
            return self._reject("skip_duration")

        ref_codes = row.get("ref_audio_codes")
        target_codes = row.get("audio_codes")
        if not isinstance(ref_codes, list) or not isinstance(target_codes, list):
            return self._reject("skip_missing_codes")
        orientation = str(row.get("orientation") or "")
        if orientation == "prefix_reference":
            full_codes = [*ref_codes, *target_codes]
            text_parts = (row.get("ref_text"), row.get("text"))
        elif orientation == "suffix_reference":
            full_codes = [*target_codes, *ref_codes]
            text_parts = (row.get("text"), row.get("ref_text"))
        else:
            return self._reject("skip_unknown_orientation")
        if not full_codes or any(
            not isinstance(frame, list) or len(frame) != 16 for frame in full_codes
        ):
            return self._reject("skip_bad_code_shape")

        speaker_embedding = row.get("ref_spk_embedding")
        if not isinstance(speaker_embedding, list) or len(speaker_embedding) != 2048:
            return self._reject("skip_bad_speaker_embedding")

        # Remove the source accent marks: every accepted row must be accented
        # again by the local Silero Stress contract, not trusted transitively.
        source_text = " ".join(
            part.strip()
            for part in (str(value or "") for value in text_parts)
            if part.strip()
        ).replace("+", "")
        if (
            len(source_text.split()) < self.args.min_words
            or len(source_text) > self.args.max_text_chars
        ):
            return self._reject("skip_text_length")
        try:
            stressed_text = apply_silero_stress(self.accentor, source_text)
        except Exception:
            return self._reject("skip_stress_error")
        if not stressed_text.strip():
            return self._reject("skip_empty_stressed_text")

        min_prefix_frames = round(self.args.min_prefix_sec * self.args.codec_fps)
        max_prefix_frames = round(self.args.max_prefix_sec * self.args.codec_fps)
        min_continuation_frames = round(
            self.args.min_continuation_sec * self.args.codec_fps
        )
        try:
            prefix_frames = deterministic_prefix_frames(
                key=stable_key,
                total_frames=len(full_codes),
                seed=self.args.seed,
                min_prefix_frames=min_prefix_frames,
                max_prefix_frames=max_prefix_frames,
                min_continuation_frames=min_continuation_frames,
            )
        except ValueError:
            return self._reject("skip_insufficient_codec_frames")

        text_ids = tokenize_tts_text(
            self.processor, build_assistant_text(stressed_text)
        )
        self.stats["accepted_candidates"] += 1
        return {
            "_accepted": True,
            "_silero_stress_applied": True,
            "full_codes": full_codes,
            "speaker_embedding": speaker_embedding,
            "text_ids": text_ids,
            "text": stressed_text,
            "source_text": source_text,
            "transcript_source": "rover_punctuated_accented+silero_stress",
            "prefix_frames": prefix_frames,
            "continuation_frames": len(full_codes) - prefix_frames,
            "duration": duration,
            "language": self.args.language,
            "dataset": self.args.dataset_name,
            "dataset_revision": self.args.dataset_revision,
            "source_key": source_key,
            "source_url": "qwen-high-agreement",
            "asr_agreement_mean": agreement,
            "quality": {"agreement_bucket": row.get("agreement_bucket")},
        }


def _accepted(value: bool) -> bool:
    return bool(value)


def _streaming_dataset(args: SimpleNamespace, token: str | None) -> Any:
    if not args.streaming:
        raise ValueError("This pipeline requires streaming=true")
    dataset_format = str(getattr(args, "dataset_format", "webdataset_audio"))
    if dataset_format == "qwen_prepared_full_utterance":
        from datasets import load_dataset

        data_file = str(getattr(args, "dataset_data_file", "") or "").strip()
        if not data_file:
            raise ValueError("dataset_data_file is required for the prepared Qwen cache")
        local_data_file = Path(data_file).expanduser()
        if local_data_file.is_absolute():
            # --- A10-qwen local edit (E10 P3, 2026-08-31): accept a glob of shard files.
            # A single local JSONL yields dataset.num_shards == 1, which caps the
            # DataLoader at ONE worker and serializes Silero Stress on the hot path
            # (~10-20 s per long row). N shard files -> N-way worker parallelism.
            if any(ch in str(local_data_file) for ch in "*?["):
                import glob as _glob
                _matches = sorted(_glob.glob(str(local_data_file)))
                if not _matches:
                    raise FileNotFoundError(
                        f"local prepared Qwen JSONL glob matched nothing: {local_data_file}"
                    )
                source = _matches
            elif not local_data_file.is_file():
                raise FileNotFoundError(
                    f"local prepared Qwen JSONL does not exist: {local_data_file}"
                )
            else:
                source = str(local_data_file.resolve())
        else:
            source = (
                f"hf://datasets/{args.dataset_name}@{args.dataset_revision}/"
                f"{data_file.lstrip('/')}"
            )
        dataset = load_dataset(
            "json",
            data_files={args.dataset_split: source},
            split=args.dataset_split,
            streaming=True,
            token=token,
        )
        return dataset
    if dataset_format != "webdataset_audio":
        raise ValueError(f"Unsupported dataset_format={dataset_format!r}")
    audio_column = str(args.audio_column)
    dataset = load_untyped_streaming_dataset(
        args.dataset_name,
        split=args.dataset_split,
        revision=args.dataset_revision,
        token=token,
        required_columns=(audio_column, "json", "__key__", "__url__"),
        max_retries=int(getattr(args, "stream_max_retries", 3)),
        retry_backoff_sec=float(getattr(args, "stream_retry_backoff_sec", 1.0)),
        skip_failed_shards=bool(getattr(args, "stream_skip_failed_shards", True)),
    )
    if args.shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return dataset


def _map_candidates(
    dataset: Any,
    args: SimpleNamespace,
    excluded_keys: set[str] | None,
    processor: Any | None,
) -> tuple[Any, Any]:
    dataset_format = str(getattr(args, "dataset_format", "webdataset_audio"))
    if dataset_format == "qwen_prepared_full_utterance":
        mapper: Any = PreparedQwenFullUtteranceMapper(args, processor, excluded_keys)
    else:
        mapper = BinarySafeSovaCandidateMapper(args, excluded_keys)
    source_columns = list(
        dataset.column_names
        or (str(args.audio_column), "json", "__key__", "__url__")
    )
    dataset = dataset.map(mapper, remove_columns=source_columns)
    dataset = dataset.filter(_accepted, input_columns=["_accepted"])
    dataset = dataset.remove_columns(["_accepted"])
    if dataset_format == "qwen_prepared_full_utterance" and args.shuffle_buffer > 1:
        # Deduplication relies on the publication's adjacent source groups, so
        # randomize only the already accepted full-utterance stream.
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return dataset, mapper


def build_tts_candidate_dataset(
    args: SimpleNamespace,
    token: str | None,
    *,
    excluded_keys: set[str] | None = None,
    repeat: bool = False,
    preflight_rows: int = 0,
    processor: Any | None = None,
) -> tuple[Any, Any]:
    """Build the lazy stream and reject an empty repeated stream before training."""

    if repeat:
        if preflight_rows <= 0:
            raise ValueError("A repeated stream requires stream_preflight_rows > 0")
        probe_source = _streaming_dataset(args, token).take(preflight_rows)
        probe, probe_mapper = _map_candidates(
            probe_source, args, excluded_keys, processor
        )
        first = next(iter(probe.take(1)), None)
        if first is None:
            raise RuntimeError(
                "No acceptable training row was found in the first "
                f"{preflight_rows} source rows; refusing to repeat an empty stream. "
                f"Filter stats: {dict(probe_mapper.stats)}"
            )
        stress_applied = bool(first.get("_silero_stress_applied")) or bool(
            (first.get("_candidate") or {}).get("_silero_stress_applied")
        )
        if not stress_applied:
            raise RuntimeError("Silero Stress preprocessing was not applied")

    dataset, mapper = _map_candidates(
        _streaming_dataset(args, token), args, excluded_keys, processor
    )
    if repeat:
        dataset = dataset.repeat(None)
    return dataset, mapper


class QwenTtsSFTTrainer(SFTTrainer):
    """SFTTrainer without its text-token-only metric assumptions.

    The Qwen task wrappers compute codec continuation loss in ``forward``.
    Calling Transformers' base implementation preserves that loss while TRL
    still owns checkpointing, Accelerate, callbacks and configuration.
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> Any:
        inputs.pop("_prediction_loss_only", None)
        return Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


class _TrackioWriter:
    """Small SummaryWriter-compatible bridge used by existing audio validation."""

    def __init__(self) -> None:
        self.pending: dict[int, dict[str, Any]] = defaultdict(dict)

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        self.pending[int(step)][tag] = float(value)

    def add_text(self, tag: str, value: Any, step: int, **_: Any) -> None:
        self.pending[int(step)][tag] = str(value)

    def add_audio(
        self,
        tag: str,
        value: Any,
        step: int,
        sample_rate: int = 24_000,
        **_: Any,
    ) -> None:
        import trackio

        audio = torch.as_tensor(value).detach().float().cpu().numpy().squeeze()
        self.pending[int(step)][tag] = trackio.Audio(
            np.asarray(audio), caption=tag, sample_rate=int(sample_rate), format="wav"
        )

    def flush(self) -> None:
        import trackio

        for step in sorted(self.pending):
            trackio.log(self.pending[step], step=step)
        self.pending.clear()

    def close(self) -> None:
        self.flush()


class TrackioAudioValidationCallback(AudioValidationCallback):
    """Reuse audio validation while sending its scalars and WAVs to Trackio."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._first_evaluation = True

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        step = int(state.global_step)
        interval = int(self.pipeline_args.audio_validation_steps)
        run_on_start = bool(self.pipeline_args.audio_validation_on_start)
        should_run = step % interval == 0 or (
            self._first_evaluation and run_on_start
        )
        self._first_evaluation = False
        if not should_run:
            return control
        return super().on_evaluate(
            args,
            state,
            control,
            metrics=metrics,
            **kwargs,
        )

    def _run(self, step: int, eval_metrics: dict[str, float]) -> None:
        if self.writer is None:
            self.writer = _TrackioWriter()
        super()._run(step, eval_metrics)


__all__ = [
    "MainTalkerDataCollator",
    "MainTalkerTrainingModel",
    "QwenTtsSFTTrainer",
    "TrackioAudioValidationCallback",
    "build_tts_candidate_dataset",
]
