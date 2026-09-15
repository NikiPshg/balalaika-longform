"""Codec-aware TRL GRPO support for Qwen3-TTS voice-clone training.

The stock :class:`trl.GRPOTrainer` assumes that a completion is a single text
token stream.  Qwen3-TTS instead samples a ``[frames, 16]`` codec tensor and
only codebook zero is produced by the trainable main Talker head.
This module keeps TRL's sampler/checkpoint/DDP/Trackio infrastructure while
replacing rollout preparation and the policy loss with that real contract.

Imports are intentionally model-light: importing this module does not load a
checkpoint, a speech tokenizer, or the ASR model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import random
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel
from trl import GRPOConfig, GRPOTrainer


NUM_CODEBOOKS = 16
SUPPORTED_TRL_VERSION = "1.9.2"


def validate_trl_runtime_version() -> str:
    """Protect the subclass contract from incompatible TRL internals."""

    try:
        actual = package_version("trl")
    except PackageNotFoundError as error:
        raise RuntimeError("TRL is required for codec GRPO") from error
    if actual != SUPPORTED_TRL_VERSION:
        raise RuntimeError(
            f"codec GRPO v1 requires trl=={SUPPORTED_TRL_VERSION}, found {actual}"
        )
    return actual


def validate_codec_codes(
    codes: torch.Tensor,
    *,
    name: str = "codec codes",
    num_codebooks: int = NUM_CODEBOOKS,
) -> torch.Tensor:
    """Validate and normalize the public Qwen codec rollout contract."""

    if not torch.is_tensor(codes):
        raise TypeError(f"{name} must be a torch.Tensor")
    if codes.ndim != 2 or codes.shape[0] <= 0 or codes.shape[1] != num_codebooks:
        raise ValueError(f"{name} must have shape [T, {num_codebooks}]")
    if codes.dtype == torch.bool or torch.is_floating_point(codes) or torch.is_complex(codes):
        raise TypeError(f"{name} must contain integer token ids")
    return codes.to(dtype=torch.long).contiguous()


def compute_group_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    *,
    scale_rewards: str = "none",
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample advantages, group means and group standard deviations.

    ``rewards`` must already be gathered in global sampler order.  Keeping this
    helper independent of Accelerate makes the most important DDP contract easy
    to unit test.
    """

    if not torch.is_tensor(rewards) or rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional tensor")
    if isinstance(num_generations, bool) or num_generations < 2:
        raise ValueError("num_generations must be an integer of at least two")
    if rewards.numel() == 0 or rewards.numel() % num_generations:
        raise ValueError("global reward count must be divisible by num_generations")
    if not torch.is_floating_point(rewards) or not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must contain finite floating-point values")
    if scale_rewards not in {"none", "group"}:
        raise ValueError("codec GRPO supports scale_rewards='none' or 'group'")

    grouped = rewards.reshape(-1, num_generations)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, correction=1, keepdim=True)
    advantages = grouped - means
    if scale_rewards == "group":
        advantages = advantages / (stds + float(epsilon))
    return advantages.reshape(-1), means.expand_as(grouped).reshape(-1), stds.expand_as(grouped).reshape(-1)


def compute_gated_pairwise_advantages(
    critical_rewards: torch.Tensor,
    utterance_rewards: torch.Tensor,
    pace_rewards: torch.Tensor,
    valid: torch.Tensor,
    type_scales: torch.Tensor,
    num_generations: int,
    *,
    critical_margin: float = 0.05,
    minimum_critical_reward: float = 0.0,
    maximum_negative_critical_reward: float = 1.0,
    minimum_utterance_reward: float = 0.45,
    minimum_pace_reward: float = 0.80,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one safe positive/negative preference pair per rollout group.

    The positive must pass critical, whole-utterance and pace gates.  The
    negative must be genuinely weak on the critical span (invalid audio has a
    zero critical reward and remains eligible).  Groups with no feasible pair
    or insufficient critical separation get zero advantages and therefore no
    policy update.
    """

    tensors = (critical_rewards, utterance_rewards, pace_rewards, valid, type_scales)
    if any(not torch.is_tensor(value) or value.ndim != 1 for value in tensors):
        raise ValueError("pairwise features must be one-dimensional tensors")
    if len({value.numel() for value in tensors}) != 1:
        raise ValueError("pairwise feature tensors must have equal lengths")
    if isinstance(num_generations, bool) or num_generations < 2:
        raise ValueError("num_generations must be an integer of at least two")
    if not critical_rewards.numel() or critical_rewards.numel() % num_generations:
        raise ValueError("pairwise feature count must be divisible by num_generations")
    numeric = (critical_rewards, utterance_rewards, pace_rewards, type_scales)
    if any(not bool(torch.isfinite(value).all()) for value in numeric):
        raise ValueError("pairwise features must be finite")
    thresholds = (
        critical_margin,
        minimum_critical_reward,
        maximum_negative_critical_reward,
        minimum_utterance_reward,
        minimum_pace_reward,
    )
    if any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("pairwise gates and margin must be finite and in [0, 1]")
    if bool((type_scales < 0).any()) or bool((type_scales > 8).any()):
        raise ValueError("pairwise type scales must be in [0, 8]")

    advantages = torch.zeros_like(critical_rewards)
    active = torch.zeros(
        critical_rewards.numel() // num_generations,
        device=critical_rewards.device,
        dtype=torch.float32,
    )
    groups = zip(
        critical_rewards.reshape(-1, num_generations),
        utterance_rewards.reshape(-1, num_generations),
        pace_rewards.reshape(-1, num_generations),
        valid.reshape(-1, num_generations).bool(),
        type_scales.reshape(-1, num_generations),
        advantages.reshape(-1, num_generations),
        strict=True,
    )
    for group_index, (critical, utterance, pace, is_valid, scales, output) in enumerate(groups):
        feasible = (
            is_valid
            & (critical >= minimum_critical_reward)
            & (utterance >= minimum_utterance_reward)
            & (pace >= minimum_pace_reward)
        )
        if not bool(feasible.any()):
            continue
        masked_positive = torch.where(feasible, critical, torch.full_like(critical, -torch.inf))
        positive = int(torch.argmax(masked_positive).item())
        negative_candidates = critical <= maximum_negative_critical_reward
        if not bool(negative_candidates.any()):
            continue
        masked_negative = torch.where(
            negative_candidates, critical, torch.full_like(critical, torch.inf)
        )
        negative = int(torch.argmin(masked_negative).item())
        if positive == negative or float(critical[positive] - critical[negative]) < critical_margin:
            continue
        scale = float(scales.mean().item())
        output[positive] = scale
        output[negative] = -scale
        active[group_index] = 1.0
    return advantages, active


def compute_gated_best_advantages(
    critical_rewards: torch.Tensor,
    utterance_rewards: torch.Tensor,
    pace_rewards: torch.Tensor,
    valid: torch.Tensor,
    type_scales: torch.Tensor,
    num_generations: int,
    *,
    minimum_critical_reward: float = 0.95,
    minimum_utterance_reward: float = 0.70,
    minimum_pace_reward: float = 0.90,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Promote only one verified best rollout, without a negative update."""

    tensors = (critical_rewards, utterance_rewards, pace_rewards, valid, type_scales)
    if any(not torch.is_tensor(value) or value.ndim != 1 for value in tensors):
        raise ValueError("best-of-group features must be one-dimensional tensors")
    if len({value.numel() for value in tensors}) != 1:
        raise ValueError("best-of-group feature tensors must have equal lengths")
    if isinstance(num_generations, bool) or num_generations < 2:
        raise ValueError("num_generations must be an integer of at least two")
    if not critical_rewards.numel() or critical_rewards.numel() % num_generations:
        raise ValueError("best-of-group feature count must be divisible by num_generations")
    numeric = (critical_rewards, utterance_rewards, pace_rewards, type_scales)
    if any(not bool(torch.isfinite(value).all()) for value in numeric):
        raise ValueError("best-of-group features must be finite")
    thresholds = (minimum_critical_reward, minimum_utterance_reward, minimum_pace_reward)
    if any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("best-of-group gates must be finite and in [0, 1]")

    advantages = torch.zeros_like(critical_rewards)
    active = torch.zeros(
        critical_rewards.numel() // num_generations,
        device=critical_rewards.device,
        dtype=torch.float32,
    )
    groups = zip(
        critical_rewards.reshape(-1, num_generations),
        utterance_rewards.reshape(-1, num_generations),
        pace_rewards.reshape(-1, num_generations),
        valid.reshape(-1, num_generations).bool(),
        type_scales.reshape(-1, num_generations),
        advantages.reshape(-1, num_generations),
        strict=True,
    )
    for group_index, (critical, utterance, pace, is_valid, scales, output) in enumerate(groups):
        feasible = (
            is_valid
            & (critical >= minimum_critical_reward)
            & (utterance >= minimum_utterance_reward)
            & (pace >= minimum_pace_reward)
        )
        if not bool(feasible.any()):
            continue
        # The non-critical terms are hard safety gates, not optimization
        # targets.  Among candidates that pass them, distill the strongest
        # critical-span pronunciation so a slightly better utterance score
        # cannot displace a more accurate name/number realization.
        masked = torch.where(feasible, critical, torch.full_like(critical, -torch.inf))
        positive = int(torch.argmax(masked).item())
        output[positive] = scales[positive]
        active[group_index] = 1.0
    return advantages, active


def process_slice(*, local_batch_size: int, process_index: int, num_processes: int, global_size: int) -> slice:
    """Resolve the rank-local contiguous slice returned by ``Accelerator.gather``."""

    values = (local_batch_size, process_index, num_processes, global_size)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("distributed slice arguments must be integers")
    if local_batch_size <= 0 or num_processes <= 0 or not 0 <= process_index < num_processes:
        raise ValueError("invalid distributed slice arguments")
    if global_size != local_batch_size * num_processes:
        raise ValueError("gathered batch size does not match world size")
    start = process_index * local_batch_size
    return slice(start, start + local_batch_size)


def pad_teacher_samples(
    samples: Sequence[Mapping[str, torch.Tensor]], *, detach: bool = True
) -> dict[str, torch.Tensor]:
    """Pad teacher-forced samples and return a tensor-only batch."""

    if not samples:
        raise ValueError("at least one teacher sample is required")
    required = {"inputs_embeds", "labels", "attention_mask"}
    for index, sample in enumerate(samples):
        if not required.issubset(sample):
            raise ValueError(f"teacher sample {index} is missing {sorted(required - set(sample))}")
        length = sample["inputs_embeds"].shape[0]
        if sample["inputs_embeds"].ndim != 2:
            raise ValueError("inputs_embeds must have shape [L, H]")
        if sample["labels"].shape != (length,) or sample["attention_mask"].shape != (length,):
            raise ValueError("teacher labels and attention mask must match the embedding length")
        if not bool((sample["labels"] != -100).any()):
            raise ValueError("every teacher sample must contain at least one supervised codebook-0 token")
    def maybe_detach(value: torch.Tensor) -> torch.Tensor:
        return value.detach() if detach else value

    return {
        "tts_inputs_embeds": pad_sequence(
            [maybe_detach(sample["inputs_embeds"]) for sample in samples], batch_first=True, padding_value=0.0
        ),
        "tts_labels": pad_sequence(
            [maybe_detach(sample["labels"]) for sample in samples], batch_first=True, padding_value=-100
        ),
        "tts_attention_mask": pad_sequence(
            [maybe_detach(sample["attention_mask"]) for sample in samples], batch_first=True, padding_value=0
        ),
    }


def codebook0_logps(logits: torch.Tensor, shifted_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather codebook-zero log-probabilities aligned to shifted teacher labels."""

    if logits.ndim != 3 or shifted_labels.ndim != 2 or logits.shape[:2] != shifted_labels.shape:
        raise ValueError("logits [B, T, V] and shifted labels [B, T] must align")
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    values = F.log_softmax(logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return values, mask


def codec_policy_loss(
    per_token_logps: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    old_per_token_logps: torch.Tensor | None,
    ref_per_token_logps: torch.Tensor | None,
    beta: float,
    epsilon_low: float,
    epsilon_high: float,
    loss_type: str,
    length_norm: float,
    global_active_tokens: torch.Tensor | None = None,
    ddp_world_size: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped codec GRPO with selectable normalization and k3 KL.

    ``grpo`` averages the token mean of each sequence, ``bnpo`` averages over
    all active tokens in the local batch, and ``dr_grpo`` uses the fixed
    ``batch_size * length_norm`` denominator from Dr.GRPO.  The variants
    therefore change the actual optimization math, not merely run labels.
    """

    if per_token_logps.ndim != 2 or mask.shape != per_token_logps.shape:
        raise ValueError("per-token logps and mask must have matching [B, T] shape")
    if advantages.shape != (per_token_logps.shape[0],):
        raise ValueError("advantages must have shape [B]")
    if length_norm <= 0 or not math.isfinite(float(length_norm)):
        raise ValueError("length_norm must be finite and positive")
    if beta < 0 or not math.isfinite(float(beta)):
        raise ValueError("beta must be finite and non-negative")
    if loss_type not in {"grpo", "bnpo", "dr_grpo"}:
        raise ValueError("loss_type must be one of: grpo, bnpo, dr_grpo")
    old = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
    if old.shape != per_token_logps.shape:
        raise ValueError("old policy logps must match current policy logps")
    log_ratio = per_token_logps - old
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    advantage = advantages.to(per_token_logps.dtype).unsqueeze(1)
    policy_per_token = -torch.minimum(ratio * advantage, clipped * advantage)

    if beta:
        if ref_per_token_logps is None or ref_per_token_logps.shape != per_token_logps.shape:
            raise ValueError("reference logps are required when beta is non-zero")
        ref_delta = ref_per_token_logps - per_token_logps
        per_token_kl = ref_delta.exp() - ref_delta - 1.0
    else:
        per_token_kl = torch.zeros_like(per_token_logps)

    per_token_loss = policy_per_token + float(beta) * per_token_kl
    float_mask = mask.to(per_token_loss.dtype)
    active = float_mask.sum().clamp(min=1.0)

    def normalize(values: torch.Tensor) -> torch.Tensor:
        if loss_type == "grpo":
            lengths = float_mask.sum(dim=1).clamp(min=1.0)
            return ((values * float_mask).sum(dim=1) / lengths).mean()
        if loss_type == "bnpo":
            denominator = active if global_active_tokens is None else global_active_tokens.clamp(min=1.0)
            return (values * float_mask).sum() * int(ddp_world_size) / denominator
        denominator = values.shape[0] * float(length_norm)
        return (values * float_mask).sum() / denominator

    loss = normalize(per_token_loss)
    stats = {
        "policy_loss": normalize(policy_per_token),
        "kl": (per_token_kl * float_mask).sum() / active,
        "clip_ratio": ((ratio != clipped).to(float_mask.dtype) * float_mask).sum() / active,
    }
    return loss, stats


def code_predictor_policy_loss(
    per_sample_nll: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Distill selected full-codebook rollouts into the small predictor.

    Only the positive part of the policy advantage is distilled.  This lets the
    main Talker use either centered GRPO or positive-only gated-best selection,
    while codebooks 1..15 are never trained to imitate a rejected rollout.
    """

    if per_sample_nll.ndim != 1 or advantages.shape != per_sample_nll.shape:
        raise ValueError("predictor NLL and advantages must have matching [B] shape")
    if not bool(torch.isfinite(per_sample_nll).all()) or bool((per_sample_nll < 0).any()):
        raise ValueError("predictor NLL must be finite and non-negative")
    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("predictor distillation requires finite advantages")
    positive_advantages = advantages.to(per_sample_nll.dtype).clamp_min(0)
    return (per_sample_nll * positive_advantages).mean()


def validate_tensor_batch(batch: Mapping[str, Any]) -> None:
    """Fail if a prepared rollout would break TRL's shuffle/split buffer."""

    if not batch:
        raise ValueError("prepared rollout batch is empty")
    batch_size: int | None = None
    for name, value in batch.items():
        if not torch.is_tensor(value):
            raise TypeError(f"prepared field {name!r} is not a tensor")
        if value.ndim == 0:
            continue
        batch_size = value.shape[0] if batch_size is None else batch_size
        if value.shape[0] != batch_size:
            raise ValueError(f"prepared field {name!r} has a different batch dimension")
    if batch_size is None or batch_size <= 0:
        raise ValueError("prepared rollout has no batch-first tensor")


@dataclass
class QwenTTSGRPOConfig(GRPOConfig):
    """GRPOConfig with explicit Qwen codec v1 runtime controls."""

    output_dir: str = "artifacts/checkpoints/trl/grpo-main-talker"
    do_train: bool = True
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    max_steps: int = 1_000
    bf16: bool | None = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    remove_unused_columns: bool | None = False
    ddp_find_unused_parameters: bool | None = False
    average_tokens_across_devices: bool = False
    report_to: Optional[str] = "trackio"
    project: str = "qwen3-tts"
    run_name: str | None = "grpo-main-talker"
    num_generations: int | None = 8
    max_completion_length: int | None = 320
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    beta: float = 0.0
    num_iterations: int = 1
    scale_rewards: str = "none"
    loss_type: str = "grpo"
    use_vllm: bool = False
    use_liger_kernel: bool = False
    use_transformers_continuous_batching: bool = False
    use_transformers_paged: bool = False
    log_completions: bool = False
    sync_ref_model: bool = False
    entropy_coef: float = 0.0
    use_adaptive_entropy: bool = False
    language: str = "Russian"
    non_streaming_mode: bool = False
    num_codebooks: int = NUM_CODEBOOKS
    logprob_micro_batch_size: int = 0
    asr_backend: str = "gigaam"
    asr_model_name: str = "gigaam-v3-rnnt"
    asr_model_path: str | None = None
    subtalker_dosample: bool = True
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    subtalker_temperature: float = 0.9
    reference_mode: str = "none"
    reward_number_wer_weight: float = 0.35
    reward_number_cer_weight: float = 0.35
    reward_utterance_wer_weight: float = 0.15
    reward_utterance_cer_weight: float = 0.15
    reward_temperature: float = 3.0
    reward_critical_component_weight: float = 0.0
    reward_utterance_component_weight: float = 1.0
    reward_duration_component_weight: float = 0.0
    reward_critical_wer_weight: float = 0.35
    reward_critical_cer_weight: float = 0.45
    reward_critical_exact_weight: float = 0.20
    reward_critical_worst_span_weight: float = 0.30
    reward_duration_target_words_per_sec: float = 2.5
    reward_duration_min_ratio: float = 0.65
    reward_duration_max_ratio: float = 1.30
    reward_duration_temperature: float = 3.0
    reward_type_default_scale: float = 1.0
    reward_type_fio_scale: float = 1.0
    reward_type_identifier_scale: float = 1.0
    reward_type_measurement_scale: float = 1.0
    reward_type_money_scale: float = 1.0
    reward_type_name_patronymic_scale: float = 1.0
    reward_type_number_scale: float = 1.0
    reward_type_percent_scale: float = 1.0
    reward_type_phone_scale: float = 1.0
    advantage_mode: str = "centered"
    pairwise_critical_margin: float = 0.05
    pairwise_minimum_critical_reward: float = 0.0
    pairwise_maximum_negative_critical_reward: float = 1.0
    pairwise_minimum_utterance_reward: float = 0.45
    pairwise_minimum_pace_reward: float = 0.80
    train_main_talker: bool = True
    train_code_predictor: bool = False
    code_predictor_policy_weight: float = 0.0
    code_predictor_lr_multiplier: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_v1_config(self)


def validate_v1_config(args: GRPOConfig) -> None:
    """Reject TRL modes that do not preserve the codec-aware v1 contract."""

    validate_trl_runtime_version()
    guards = {
        "use_vllm": False,
        "use_transformers_continuous_batching": False,
        "use_liger_kernel": False,
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
        "num_iterations": 1,
        "log_completions": False,
        "sync_ref_model": False,
        "use_adaptive_entropy": False,
        "entropy_coef": 0.0,
    }
    for name, expected in guards.items():
        actual = getattr(args, name, None)
        if actual != expected:
            raise ValueError(f"Qwen codec GRPO v1 requires {name}={expected!r}, got {actual!r}")
    if getattr(args, "use_transformers_paged", False):
        raise ValueError("Qwen codec GRPO v1 requires use_transformers_paged=False")
    if getattr(args, "loss_type", None) not in {"grpo", "bnpo", "dr_grpo"}:
        raise ValueError("Qwen codec GRPO v1 supports loss_type grpo, bnpo, or dr_grpo")
    if getattr(args, "scale_rewards", None) not in {"none", "group"}:
        raise ValueError("Qwen codec GRPO v1 supports scale_rewards='none' or 'group'")
    if getattr(args, "num_generations", None) is None or args.num_generations < 2:
        raise ValueError("Qwen codec GRPO requires num_generations >= 2")
    if getattr(args, "max_completion_length", None) is None or args.max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive")
    if getattr(args, "num_codebooks", NUM_CODEBOOKS) != NUM_CODEBOOKS:
        raise ValueError(f"Qwen codec GRPO v1 requires exactly {NUM_CODEBOOKS} codebooks")
    micro = getattr(args, "logprob_micro_batch_size", 0)
    if isinstance(micro, bool) or not isinstance(micro, int) or micro < 0:
        raise ValueError("logprob_micro_batch_size must be a non-negative integer")
    reference_mode = getattr(args, "reference_mode", "none")
    if reference_mode not in {"none", "initial_policy"}:
        raise ValueError("reference_mode must be 'none' or 'initial_policy'")
    beta = float(getattr(args, "beta", 0.0))
    if (beta > 0) != (reference_mode == "initial_policy"):
        raise ValueError("beta > 0 requires reference_mode='initial_policy'; beta=0 requires 'none'")
    if getattr(args, "deepspeed", None) and beta > 0:
        raise ValueError("frozen full-reference KL v1 does not support DeepSpeed")
    if getattr(args, "fsdp", None) and beta > 0:
        raise ValueError("frozen full-reference KL v1 does not support FSDP")
    weights = [
        float(getattr(args, name, 0.0))
        for name in (
            "reward_number_wer_weight",
            "reward_number_cer_weight",
            "reward_utterance_wer_weight",
            "reward_utterance_cer_weight",
        )
    ]
    if any(not math.isfinite(value) or value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("reward weights must be finite, non-negative, and have a positive sum")
    temperature = float(getattr(args, "reward_temperature", 3.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("reward_temperature must be finite and positive")
    component_weights = [
        float(getattr(args, "reward_critical_component_weight", 0.0)),
        float(getattr(args, "reward_utterance_component_weight", 1.0)),
        float(getattr(args, "reward_duration_component_weight", 0.0)),
    ]
    critical_weights = [
        float(getattr(args, "reward_critical_wer_weight", 0.35)),
        float(getattr(args, "reward_critical_cer_weight", 0.45)),
        float(getattr(args, "reward_critical_exact_weight", 0.20)),
    ]
    if (
        any(not math.isfinite(value) or value < 0 for value in component_weights)
        or sum(component_weights) <= 0
    ):
        raise ValueError("reward component weights must be finite/non-negative with positive sum")
    if (
        any(not math.isfinite(value) or value < 0 for value in critical_weights)
        or sum(critical_weights) <= 0
    ):
        raise ValueError("critical reward weights must be finite/non-negative with positive sum")
    worst = float(getattr(args, "reward_critical_worst_span_weight", 0.30))
    if not math.isfinite(worst) or not 0 <= worst <= 1:
        raise ValueError("reward_critical_worst_span_weight must be in [0, 1]")
    target_wps = float(getattr(args, "reward_duration_target_words_per_sec", 2.5))
    minimum_ratio = float(getattr(args, "reward_duration_min_ratio", 0.65))
    maximum_ratio = float(getattr(args, "reward_duration_max_ratio", 1.30))
    duration_temperature = float(getattr(args, "reward_duration_temperature", 3.0))
    if not math.isfinite(target_wps) or target_wps <= 0:
        raise ValueError("reward_duration_target_words_per_sec must be finite and positive")
    if (
        not math.isfinite(minimum_ratio)
        or not math.isfinite(maximum_ratio)
        or not 0 < minimum_ratio < maximum_ratio
    ):
        raise ValueError("duration reward ratios must satisfy 0 < min < max")
    if not math.isfinite(duration_temperature) or duration_temperature <= 0:
        raise ValueError("reward_duration_temperature must be finite and positive")
    type_scales = [
        float(getattr(args, name, 1.0))
        for name in (
            "reward_type_default_scale",
            "reward_type_fio_scale",
            "reward_type_identifier_scale",
            "reward_type_measurement_scale",
            "reward_type_money_scale",
            "reward_type_name_patronymic_scale",
            "reward_type_number_scale",
            "reward_type_percent_scale",
            "reward_type_phone_scale",
        )
    ]
    if any(not math.isfinite(value) or not 0 <= value <= 8 for value in type_scales):
        raise ValueError("reward type scales must be finite and in [0, 8]")
    if getattr(args, "advantage_mode", "centered") not in {
        "centered",
        "gated_pairwise",
        "gated_best",
    }:
        raise ValueError("advantage_mode must be centered, gated_pairwise, or gated_best")
    pairwise_values = [
        float(getattr(args, name, default))
        for name, default in (
            ("pairwise_critical_margin", 0.05),
            ("pairwise_minimum_critical_reward", 0.0),
            ("pairwise_maximum_negative_critical_reward", 1.0),
            ("pairwise_minimum_utterance_reward", 0.45),
            ("pairwise_minimum_pace_reward", 0.80),
        )
    ]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in pairwise_values):
        raise ValueError("pairwise gates and margin must be finite and in [0, 1]")
    train_code_predictor = bool(getattr(args, "train_code_predictor", False))
    train_main_talker = bool(getattr(args, "train_main_talker", True))
    predictor_weight = float(getattr(args, "code_predictor_policy_weight", 0.0))
    predictor_lr_multiplier = float(getattr(args, "code_predictor_lr_multiplier", 1.0))
    if not math.isfinite(predictor_weight) or not 0 <= predictor_weight <= 1:
        raise ValueError("code_predictor_policy_weight must be finite and in [0, 1]")
    if not math.isfinite(predictor_lr_multiplier) or not 0 < predictor_lr_multiplier <= 100:
        raise ValueError("code_predictor_lr_multiplier must be finite and in (0, 100]")
    if train_code_predictor != (predictor_weight > 0):
        raise ValueError(
            "train_code_predictor=true requires positive code_predictor_policy_weight; "
            "the weight must be zero when the predictor is frozen"
        )
    if not train_main_talker and not train_code_predictor:
        raise ValueError("at least one of main Talker or code predictor must be trainable")
    if not train_code_predictor and predictor_lr_multiplier != 1.0:
        raise ValueError(
            "code_predictor_lr_multiplier must be 1 when the code predictor is frozen"
        )
    if getattr(args, "subtalker_top_k", 0) < 0:
        raise ValueError("subtalker_top_k must be non-negative")
    if not 0 < float(getattr(args, "subtalker_top_p", 1.0)) <= 1:
        raise ValueError("subtalker_top_p must be in (0, 1]")
    if float(getattr(args, "subtalker_temperature", 0.0)) <= 0:
        raise ValueError("subtalker_temperature must be positive")


def configure_full_main_talker(
    tts_model: PreTrainedModel,
    *,
    train_main_talker: bool = True,
    train_code_predictor: bool = False,
) -> tuple[int, int]:
    """Configure main-Talker and 15-codebook-predictor trainable scopes."""

    for parameter in tts_model.parameters():
        parameter.requires_grad_(False)
    for parameter in tts_model.talker.parameters():
        parameter.requires_grad_(train_main_talker)
    for parameter in tts_model.talker.code_predictor.parameters():
        parameter.requires_grad_(train_code_predictor)
    if hasattr(tts_model.talker, "config"):
        tts_model.talker.config.use_cache = False
    tts_model.talker.code_predictor.train(train_code_predictor)
    speaker_encoder = getattr(tts_model, "speaker_encoder", None)
    if speaker_encoder is not None:
        speaker_encoder.eval()
    speech_tokenizer = getattr(tts_model, "speech_tokenizer", None)
    speech_model = getattr(speech_tokenizer, "model", None)
    if speech_model is not None:
        speech_model.requires_grad_(False)
        speech_model.eval()
    total = sum(parameter.numel() for parameter in tts_model.parameters())
    trainable = sum(parameter.numel() for parameter in tts_model.parameters() if parameter.requires_grad)
    if not trainable or trainable > total:
        raise RuntimeError("invalid Talker trainable scope")
    invalid_names = [
        name
        for name, parameter in tts_model.named_parameters()
        if parameter.requires_grad
        and (
            not name.startswith("talker.")
            or (
                name.startswith("talker.code_predictor.")
                and not train_code_predictor
            )
            or (
                not name.startswith("talker.code_predictor.")
                and not train_main_talker
            )
        )
    ]
    if invalid_names:
        preview = ", ".join(invalid_names[:8])
        raise RuntimeError(f"parameters outside the requested Talker scope are trainable: {preview}")
    predictor_trainable = any(
        parameter.requires_grad for parameter in tts_model.talker.code_predictor.parameters()
    )
    if predictor_trainable != train_code_predictor:
        raise RuntimeError("code predictor trainable scope does not match the requested mode")
    return trainable, total


def build_joint_optimizer_groups(
    model: torch.nn.Module,
    decay_parameter_names: set[str],
    *,
    base_learning_rate: float,
    weight_decay: float,
    code_predictor_lr_multiplier: float,
) -> list[dict[str, Any]]:
    """Split main/predictor parameters so Adam receives genuinely different LRs."""

    numeric = (base_learning_rate, weight_decay, code_predictor_lr_multiplier)
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("optimizer group values must be finite")
    if base_learning_rate <= 0 or weight_decay < 0 or code_predictor_lr_multiplier <= 0:
        raise ValueError("optimizer LR values must be positive and weight decay non-negative")

    buckets: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    expected = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        expected += parameter.numel()
        is_predictor = ".code_predictor." in f".{name}."
        uses_decay = name in decay_parameter_names
        buckets.setdefault((is_predictor, uses_decay), []).append(parameter)

    groups: list[dict[str, Any]] = []
    grouped = 0
    for (is_predictor, uses_decay), parameters in sorted(buckets.items()):
        grouped += sum(parameter.numel() for parameter in parameters)
        groups.append(
            {
                "params": parameters,
                "lr": float(base_learning_rate)
                * (float(code_predictor_lr_multiplier) if is_predictor else 1.0),
                "weight_decay": float(weight_decay) if uses_decay else 0.0,
                "is_code_predictor": is_predictor,
            }
        )
    if grouped != expected or not groups:
        raise RuntimeError("joint optimizer groups do not cover every trainable parameter exactly once")
    return groups


class TalkerPolicyModel(PreTrainedModel):
    """DDP-visible teacher-forced facade over Qwen3-TTS' main Talker."""

    base_model_prefix = "tts_model"
    main_input_name = "inputs_embeds"
    # The wrapped Qwen checkpoint is already constructed with SDPA.  Without
    # advertising the same capability, PreTrainedModel rejects its config
    # while building this DDP facade even though attention is delegated to the
    # underlying Talker.
    _supports_sdpa = True

    def __init__(
        self,
        tts_model: PreTrainedModel,
        *,
        language: str = "Russian",
        non_streaming_mode: bool = False,
        train_code_predictor: bool = False,
    ) -> None:
        super().__init__(tts_model.config)
        self.tts_model = tts_model
        self.language = str(language)
        self.non_streaming_mode = bool(non_streaming_mode)
        self.train_code_predictor = bool(train_code_predictor)

    @property
    def talker(self):
        return self.tts_model.talker

    @property
    def speech_tokenizer(self):
        return self.tts_model.speech_tokenizer

    def train(self, mode: bool = True):
        super().train(mode)
        self.talker.code_predictor.train(mode and self.train_code_predictor)
        speaker_encoder = getattr(self.tts_model, "speaker_encoder", None)
        if speaker_encoder is not None:
            speaker_encoder.eval()
        speech_tokenizer = getattr(self.tts_model, "speech_tokenizer", None)
        speech_model = getattr(speech_tokenizer, "model", None)
        if speech_model is not None:
            speech_model.eval()
        return self

    def get_input_embeddings(self):
        return self.talker.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.talker.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.talker.get_output_embeddings()

    def set_output_embeddings(self, value):
        return self.talker.set_output_embeddings(value)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.tts_model.state_dict(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        try:
            return self.tts_model.load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return self.tts_model.load_state_dict(state_dict, strict=strict)

    def save_pretrained(self, save_directory: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("max_shard_size", "5GB")
        return self.tts_model.save_pretrained(save_directory, *args, **kwargs)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        reference_ids: torch.Tensor,
        reference_id_lengths: torch.Tensor,
        reference_codes: torch.Tensor,
        reference_code_lengths: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        target_codes: torch.Tensor,
        target_code_lengths: torch.Tensor,
        truncated: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build policy-specific embeddings and run codebook-zero teacher forcing.

        Building embeddings inside this forwarded module keeps every trainable
        main-Talker parameter inside DDP's reducer lifecycle.
        """

        from grpo_finetuning.modeling_utils import build_voice_clone_teacher_sample

        batch_size = int(text_ids.shape[0])
        fields = (
            text_lengths,
            reference_id_lengths,
            reference_code_lengths,
            target_code_lengths,
            truncated,
        )
        if any(value.shape != (batch_size,) for value in fields):
            raise ValueError("teacher length and truncation tensors must have shape [B]")
        samples: list[dict[str, torch.Tensor]] = []
        codec_labels: list[torch.Tensor] = []
        for index in range(batch_size):
            target_length = int(target_code_lengths[index].item())
            target = target_codes[index, :target_length]
            sample = build_voice_clone_teacher_sample(
                self.tts_model,
                text_ids[index, : int(text_lengths[index].item())],
                reference_ids[index, : int(reference_id_lengths[index].item())],
                reference_codes[index, : int(reference_code_lengths[index].item())],
                speaker_embeddings[index],
                target,
                self.language,
                self.non_streaming_mode,
            )
            if bool(truncated[index].item()):
                labels = sample["labels"].clone()
                supervised = torch.nonzero(labels != -100, as_tuple=False).flatten()
                if supervised.numel() == 0:
                    raise ValueError("truncated teacher sample contains no supervised target")
                labels[int(supervised[-1].item())] = -100
                sample = dict(sample)
                sample["labels"] = labels
            frame_positions = torch.nonzero(
                sample["labels"] != -100, as_tuple=False
            ).flatten()[:target_length]
            if frame_positions.numel() != target_length:
                raise RuntimeError("main and code-predictor rollout targets are misaligned")
            all_codes = torch.full(
                (sample["labels"].shape[0], NUM_CODEBOOKS),
                -100,
                device=sample["labels"].device,
                dtype=torch.long,
            )
            all_codes[frame_positions] = target
            samples.append(sample)
            codec_labels.append(all_codes)
        teacher = pad_teacher_samples(samples, detach=False)
        all_codec_labels = pad_sequence(codec_labels, batch_first=True, padding_value=-100)
        outputs = self.talker(
            inputs_embeds=teacher["tts_inputs_embeds"][:, :-1],
            attention_mask=teacher["tts_attention_mask"][:, :-1],
            labels=None,
            use_cache=False,
            output_hidden_states=self.train_code_predictor,
            return_dict=True,
        )
        result = {
            "logits": outputs.logits,
            "labels": teacher["tts_labels"][:, 1:],
            "attention_mask": teacher["tts_attention_mask"][:, 1:],
        }
        if self.train_code_predictor:
            hidden_stack = outputs.hidden_states[0]
            if not hidden_stack:
                raise RuntimeError("main Talker did not return hidden states for predictor RL")
            main_hidden = hidden_stack[-1]
            fast_targets_full = all_codec_labels[:, 1:]
            fast_frame_mask = fast_targets_full[..., 0] != -100
            selected_hidden = main_hidden[fast_frame_mask]
            selected_codes = fast_targets_full[fast_frame_mask]
            if selected_hidden.numel() == 0:
                raise RuntimeError("joint Talker rollout contains no codec frames")
            fast_logits, _ = self.talker.forward_sub_talker_finetune(
                selected_codes, selected_hidden
            )
            fast_targets = selected_codes[:, 1:]
            fast_losses = F.cross_entropy(
                fast_logits.float().reshape(-1, fast_logits.shape[-1]),
                fast_targets.reshape(-1),
                reduction="none",
            ).reshape_as(fast_targets)
            owners = fast_frame_mask.nonzero(as_tuple=False)[:, 0]
            per_sample_sum = fast_losses.new_zeros((batch_size,))
            per_sample_count = fast_losses.new_zeros((batch_size,))
            per_frame_sum = fast_losses.sum(dim=1)
            per_frame_count = torch.full_like(per_frame_sum, fast_targets.shape[1])
            per_sample_sum.scatter_add_(0, owners, per_frame_sum)
            per_sample_count.scatter_add_(0, owners, per_frame_count)
            result["code_predictor_nll"] = per_sample_sum / per_sample_count.clamp_min(1)
        return result


def codec_accuracy_reward(
    metrics: Any,
    *,
    number_wer_weight: float = 0.35,
    number_cer_weight: float = 0.35,
    utterance_wer_weight: float = 0.15,
    utterance_cer_weight: float = 0.15,
    temperature: float = 3.0,
) -> float:
    """Convert real ASR alignment metrics into a clipped accuracy reward."""

    weighted = (
        (number_wer_weight, "number_wer"),
        (number_cer_weight, "number_cer"),
        (utterance_wer_weight, "utterance_wer"),
        (utterance_cer_weight, "utterance_cer"),
    )
    total_weight = sum(weight for weight, _ in weighted)
    if total_weight <= 0:
        raise ValueError("reward weights must have a positive sum")
    error = 0.0
    for weight, name in weighted:
        value = float(getattr(metrics, name))
        if not math.isfinite(value):
            return 0.0
        error += weight * min(1.0, max(0.0, value))
    return 1.0 - math.tanh(float(temperature) * error / total_weight)


def codec_composite_accuracy_reward(
    metrics: Any,
    typed_result: Any | None,
    *,
    number_wer_weight: float,
    number_cer_weight: float,
    utterance_wer_weight: float,
    utterance_cer_weight: float,
    temperature: float,
    critical_component_weight: float,
    utterance_component_weight: float,
    critical_wer_weight: float,
    critical_cer_weight: float,
    critical_exact_weight: float,
    critical_worst_span_weight: float,
    duration_reward: float | None = None,
    duration_component_weight: float = 0.0,
) -> float:
    """Combine backward-compatible accuracy with typed dense critical reward."""

    utterance = codec_accuracy_reward(
        metrics,
        number_wer_weight=number_wer_weight,
        number_cer_weight=number_cer_weight,
        utterance_wer_weight=utterance_wer_weight,
        utterance_cer_weight=utterance_cer_weight,
        temperature=temperature,
    )
    total = (
        float(critical_component_weight)
        + float(utterance_component_weight)
        + float(duration_component_weight)
    )
    if total <= 0:
        raise ValueError("reward component weights must have a positive sum")
    critical = 0.0
    if critical_component_weight > 0:
        if typed_result is None:
            return 0.0
        from critical_span_metrics import typed_critical_reward

        critical = typed_critical_reward(
            typed_result,
            wer_weight=critical_wer_weight,
            cer_weight=critical_cer_weight,
            exact_weight=critical_exact_weight,
            temperature=temperature,
            worst_span_weight=critical_worst_span_weight,
        )
    pace = 0.0 if duration_reward is None else float(duration_reward)
    return (
        float(utterance_component_weight) * utterance
        + float(critical_component_weight) * critical
        + float(duration_component_weight) * pace
    ) / total


def critical_type_gradient_scale(
    critical_spans: Any,
    type_scales: Mapping[str, float],
    *,
    default_scale: float = 1.0,
) -> float:
    """Return the strongest configured scale for the typed spans in one prompt.

    Every GRPO group contains generations for one prompt, so multiplying its
    rewards by this value scales that prompt's centered advantages without
    changing the ordering of candidates inside the group.  This provides a
    type-aware curriculum while preserving the immutable source sampling order.
    """

    values = [float(default_scale), *(float(value) for value in type_scales.values())]
    if any(not math.isfinite(value) or not 0 <= value <= 8 for value in values):
        raise ValueError("critical type scales must be finite and in [0, 8]")
    if not isinstance(critical_spans, Sequence) or isinstance(critical_spans, (str, bytes)):
        return float(default_scale)
    matched = [
        float(type_scales.get(str(span.get("type", "")).strip(), default_scale))
        for span in critical_spans
        if isinstance(span, Mapping)
    ]
    return max(matched, default=float(default_scale))


def configured_critical_type_scale(args: Any, critical_spans: Any) -> float:
    """Resolve per-type GRPO advantage scaling from the frozen trial config."""

    return critical_type_gradient_scale(
        critical_spans,
        {
            name: float(getattr(args, f"reward_type_{name}_scale", 1.0))
            for name in (
                "fio",
                "identifier",
                "measurement",
                "money",
                "name_patronymic",
                "number",
                "percent",
                "phone",
            )
        },
        default_scale=float(getattr(args, "reward_type_default_scale", 1.0)),
    )


def codec_duration_reward(
    duration_seconds: float,
    word_count: int,
    *,
    target_words_per_sec: float = 2.5,
    minimum_ratio: float = 0.65,
    maximum_ratio: float = 1.30,
    temperature: float = 3.0,
) -> tuple[float, float, float]:
    """Return asymmetric pace reward, duration ratio, and observed words/sec."""

    values = (duration_seconds, target_words_per_sec, minimum_ratio, maximum_ratio, temperature)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("duration reward inputs must be finite")
    if duration_seconds <= 0 or isinstance(word_count, bool) or word_count <= 0:
        raise ValueError("duration and word count must be positive")
    if target_words_per_sec <= 0 or not 0 < minimum_ratio < maximum_ratio or temperature <= 0:
        raise ValueError("invalid duration reward configuration")
    expected = max(0.5, float(word_count) / target_words_per_sec)
    ratio = duration_seconds / expected
    if minimum_ratio <= ratio <= maximum_ratio:
        reward = 1.0
    elif ratio > maximum_ratio:
        reward = math.exp(-temperature * math.log(ratio / maximum_ratio))
    else:
        reward = math.exp(-temperature * math.log(minimum_ratio / ratio))
    return min(1.0, max(0.0, reward)), ratio, float(word_count) / duration_seconds


def wav_duration_seconds(wav_bytes: bytes) -> float:
    import io
    import soundfile as sf

    info = sf.info(io.BytesIO(wav_bytes))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("decoded WAV has invalid duration metadata")
    return float(info.frames) / float(info.samplerate)


def clone_frozen_reference(
    tts_model: PreTrainedModel,
    *,
    language: str = "Russian",
    non_streaming_mode: bool = False,
) -> TalkerPolicyModel:
    """Create an exact frozen initial-policy reference without decoder duplication."""

    speech_tokenizer = getattr(tts_model, "speech_tokenizer", None)
    supported_speakers = getattr(tts_model, "supported_speakers", None)
    try:
        # The codec decoder is irrelevant to reference log-probabilities and is
        # often larger than the state we actually need to copy.
        tts_model.speech_tokenizer = None
        # Qwen3TTSForConditionalGeneration stores ``dict.keys()`` here. A
        # mapping view is not pickleable, so module deepcopy fails before step
        # zero. Materialize it only for the copy and restore the live policy.
        if isinstance(supported_speakers, type({}.keys())):
            tts_model.supported_speakers = tuple(supported_speakers)
        reference_tts = copy.deepcopy(tts_model)
    finally:
        tts_model.speech_tokenizer = speech_tokenizer
        tts_model.supported_speakers = supported_speakers
    reference_tts.speech_tokenizer = None
    reference = TalkerPolicyModel(
        reference_tts,
        language=language,
        non_streaming_mode=non_streaming_mode,
    )
    reference.requires_grad_(False)
    reference.eval()
    return reference


class RankLocalGigaAM:
    """Lazy GigaAM runtime whose CUDA provider follows the Accelerate rank."""

    def __init__(self, model_name: str, model_path: str | None = None) -> None:
        self.model_name = str(model_name)
        self.model_path = model_path
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from onnx_asr import load_model

            providers: list[Any]
            if torch.cuda.is_available():
                providers = [
                    ("CUDAExecutionProvider", {"device_id": int(torch.cuda.current_device())}),
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]
            self._model = load_model(
                self.model_name,
                path=self.model_path,
                providers=providers,
            )
        return self._model

    @staticmethod
    def _waveform(wav_bytes: bytes) -> np.ndarray:
        import io
        import soundfile as sf
        import soxr

        waveform, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            waveform = soxr.resample(waveform, sample_rate, 16_000).astype(np.float32)
        return np.asarray(waveform, dtype=np.float32)

    def transcribe_batch(self, wav_bytes: Sequence[bytes]) -> list[str | None]:
        result: list[str | None] = [None] * len(wav_bytes)
        waveforms: list[np.ndarray] = []
        indices: list[int] = []
        for index, audio in enumerate(wav_bytes):
            try:
                waveforms.append(self._waveform(audio))
                indices.append(index)
            except Exception:
                continue
        if not waveforms:
            return result
        try:
            values = self._load().recognize(waveforms, sample_rate=16_000)
            if len(values) != len(waveforms):
                raise ValueError("GigaAM returned the wrong batch size")
            for index, value in zip(indices, values, strict=True):
                text = str(value).strip()
                result[index] = text or None
        except Exception:
            pass
        return result


class QwenTTSGRPOTrainer(GRPOTrainer):
    """TRL GRPOTrainer specialized for real 16-codebook Qwen3-TTS rollouts."""

    _TEACHER_KEYS = (
        "text_ids",
        "text_lengths",
        "reference_ids",
        "reference_id_lengths",
        "reference_codes",
        "reference_code_lengths",
        "speaker_embeddings",
        "target_codes",
        "target_code_lengths",
        "truncated",
    )

    def __init__(
        self,
        *,
        qwen: Any,
        args: QwenTTSGRPOConfig,
        train_dataset: Any,
        references_path: str | None = None,
        references: Sequence[Any] | None = None,
        asr: Any | None = None,
        eval_dataset: Any | None = None,
        callbacks: Sequence[Any] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
    ) -> None:
        validate_v1_config(args)
        if (references_path is None) == (references is None):
            raise ValueError("provide exactly one of references_path or references")
        requested_beta = float(args.beta)
        frozen_reference = (
            clone_frozen_reference(
                qwen.model,
                language=args.language,
                non_streaming_mode=args.non_streaming_mode,
            )
            if requested_beta > 0
            else None
        )
        trainable, total = configure_full_main_talker(
            qwen.model,
            train_main_talker=args.train_main_talker,
            train_code_predictor=args.train_code_predictor,
        )
        policy = TalkerPolicyModel(
            qwen.model,
            language=args.language,
            non_streaming_mode=args.non_streaming_mode,
            train_code_predictor=args.train_code_predictor,
        )
        self.qwen = qwen
        self._trainable_parameter_count = trainable
        self._total_parameter_count = total

        # TRL attempts to reload an ordinary CausalLM when beta is non-zero.
        # Qwen TTS needs the exact frozen Talker facade above, so suppress that
        # stock construction and restore the user-visible value afterwards.
        args.beta = 0.0
        try:
            super().__init__(
                model=policy,
                reward_funcs=codec_accuracy_reward,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=qwen.processor,
                callbacks=list(callbacks or ()),
                optimizers=optimizers,
            )
        finally:
            args.beta = requested_beta
        self.beta = requested_beta
        self.args.beta = requested_beta
        self._reference_policy = frozen_reference
        if self._reference_policy is not None:
            self._reference_policy.to(self.accelerator.device)
            self._reference_policy.eval()
        self.ref_model = self._reference_policy
        self._bind_runtime(references_path=references_path, references=references, asr=asr)

    def create_optimizer(self):
        """Create standard Trainer optimizer groups with a real predictor LR."""

        multiplier = float(self.args.code_predictor_lr_multiplier)
        if self.optimizer is not None or multiplier == 1.0:
            return super().create_optimizer()
        decay_parameters = self.get_decay_parameter_names(self.model)
        grouped_parameters = build_joint_optimizer_groups(
            self.model,
            decay_parameters,
            base_learning_rate=float(self.args.learning_rate),
            weight_decay=float(self.args.weight_decay),
            code_predictor_lr_multiplier=multiplier,
        )
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, self.model
            )
        optimizer_kwargs = dict(optimizer_kwargs)
        for unsupported in ("params", "model", "optimizer_dict"):
            if unsupported in optimizer_kwargs:
                raise ValueError(
                    f"code_predictor_lr_multiplier does not support optimizer kwarg {unsupported!r}"
                )
        self.optimizer = optimizer_cls(grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _bind_runtime(
        self,
        *,
        references_path: str | None,
        references: Sequence[Any] | None,
        asr: Any | None,
    ) -> None:
        device = self.accelerator.device
        self.qwen.device = device
        tokenizer = getattr(self.qwen.model, "speech_tokenizer", None)
        tokenizer_model = getattr(tokenizer, "model", None)
        if tokenizer_model is None:
            raise RuntimeError("Qwen speech tokenizer is not loaded")
        tokenizer_model.requires_grad_(False)
        tokenizer_model.to(device)
        tokenizer_model.eval()
        tokenizer.device = device
        from grpo_finetuning.modeling_utils import force_eager_tokenizer_decode

        force_eager_tokenizer_decode(self.qwen.model)
        if references is None:
            from grpo_dpo_finetuning.qwen_runtime import load_reference_assets

            references = load_reference_assets(self.qwen, str(references_path))
        self.references = list(references)
        if not self.references:
            raise ValueError("reference manifest must contain at least one voice reference")
        for reference in self.references:
            validate_codec_codes(reference.reference_codes, name="reference codes")
        if asr is not None:
            self.asr = asr
        elif self.args.asr_backend == "gigaam":
            self.asr = RankLocalGigaAM(self.args.asr_model_name, self.args.asr_model_path)
        else:
            raise ValueError(f"unsupported ASR backend: {self.args.asr_backend!r}")

    @staticmethod
    def _row_text(row: Mapping[str, Any]) -> str:
        for key in ("stressed", "prompt", "words", "normalized_gold", "gold", "text", "input"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        raise ValueError("GRPO row does not contain prompt text")

    @staticmethod
    def _row_id(row: Mapping[str, Any]) -> str:
        for key in ("id", "prompt_id", "key", "source_key"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value)
        stable = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    def _reference_for_prompt(self, prompt_id: str) -> Any:
        digest = hashlib.sha256(f"{self.args.seed}\0{prompt_id}".encode("utf-8")).digest()
        return self.references[int.from_bytes(digest[:8], "big") % len(self.references)]

    def _generation_seed(self, rows: Sequence[Mapping[str, Any]], mode: str) -> int:
        identity = "\0".join(self._row_id(row) for row in rows)
        payload = (
            f"{self.args.seed}\0{self.state.global_step}\0{self._step}\0"
            f"{self.accelerator.process_index}\0{mode}\0{identity}"
        )
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)

    @staticmethod
    def _one_dimensional_ids(value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        while value.ndim > 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"{name} must contain one non-empty token sequence")
        return value.to(dtype=torch.long).contiguous()

    @staticmethod
    def _one_item_batch_ids(value: torch.Tensor, name: str) -> torch.Tensor:
        """Normalize one Qwen generation sequence to the required ``[1, T]``."""

        return QwenTTSGRPOTrainer._one_dimensional_ids(value, name).unsqueeze(0)

    @staticmethod
    def _pad_context(
        text_ids: Sequence[torch.Tensor],
        references: Sequence[Any],
        targets: Sequence[torch.Tensor],
        truncated: Sequence[bool],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        reference_ids = [
            QwenTTSGRPOTrainer._one_dimensional_ids(item.ref_ids, "reference ids")
            for item in references
        ]
        reference_codes = [validate_codec_codes(item.reference_codes, name="reference codes") for item in references]
        speakers = [torch.as_tensor(item.speaker_embedding).reshape(-1) for item in references]
        if len({int(value.numel()) for value in speakers}) != 1:
            raise ValueError("reference speaker embeddings must have one common width")
        result = {
            "text_ids": pad_sequence(text_ids, batch_first=True, padding_value=0),
            "text_lengths": torch.tensor([value.numel() for value in text_ids], dtype=torch.long),
            "reference_ids": pad_sequence(reference_ids, batch_first=True, padding_value=0),
            "reference_id_lengths": torch.tensor([value.numel() for value in reference_ids], dtype=torch.long),
            "reference_codes": pad_sequence(reference_codes, batch_first=True, padding_value=0),
            "reference_code_lengths": torch.tensor(
                [value.shape[0] for value in reference_codes], dtype=torch.long
            ),
            "speaker_embeddings": torch.stack(speakers),
            "target_codes": pad_sequence(targets, batch_first=True, padding_value=0),
            "target_code_lengths": torch.tensor([value.shape[0] for value in targets], dtype=torch.long),
            "truncated": torch.tensor(truncated, dtype=torch.bool),
        }
        return {name: value.to(device) for name, value in result.items()}

    def _teacher_kwargs(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: inputs[name] for name in self._TEACHER_KEYS}

    @torch.no_grad()
    def _reference_logps(self, context: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self._reference_policy is None:
            raise RuntimeError("reference policy is unavailable")
        batch_size = int(context["text_ids"].shape[0])
        micro = int(self.args.logprob_micro_batch_size) or batch_size
        rows: list[torch.Tensor] = []
        for start in range(0, batch_size, micro):
            stop = min(start + micro, batch_size)
            output = self._reference_policy(
                **{name: value[start:stop] for name, value in self._teacher_kwargs(context).items()}
            )
            values, _ = codebook0_logps(output["logits"], output["labels"])
            rows.extend(values[index] for index in range(values.shape[0]))
        return pad_sequence(rows, batch_first=True, padding_value=0.0)

    @torch.no_grad()
    def _generate_and_score_completions(
        self, inputs: Sequence[Mapping[str, Any]]
    ) -> dict[str, torch.Tensor]:
        if not isinstance(inputs, Sequence) or not inputs:
            raise ValueError("codec rollout expects a non-empty sequence of dataset rows")
        mode = "train" if self.model.training else "eval"
        rows = [dict(row) for row in inputs]
        texts = [self._row_text(row) for row in rows]
        prompt_ids = [self._row_id(row) for row in rows]
        references = [self._reference_for_prompt(prompt_id) for prompt_id in prompt_ids]
        device = self.accelerator.device
        text_ids = [
            self._one_dimensional_ids(value, "assistant ids")
            for value in self.qwen._tokenize_texts(
                [self.qwen._build_assistant_text(text) for text in texts]
            )
        ]
        reference_ids = [
            self._one_dimensional_ids(reference.ref_ids, "reference ids") for reference in references
        ]
        voice_prompt = self.qwen._prompt_items_to_voice_clone_prompt(
            [reference.prompt_item for reference in references]
        )
        seed = self._generation_seed(rows, mode)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        unwrapped = self.accelerator.unwrap_model(self.model)
        tts_model = unwrapped.tts_model
        was_training = unwrapped.training
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            unwrapped.eval()
            generated, _ = tts_model.generate(
                input_ids=[
                    self._one_item_batch_ids(value, "assistant ids").to(device)
                    for value in text_ids
                ],
                ref_ids=[
                    self._one_item_batch_ids(value, "reference ids").to(device)
                    for value in reference_ids
                ],
                voice_clone_prompt=voice_prompt,
                languages=[self.args.language] * len(rows),
                non_streaming_mode=self.args.non_streaming_mode,
                max_new_tokens=self.args.max_completion_length,
                do_sample=True,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                repetition_penalty=self.args.repetition_penalty,
                subtalker_dosample=self.args.subtalker_dosample,
                subtalker_temperature=self.args.subtalker_temperature,
                subtalker_top_k=self.args.subtalker_top_k,
                subtalker_top_p=self.args.subtalker_top_p,
            )
        finally:
            unwrapped.train(was_training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
        if len(generated) != len(rows):
            raise ValueError("Qwen returned a different number of codec rollouts than prompts")
        targets = [validate_codec_codes(value, name="generated codes") for value in generated]
        truncated = [value.shape[0] >= self.args.max_completion_length for value in targets]

        from grpo_dpo_finetuning.metrics import MetricResult, score_text_pair
        from grpo_dpo_finetuning.qwen_runtime import decode_full_codes
        from critical_span_metrics import normalize_text, score_typed_text_pair

        try:
            decoded = decode_full_codes(
                tts_model,
                targets,
                [reference.reference_codes.to(device) for reference in references],
            )
        except Exception:
            decoded = []
            for target, reference in zip(targets, references, strict=True):
                try:
                    decoded.append(
                        decode_full_codes(tts_model, [target], [reference.reference_codes.to(device)])[0]
                    )
                except Exception:
                    decoded.append(b"")
        hypotheses: list[str | None] = [None] * len(decoded)
        valid_indices = [index for index, audio in enumerate(decoded) if audio]
        if valid_indices:
            transcribed = self.asr.transcribe_batch([decoded[index] for index in valid_indices])
            if len(transcribed) != len(valid_indices):
                raise ValueError("ASR returned a different number of hypotheses than audio inputs")
            for index, hypothesis in zip(valid_indices, transcribed, strict=True):
                text = str(hypothesis).strip() if hypothesis is not None else ""
                hypotheses[index] = text or None

        metric_rows: list[MetricResult] = []
        typed_metric_rows: list[tuple[float, float, float, float]] = []
        pace_metric_rows: list[tuple[float, float, float, float, float]] = []
        preference_rows: list[tuple[float, float, float, float, float]] = []
        valid: list[bool] = []
        rewards: list[float] = []
        for row, target_text, audio, hypothesis, is_truncated in zip(
            rows, texts, decoded, hypotheses, truncated, strict=True
        ):
            is_valid = bool(audio) and hypothesis is not None and not is_truncated
            typed_result = None
            critical_spans = row.get("critical_spans")
            type_scale = configured_critical_type_scale(self.args, critical_spans)
            duration_seconds = 0.0
            duration_ratio = 0.0
            words_per_sec = 0.0
            duration_reward_value = 0.0
            try:
                metrics = (
                    score_text_pair(
                        str(row.get("source_text", row.get("words", target_text))),
                        hypothesis or "",
                        row.get("num_words", ()),
                    )
                    if is_valid
                    else MetricResult(
                        number_wer=1.0,
                        number_cer=1.0,
                        utterance_wer=1.0,
                        utterance_cer=1.0,
                        number_exact=False,
                    )
                )
                if is_valid and critical_spans:
                    typed_result = score_typed_text_pair(
                        str(row.get("source_text", row.get("words", target_text))),
                        hypothesis or "",
                        critical_spans,
                    )
                if (
                    is_valid
                    and self.args.reward_critical_component_weight > 0
                    and typed_result is None
                ):
                    raise ValueError("typed critical reward requires critical_spans")
                if is_valid:
                    duration_seconds = wav_duration_seconds(audio)
                    word_count = len(
                        normalize_text(
                            str(row.get("source_text", row.get("words", target_text)))
                        ).split()
                    )
                    duration_reward_value, duration_ratio, words_per_sec = codec_duration_reward(
                        duration_seconds,
                        word_count,
                        target_words_per_sec=self.args.reward_duration_target_words_per_sec,
                        minimum_ratio=self.args.reward_duration_min_ratio,
                        maximum_ratio=self.args.reward_duration_max_ratio,
                        temperature=self.args.reward_duration_temperature,
                    )
            except (TypeError, ValueError):
                is_valid = False
                typed_result = None
                duration_seconds = 0.0
                duration_ratio = 0.0
                words_per_sec = 0.0
                duration_reward_value = 0.0
                metrics = MetricResult(
                    number_wer=1.0,
                    number_cer=1.0,
                    utterance_wer=1.0,
                    utterance_cer=1.0,
                    number_exact=False,
                )
            reward = (
                codec_composite_accuracy_reward(
                    metrics,
                    typed_result,
                    number_wer_weight=self.args.reward_number_wer_weight,
                    number_cer_weight=self.args.reward_number_cer_weight,
                    utterance_wer_weight=self.args.reward_utterance_wer_weight,
                    utterance_cer_weight=self.args.reward_utterance_cer_weight,
                    temperature=self.args.reward_temperature,
                    critical_component_weight=self.args.reward_critical_component_weight,
                    utterance_component_weight=self.args.reward_utterance_component_weight,
                    critical_wer_weight=self.args.reward_critical_wer_weight,
                    critical_cer_weight=self.args.reward_critical_cer_weight,
                    critical_exact_weight=self.args.reward_critical_exact_weight,
                    critical_worst_span_weight=self.args.reward_critical_worst_span_weight,
                    duration_reward=duration_reward_value,
                    duration_component_weight=self.args.reward_duration_component_weight,
                )
                if is_valid
                else 0.0
            )
            from critical_span_metrics import typed_critical_reward

            critical_reward_value = (
                typed_critical_reward(
                    typed_result,
                    wer_weight=self.args.reward_critical_wer_weight,
                    cer_weight=self.args.reward_critical_cer_weight,
                    exact_weight=self.args.reward_critical_exact_weight,
                    temperature=self.args.reward_temperature,
                    worst_span_weight=self.args.reward_critical_worst_span_weight,
                )
                if is_valid and typed_result is not None
                else 0.0
            )
            utterance_reward_value = (
                codec_accuracy_reward(
                    metrics,
                    number_wer_weight=self.args.reward_number_wer_weight,
                    number_cer_weight=self.args.reward_number_cer_weight,
                    utterance_wer_weight=self.args.reward_utterance_wer_weight,
                    utterance_cer_weight=self.args.reward_utterance_cer_weight,
                    temperature=self.args.reward_temperature,
                )
                if is_valid
                else 0.0
            )
            if self.args.advantage_mode == "centered":
                reward *= type_scale
            metric_rows.append(metrics)
            typed_metric_rows.append(
                (
                    typed_result.critical_wer if typed_result is not None else 1.0,
                    typed_result.critical_cer if typed_result is not None else 1.0,
                    typed_result.critical_exact_rate if typed_result is not None else 0.0,
                    float(typed_result is not None),
                )
            )
            pace_metric_rows.append(
                (
                    duration_seconds,
                    words_per_sec,
                    duration_ratio,
                    duration_reward_value,
                    type_scale,
                )
            )
            preference_rows.append(
                (
                    critical_reward_value,
                    utterance_reward_value,
                    duration_reward_value,
                    float(is_valid),
                    type_scale,
                )
            )
            valid.append(is_valid)
            rewards.append(reward)
        local_rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        all_rewards = self.accelerator.gather(local_rewards)
        generations = self.num_generations if mode == "train" else self.num_generations_eval
        all_advantages, _, all_stds = compute_group_advantages(
            all_rewards,
            generations,
            scale_rewards=self.args.scale_rewards,
        )
        pairwise_active = torch.zeros(1, device=device, dtype=torch.float32)
        if self.args.advantage_mode in {"gated_pairwise", "gated_best"}:
            local_preferences = torch.tensor(preference_rows, device=device, dtype=torch.float32)
            all_preferences = self.accelerator.gather(local_preferences)
            common = (
                all_preferences[:, 0],
                all_preferences[:, 1],
                all_preferences[:, 2],
                all_preferences[:, 3],
                all_preferences[:, 4],
                generations,
            )
            if self.args.advantage_mode == "gated_pairwise":
                all_advantages, pairwise_groups = compute_gated_pairwise_advantages(
                    *common,
                    critical_margin=self.args.pairwise_critical_margin,
                    minimum_critical_reward=self.args.pairwise_minimum_critical_reward,
                    maximum_negative_critical_reward=(
                        self.args.pairwise_maximum_negative_critical_reward
                    ),
                    minimum_utterance_reward=self.args.pairwise_minimum_utterance_reward,
                    minimum_pace_reward=self.args.pairwise_minimum_pace_reward,
                )
            else:
                all_advantages, pairwise_groups = compute_gated_best_advantages(
                    *common,
                    minimum_critical_reward=self.args.pairwise_minimum_critical_reward,
                    minimum_utterance_reward=self.args.pairwise_minimum_utterance_reward,
                    minimum_pace_reward=self.args.pairwise_minimum_pace_reward,
                )
            pairwise_active = pairwise_groups.mean().reshape(1)
        rank_slice = process_slice(
            local_batch_size=len(rows),
            process_index=self.accelerator.process_index,
            num_processes=self.accelerator.num_processes,
            global_size=all_rewards.numel(),
        )
        context = self._pad_context(text_ids, references, targets, truncated, device)
        context["advantages"] = all_advantages[rank_slice]
        context["rewards"] = local_rewards
        if self.beta > 0:
            context["ref_per_token_logps"] = self._reference_logps(context)

        local_metrics = torch.tensor(
            [
                [
                    item.number_wer,
                    item.number_cer,
                    item.utterance_wer,
                    item.utterance_cer,
                    float(ok),
                    float(target.shape[0]),
                    float(cut),
                ]
                + list(typed)
                + list(pace)
                for item, typed, pace, ok, target, cut in zip(
                    metric_rows,
                    typed_metric_rows,
                    pace_metric_rows,
                    valid,
                    targets,
                    truncated,
                    strict=True,
                )
            ],
            device=device,
            dtype=torch.float32,
        )
        all_metrics = self.accelerator.gather(local_metrics)
        if not bool(all_metrics[:, 4].bool().any()) and not bool(all_metrics[:, 6].bool().all()):
            raise RuntimeError(
                "global rollout batch has no decodable, transcribed, non-truncated candidate; "
                "check the speech decoder and rank-local ASR runtime"
            )
        names = (
            "audio/number_wer",
            "audio/number_cer",
            "audio/utterance_wer",
            "audio/utterance_cer",
            "audio/valid_rate",
            "codec/frames",
            "codec/truncated_rate",
            "audio/critical_wer",
            "audio/critical_cer",
            "audio/critical_exact_rate",
            "audio/critical_coverage",
            "audio/duration_sec",
            "audio/words_per_sec",
            "audio/duration_ratio",
            "audio/duration_reward",
            "audio/critical_type_scale",
        )
        for index, name in enumerate(names):
            self._metrics[mode][name].append(all_metrics[:, index].mean().item())
        self._metrics[mode]["reward"].append(all_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(all_rewards.std(correction=1).item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            torch.isclose(all_stds, torch.zeros_like(all_stds)).float().mean().item()
        )
        self._metrics[mode]["pairwise/active_group_rate"].append(pairwise_active.item())
        validate_tensor_batch(context)
        return context

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> torch.Tensor:
        if return_outputs:
            raise ValueError("Qwen codec GRPO does not return model outputs")
        outputs = model(**self._teacher_kwargs(inputs))
        per_token_logps, mask = codebook0_logps(outputs["logits"], outputs["labels"])
        global_active_tokens = self.accelerator.reduce(
            mask.sum().to(dtype=torch.float32), reduction="sum"
        )
        ref_logps = inputs.get("ref_per_token_logps")
        if ref_logps is not None:
            if ref_logps.shape[1] < per_token_logps.shape[1]:
                raise ValueError("policy sequence exceeds the frozen-reference teacher sequence")
            # The rollout buffer is padded before it is shuffled/split.  A
            # resulting microbatch can have a smaller teacher-sequence maximum.
            ref_logps = ref_logps[:, : per_token_logps.shape[1]]
        loss, stats = codec_policy_loss(
            per_token_logps,
            mask,
            inputs["advantages"],
            old_per_token_logps=None,
            ref_per_token_logps=ref_logps,
            beta=self.beta,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            loss_type=self.args.loss_type,
            length_norm=float(self.args.max_completion_length),
            global_active_tokens=global_active_tokens,
            ddp_world_size=self.accelerator.num_processes,
        )
        predictor_policy = loss.new_zeros(())
        predictor_nll = loss.new_zeros(())
        if self.args.train_code_predictor:
            if "code_predictor_nll" not in outputs:
                raise RuntimeError("joint predictor mode did not return per-sample NLL")
            predictor_nll = outputs["code_predictor_nll"].mean()
            predictor_policy = code_predictor_policy_loss(
                outputs["code_predictor_nll"], inputs["advantages"]
            )
            loss = loss + float(self.args.code_predictor_policy_weight) * predictor_policy
        mode = "train" if self.model.training else "eval"
        values = torch.stack(
            [
                stats["policy_loss"].detach(),
                stats["kl"].detach(),
                stats["clip_ratio"].detach(),
                predictor_policy.detach(),
                predictor_nll.detach(),
            ]
        )
        values = self.accelerator.reduce(values, reduction="mean")
        self._metrics[mode]["policy_loss"].append(values[0].item())
        self._metrics[mode]["kl"].append(values[1].item())
        self._metrics[mode]["clip_ratio"].append(values[2].item())
        self._metrics[mode]["code_predictor/policy_loss"].append(values[3].item())
        self._metrics[mode]["code_predictor/nll"].append(values[4].item())
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1
        return loss / normalizer


__all__ = [
    "NUM_CODEBOOKS",
    "SUPPORTED_TRL_VERSION",
    "QwenTTSGRPOConfig",
    "QwenTTSGRPOTrainer",
    "RankLocalGigaAM",
    "TalkerPolicyModel",
    "clone_frozen_reference",
    "codebook0_logps",
    "codec_accuracy_reward",
    "codec_composite_accuracy_reward",
    "codec_duration_reward",
    "configured_critical_type_scale",
    "critical_type_gradient_scale",
    "codec_policy_loss",
    "code_predictor_policy_loss",
    "compute_gated_pairwise_advantages",
    "compute_gated_best_advantages",
    "build_joint_optimizer_groups",
    "compute_group_advantages",
    "configure_full_main_talker",
    "pad_teacher_samples",
    "process_slice",
    "validate_codec_codes",
    "validate_tensor_batch",
    "validate_trl_runtime_version",
    "validate_v1_config",
    "wav_duration_seconds",
]
