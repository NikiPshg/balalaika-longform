# coding=utf-8
"""Full main-Talker SFT with Hugging Face streaming datasets and Trainer.

The source WebDataset stays lazy. CPU-only filtering, audio decoding and stress
marking run in the Hugging Face ``IterableDataset``/DataLoader workers. Codec
and speaker feature extraction run under ``no_grad`` inside the model forward,
so the trainable teacher-forcing graph is always entered through DDP.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch
from transformers import (
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_finetuning.full_utterance_utils import dotenv_value  # noqa: E402
from lora_finetuning.main_talker_training import (  # noqa: E402
    FullUtteranceDataset,
    GigaAMValidator,
    compute_loss,
    core_model,
    generate_validation,
    normalize_codes,
    normalize_ids,
    validation_texts,
)
from lora_finetuning.sova_streaming import (  # noqa: E402
    candidate_from_row,
    prepared_rows,
    streaming_dataset,
)


def log(message: str) -> None:
    print(
        f"[hf-streaming-main-talker] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}",
        flush=True,
    )


@dataclass
class ModelArguments:
    model_path: str = "bitmanagerai/Qwen3TTS-RL-2"
    attn_implementation: str = "sdpa"
    hf_token_env: str = "HF_TOKEN"
    env_file: str = str(REPO_ROOT / ".env")
    save_final_model: bool = True


@dataclass
class DataArguments:
    dataset_name: str = "lab260/sova_balalaika"
    dataset_split: str = "train"
    dataset_revision: str = "be89f9bbc9908afd34b28e05441bbb9f655c0183"
    streaming: bool = True
    shuffle_buffer: int = 512
    validation_samples: int = 32
    validation_encode_batch_size: int = 8

    min_duration_sec: float = 5.0
    max_duration_sec: float = 20.0
    min_prefix_sec: float = 3.0
    max_prefix_sec: float = 8.0
    min_continuation_sec: float = 2.0
    codec_fps: float = 12.0
    min_distill_mos: float = 3.5
    max_music_prob: float = 0.30
    max_asr_wer: float = 0.25
    max_e2e_rover_wer: float = 0.0
    require_single_speaker: bool = True
    language: str = "Russian"
    min_words: int = 4
    max_text_chars: int = 800


@dataclass
class ValidationArguments:
    validation_texts_file: str = str(
        Path(__file__).with_name("validation_main_talker_ru.txt")
    )
    validation_references: int = 8
    validation_texts: int = 26
    validation_generation_batch_size: int = 4
    generate_audio: bool = True
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_dosample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0

    asr_enabled: bool = True
    asr_model_name: str = "gigaam-v3-e2e-ctc"
    asr_model_path: str = str(REPO_ROOT / "artifacts/models/gigaam-v3-e2e-ctc-onnx")
    asr_device_id: int = 0
    hf_upload_samples: bool = True
    hf_samples_repo: str = "bitmanagerai/etc"
    hf_samples_path: str = "qwen3tts-training"


@dataclass
class MainTalkerTrainingArguments(TrainingArguments):
    """HF TrainingArguments with the previous pipeline's useful defaults."""

    output_dir: str = "artifacts/checkpoints/sova_streaming_main_talker_full"
    do_train: bool = True
    do_eval: bool = True
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 10_000
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch_fused"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[dict[str, Any]] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    logging_strategy: str = "steps"
    logging_steps: float = 10
    eval_strategy: str = "steps"
    eval_steps: Optional[float] = 100
    eval_on_start: bool = True
    save_strategy: str = "steps"
    save_steps: float = 100
    save_total_limit: Optional[int] = 3
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: Optional[int] = None
    dataloader_drop_last: bool = False
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: Optional[bool] = False
    ddp_broadcast_buffers: Optional[bool] = False
    ddp_timeout: int = 14_400
    average_tokens_across_devices: bool = False
    report_to: Optional[str] = "tensorboard"
    run_name: Optional[str] = "full-main-talker-streaming-rl2"


def merged_pipeline_args(
    model_args: ModelArguments,
    data_args: DataArguments,
    validation_args: ValidationArguments,
    training_args: MainTalkerTrainingArguments,
) -> SimpleNamespace:
    values: dict[str, Any] = {}
    for arguments in (model_args, data_args, validation_args):
        values.update(asdict(arguments))
    values.update(
        output_dir=training_args.output_dir,
        seed=training_args.seed,
        run_name=training_args.run_name or "main-talker",
    )
    return SimpleNamespace(**values)


def row_key(row: dict[str, Any]) -> str:
    source_url = row.get("source_url", row.get("__url__", ""))
    source_key = row.get("source_key", row.get("__key__", ""))
    return f"{source_url}#{source_key}"


class SovaCandidateMapper:
    """Pickle-safe CPU transform used lazily by a HF IterableDataset."""

    def __init__(self, args: SimpleNamespace, excluded_keys: set[str] | None = None):
        self.args = args
        self.excluded_keys = frozenset(excluded_keys or ())
        self.stats: Counter[str] = Counter()
        self._accentor: Any = None
        self._librosa: Any = None
        self._soundfile: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["stats"] = Counter()
        state["_accentor"] = None
        state["_librosa"] = None
        state["_soundfile"] = None
        return state

    def _ensure_runtime(self) -> None:
        if self._accentor is not None:
            return
        import librosa
        import soundfile
        from silero_stress import load_accentor

        # silero-stress 1.4 changes global grad mode while loading. A DataLoader
        # worker does not train, but preserving the caller's mode is still safer.
        grad_enabled = torch.is_grad_enabled()
        try:
            with torch.no_grad():
                self._accentor = load_accentor()
        finally:
            torch.set_grad_enabled(grad_enabled)
        self._librosa = librosa
        self._soundfile = soundfile

    @property
    def accentor(self) -> Any:
        self._ensure_runtime()
        return self._accentor

    def __call__(self, source_row: dict[str, Any]) -> dict[str, Any]:
        key = row_key(source_row)
        if key in self.excluded_keys:
            self.stats["skip_validation_key"] += 1
            return {"_accepted": False, "_candidate": {}}
        self._ensure_runtime()
        candidate = candidate_from_row(
            source_row,
            self.args,
            self._accentor,
            self._librosa,
            self._soundfile,
            self.stats,
        )
        if candidate is None:
            return {"_accepted": False, "_candidate": {}}
        self.stats["accepted_candidates"] += 1
        return {"_accepted": True, "_candidate": candidate}


def _is_accepted(value: bool) -> bool:
    return bool(value)


def build_candidate_dataset(
    args: SimpleNamespace,
    token: str | None,
    *,
    excluded_keys: set[str] | None = None,
    repeat: bool = False,
) -> tuple[Any, SovaCandidateMapper]:
    """Return a real HF streaming dataset of accepted, decoded candidates."""

    dataset = streaming_dataset(args, token)
    mapper = SovaCandidateMapper(args, excluded_keys)
    source_columns = list(dataset.column_names or dataset.features.keys())
    dataset = dataset.map(mapper, remove_columns=source_columns)
    dataset = dataset.filter(_is_accepted, input_columns=["_accepted"])
    dataset = dataset.remove_columns(["_accepted"])
    if repeat:
        dataset = dataset.repeat(None)
    return dataset, mapper


def prepared_item(row: dict[str, Any], index: int) -> dict[str, Any]:
    codes = normalize_codes(row["full_codes"], "full_codes")
    prefix_frames = int(row["prefix_frames"])
    if not 0 < prefix_frames < codes.shape[0]:
        raise ValueError(
            f"Invalid prefix_frames={prefix_frames} for T={codes.shape[0]}"
        )
    return {
        "text_ids": normalize_ids(row["text_ids"], "text_ids"),
        "full_codes": codes,
        "speaker_embedding": torch.as_tensor(
            row["speaker_embedding"], dtype=torch.float32
        ).reshape(-1),
        "prefix_frames": prefix_frames,
        "language": row.get("language", "Russian") or "Russian",
        "text": row.get("text", ""),
        "key": row_key(row),
        "index": index,
    }


def collate_prepared_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return FullUtteranceDataset.collate(
        [prepared_item(row, index) for index, row in enumerate(rows)]
    )


class MainTalkerDataCollator:
    """CPU-only collator; online GPU encoding deliberately lives in forward."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        labels = torch.zeros(len(features), dtype=torch.long)
        if "_candidate" in features[0]:
            return {
                "candidates": [feature["_candidate"] for feature in features],
                "labels": labels,
            }
        batch = collate_prepared_rows(features)
        return {
            "text_ids": batch["text_ids"],
            "text_lengths": batch["text_lengths"],
            "full_codes": batch["full_codes"],
            "code_lengths": batch["code_lengths"],
            "speaker_embeddings": batch["speaker_embeddings"],
            "prefix_frames": batch["prefix_frames"],
            "languages": batch["languages"],
            "labels": labels,
        }


def configure_full_main_talker(model: Any) -> tuple[int, int]:
    base = core_model(model)
    base.requires_grad_(False)
    base.talker.requires_grad_(True)
    base.talker.code_predictor.requires_grad_(False)
    speech_tokenizer_model = getattr(base.speech_tokenizer, "model", None)
    if speech_tokenizer_model is not None:
        speech_tokenizer_model.requires_grad_(False)
    if base.speaker_encoder is not None:
        base.speaker_encoder.requires_grad_(False)
    base.talker.config.use_cache = False

    trainable = [
        (name, parameter)
        for name, parameter in base.named_parameters()
        if parameter.requires_grad
    ]
    forbidden = [
        name
        for name, _ in trainable
        if not name.startswith("talker.") or "talker.code_predictor" in name
    ]
    if forbidden:
        raise RuntimeError(f"Unexpected trainable parameters: {forbidden[:10]}")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    total_count = sum(parameter.numel() for parameter in base.parameters())
    if not trainable_count:
        raise RuntimeError("No main-Talker parameters were enabled")
    return trainable_count, total_count


def set_training_modes(model: Any) -> None:
    base = core_model(model)
    base.train()
    speech_tokenizer_model = getattr(base.speech_tokenizer, "model", None)
    if speech_tokenizer_model is not None:
        speech_tokenizer_model.eval()
    if base.speaker_encoder is not None:
        base.speaker_encoder.eval()
    base.talker.code_predictor.eval()


class MainTalkerTrainingModel(PreTrainedModel):
    """Thin task adapter that lets the unmodified HF Trainer own the loop."""

    accepts_loss_kwargs = False

    def __init__(self, qwen3tts: Any, pipeline_args: SimpleNamespace):
        super().__init__(qwen3tts.model.config)
        self.qwen_model = qwen3tts.model
        self.qwen3tts = qwen3tts
        self.pipeline_args = pipeline_args

        # Qwen3TTSTokenizer is a plain Python wrapper, so its torch model is not
        # registered below the root Qwen model. Register it here so Trainer.to()
        # moves the frozen online encoder to the rank-local device as well.
        speech_model = getattr(self.qwen_model.speech_tokenizer, "model", None)
        if speech_model is None:
            raise RuntimeError("The Qwen speech tokenizer was not loaded")
        self.speech_tokenizer_model = speech_model

    def get_base_model(self) -> Any:
        return self.qwen_model

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Trainer saves the original Qwen key layout, without adapter prefixes or
        # a duplicate frozen speech-tokenizer state.
        return self.qwen_model.state_dict(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        try:
            return self.qwen_model.load_state_dict(
                state_dict, strict=strict, assign=assign
            )
        except TypeError:
            return self.qwen_model.load_state_dict(state_dict, strict=strict)

    def save_pretrained(
        self, save_directory: str | Path, *args: Any, **kwargs: Any
    ) -> Any:
        kwargs.setdefault("max_shard_size", "5GB")
        return self.qwen_model.save_pretrained(save_directory, *args, **kwargs)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        kwargs = gradient_checkpointing_kwargs or {}
        try:
            self.qwen_model.talker.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=kwargs
            )
        except TypeError:
            self.qwen_model.talker.gradient_checkpointing_enable()
        self.qwen_model.talker.config.use_cache = False

    def gradient_checkpointing_disable(self) -> None:
        self.qwen_model.talker.gradient_checkpointing_disable()

    def train(self, mode: bool = True) -> "MainTalkerTrainingModel":
        super().train(mode)
        if mode:
            set_training_modes(self.qwen_model)
        return self

    def _sync_speech_tokenizer_device(self) -> torch.device:
        device = next(self.qwen_model.talker.parameters()).device
        self.qwen_model.speech_tokenizer.device = device
        return device

    def _encode_candidates(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        device = self._sync_speech_tokenizer_device()
        rows = prepared_rows(
            candidates,
            qwen3tts=self.qwen3tts,
            args=self.pipeline_args,
            torch=torch,
            serialize=False,
            output_device=device,
        )
        return collate_prepared_rows(rows)

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
        if candidates is not None:
            batch = self._encode_candidates(candidates)
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

        if self.training:
            loss_sum, loss_components = compute_loss(
                self.qwen_model, batch, return_loss_components=True
            )
            loss = distributed_token_mean(
                loss_sum, loss_components["token_count"]
            )
            batch_size = (
                len(candidates) if candidates is not None else int(text_ids.shape[0])
            )
            metric_logits = loss.new_zeros((batch_size, 4))
        else:
            loss, token_metrics = compute_loss(
                self.qwen_model, batch, return_token_metrics=True
            )
            metric_logits = token_metrics["per_sample_metrics"]
        return {"loss": loss, "logits": metric_logits}


def distributed_token_mean(
    local_loss_sum: torch.Tensor, local_token_count: torch.Tensor
) -> torch.Tensor:
    """Produce a global token mean after DDP's gradient averaging."""

    global_token_count = local_token_count.detach().to(local_loss_sum.device).clone()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_token_count, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    # DDP averages gradients across ranks. Multiplying each local loss sum by
    # world_size makes that average equal sum(local gradients) / global tokens.
    return local_loss_sum * world_size / global_token_count.clamp_min(1)


def compute_validation_metrics(prediction: Any) -> dict[str, float]:
    predictions = prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    values = np.asarray(predictions, dtype=np.float64).reshape(-1, 4).sum(axis=0)
    loss_sum, token_count, top1_correct, top5_correct = values.tolist()
    denominator = max(1.0, token_count)
    return {
        "codec_token_loss": loss_sum / denominator,
        "token_accuracy_top1": top1_correct / denominator,
        "token_accuracy_top5": top5_correct / denominator,
        "target_tokens": token_count,
    }


def encode_validation_rows(
    dataset: Any,
    mapper: SovaCandidateMapper,
    qwen3tts: Any,
    args: SimpleNamespace,
    *,
    sample_count: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    candidates = [item["_candidate"] for item in dataset.take(sample_count)]
    if len(candidates) < sample_count:
        raise RuntimeError(
            f"Only {len(candidates)} validation samples were found; requested {sample_count}; "
            f"filter stats={dict(mapper.stats)}"
        )
    rows: list[dict[str, Any]] = []
    for start in range(0, len(candidates), max(1, batch_size)):
        rows.extend(
            prepared_rows(
                candidates[start : start + batch_size],
                qwen3tts=qwen3tts,
                args=args,
                torch=torch,
                serialize=True,
            )
        )
    return rows


def _copy_speech_tokenizer(qwen3tts: Any, destination: Path) -> None:
    speech_tokenizer = qwen3tts.model.speech_tokenizer
    source = Path(str(getattr(speech_tokenizer.model.config, "_name_or_path", "")))
    if source.is_dir():
        if source.resolve() == destination.resolve():
            return
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    speech_tokenizer.model.save_pretrained(
        destination, safe_serialization=True, max_shard_size="5GB"
    )
    if speech_tokenizer.feature_extractor is not None:
        speech_tokenizer.feature_extractor.save_pretrained(destination)


STANDALONE_CHECKPOINT_MARKER = "standalone_complete.json"


def _has_complete_model_weights(directory: Path) -> bool:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = directory / filename
        if path.is_file() and path.stat().st_size > 0:
            return True
    for filename in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = directory / filename
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get(
                "weight_map", {}
            )
        except (OSError, ValueError, TypeError):
            continue
        shards = (
            set(weight_map.values())
            if isinstance(weight_map, dict)
            and all(isinstance(shard, str) for shard in weight_map.values())
            else set()
        )
        if shards and all(
            (directory / shard).is_file()
            and (directory / shard).stat().st_size > 0
            for shard in shards
        ):
            return True
    return False


def stage_checkpoint_assets(qwen3tts: Any, output_dir: str | Path) -> Path:
    """Write costly inference assets before Trainer rotates older checkpoints."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / STANDALONE_CHECKPOINT_MARKER).unlink(missing_ok=True)
    qwen3tts.model.config.save_pretrained(destination)
    qwen3tts.processor.save_pretrained(destination)
    generation_config = qwen3tts.model.generate_config
    if hasattr(generation_config, "to_dict"):
        generation_config = generation_config.to_dict()
    (destination / "generation_config.json").write_text(
        json.dumps(generation_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _copy_speech_tokenizer(qwen3tts, destination / "speech_tokenizer")

    speech_directory = destination / "speech_tokenizer"
    required = [
        destination / "config.json",
        destination / "generation_config.json",
        destination / "tokenizer_config.json",
        speech_directory / "config.json",
        speech_directory / "preprocessor_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if not _has_complete_model_weights(speech_directory):
        missing.append(f"{speech_directory}/<model weights or complete index>")
    if missing:
        raise RuntimeError(f"Incomplete staged checkpoint assets: {missing}")
    return destination


def complete_checkpoint_assets(output_dir: str | Path) -> Path:
    """Validate a Trainer checkpoint and atomically mark it directly reloadable."""

    destination = Path(output_dir)
    speech_directory = destination / "speech_tokenizer"
    required = [
        destination / "config.json",
        destination / "generation_config.json",
        destination / "tokenizer_config.json",
        speech_directory / "config.json",
        speech_directory / "preprocessor_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if not _has_complete_model_weights(destination):
        missing.append(f"{destination}/<model weights or complete index>")
    if not _has_complete_model_weights(speech_directory):
        missing.append(f"{speech_directory}/<model weights or complete index>")
    if missing:
        raise RuntimeError(f"Incomplete standalone checkpoint: {missing}")

    marker = destination / STANDALONE_CHECKPOINT_MARKER
    temporary_marker = destination / f".{STANDALONE_CHECKPOINT_MARKER}.tmp"
    temporary_marker.write_text(
        json.dumps(
            {
                "format": "qwen3tts-standalone-checkpoint",
                "version": 1,
                "completed_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_marker, marker)
    return destination


def save_checkpoint_assets(qwen3tts: Any, output_dir: str | Path) -> Path:
    """Complete a Trainer checkpoint so Qwen3TTSModel can reload it directly."""

    destination = stage_checkpoint_assets(qwen3tts, output_dir)
    return complete_checkpoint_assets(destination)


def raise_distributed_error(error: str | None, context: str) -> None:
    """Raise the same rank-local failure everywhere without stranding DDP peers."""

    errors = [error]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, error)
    for rank, rank_error in enumerate(errors):
        if rank_error is not None:
            raise RuntimeError(f"{context} failed on rank {rank}:\n{rank_error}")


class StandaloneCheckpointCallback(TrainerCallback):
    def __init__(self, qwen3tts: Any):
        self.qwen3tts = qwen3tts
        self.staged_steps: set[int] = set()

    @staticmethod
    def _checkpoint(args: TrainingArguments, state: Any) -> Path:
        return Path(args.output_dir) / f"checkpoint-{state.global_step}"

    def _stage_if_needed(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        *,
        force: bool = False,
    ) -> None:
        step = int(state.global_step)
        if step in self.staged_steps or (not force and not control.should_save):
            return
        error = None
        if state.is_world_process_zero:
            try:
                stage_checkpoint_assets(self.qwen3tts, self._checkpoint(args, state))
            except Exception:
                error = traceback.format_exc()
        raise_distributed_error(error, "standalone checkpoint asset staging")
        self.staged_steps.add(step)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        self._stage_if_needed(args, state, control)
        return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        self._stage_if_needed(args, state, control)
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        self._stage_if_needed(args, state, control, force=True)
        error = None
        if state.is_world_process_zero:
            try:
                checkpoint = self._checkpoint(args, state)
                complete_checkpoint_assets(checkpoint)
                log(f"saved standalone Trainer checkpoint {checkpoint}")
            except Exception:
                error = traceback.format_exc()
        raise_distributed_error(error, "standalone checkpoint asset save")
        self.staged_steps.discard(int(state.global_step))
        return control


def upload_validation_samples(
    *,
    validation_dir: Path,
    step: int,
    metrics: dict[str, float],
    args: SimpleNamespace,
    token: str | None,
) -> str | None:
    if not args.hf_upload_samples:
        return None
    if not token:
        raise RuntimeError(
            f"{args.hf_token_env} is required to upload validation samples"
        )
    from huggingface_hub import HfApi

    safe_run_name = re.sub(r"[^A-Za-z0-9._-]+", "-", args.run_name).strip("-.")
    if not safe_run_name:
        raise ValueError(f"Invalid run_name: {args.run_name!r}")
    root = args.hf_samples_path.strip("/")
    remote_dir = f"{root}/{safe_run_name}/step-{step:08d}"
    payload = {
        "step": step,
        "metrics": metrics,
        "model_path": args.model_path,
        "dataset": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "remote_path": remote_dir,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (validation_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=args.hf_samples_repo,
        repo_type="dataset",
        folder_path=str(validation_dir),
        path_in_repo=remote_dir,
        commit_message=f"qwen3tts validation samples at step {step}",
    )
    api.upload_file(
        repo_id=args.hf_samples_repo,
        repo_type="dataset",
        path_or_fileobj=io.BytesIO(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        ),
        path_in_repo=f"{root}/{safe_run_name}/latest.json",
        commit_message=f"update qwen3tts validation pointer to step {step}",
    )
    url = (
        f"https://huggingface.co/datasets/{args.hf_samples_repo}/tree/main/{remote_dir}"
    )
    log(f"uploaded validation samples: {url}")
    return url


class AudioValidationCallback(TrainerCallback):
    """Keep expensive listening probes outside the generic Trainer loop."""

    def __init__(
        self,
        *,
        training_model: MainTalkerTrainingModel,
        processor: Any,
        validation_rows: list[dict[str, Any]],
        texts: list[str],
        pipeline_args: SimpleNamespace,
        token: str | None,
    ):
        self.training_model = training_model
        self.processor = processor
        self.validation_rows = validation_rows
        self.texts = texts
        self.pipeline_args = pipeline_args
        self.token = token
        self.completed_steps: set[int] = set()
        self.writer: Any = None
        self.asr_validator: Any = None

    def _run(self, step: int, eval_metrics: dict[str, float]) -> None:
        from torch.utils.tensorboard import SummaryWriter

        if self.writer is None:
            self.writer = SummaryWriter(self.pipeline_args.logging_dir)
        if self.pipeline_args.asr_enabled and self.asr_validator is None:
            self.asr_validator = GigaAMValidator(self.pipeline_args)
        output_dir = Path(self.pipeline_args.output_dir) / f"validation-step-{step}"
        audio_metrics = generate_validation(
            model=self.training_model,
            processor=self.processor,
            val_rows=self.validation_rows,
            texts=self.texts,
            output_dir=output_dir,
            step=step,
            args=self.pipeline_args,
            writer=self.writer,
            asr_validator=self.asr_validator,
        )
        metrics = {**eval_metrics, **audio_metrics}
        upload_validation_samples(
            validation_dir=output_dir,
            step=step,
            metrics=metrics,
            args=self.pipeline_args,
            token=self.token,
        )
        self.training_model.train()
        log(f"step={step} audio validation metrics: {audio_metrics}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        step = int(state.global_step)
        if step in self.completed_steps:
            return control
        error = None
        if state.is_world_process_zero:
            try:
                self._run(step, dict(metrics or {}))
            except Exception:
                error = traceback.format_exc()
        raise_distributed_error(error, "audio validation")
        self.completed_steps.add(step)
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        return control


def _parse_arguments() -> tuple[
    ModelArguments,
    DataArguments,
    ValidationArguments,
    MainTalkerTrainingArguments,
]:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            ValidationArguments,
            MainTalkerTrainingArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return tuple(parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])))
    return tuple(parser.parse_args_into_dataclasses())


def _dtype_for(arguments: MainTalkerTrainingArguments) -> torch.dtype:
    if arguments.bf16:
        return torch.bfloat16
    if arguments.fp16:
        return torch.float16
    return torch.float32


def checkpoint_step(model_path: str | Path) -> int:
    state_path = Path(model_path) / "trainer_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("global_step", 0))
    match = re.search(r"checkpoint-(\d+)$", str(model_path).rstrip("/"))
    return int(match.group(1)) if match else 0


def main() -> None:
    model_args, data_args, validation_args, training_args = _parse_arguments()
    if not data_args.streaming:
        raise ValueError("This trainer requires --streaming true")
    if training_args.do_train and training_args.max_steps <= 0:
        raise ValueError("A streaming train dataset requires --max_steps > 0")
    if training_args.label_smoothing_factor != 0:
        raise ValueError(
            "--label_smoothing_factor must stay 0: collator labels are only a "
            "Trainer evaluation marker, while codec targets are built inside forward"
        )
    if training_args.world_size == 1 and training_args.n_gpu > 1:
        raise RuntimeError(
            "Multiple CUDA devices are visible, but distributed training was not "
            "launched. This online-audio adapter does not support nn.DataParallel; "
            "use NUM_GPUS=<count>, torchrun, or restrict CUDA_VISIBLE_DEVICES to one GPU."
        )
    if training_args.do_eval and data_args.validation_samples <= 0:
        raise ValueError("Evaluation requires --validation_samples > 0")
    if data_args.validation_encode_batch_size <= 0:
        raise ValueError("validation_encode_batch_size must be positive")

    # Iterable batches contain numpy audio and strings. They must be fetched on
    # every rank; with dispatch disabled Accelerate efficiently calls
    # datasets.IterableDataset.shard(...) when the source has enough shards.
    if training_args.accelerator_config.dispatch_batches is True:
        raise ValueError(
            "accelerator_config.dispatch_batches must be false for online audio encoding"
        )
    training_args.accelerator_config.dispatch_batches = False
    training_args.remove_unused_columns = False
    if training_args.dataloader_num_workers == 0:
        training_args.dataloader_prefetch_factor = None
    elif training_args.dataloader_prefetch_factor is None:
        training_args.dataloader_prefetch_factor = 1
    if not training_args.do_eval:
        training_args.eval_strategy = "no"
        training_args.eval_on_start = False

    set_seed(training_args.seed)
    pipeline_args = merged_pipeline_args(
        model_args, data_args, validation_args, training_args
    )
    pipeline_args.logging_dir = training_args.logging_dir
    token = os.environ.get(model_args.hf_token_env) or dotenv_value(
        model_args.env_file, model_args.hf_token_env
    )

    from datasets import Dataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    dtype = _dtype_for(training_args)
    if training_args.process_index == 0:
        log(
            f"loading {model_args.model_path} as Base ({dtype}, "
            f"{model_args.attn_implementation}); world_size={training_args.world_size}"
        )
    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_args.model_path,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation=model_args.attn_implementation,
        token=token,
    )
    qwen3tts.model.config.tts_model_type = "base"
    qwen3tts.model.tts_model_type = "base"
    trainable_count, total_count = configure_full_main_talker(qwen3tts.model)
    training_model = MainTalkerTrainingModel(qwen3tts, pipeline_args)
    training_model.to(training_args.device)
    training_model._sync_speech_tokenizer_device()
    if training_args.process_index == 0:
        log(
            f"main Talker trainable={trainable_count:,}/{total_count:,} "
            f"({100 * trainable_count / total_count:.2f}%); code predictor frozen"
        )

    validation_rows: list[dict[str, Any]] = []
    validation_dataset = None
    fixed_texts: list[str] = []
    validation_keys: set[str] = set()
    if training_args.do_eval:
        if training_args.process_index == 0:
            log("collecting the fixed validation set from the HF stream")
        validation_stream, validation_mapper = build_candidate_dataset(
            pipeline_args, token
        )
        validation_rows = encode_validation_rows(
            validation_stream,
            validation_mapper,
            qwen3tts,
            pipeline_args,
            sample_count=data_args.validation_samples,
            batch_size=data_args.validation_encode_batch_size,
        )
        validation_keys = {row_key(row) for row in validation_rows}
        validation_dataset = Dataset.from_list(validation_rows)
        if validation_args.generate_audio and training_args.process_index == 0:
            fixed_texts = validation_texts(
                validation_args.validation_texts_file,
                validation_args.validation_texts,
                validation_mapper.accentor,
            )
        if training_args.process_index == 0:
            log(
                f"validation ready: {len(validation_rows)} rows; "
                f"source rows inspected={validation_mapper.stats['seen']}"
            )

    train_dataset = None
    if training_args.do_train:
        train_dataset, _ = build_candidate_dataset(
            pipeline_args,
            token,
            excluded_keys=validation_keys,
            repeat=True,
        )

    callbacks: list[TrainerCallback] = [StandaloneCheckpointCallback(qwen3tts)]
    if training_args.do_eval and validation_args.generate_audio:
        callbacks.append(
            AudioValidationCallback(
                training_model=training_model,
                processor=qwen3tts.processor,
                validation_rows=validation_rows,
                texts=fixed_texts,
                pipeline_args=pipeline_args,
                token=token,
            )
        )

    trainer = Trainer(
        model=training_model,
        args=training_args,
        data_collator=MainTalkerDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=qwen3tts.processor,
        compute_metrics=compute_validation_metrics if training_args.do_eval else None,
        callbacks=callbacks,
    )

    if not training_args.do_train:
        trainer.state.global_step = checkpoint_step(model_args.model_path)

    if training_args.do_train:
        result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
    elif training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if model_args.save_final_model:
        error = None
        try:
            trainer.save_model(training_args.output_dir)
            if training_args.should_save:
                save_checkpoint_assets(qwen3tts, training_args.output_dir)
                log(f"saved final standalone model to {training_args.output_dir}")
        except Exception:
            error = traceback.format_exc()
        raise_distributed_error(error, "final model save")
    if training_args.process_index == 0:
        log(f"finished at optimizer step {trainer.state.global_step}")


if __name__ == "__main__":
    main()
