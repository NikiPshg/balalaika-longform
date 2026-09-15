# coding=utf-8
"""Config-driven TRL SFT for the streaming Qwen3-TTS main Talker."""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# This script lives in a directory named ``trl``. Importing from the external
# package is safe because Python searches for ``trl`` *inside* the script
# directory, not the directory itself. ``trainers`` is the local namespace.
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, TrlParser

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
FINETUNING_ROOT = Path(__file__).resolve().parent.parent
if str(FINETUNING_ROOT) not in sys.path:
    sys.path.insert(0, str(FINETUNING_ROOT))

from sft_streaming_main_talker_full import (  # noqa: E402
    AudioValidationCallback,
    DataArguments,
    ModelArguments,
    StandaloneCheckpointCallback,
    ValidationArguments,
    checkpoint_step,
    compute_validation_metrics,
    configure_full_main_talker,
    dotenv_value,
    encode_validation_rows,
    merged_pipeline_args,
    raise_distributed_error,
    row_key,
    save_checkpoint_assets,
    validation_texts,
)
from trainers.qwen_tts_sft import (  # noqa: E402
    MainTalkerDataCollator,
    MainTalkerTrainingModel,
    QwenTtsSFTTrainer,
    TalkerGradientMetricsCallback,
    TrackioAudioValidationCallback,
    build_tts_candidate_dataset,
    compute_all_talker_validation_metrics,
    configure_all_talker,
)


def log(message: str) -> None:
    print(
        f"[trl-sft-main-talker] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}",
        flush=True,
    )


@dataclass
class TtsModelArguments(ModelArguments):
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_revision: str = "fd4b254389122332181a7c3db7f27e918eec64e3"
    model_dtype: str = "auto"


@dataclass
class TtsDataArguments(DataArguments):
    dataset_name: str = "lab260/youtube_balalaika"
    dataset_revision: str = "f847a349ac9cbf2952726c12b5aab1fd70b8a4cf"
    audio_column: str = "flac"
    stream_preflight_rows: int = 2_048
    stream_max_retries: int = 3
    stream_retry_backoff_sec: float = 1.0
    stream_skip_failed_shards: bool = True
    asr_consistency_threshold: float = 75.0
    dataset_format: str = "webdataset_audio"
    dataset_data_file: str | None = None
    min_asr_agreement: float = 0.97
    prepared_orientation: str = "prefix_reference"
    prepared_profile: str = "balanced"
    prepared_boundary_type: str = "punctuation"
    prepared_boundary_tier: str = "primary"
    prepared_text_source: str = "rover_punctuated_accented"
    prepared_deduplicate_consecutive: bool = True


@dataclass
class TtsValidationArguments(ValidationArguments):
    """Listening-probe controls kept separate from inexpensive loss eval."""

    audio_validation_steps: int = 1_000
    audio_validation_on_start: bool = True


@dataclass
class TtsSFTConfig(SFTConfig):
    """SFTConfig defaults that are valid for online codec teacher forcing."""

    output_dir: str = "lora_finetuning/trl/outputs/sft-main-talker-full-base"
    do_train: bool = True
    do_eval: bool = True
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 10_000
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch_fused"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
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
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: Optional[int] = 2
    dataloader_persistent_workers: bool = True
    dataloader_drop_last: bool = False
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: Optional[bool] = False
    ddp_broadcast_buffers: Optional[bool] = False
    ddp_timeout: int = 14_400
    average_tokens_across_devices: bool = False
    run_name: Optional[str] = "sft-main-talker-full-upstream-base"
    project: str = "qwen3-tts"
    trackio_space_id: Optional[str] = None
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default_factory=lambda: {"skip_prepare_dataset": True}
    )
    max_length: Optional[int] = None
    packing: bool = False
    padding_free: bool = False
    loss_type: Optional[str] = "nll"
    use_liger_kernel: bool = False
    training_scope: str = "main_talker"
    code_predictor_loss_weight: float = 1.0


def _parse_arguments() -> tuple[
    TtsModelArguments,
    TtsDataArguments,
    TtsValidationArguments,
    TtsSFTConfig,
]:
    parser = TrlParser(
        (
            TtsModelArguments,
            TtsDataArguments,
            TtsValidationArguments,
            TtsSFTConfig,
        )
    )
    return tuple(parser.parse_args_and_config())


def _dtype_for(arguments: TtsSFTConfig, model_arguments: TtsModelArguments) -> torch.dtype:
    explicit = model_arguments.model_dtype.lower()
    if explicit == "float32":
        return torch.float32
    if explicit == "bfloat16":
        return torch.bfloat16
    if explicit == "float16":
        return torch.float16
    if explicit != "auto":
        raise ValueError("model_dtype must be auto, float32, bfloat16, or float16")
    if arguments.bf16:
        return torch.bfloat16
    if arguments.fp16:
        return torch.float16
    return torch.float32


def _reporting_to(arguments: TtsSFTConfig, integration: str) -> bool:
    reports = arguments.report_to
    if reports is None:
        return False
    if isinstance(reports, str):
        reports = [reports]
    return integration in reports or "all" in reports


def _resolved_resume_checkpoint(arguments: TtsSFTConfig) -> str | None:
    value = arguments.resume_from_checkpoint
    if value is None:
        return None
    if isinstance(value, bool):
        if not value:
            return None
        checkpoint = get_last_checkpoint(arguments.output_dir)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found below {arguments.output_dir}")
        return checkpoint
    return str(value)


def _resolved_model_source(
    model_path: str, revision: str, token: str | None
) -> str:
    """Resolve Hub models once so every nested TTS asset uses one revision."""

    local_path = Path(model_path).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_path, revision=revision, token=token)


def _validate_configuration(
    model_args: TtsModelArguments,
    data_args: TtsDataArguments,
    validation_args: TtsValidationArguments,
    training_args: TtsSFTConfig,
) -> None:
    if not data_args.streaming:
        raise ValueError("This trainer requires streaming=true")
    if not training_args.do_train and not training_args.do_eval:
        raise ValueError("At least one of do_train/do_eval must be true")
    if not _reporting_to(training_args, "trackio"):
        raise ValueError("report_to must include trackio")
    if training_args.push_to_hub and not training_args.hub_model_id:
        raise ValueError("push_to_hub=true requires an explicit hub_model_id")
    if not training_args.do_eval and training_args.eval_strategy != "no":
        raise ValueError("eval_strategy requires do_eval=true and an eval dataset")
    if training_args.do_train and training_args.max_steps <= 0:
        raise ValueError("A streaming train dataset requires max_steps > 0")
    if training_args.label_smoothing_factor != 0:
        raise ValueError("label_smoothing_factor must be 0 for codec targets")
    if training_args.loss_type not in (None, "nll"):
        raise ValueError("Only loss_type=nll is valid for the custom codec loss")
    if training_args.use_liger_kernel:
        raise ValueError("use_liger_kernel is incompatible with the custom TTS forward")
    if training_args.packing or training_args.padding_free:
        raise ValueError("packing and padding_free are text-only and must stay false")
    if not (training_args.dataset_kwargs or {}).get("skip_prepare_dataset", False):
        raise ValueError("dataset_kwargs.skip_prepare_dataset must be true")
    if training_args.remove_unused_columns:
        raise ValueError("remove_unused_columns must be false")
    if training_args.average_tokens_across_devices:
        raise ValueError(
            "average_tokens_across_devices must be false; codec loss already uses a global token mean"
        )
    if training_args.training_scope not in {"main_talker", "all_talker"}:
        raise ValueError("training_scope must be 'main_talker' or 'all_talker'")
    if (
        not torch.isfinite(torch.tensor(training_args.code_predictor_loss_weight))
        or training_args.code_predictor_loss_weight <= 0
    ):
        raise ValueError("code_predictor_loss_weight must be finite and positive")
    if model_args.model_dtype.lower() not in {
        "auto",
        "float32",
        "bfloat16",
        "float16",
    }:
        raise ValueError("model_dtype must be auto, float32, bfloat16, or float16")
    if training_args.training_scope == "all_talker" and model_args.model_dtype != "float32":
        raise ValueError(
            "all_talker SFT requires model_dtype=float32 so small optimizer updates "
            "are not rounded away; bf16 training autocast remains enabled"
        )
    if training_args.world_size == 1 and training_args.n_gpu > 1:
        raise RuntimeError(
            "Multiple GPUs are visible without a distributed launcher; use accelerate launch"
        )
    if training_args.do_eval and data_args.validation_samples <= 0:
        raise ValueError("Evaluation requires validation_samples > 0")
    if data_args.validation_encode_batch_size <= 0:
        raise ValueError("validation_encode_batch_size must be positive")
    if training_args.do_train and data_args.stream_preflight_rows <= 0:
        raise ValueError("stream_preflight_rows must be positive")
    if data_args.stream_max_retries < 0:
        raise ValueError("stream_max_retries must be non-negative")
    if data_args.stream_retry_backoff_sec < 0:
        raise ValueError("stream_retry_backoff_sec must be non-negative")
    if data_args.dataset_format not in {
        "webdataset_audio",
        "qwen_prepared_full_utterance",
    }:
        raise ValueError(f"Unsupported dataset_format={data_args.dataset_format!r}")
    if (
        data_args.dataset_format == "qwen_prepared_full_utterance"
        and not data_args.dataset_data_file
    ):
        raise ValueError(
            "dataset_data_file is required for qwen_prepared_full_utterance"
        )
    if not 0.0 <= data_args.min_asr_agreement <= 1.0:
        raise ValueError("min_asr_agreement must be in the 0..1 range")
    if validation_args.generate_audio and not training_args.do_eval:
        raise ValueError("generate_audio requires do_eval=true")
    if validation_args.audio_validation_steps <= 0:
        raise ValueError("audio_validation_steps must be positive")


def main() -> None:
    model_args, data_args, validation_args, training_args = _parse_arguments()
    _validate_configuration(model_args, data_args, validation_args, training_args)
    training_args.resume_from_checkpoint = _resolved_resume_checkpoint(training_args)
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
    # ``merged_pipeline_args`` intentionally carries only generic Trainer
    # fields. Joint Talker SFT needs these task-specific controls inside the
    # model wrapper as well.
    pipeline_args.training_scope = training_args.training_scope
    pipeline_args.code_predictor_loss_weight = training_args.code_predictor_loss_weight
    token = os.environ.get(model_args.hf_token_env) or dotenv_value(
        model_args.env_file, model_args.hf_token_env
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
    if (
        training_args.push_to_hub
        or training_args.trackio_space_id
        or validation_args.hf_upload_samples
    ) and not token:
        raise RuntimeError(
            f"{model_args.hf_token_env} is required for Hub or remote Trackio persistence"
        )
    if training_args.push_to_hub and not training_args.hub_token and token:
        training_args.hub_token = token

    from datasets import Dataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    dtype = _dtype_for(training_args, model_args)
    if training_args.process_index == 0:
        log(
            f"loading {model_args.model_path}@{model_args.model_revision} "
            f"({dtype}, {model_args.attn_implementation}); full main-Talker; "
            f"world_size={training_args.world_size}"
        )
    model_source = _resolved_model_source(
        model_args.model_path, model_args.model_revision, token
    )
    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_source,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation=model_args.attn_implementation,
        token=token,
    )
    qwen3tts.model.config.tts_model_type = "base"
    qwen3tts.model.tts_model_type = "base"
    if training_args.training_scope == "all_talker":
        trainable_count, total_count, trainable_groups = configure_all_talker(
            qwen3tts.model
        )
    else:
        trainable_count, total_count = configure_full_main_talker(qwen3tts.model)
        trainable_groups = {
            "main_talker": trainable_count,
            "code_predictor": 0,
        }
    training_model = MainTalkerTrainingModel(qwen3tts, pipeline_args)
    training_model.to(training_args.device)
    training_model._sync_speech_tokenizer_device()
    # --- A10-qwen local edit (E10 P3, 2026-08-31): wire gradient checkpointing to the
    # talker explicitly. MainTalkerTrainingModel is a plain nn.Module wrapper, so the
    # HF Trainer cannot enable checkpointing through it, and GradientCheckpointingLayer
    # only engages in train() mode (measured: reports/qwen_p3_plan.md §6 in
    # .).
    if training_args.gradient_checkpointing:
        qwen3tts.model.talker.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=(training_args.gradient_checkpointing_kwargs
                                           or {"use_reentrant": False}))
        training_args.gradient_checkpointing = False
    if training_args.process_index == 0:
        log(
            f"trainable={trainable_count:,}/{total_count:,} "
            f"({100 * trainable_count / total_count:.3f}%); "
            f"scope={training_args.training_scope}; groups={trainable_groups}; "
            "speech tokenizer frozen; speaker encoder frozen"
        )

    validation_rows: list[dict[str, Any]] = []
    validation_dataset = None
    fixed_texts: list[str] = []
    validation_keys: set[str] = set()
    if training_args.do_eval:
        if training_args.process_index == 0:
            log("collecting a fixed validation set from the HF stream")
        validation_stream, validation_mapper = build_tts_candidate_dataset(
            pipeline_args, token, processor=qwen3tts.processor
        )
        if getattr(validation_mapper, "prepared_rows", False):
            validation_rows = list(
                validation_stream.take(data_args.validation_samples)
            )
            if len(validation_rows) < data_args.validation_samples:
                raise RuntimeError(
                    f"Only {len(validation_rows)} prepared validation samples were "
                    f"found; requested {data_args.validation_samples}; "
                    f"filter stats={dict(validation_mapper.stats)}"
                )
        else:
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

    train_dataset = None
    if training_args.do_train:
        train_dataset, _ = build_tts_candidate_dataset(
            pipeline_args,
            token,
            excluded_keys=validation_keys,
            repeat=True,
            preflight_rows=data_args.stream_preflight_rows,
            processor=qwen3tts.processor,
        )

    # --- A10-qwen local edit (E10 P3, 2026-08-31, disk rule): standalone inference
    # assets per intermediate save cost several GB each and the final save_model path
    # already stages them once via save_checkpoint_assets. Intermediate standalone
    # exports are therefore OFF unless explicitly requested.
    callbacks: list[TrainerCallback] = (
        [StandaloneCheckpointCallback(qwen3tts)]
        if os.environ.get("QWEN_STANDALONE_EVERY_SAVE") == "1" else []
    )
    if training_args.training_scope == "all_talker":
        callbacks.append(TalkerGradientMetricsCallback())
    if training_args.do_eval and validation_args.generate_audio:
        callback_type = (
            TrackioAudioValidationCallback
            if _reporting_to(training_args, "trackio")
            else AudioValidationCallback
        )
        callbacks.append(
            callback_type(
                training_model=training_model,
                processor=qwen3tts.processor,
                validation_rows=validation_rows,
                texts=fixed_texts,
                pipeline_args=pipeline_args,
                token=token,
            )
        )

    trainer = QwenTtsSFTTrainer(
        model=training_model,
        args=training_args,
        data_collator=MainTalkerDataCollator(),
        train_dataset=(
            train_dataset if train_dataset is not None else validation_dataset
        ),
        eval_dataset=validation_dataset,
        processing_class=qwen3tts.processor.tokenizer,
        compute_metrics=(
            compute_all_talker_validation_metrics
            if training_args.do_eval and training_args.training_scope == "all_talker"
            else compute_validation_metrics if training_args.do_eval else None
        ),
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
        except Exception:
            error = traceback.format_exc()
        raise_distributed_error(error, "final model save")
    if training_args.push_to_hub:
        trainer.push_to_hub(tags=["trl", "sft", "qwen3-tts", "full-finetune"])
    if training_args.process_index == 0:
        log(f"finished at optimizer step {trainer.state.global_step}")


if __name__ == "__main__":
    main()
