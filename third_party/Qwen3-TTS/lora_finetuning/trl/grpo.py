# coding=utf-8
"""Config-driven full main-Talker GRPO for 16-codebook Qwen3-TTS."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
FINETUNING_ROOT = SCRIPT_ROOT.parent
for path in (REPO_ROOT, SCRIPT_ROOT, FINETUNING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The script directory is itself named ``trl``.  Python searches *inside* that
# directory for a top-level package, so this resolves the installed TRL 1.9.2;
# local code is imported through the ``trainers`` namespace below.
from transformers import TrainerCallback, set_seed  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import TrlParser  # noqa: E402

from full_utterance_utils import (  # noqa: E402
    apply_silero_stress,
    asr_consensus,
    dotenv_value,
    select_training_transcript,
)
from sft_streaming_main_talker_full import (  # noqa: E402
    StandaloneCheckpointCallback,
    raise_distributed_error,
    save_checkpoint_assets,
)
from trainers.qwen_tts_grpo import QwenTTSGRPOConfig, QwenTTSGRPOTrainer  # noqa: E402
from data_stream import load_untyped_streaming_dataset  # noqa: E402


def log(message: str) -> None:
    print(f"[trl-grpo-main-talker] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


@dataclass
class ModelArguments:
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_revision: str = "fd4b254389122332181a7c3db7f27e918eec64e3"
    processor_source_path: str | None = None
    attn_implementation: str = "sdpa"
    local_files_only: bool = False
    cache_dir: str | None = None
    hf_token_env: str = "HF_TOKEN"
    env_file: str = str(REPO_ROOT / ".env")
    save_final_model: bool = True


@dataclass
class DataArguments:
    dataset_name: str = "lab260/youtube_balalaika"
    dataset_split: str = "train"
    dataset_revision: str = "f847a349ac9cbf2952726c12b5aab1fd70b8a4cf"
    streaming: bool = True
    shuffle_buffer: int = 2_048
    stream_preflight_rows: int = 256
    stream_max_retries: int = 3
    stream_retry_backoff_sec: float = 1.0
    stream_skip_failed_shards: bool = True
    audio_column: str = "flac"
    references_path: str | None = None
    auto_references: bool = True
    reference_scan_rows: int = 50_000
    reference_output_dir: str | None = None
    min_duration_sec: float = 5.0
    max_duration_sec: float = 30.0
    min_distill_mos: float = 3.5
    max_music_prob: float = 0.30
    max_asr_wer: float = 0.25
    max_e2e_rover_wer: float = 0.0
    require_single_speaker: bool = True
    min_words: int = 4
    max_text_chars: int = 800
    dataset_format: str = "webdataset_audio"
    dataset_data_file: str | None = None
    min_asr_agreement: float = 0.97
    prepared_text_source: str = "rover_punctuated_accented"
    prepared_deduplicate_consecutive: bool = True


class SileroPromptMapper:
    """Lazy prompt-only counterpart of the SFT stream preprocessing path."""

    def __init__(self, args: DataArguments) -> None:
        self.args = args
        self.stats: Counter[str] = Counter()
        self._accentor: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["stats"] = Counter()
        state["_accentor"] = None
        return state

    def _ensure_accentor(self) -> None:
        if self._accentor is not None:
            return
        from silero_stress import load_accentor

        grad_enabled = torch.is_grad_enabled()
        try:
            with torch.no_grad():
                self._accentor = load_accentor()
        finally:
            # silero-stress 1.4 changes the process-wide grad mode on load.
            torch.set_grad_enabled(grad_enabled)

    @staticmethod
    def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
        value = row.get("json") or {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
        if not isinstance(value, Mapping):
            return {}
        metadata = dict(value)
        asr = metadata.get("asr")
        asr = asr if isinstance(asr, Mapping) else {}
        metadata.update(
            {
                "giga_ctc.txt": asr.get("gigaam-v3-ctc"),
                "giga_rnnt.txt": asr.get("gigaam-v3-rnnt"),
                "vosk.txt": asr.get("vosk-model-ru"),
                "gigaam-v3-e2e-ctc.txt": asr.get("gigaam-v3-e2e-ctc"),
                "rover.txt": metadata.get("rover"),
                "punct.txt": metadata.get("punct"),
                "accent.txt": metadata.get("accent"),
            }
        )
        return metadata

    @staticmethod
    def _number(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if math.isfinite(result) else float(default)

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        self.stats["seen"] += 1
        metadata = self._metadata(row)
        duration = self._number(metadata.get("total_duration"), 0.0)
        if not self.args.min_duration_sec <= duration <= self.args.max_duration_sec:
            self.stats["skip_duration"] += 1
            return {"_accepted": False, "prompt": ""}
        if self.args.require_single_speaker and not bool(metadata.get("is_single_speaker")):
            self.stats["skip_multispeaker"] += 1
            return {"_accepted": False, "prompt": ""}
        if self._number(metadata.get("DistillMOS"), 0.0) < self.args.min_distill_mos:
            self.stats["skip_mos"] += 1
            return {"_accepted": False, "prompt": ""}
        music_prob = metadata.get("music_prob")
        if self._number(music_prob, 1.0) > self.args.max_music_prob:
            self.stats["skip_music"] += 1
            return {"_accepted": False, "prompt": ""}
        consensus_ok, _ = asr_consensus(metadata, max_wer=self.args.max_asr_wer)
        if not consensus_ok:
            self.stats["skip_asr_consensus"] += 1
            return {"_accepted": False, "prompt": ""}
        source_text, transcript_source, _ = select_training_transcript(
            metadata, max_wer=self.args.max_e2e_rover_wer
        )
        if not source_text:
            self.stats["skip_missing_transcript"] += 1
            return {"_accepted": False, "prompt": ""}
        if len(source_text.split()) < self.args.min_words or len(source_text) > self.args.max_text_chars:
            self.stats["skip_text_length"] += 1
            return {"_accepted": False, "prompt": ""}
        self._ensure_accentor()
        try:
            stressed = apply_silero_stress(self._accentor, source_text)
        except Exception:
            self.stats["skip_stress_error"] += 1
            return {"_accepted": False, "prompt": ""}
        source_url = str(row.get("__url__") or "")
        source_key = str(row.get("__key__") or "")
        self.stats["accepted"] += 1
        return {
            "_accepted": True,
            "id": f"{source_url}#{source_key}",
            "source_key": source_key,
            "prompt": stressed,
            "stressed": stressed,
            "source_text": source_text,
            "transcript_source": transcript_source,
            "num_words": [],
        }


class PreparedHighAgreementPromptMapper:
    """Prompt-only view of one reconstructed row per prepared source group."""

    def __init__(self, args: DataArguments) -> None:
        self.args = args
        self.stats: Counter[str] = Counter()
        self._accentor: Any = None
        self._last_source_key: str | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["stats"] = Counter()
        state["_accentor"] = None
        state["_last_source_key"] = None
        return state

    def _reject(self, reason: str) -> dict[str, Any]:
        self.stats[reason] += 1
        return {"_accepted": False, "prompt": ""}

    def _ensure_accentor(self) -> None:
        if self._accentor is not None:
            return
        from silero_stress import load_accentor

        grad_enabled = torch.is_grad_enabled()
        try:
            with torch.no_grad():
                self._accentor = load_accentor()
        finally:
            torch.set_grad_enabled(grad_enabled)

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        source_key = str(row.get("source_record_id") or "").strip()
        if not source_key:
            return self._reject("skip_missing_source_record_id")
        if self.args.prepared_deduplicate_consecutive:
            if source_key == self._last_source_key:
                return self._reject("skip_duplicate_source_record_id")
            self._last_source_key = source_key
        if str(row.get("agreement_bucket") or "") != "agreement_ge_0_95":
            return self._reject("skip_agreement_bucket")
        if str(row.get("text_source") or "") != self.args.prepared_text_source:
            return self._reject("skip_text_source")
        agreement = SileroPromptMapper._number(row.get("asr_agreement_mean"), 0.0)
        if agreement < self.args.min_asr_agreement:
            return self._reject("skip_asr_agreement")
        duration = SileroPromptMapper._number(
            row.get("ref_duration"), 0.0
        ) + SileroPromptMapper._number(row.get("duration"), 0.0)
        if not self.args.min_duration_sec <= duration <= self.args.max_duration_sec:
            return self._reject("skip_duration")

        orientation = str(row.get("orientation") or "")
        if orientation == "prefix_reference":
            text_parts = (row.get("ref_text"), row.get("text"))
        elif orientation == "suffix_reference":
            text_parts = (row.get("text"), row.get("ref_text"))
        else:
            return self._reject("skip_unknown_orientation")
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
        self._ensure_accentor()
        try:
            stressed = apply_silero_stress(self._accentor, source_text)
        except Exception:
            return self._reject("skip_stress_error")
        if not stressed.strip():
            return self._reject("skip_empty_stressed_text")
        self.stats["accepted"] += 1
        return {
            "_accepted": True,
            "id": f"qwen-high-agreement#{source_key}",
            "source_key": source_key,
            "prompt": stressed,
            "stressed": stressed,
            "source_text": source_text,
            "transcript_source": "rover_punctuated_accented+silero_stress",
            "asr_agreement_mean": agreement,
            "duration": duration,
            "num_words": [],
        }


class LocalCriticalPromptMapper:
    """Validate already stressed, typed prompts from an immutable local JSONL."""

    def __init__(self, args: DataArguments) -> None:
        self.args = args
        self.stats: Counter[str] = Counter()

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        row_id = str(row.get("id") or "").strip()
        prompt = str(row.get("stressed") or row.get("prompt") or "").strip()
        source_text = str(row.get("source_text") or "").strip()
        spans = row.get("critical_spans")
        if not row_id:
            self.stats["skip_missing_id"] += 1
            return {"_accepted": False, "prompt": ""}
        if not prompt or not source_text:
            self.stats["skip_missing_text"] += 1
            return {"_accepted": False, "prompt": ""}
        if not isinstance(spans, list) or not spans:
            self.stats["skip_missing_critical_spans"] += 1
            return {"_accepted": False, "prompt": ""}
        if any(not isinstance(span, Mapping) for span in spans):
            self.stats["skip_invalid_critical_spans"] += 1
            return {"_accepted": False, "prompt": ""}
        self.stats["accepted"] += 1
        return {
            "_accepted": True,
            "id": row_id,
            "source_key": str(row.get("source_id") or row_id),
            "prompt": prompt,
            "stressed": prompt,
            "source_text": source_text,
            "transcript_source": "immutable_local_critical_jsonl",
            "critical_spans": [dict(span) for span in spans],
            "num_words": list(row.get("num_words") or ()),
            "groups": dict(row.get("groups") or {}),
        }


def _accepted(value: bool) -> bool:
    return bool(value)


def _source_dataset(
    data_args: DataArguments,
    *,
    token: str | None,
    seed: int,
    shuffle: bool,
) -> Any:
    if data_args.dataset_format == "local_critical_jsonl":
        from datasets import load_dataset

        if not data_args.dataset_data_file:
            raise ValueError("dataset_data_file is required for local critical JSONL")
        path = Path(data_args.dataset_data_file).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"local critical JSONL does not exist: {path}")
        return load_dataset(
            "json",
            data_files={data_args.dataset_split: str(path.resolve())},
            split=data_args.dataset_split,
            streaming=True,
        )
    if data_args.dataset_format == "qwen_prepared_full_utterance":
        from datasets import load_dataset

        if not data_args.dataset_data_file:
            raise ValueError("dataset_data_file is required for the prepared prompt cache")
        local_data_file = Path(data_args.dataset_data_file).expanduser()
        if local_data_file.is_absolute():
            if not local_data_file.is_file():
                raise FileNotFoundError(
                    f"local prepared Qwen JSONL does not exist: {local_data_file}"
                )
            source = str(local_data_file.resolve())
        else:
            source = (
                f"hf://datasets/{data_args.dataset_name}@{data_args.dataset_revision}/"
                f"{data_args.dataset_data_file.lstrip('/')}"
            )
        return load_dataset(
            "json",
            data_files={data_args.dataset_split: source},
            split=data_args.dataset_split,
            streaming=True,
            token=token,
        )
    if data_args.dataset_format != "webdataset_audio":
        raise ValueError(f"Unsupported dataset_format={data_args.dataset_format!r}")
    source = load_untyped_streaming_dataset(
        data_args.dataset_name,
        split=data_args.dataset_split,
        revision=data_args.dataset_revision,
        token=token,
        required_columns=(data_args.audio_column, "json", "__key__", "__url__"),
        max_retries=data_args.stream_max_retries,
        retry_backoff_sec=data_args.stream_retry_backoff_sec,
        skip_failed_shards=data_args.stream_skip_failed_shards,
    )
    if shuffle and data_args.shuffle_buffer > 1:
        source = source.shuffle(seed=seed, buffer_size=data_args.shuffle_buffer)
    return source


def _map_prompt_source(source: Any, mapper: Any, *, seed: int, shuffle: bool) -> Any:
    source_columns = list(
        source.column_names
        or (mapper.args.audio_column, "json", "__key__", "__url__")
    )
    dataset = source.map(mapper, remove_columns=source_columns)
    dataset = dataset.filter(_accepted, input_columns=["_accepted"])
    dataset = dataset.remove_columns(["_accepted"])
    if (
        shuffle
        and mapper.args.dataset_format in {
            "qwen_prepared_full_utterance",
            "local_critical_jsonl",
        }
        and mapper.args.shuffle_buffer > 1
    ):
        dataset = dataset.shuffle(seed=seed, buffer_size=mapper.args.shuffle_buffer)
    return dataset


def build_prompt_dataset(
    data_args: DataArguments,
    training_args: QwenTTSGRPOConfig,
    token: str | None,
) -> tuple[Any, SileroPromptMapper]:
    """Build a lazy HF stream and preflight the real Silero prompt transform."""

    if not data_args.streaming:
        raise ValueError("Qwen codec GRPO requires streaming=true")
    probe_source = _source_dataset(
        data_args,
        token=token,
        seed=training_args.seed,
        shuffle=True,
    ).take(data_args.stream_preflight_rows)
    mapper_type = {
        "qwen_prepared_full_utterance": PreparedHighAgreementPromptMapper,
        "local_critical_jsonl": LocalCriticalPromptMapper,
    }.get(data_args.dataset_format, SileroPromptMapper)
    probe_mapper = mapper_type(data_args)
    probe = _map_prompt_source(
        probe_source, probe_mapper, seed=training_args.seed, shuffle=False
    )
    if next(iter(probe.take(1)), None) is None:
        raise RuntimeError(
            f"No valid GRPO prompt was found in the first {data_args.stream_preflight_rows} "
            f"source rows; filter stats={dict(probe_mapper.stats)}"
        )
    if training_args.max_steps <= 0:
        raise ValueError("streaming GRPO requires max_steps > 0")
    source = _source_dataset(
        data_args,
        token=token,
        seed=training_args.seed,
        shuffle=True,
    )
    mapper = mapper_type(data_args)
    dataset = _map_prompt_source(
        source, mapper, seed=training_args.seed, shuffle=True
    )
    return dataset, mapper


REFERENCE_DURATION_BANDS: tuple[tuple[str, float, float], ...] = (
    ("3-4", 3.0, 4.0),
    ("5-6", 5.0, 6.0),
    ("8-10", 8.0, 10.0),
    ("12-14", 12.0, 14.0),
    ("16-18", 16.0, 18.0),
    ("20-22", 20.0, 22.0),
)


def _binary_payload(value: Any) -> bytes | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
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


def _reference_band(duration: float) -> str | None:
    for name, lower, upper in REFERENCE_DURATION_BANDS:
        if lower <= duration <= upper:
            return name
    return None


def _reference_text(
    row: Mapping[str, Any],
    mapper: SileroPromptMapper,
) -> tuple[str, str] | None:
    metadata = mapper._metadata(row)
    if mapper.args.require_single_speaker and not bool(metadata.get("is_single_speaker")):
        return None
    if mapper._number(metadata.get("DistillMOS"), 0.0) < mapper.args.min_distill_mos:
        return None
    music_prob = metadata.get("music_prob")
    if mapper._number(music_prob, 1.0) > mapper.args.max_music_prob:
        return None
    consensus_ok, _ = asr_consensus(metadata, max_wer=mapper.args.max_asr_wer)
    if not consensus_ok:
        return None
    source_text, _, _ = select_training_transcript(
        metadata, max_wer=mapper.args.max_e2e_rover_wer
    )
    if not source_text or len(source_text) > mapper.args.max_text_chars:
        return None
    mapper._ensure_accentor()
    try:
        return source_text, apply_silero_stress(mapper._accentor, source_text)
    except Exception:
        return None


def materialize_reference_manifest(
    data_args: DataArguments,
    training_args: QwenTTSGRPOConfig,
    token: str | None,
) -> str:
    """Materialize two deterministic clean references for each duration band."""

    destination = Path(
        data_args.reference_output_dir
        or (Path(training_args.output_dir) / "runtime_references")
    ).expanduser().resolve()
    manifest = destination / "references.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        cached = payload.get("references", ()) if isinstance(payload, Mapping) else ()
        if (
            len(cached) == 12
            and payload.get("dataset_name") == data_args.dataset_name
            and payload.get("dataset_revision") == data_args.dataset_revision
            and payload.get("seed") == training_args.seed
            and all(Path(str(row.get("wav", ""))).is_file() for row in cached)
            and all(str(row.get("ref_text", "")).strip() for row in cached)
        ):
            return str(manifest)

    destination.mkdir(parents=True, exist_ok=True)
    source = _source_dataset(
        data_args,
        token=token,
        seed=training_args.seed + 10_003,
        shuffle=True,
    )
    mapper = SileroPromptMapper(data_args)
    chosen: dict[str, list[dict[str, Any]]] = {
        name: [] for name, _, _ in REFERENCE_DURATION_BANDS
    }
    scanned = 0
    for row in source:
        scanned += 1
        metadata = mapper._metadata(row)
        duration = mapper._number(metadata.get("total_duration"), 0.0)
        band = _reference_band(duration)
        if band is None or len(chosen[band]) >= 2:
            if scanned >= data_args.reference_scan_rows:
                break
            continue
        audio = _binary_payload(row.get(data_args.audio_column))
        texts = _reference_text(row, mapper)
        if audio and texts is not None:
            source_text, stressed_text = texts
            key = str(row.get("__key__") or f"row-{scanned}")
            chosen[band].append(
                {
                    "key": key,
                    "audio": audio,
                    "source_text": source_text,
                    "ref_text": stressed_text,
                    "duration": duration,
                }
            )
        if all(len(values) == 2 for values in chosen.values()):
            break
        if scanned >= data_args.reference_scan_rows:
            break
    missing = {name: 2 - len(values) for name, values in chosen.items() if len(values) != 2}
    if missing:
        raise RuntimeError(
            f"Could not materialize all reference bands after {scanned} rows; missing={missing}"
        )

    rows: list[dict[str, Any]] = []
    for band, _, _ in REFERENCE_DURATION_BANDS:
        for index, item in enumerate(chosen[band]):
            audio_path = destination / f"reference-{band}-{index}.flac"
            temporary_audio = audio_path.with_suffix(".flac.tmp")
            temporary_audio.write_bytes(item.pop("audio"))
            os.replace(temporary_audio, audio_path)
            rows.append(
                {
                    "id": f"{band}-{index}-{item['key']}",
                    "duration_band": band,
                    "wav": str(audio_path),
                    "ref_text": item["ref_text"],
                    "source_text": item["source_text"],
                    "duration": item["duration"],
                }
            )
    temporary_manifest = manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(
            {
                "dataset_name": data_args.dataset_name,
                "dataset_revision": data_args.dataset_revision,
                "seed": training_args.seed,
                "references": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest)
    return str(manifest)


def _parse_arguments() -> tuple[ModelArguments, DataArguments, QwenTTSGRPOConfig]:
    parser = TrlParser((ModelArguments, DataArguments, QwenTTSGRPOConfig))
    return tuple(parser.parse_args_and_config())


def _reporting_to(args: QwenTTSGRPOConfig, integration: str) -> bool:
    reports = args.report_to
    if reports is None:
        return False
    if isinstance(reports, str):
        reports = [reports]
    return integration in reports or "all" in reports


def _validate_configuration(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: QwenTTSGRPOConfig,
) -> None:
    if not training_args.do_train:
        raise ValueError("GRPO entrypoint requires do_train=true")
    if not _reporting_to(training_args, "trackio"):
        raise ValueError("report_to must include trackio")
    if training_args.push_to_hub and not training_args.hub_model_id:
        raise ValueError("push_to_hub=true requires an explicit hub_model_id")
    if not data_args.streaming:
        raise ValueError("streaming must be true")
    if data_args.dataset_format not in {
        "webdataset_audio",
        "qwen_prepared_full_utterance",
        "local_critical_jsonl",
    }:
        raise ValueError(f"Unsupported dataset_format={data_args.dataset_format!r}")
    if (
        data_args.dataset_format in {
            "qwen_prepared_full_utterance",
            "local_critical_jsonl",
        }
        and not data_args.dataset_data_file
    ):
        raise ValueError(
            "dataset_data_file is required for qwen_prepared_full_utterance"
        )
    if not 0.0 <= data_args.min_asr_agreement <= 1.0:
        raise ValueError("min_asr_agreement must be in the 0..1 range")
    if data_args.stream_preflight_rows <= 0:
        raise ValueError("stream_preflight_rows must be positive")
    if data_args.stream_max_retries < 0:
        raise ValueError("stream_max_retries must be non-negative")
    if data_args.stream_retry_backoff_sec < 0:
        raise ValueError("stream_retry_backoff_sec must be non-negative")
    if bool(data_args.references_path) == bool(data_args.auto_references):
        raise ValueError("select exactly one reference mode: references_path or auto_references=true")
    if data_args.reference_scan_rows <= 0:
        raise ValueError("reference_scan_rows must be positive")
    if training_args.world_size == 1 and training_args.n_gpu > 1:
        raise RuntimeError("Multiple GPUs are visible without a distributed launcher; use accelerate launch")
    if training_args.eval_strategy != "no":
        raise ValueError("codec GRPO v1 has no offline eval dataset; eval_strategy must be 'no'")
    if training_args.steps_per_generation > training_args.gradient_accumulation_steps:
        raise ValueError("steps_per_generation must not exceed gradient_accumulation_steps")
    if training_args.gradient_accumulation_steps % training_args.steps_per_generation:
        raise ValueError("gradient_accumulation_steps must be divisible by steps_per_generation")


def _dtype_for(args: QwenTTSGRPOConfig) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def _resolved_model_source(model_args: ModelArguments, token: str | None) -> str:
    local = Path(model_args.model_path).expanduser()
    if local.exists():
        return str(local.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_args.model_path,
        revision=model_args.model_revision,
        token=token,
        cache_dir=model_args.cache_dir,
        local_files_only=model_args.local_files_only,
    )


def _resolved_resume_checkpoint(args: QwenTTSGRPOConfig) -> str | None:
    value = args.resume_from_checkpoint
    if value is None:
        return None
    if isinstance(value, bool):
        if not value:
            return None
        checkpoint = get_last_checkpoint(args.output_dir)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found below {args.output_dir}")
        return checkpoint
    return str(value)


def main() -> None:
    model_args, data_args, training_args = _parse_arguments()
    _validate_configuration(model_args, data_args, training_args)
    training_args.resume_from_checkpoint = _resolved_resume_checkpoint(training_args)
    training_args.accelerator_config.dispatch_batches = False
    training_args.remove_unused_columns = False
    set_seed(training_args.seed)

    token = os.environ.get(model_args.hf_token_env) or dotenv_value(
        model_args.env_file, model_args.hf_token_env
    )
    if (training_args.push_to_hub or training_args.trackio_space_id) and not token:
        raise RuntimeError(
            f"{model_args.hf_token_env} is required for Hub and persistent Trackio output"
        )
    if training_args.push_to_hub and not training_args.hub_token:
        training_args.hub_token = token

    train_dataset, mapper = build_prompt_dataset(data_args, training_args, token)
    if data_args.references_path:
        references_path = str(Path(data_args.references_path).expanduser().resolve())
        if not Path(references_path).is_file():
            raise FileNotFoundError(f"reference manifest does not exist: {references_path}")
    else:
        reference_error = None
        if training_args.process_index == 0:
            try:
                log("materializing 12 rank-shared voice references from the pinned FLAC stream")
                references_path = materialize_reference_manifest(data_args, training_args, token)
            except Exception:
                reference_error = traceback.format_exc()
                references_path = ""
        else:
            references_path = str(
                Path(
                    data_args.reference_output_dir
                    or (Path(training_args.output_dir) / "runtime_references")
                ).expanduser().resolve()
                / "references.json"
            )
        raise_distributed_error(reference_error, "automatic reference materialization")
        if not Path(references_path).is_file():
            raise FileNotFoundError(f"rank-shared reference manifest was not created: {references_path}")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    model_source = _resolved_model_source(model_args, token)
    if training_args.process_index == 0:
        log(
            f"loading {model_args.model_path}@{model_args.model_revision}; "
            f"dataset={data_args.dataset_name}; loss={training_args.loss_type}; "
            f"beta={training_args.beta}; world_size={training_args.world_size}"
        )
    qwen = Qwen3TTSModel.from_pretrained(
        model_source,
        processor_source_path=model_args.processor_source_path,
        torch_dtype=_dtype_for(training_args),
        device_map=None,
        attn_implementation=model_args.attn_implementation,
        token=token,
        local_files_only=model_args.local_files_only,
    )
    qwen.model.config.tts_model_type = "base"
    qwen.model.tts_model_type = "base"
    callbacks: list[TrainerCallback] = [StandaloneCheckpointCallback(qwen)]
    trainer = QwenTTSGRPOTrainer(
        qwen=qwen,
        args=training_args,
        train_dataset=train_dataset,
        references_path=references_path,
        callbacks=callbacks,
    )
    if training_args.process_index == 0:
        log(
            f"full main-Talker trainable={trainer._trainable_parameter_count:,}/"
            f"{trainer._total_parameter_count:,}; preflight={len(list(train_dataset.take(1)))}; "
            f"filters={dict(mapper.stats)}"
        )
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    if model_args.save_final_model:
        error = None
        try:
            trainer.save_model(training_args.output_dir)
            if training_args.should_save:
                save_checkpoint_assets(qwen, training_args.output_dir)
        except Exception:
            error = traceback.format_exc()
        raise_distributed_error(error, "final model save")
    if training_args.push_to_hub:
        trainer.push_to_hub(tags=["trl", "grpo", "qwen3-tts", "full-finetune"])
    if training_args.process_index == 0:
        log(f"finished at optimizer step {trainer.state.global_step}")


if __name__ == "__main__":
    main()
