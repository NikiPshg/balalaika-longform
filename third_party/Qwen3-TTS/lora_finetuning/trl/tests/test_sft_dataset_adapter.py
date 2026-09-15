from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml
from datasets import Features, IterableDataset, Value


TRL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRL_ROOT.parents[1]


def _broken_feature_rows():
    yield {
        "flac": {"path": "sample.flac", "bytes": b"fLaC"},
        "json": {"score": ""},
        "__key__": "sample",
        "__url__": "shard.tar",
    }


class _FailOnceAfterFirstRow:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        yield "row-0", {"value": 0}
        if self.calls == 1:
            raise tarfile.ReadError("unexpected end of data")
        yield "row-1", {"value": 1}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sft_helpers = _load_module(
    "qwen_tts_sft_test_target", TRL_ROOT / "trainers" / "qwen_tts_sft.py"
)
grpo_helpers = _load_module(
    "qwen_tts_grpo_entrypoint_test_target", TRL_ROOT / "grpo.py"
)
data_stream = _load_module("data_stream_test_target", TRL_ROOT / "data_stream.py")
sova_streaming = _load_module(
    "sova_streaming_test_target", REPO_ROOT / "lora_finetuning" / "sova_streaming.py"
)


class WavBytesTests(unittest.TestCase):
    def test_accepts_dataset_binary_shapes(self) -> None:
        self.assertEqual(sft_helpers._wav_bytes(b"RIFF"), b"RIFF")
        self.assertEqual(sft_helpers._wav_bytes(bytearray(b"WAVE")), b"WAVE")
        self.assertEqual(
            sft_helpers._wav_bytes({"path": "unused.flac", "bytes": b"fLaC"}),
            b"fLaC",
        )

    def test_rejects_empty_or_unknown_payloads(self) -> None:
        for value in (None, b"", {}, {"bytes": b""}, 42):
            with self.subTest(value=value):
                self.assertIsNone(sft_helpers._wav_bytes(value))


class UntypedStreamTests(unittest.TestCase):
    def test_bypasses_invalid_nested_float_without_losing_shards(self) -> None:
        features = Features(
            {
                "flac": Value("binary"),
                "json": {"score": Value("float64")},
                "__key__": Value("string"),
                "__url__": Value("string"),
            }
        )
        source = IterableDataset.from_generator(_broken_feature_rows, features=features)
        with self.assertRaisesRegex(ValueError, "could not convert string to float"):
            next(iter(source))

        raw = data_stream.without_feature_casting(
            source, ("flac", "json", "__key__", "__url__")
        )
        row = next(iter(raw))
        self.assertEqual(row["json"]["score"], "")
        self.assertEqual(raw.n_shards, source.n_shards)

    def test_shard_retry_resumes_without_replaying_emitted_rows(self) -> None:
        flaky = _FailOnceAfterFirstRow()
        retrying = data_stream.RetryingExamplesGenerator(
            flaky, max_retries=1, backoff_sec=0.0, skip_failed_shards=False
        )
        with self.assertWarnsRegex(RuntimeWarning, "retry 1/1"):
            rows = list(retrying(tar_paths=["shard.tar"]))
        self.assertEqual([key for key, _ in rows], ["row-0", "row-1"])
        self.assertEqual(flaky.calls, 2)


class TrackioWriterTests(unittest.TestCase):
    def test_text_is_buffered_for_audio_validation(self) -> None:
        writer = sft_helpers._TrackioWriter()
        writer.add_text("validation/sample/target", "т+екст", 7)
        self.assertEqual(
            writer.pending[7]["validation/sample/target"],
            "т+екст",
        )

    def test_audio_validation_runs_on_start_and_each_interval(self) -> None:
        callback = sft_helpers.TrackioAudioValidationCallback(
            training_model=object(),
            processor=object(),
            validation_rows=[],
            texts=[],
            pipeline_args=SimpleNamespace(
                audio_validation_steps=1000,
                audio_validation_on_start=True,
            ),
            token=None,
        )
        callback._run = mock.Mock()
        state = SimpleNamespace(global_step=7400, is_world_process_zero=True)
        control = object()

        callback.on_evaluate(object(), state, control, metrics={"eval_loss": 1.0})
        state.global_step = 7500
        callback.on_evaluate(object(), state, control, metrics={"eval_loss": 2.0})
        state.global_step = 8000
        callback.on_evaluate(object(), state, control, metrics={"eval_loss": 3.0})

        self.assertEqual(
            callback._run.call_args_list,
            [
                mock.call(7400, {"eval_loss": 1.0}),
                mock.call(8000, {"eval_loss": 3.0}),
            ],
        )


class YoutubeBalalaikaAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            audio_column="flac",
            min_duration_sec=5.0,
            max_duration_sec=20.0,
            require_single_speaker=True,
            min_distill_mos=3.5,
            max_music_prob=0.30,
            min_words=2,
            max_text_chars=800,
            asr_consistency_threshold=75.0,
        )

    @staticmethod
    def _row(consistency: float = 75.0) -> dict:
        return {
            "__url__": "dataset.tar",
            "__key__": "sample-1",
            "flac": {"bytes": b"encoded-flac", "path": None},
            "json": {
                "total_duration": 6.0,
                "is_single_speaker": True,
                "DistillMOS": 4.0,
                "music_prob": 0.0,
                "asr_consistency": consistency,
                "asr": {
                    "gigaam-v3-e2e-ctc": "текст гигаам",
                },
                "punct": "текст пункт",
                "accent": "metadata accent",
            },
        }

    def _mapper(self) -> sft_helpers.BinarySafeSovaCandidateMapper:
        mapper = sft_helpers.BinarySafeSovaCandidateMapper(self.args)
        mapper._accentor = object()
        return mapper

    def test_at_threshold_selects_gigaam_e2e_and_marks_stressed_text(self) -> None:
        row = self._row(75.0)
        original_json = dict(row["json"])
        mapper = self._mapper()
        with (
            mock.patch.object(
                sft_helpers, "apply_silero_stress", return_value="те+кст гига+ам"
            ),
            mock.patch.object(
                mapper, "_decode_audio", return_value=np.zeros(6 * 24_000, np.float32)
            ),
        ):
            result = mapper(row)

        candidate = result["_candidate"]
        self.assertTrue(result["_accepted"])
        self.assertEqual(candidate["source_text"], "текст гигаам")
        self.assertEqual(candidate["transcript_source"], "gigaam-v3-e2e-ctc.txt")
        self.assertEqual(candidate["text"], "те+кст гига+ам")
        self.assertEqual(row["json"], original_json, "source row must stay immutable")
        self.assertTrue(candidate["_silero_stress_applied"])

    def test_below_threshold_selects_punct_without_other_asr_hypotheses(self) -> None:
        mapper = self._mapper()
        with (
            mock.patch.object(
                sft_helpers, "apply_silero_stress", return_value="те+кст пу+нкт"
            ),
            mock.patch.object(
                mapper, "_decode_audio", return_value=np.zeros(6 * 24_000, np.float32)
            ),
        ):
            result = mapper(self._row(74.99))

        candidate = result["_candidate"]
        self.assertTrue(result["_accepted"])
        self.assertEqual(candidate["source_text"], "текст пункт")
        self.assertEqual(candidate["transcript_source"], "punct.txt")
        self.assertEqual(candidate["asr_consensus_wer"], {"asr_consistency_percent": 74.99})

    def test_preserves_direct_sova_transcript_columns(self) -> None:
        row = self._row(0.0)
        metadata = row["json"]
        metadata.pop("asr")
        metadata.pop("punct")
        metadata.pop("accent")
        metadata.update(
            {
                "gigaam-v3-e2e-ctc.txt": "прямой текст гигаам",
                "punct.txt": "прямой текст пунктуации",
                "accent.txt": "прямой текст ударений",
            }
        )
        row["wav"] = row.pop("flac")
        self.args.audio_column = "wav"
        mapper = self._mapper()
        with (
            mock.patch.object(
                sft_helpers, "apply_silero_stress", return_value="прямо+й те+кст"
            ),
            mock.patch.object(
                mapper, "_decode_audio", return_value=np.zeros(6 * 24_000, np.float32)
            ),
        ):
            result = mapper(row)

        self.assertTrue(result["_accepted"])
        self.assertEqual(result["_candidate"]["source_text"], "прямой текст пунктуации")
        self.assertEqual(result["_candidate"]["metadata_accent_text"], "прямой текст ударений")

    def test_invalid_numeric_metadata_is_normalized_before_selection(self) -> None:
        observed: dict = {}

        def inspect_normalized(normalized):
            observed.update(normalized["json"])
            return None

        mapper = self._mapper()
        with mock.patch.object(
            mapper, "_candidate_from_normalized_row", new=inspect_normalized
        ):
            mapper(
                {
                    "flac": b"encoded-flac",
                    "json": {"total_duration": "", "DistillMOS": "nan", "music_prob": ""},
                }
            )

        self.assertEqual(observed["total_duration"], 0.0)
        self.assertEqual(observed["DistillMOS"], 0.0)
        self.assertEqual(observed["music_prob"], 1.0)

    def test_rejects_accepted_candidate_when_stressed_text_is_empty(self) -> None:
        mapper = self._mapper()
        with (
            mock.patch.object(sft_helpers, "apply_silero_stress", return_value="   "),
            mock.patch.object(
                mapper, "_decode_audio", return_value=np.zeros(6 * 24_000, np.float32)
            ),
        ):
            result = mapper(self._row())

        self.assertEqual(result, {"_accepted": False, "_candidate": {}})
        self.assertEqual(mapper.stats["skip_empty_stressed_text"], 1)


class ProprietaryPreparedAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            prepared_orientation="prefix_reference",
            prepared_profile="balanced",
            prepared_boundary_type="punctuation",
            prepared_boundary_tier="primary",
            prepared_text_source="rover_punctuated_accented",
            min_asr_agreement=0.97,
            min_duration_sec=5.0,
            max_duration_sec=20.0,
            min_prefix_sec=3.0,
            max_prefix_sec=8.0,
            min_continuation_sec=2.0,
            codec_fps=12.0,
            min_words=4,
            max_text_chars=800,
            seed=42,
            language="Russian",
            dataset_name="bitmanagerai/balalaika_proprietary_v2",
            dataset_revision="91a5696ce6125dad5e771f99fec848c060d4f622",
        )
        self.processor = mock.Mock(
            return_value={"input_ids": __import__("torch").tensor([[1, 2, 3]])}
        )

    @staticmethod
    def _row() -> dict:
        return {
            "source_record_id": "000000/example.mp3",
            "agreement_bucket": "agreement_ge_0_95",
            "asr_agreement_mean": 0.99,
            "orientation": "prefix_reference",
            "profile": "balanced",
            "boundary_type": "punctuation",
            "boundary_tier": "primary",
            "text_source": "rover_punctuated_accented",
            "ref_duration": 3.0,
            "duration": 4.0,
            "ref_text": "П+ервая ч+асть фр+азы,",
            "text": "а такж+е втор+ая ч+асть.",
            "ref_audio_codes": [[1] * 16 for _ in range(38)],
            "audio_codes": [[2] * 16 for _ in range(50)],
            "ref_spk_embedding": [0.0] * 2048,
        }

    def test_reconstructs_full_codes_and_reapplies_silero(self) -> None:
        mapper = sft_helpers.PreparedQwenFullUtteranceMapper(
            self.args, self.processor
        )
        mapper._accentor = object()
        with mock.patch.object(
            sft_helpers,
            "apply_silero_stress",
            return_value="П+ервая ч+асть фр+азы, а т+акже втор+ая ч+асть.",
        ) as stress:
            result = mapper(self._row())

        self.assertTrue(result["_accepted"])
        self.assertTrue(result["_silero_stress_applied"])
        self.assertEqual(len(result["full_codes"]), 88)
        self.assertEqual(result["full_codes"][:38], [[1] * 16 for _ in range(38)])
        self.assertEqual(result["full_codes"][38:], [[2] * 16 for _ in range(50)])
        self.assertGreaterEqual(result["prefix_frames"], 36)
        self.assertLessEqual(result["prefix_frames"], 64)
        self.assertEqual(result["text_ids"], [1, 2, 3])
        stress.assert_called_once_with(
            mapper._accentor, "Первая часть фразы, а также вторая часть."
        )
        called_text = self.processor.call_args.kwargs["text"]
        self.assertIn("П+ервая ч+асть", called_text)

    def test_rejects_timestamp_variant_and_subthreshold_agreement(self) -> None:
        self.args.prepared_deduplicate_consecutive = False
        mapper = sft_helpers.PreparedQwenFullUtteranceMapper(
            self.args, self.processor
        )
        row = self._row()
        row["orientation"] = "suffix_reference"
        self.assertFalse(mapper(row)["_accepted"])
        row = self._row()
        row["asr_agreement_mean"] = 0.969
        self.assertFalse(mapper(row)["_accepted"])
        self.assertEqual(mapper.stats["skip_orientation"], 1)
        self.assertEqual(mapper.stats["skip_asr_agreement"], 1)

    def test_suffix_reference_is_restored_in_chronological_order(self) -> None:
        self.args.prepared_orientation = None
        mapper = sft_helpers.PreparedQwenFullUtteranceMapper(
            self.args, self.processor
        )
        mapper._accentor = object()
        row = self._row()
        row["orientation"] = "suffix_reference"
        with mock.patch.object(
            sft_helpers, "apply_silero_stress", return_value="г+отовый т+екст"
        ) as stress:
            result = mapper(row)
        self.assertTrue(result["_accepted"])
        self.assertEqual(result["full_codes"][:50], [[2] * 16 for _ in range(50)])
        self.assertEqual(result["full_codes"][50:], [[1] * 16 for _ in range(38)])
        stress.assert_called_once_with(
            mapper._accentor, "а также вторая часть. Первая часть фразы,"
        )

    def test_excludes_the_fixed_validation_record(self) -> None:
        key = "qwen-high-agreement#000000/example.mp3"
        mapper = sft_helpers.PreparedQwenFullUtteranceMapper(
            self.args, self.processor, {key}
        )
        self.assertFalse(mapper(self._row())["_accepted"])
        self.assertEqual(mapper.stats["skip_validation_key"], 1)


class PreparedDatasetSourceTests(unittest.TestCase):
    def test_sft_uses_an_explicit_local_prepared_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                streaming=True,
                dataset_format="qwen_prepared_full_utterance",
                dataset_data_file=str(path),
                dataset_name="bitmanagerai/balalaika_proprietary_v2",
                dataset_revision="91a5696ce6125dad5e771f99fec848c060d4f622",
                dataset_split="train",
            )
            sentinel = object()
            with mock.patch("datasets.load_dataset", return_value=sentinel) as load:
                result = sft_helpers._streaming_dataset(args, token="secret")

        self.assertIs(result, sentinel)
        self.assertEqual(
            load.call_args.kwargs["data_files"], {"train": str(path.resolve())}
        )
        self.assertTrue(load.call_args.kwargs["streaming"])

    def test_grpo_uses_an_explicit_local_prepared_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                dataset_format="qwen_prepared_full_utterance",
                dataset_data_file=str(path),
                dataset_name="bitmanagerai/balalaika_proprietary_v2",
                dataset_revision="91a5696ce6125dad5e771f99fec848c060d4f622",
                dataset_split="train",
            )
            sentinel = object()
            with mock.patch("datasets.load_dataset", return_value=sentinel) as load:
                result = grpo_helpers._source_dataset(
                    args, token="secret", seed=42, shuffle=True
                )

        self.assertIs(result, sentinel)
        self.assertEqual(
            load.call_args.kwargs["data_files"], {"train": str(path.resolve())}
        )

    def test_missing_absolute_prepared_jsonl_fails_loudly(self) -> None:
        missing = Path("/tmp/qwen-tts-missing-prepared-dataset.jsonl")
        missing.unlink(missing_ok=True)
        args = SimpleNamespace(
            streaming=True,
            dataset_format="qwen_prepared_full_utterance",
            dataset_data_file=str(missing),
            dataset_name="bitmanagerai/balalaika_proprietary_v2",
            dataset_revision="91a5696ce6125dad5e771f99fec848c060d4f622",
            dataset_split="train",
        )
        with self.assertRaisesRegex(FileNotFoundError, "local prepared Qwen JSONL"):
            sft_helpers._streaming_dataset(args, token=None)


class ProprietaryGrpoPromptAdapterTests(unittest.TestCase):
    def test_reconstructs_suffix_text_and_deduplicates_source_group(self) -> None:
        args = grpo_helpers.DataArguments(
            dataset_name="bitmanagerai/balalaika_proprietary_v2",
            dataset_revision="91a5696ce6125dad5e771f99fec848c060d4f622",
            dataset_format="qwen_prepared_full_utterance",
            dataset_data_file="tokenizations/qwen/agreement_ge_0_95.train.jsonl",
            min_asr_agreement=0.97,
            min_duration_sec=5.0,
            max_duration_sec=20.0,
            min_words=4,
        )
        mapper = grpo_helpers.PreparedHighAgreementPromptMapper(args)
        mapper._accentor = object()
        row = {
            "source_record_id": "000000/example.mp3",
            "agreement_bucket": "agreement_ge_0_95",
            "asr_agreement_mean": 0.99,
            "text_source": "rover_punctuated_accented",
            "orientation": "suffix_reference",
            "duration": 4.0,
            "ref_duration": 3.0,
            "text": "П+ервая ч+асть предлож+ения,",
            "ref_text": "а зд+есь ег+о кон+ец.",
        }
        with mock.patch.object(
            grpo_helpers,
            "apply_silero_stress",
            return_value="П+ервая ч+асть предлож+ения, а зд+есь ег+о кон+ец.",
        ) as stress:
            accepted = mapper(row)
            duplicate = mapper(row)
        self.assertTrue(accepted["_accepted"])
        self.assertFalse(duplicate["_accepted"])
        self.assertEqual(
            accepted["source_text"], "Первая часть предложения, а здесь его конец."
        )
        stress.assert_called_once_with(
            mapper._accentor, "Первая часть предложения, а здесь его конец."
        )
        self.assertEqual(mapper.stats["skip_duplicate_source_record_id"], 1)


class SileroStressContractTests(unittest.TestCase):
    def test_candidate_text_is_the_silero_stress_result(self) -> None:
        accentor = object()
        apply_calls: list[tuple[object, str]] = []

        def fake_stress(actual_accentor, text):
            apply_calls.append((actual_accentor, text))
            return "балала+йка игра+ет"

        args = SimpleNamespace(
            min_duration_sec=1.0,
            max_duration_sec=20.0,
            require_single_speaker=True,
            min_distill_mos=3.5,
            max_music_prob=0.3,
            max_asr_wer=0.25,
            max_e2e_rover_wer=0.0,
            min_words=2,
            max_text_chars=100,
        )
        row = {
            "__url__": "dataset.tar",
            "__key__": "sample-1",
            "wav": b"encoded-flac",
            "json": {
                "total_duration": 5.0,
                "is_single_speaker": True,
                "DistillMOS": 4.0,
                "music_prob": 0.0,
            },
        }
        stats: Counter[str] = Counter()

        with (
            mock.patch.object(
                sova_streaming, "asr_consensus", return_value=(True, {"wer": 0.0})
            ),
            mock.patch.object(
                sova_streaming,
                "select_training_transcript",
                return_value=("балалайка играет", "punct.txt", 0.0),
            ),
            mock.patch.object(sova_streaming, "apply_silero_stress", new=fake_stress),
            mock.patch.object(
                sova_streaming,
                "decode_wav_bytes",
                return_value=([0.0, 0.1], 5.0),
            ),
        ):
            candidate = sova_streaming.candidate_from_row(
                row, args, accentor, object(), object(), stats
            )

        self.assertIsNotNone(candidate)
        self.assertEqual(apply_calls, [(accentor, "балалайка играет")])
        self.assertEqual(candidate["source_text"], "балалайка играет")
        self.assertEqual(candidate["text"], "балала+йка игра+ет")


class TargetDatasetConfigTests(unittest.TestCase):
    def test_full_sft_config_pins_requested_dataset_contract(self) -> None:
        config = yaml.safe_load(
            (TRL_ROOT / "configs" / "sft_full.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["dataset_name"], "lab260/youtube_balalaika")
        self.assertEqual(config["model_path"], "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        self.assertEqual(
            config["model_revision"],
            "fd4b254389122332181a7c3db7f27e918eec64e3",
        )
        self.assertEqual(config["audio_column"], "flac")
        self.assertEqual(config["asr_consistency_threshold"], 75.0)
        self.assertEqual(
            config["dataset_revision"],
            "f847a349ac9cbf2952726c12b5aab1fd70b8a4cf",
        )
        self.assertIs(config["streaming"], True)
        self.assertEqual(config["report_to"], ["trackio"])
        self.assertEqual(config["stream_max_retries"], 3)
        self.assertIs(config["stream_skip_failed_shards"], True)
        self.assertIs(config["generate_audio"], True)
        self.assertEqual(config["audio_validation_steps"], 1000)
        self.assertEqual(config["validation_references"], 1)
        self.assertEqual(config["validation_texts"], 10)
        self.assertEqual(config["max_new_tokens"], 1024)
        validation_texts = [
            line
            for line in (TRL_ROOT / "configs" / "hard_numeric_validation.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(validation_texts), 10)

    def test_proprietary_continuation_config_is_pinned_and_fresh_stage(self) -> None:
        config = yaml.safe_load(
            (TRL_ROOT / "configs" / "sft_proprietary_v2_best.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            config["dataset_name"], "bitmanagerai/balalaika_proprietary_v2"
        )
        self.assertEqual(
            config["dataset_revision"],
            "91a5696ce6125dad5e771f99fec848c060d4f622",
        )
        self.assertEqual(config["dataset_format"], "qwen_prepared_full_utterance")
        self.assertEqual(config["min_asr_agreement"], 0.97)
        self.assertIsNone(config["prepared_orientation"])
        self.assertIsNone(config["prepared_profile"])
        self.assertIsNone(config["prepared_boundary_tier"])
        self.assertIs(config["prepared_deduplicate_consecutive"], True)
        self.assertIn("checkpoint-10000", config["model_path"])
        self.assertIsNone(config["resume_from_checkpoint"])
        self.assertEqual(config["learning_rate"], 5.0e-6)
        self.assertEqual(config["audio_validation_steps"], 1000)

    def test_two_stage_proprietary_continuation_is_10k_plus_10k(self) -> None:
        stage1 = yaml.safe_load(
            (
                TRL_ROOT
                / "configs"
                / "sft_proprietary_v2_stage1_main_from_youtube20k_10k.yaml"
            ).read_text(encoding="utf-8")
        )
        stage2 = yaml.safe_load(
            (
                TRL_ROOT
                / "configs"
                / "sft_proprietary_v2_stage2_all_from_stage1_10k.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(stage1["max_steps"] + stage2["max_steps"], 20_000)
        self.assertEqual(stage1["training_scope"], "main_talker")
        self.assertEqual(stage2["training_scope"], "all_talker")
        self.assertIn("youtube-all-talker", stage1["model_path"])
        self.assertEqual(
            stage2["model_path"],
            "lora_finetuning/trl/outputs/"
            "sft-proprietary-v2-stage1-main-from-youtube20k-10k/"
            "checkpoint-10000",
        )
        self.assertEqual(stage1["dataset_revision"], stage2["dataset_revision"])
        self.assertEqual(stage1["dataset_data_file"], stage2["dataset_data_file"])
        self.assertTrue(Path(stage1["dataset_data_file"]).is_absolute())
        self.assertEqual(stage1["save_steps"], 5000)
        self.assertEqual(stage2["save_steps"], 5000)
        self.assertEqual(stage1["audio_validation_steps"], 1000)
        self.assertEqual(stage2["audio_validation_steps"], 1000)

    def test_proprietary_grpo_config_consumes_completed_sft_and_fixed_voices(self) -> None:
        config = yaml.safe_load(
            (TRL_ROOT / "configs" / "grpo_proprietary_v2.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            config["model_path"],
            "lora_finetuning/trl/outputs/sft-proprietary-v2-best",
        )
        self.assertEqual(config["dataset_format"], "qwen_prepared_full_utterance")
        self.assertEqual(config["min_asr_agreement"], 0.97)
        self.assertEqual(config["loss_type"], "dr_grpo")
        self.assertEqual(config["beta"], 0.03)
        self.assertEqual(config["reference_mode"], "initial_policy")
        self.assertFalse(config["auto_references"])
        manifest = yaml.safe_load(
            (TRL_ROOT / "assets" / "tts-voices" / "references.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["revision"], "f57528db42342b79201cc0d360355e2532b088b7")
        self.assertEqual({row["id"] for row in manifest["references"]}, {"valera-1", "anastasia-1"})
        self.assertTrue(all("+" not in row["ref_text"] for row in manifest["references"]))


if __name__ == "__main__":
    unittest.main()
