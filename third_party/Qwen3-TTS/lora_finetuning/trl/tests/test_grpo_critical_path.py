from __future__ import annotations

import unittest

import torch
from transformers import PretrainedConfig, PreTrainedModel

from grpo_dpo_finetuning.metrics import score_text_pair
from lora_finetuning.trl.critical_span_metrics import score_typed_text_pair
from lora_finetuning.trl.grpo import DataArguments, LocalCriticalPromptMapper
from lora_finetuning.trl.trainers.qwen_tts_grpo import (
    QwenTTSGRPOTrainer,
    build_joint_optimizer_groups,
    clone_frozen_reference,
    code_predictor_policy_loss,
    codec_composite_accuracy_reward,
    codec_duration_reward,
    compute_gated_pairwise_advantages,
    compute_gated_best_advantages,
    configure_full_main_talker,
    critical_type_gradient_scale,
)


class _DummyTalker(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.code_predictor = torch.nn.Linear(2, 2)


class _DummyTTS(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self) -> None:
        super().__init__(PretrainedConfig())
        self.talker = _DummyTalker()
        self.speech_tokenizer = object()
        self.supported_speakers = {"speaker": 1}.keys()


class GRPOCriticalPathTests(unittest.TestCase):
    def test_joint_optimizer_groups_apply_distinct_predictor_lr(self) -> None:
        policy = _DummyTTS()
        configure_full_main_talker(policy, train_code_predictor=True)
        decay_names = {name for name, _ in policy.named_parameters() if name.endswith("weight")}
        groups = build_joint_optimizer_groups(
            policy,
            decay_names,
            base_learning_rate=2e-8,
            weight_decay=0.01,
            code_predictor_lr_multiplier=5.0,
        )
        seen = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(seen), len({id(parameter) for parameter in seen}))
        self.assertEqual(
            sum(parameter.numel() for parameter in seen),
            sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad),
        )
        predictor_lrs = {group["lr"] for group in groups if group["is_code_predictor"]}
        main_lrs = {group["lr"] for group in groups if not group["is_code_predictor"]}
        self.assertEqual(predictor_lrs, {1e-7})
        self.assertEqual(main_lrs, {2e-8})

    def test_joint_scope_trains_main_and_code_predictor(self) -> None:
        policy = _DummyTTS()
        trainable, total = configure_full_main_talker(
            policy, train_code_predictor=True
        )
        self.assertEqual(trainable, total)
        self.assertTrue(policy.talker.projection.weight.requires_grad)
        self.assertTrue(policy.talker.code_predictor.weight.requires_grad)

    def test_predictor_only_scope_freezes_main_talker(self) -> None:
        policy = _DummyTTS()
        trainable, total = configure_full_main_talker(
            policy, train_main_talker=False, train_code_predictor=True
        )
        self.assertLess(trainable, total)
        self.assertFalse(policy.talker.projection.weight.requires_grad)
        self.assertTrue(policy.talker.code_predictor.weight.requires_grad)

    def test_code_predictor_policy_loss_uses_positive_advantages_only(self) -> None:
        nll = torch.tensor([2.0, 4.0, 8.0], requires_grad=True)
        loss = code_predictor_policy_loss(nll, torch.tensor([-0.5, 0.25, 0.0]))
        self.assertAlmostEqual(loss.item(), 1.0 / 3.0)
        loss.backward()
        self.assertTrue(torch.equal(nll.grad, torch.tensor([0.0, 1.0 / 12.0, 0.0])))
        with self.assertRaisesRegex(ValueError, "finite advantages"):
            code_predictor_policy_loss(
                nll.detach(), torch.tensor([0.0, float("nan"), 0.0])
            )

    def test_qwen_generation_ids_have_one_item_batch_dimension(self) -> None:
        for value in (torch.tensor([1, 2, 3]), torch.tensor([[1, 2, 3]])):
            normalized = QwenTTSGRPOTrainer._one_item_batch_ids(value, "ids")
            self.assertEqual(tuple(normalized.shape), (1, 3))
            self.assertEqual(normalized.dtype, torch.long)

    def test_frozen_reference_materializes_unpickleable_speaker_keys(self) -> None:
        policy = _DummyTTS()
        original_keys = policy.supported_speakers
        reference = clone_frozen_reference(policy)

        self.assertIs(policy.supported_speakers, original_keys)
        self.assertIsNotNone(policy.speech_tokenizer)
        self.assertEqual(reference.tts_model.supported_speakers, ("speaker",))
        self.assertIsNone(reference.tts_model.speech_tokenizer)
        self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))
        self.assertIsNot(
            policy.talker.projection.weight,
            reference.tts_model.talker.projection.weight,
        )

    def test_local_mapper_preserves_typed_contract(self) -> None:
        mapper = LocalCriticalPromptMapper(
            DataArguments(dataset_format="local_critical_jsonl", dataset_data_file="x")
        )
        mapped = mapper(
            {
                "id": "row-1",
                "source_id": "source-1",
                "source_text": "Код ноль ноль принят",
                "stressed": "К+од н+оль н+оль пр+инят",
                "critical_spans": [
                    {
                        "type": "identifier",
                        "spoken_gold": ["ноль", "ноль"],
                        "token_start": 1,
                        "token_end": 3,
                    }
                ],
                "num_words": ["ноль", "ноль"],
                "groups": {"category": "identifier"},
            }
        )
        self.assertTrue(mapped["_accepted"])
        self.assertEqual(mapped["critical_spans"][0]["type"], "identifier")
        self.assertEqual(mapped["num_words"], ["ноль", "ноль"])

    def test_composite_reward_penalizes_critical_error(self) -> None:
        reference = "Позвонил Иван Иванович Петров сегодня"
        hypothesis = "Позвонил Иван Петрович Петров сегодня"
        metrics = score_text_pair(reference, hypothesis, [])
        typed = score_typed_text_pair(
            reference,
            hypothesis,
            [
                {
                    "type": "fio",
                    "spoken_gold": "Иван Иванович Петров",
                    "token_start": 1,
                    "token_end": 4,
                }
            ],
        )
        general_only = codec_composite_accuracy_reward(
            metrics,
            typed,
            number_wer_weight=0.0,
            number_cer_weight=0.0,
            utterance_wer_weight=0.5,
            utterance_cer_weight=0.5,
            temperature=3.0,
            critical_component_weight=0.0,
            utterance_component_weight=1.0,
            critical_wer_weight=0.35,
            critical_cer_weight=0.45,
            critical_exact_weight=0.20,
            critical_worst_span_weight=0.30,
        )
        typed_dense = codec_composite_accuracy_reward(
            metrics,
            typed,
            number_wer_weight=0.0,
            number_cer_weight=0.0,
            utterance_wer_weight=0.5,
            utterance_cer_weight=0.5,
            temperature=3.0,
            critical_component_weight=0.5,
            utterance_component_weight=0.5,
            critical_wer_weight=0.35,
            critical_cer_weight=0.45,
            critical_exact_weight=0.20,
            critical_worst_span_weight=0.30,
        )
        self.assertLess(typed_dense, general_only)

    def test_duration_reward_penalizes_slow_motion(self) -> None:
        on_pace, ratio, words_per_sec = codec_duration_reward(4.0, 10)
        slow, slow_ratio, slow_words_per_sec = codec_duration_reward(8.0, 10)
        too_fast, _, _ = codec_duration_reward(1.0, 10)
        self.assertEqual(on_pace, 1.0)
        self.assertEqual(ratio, 1.0)
        self.assertEqual(words_per_sec, 2.5)
        self.assertLess(slow, on_pace)
        self.assertGreater(slow_ratio, 1.30)
        self.assertLess(slow_words_per_sec, words_per_sec)
        self.assertLess(too_fast, on_pace)

    def test_type_scale_uses_strongest_typed_span_without_resampling(self) -> None:
        scales = {"fio": 0.4, "measurement": 1.5, "phone": 2.25}
        self.assertEqual(
            critical_type_gradient_scale(
                [{"type": "fio"}, {"type": "measurement"}], scales
            ),
            1.5,
        )
        self.assertEqual(critical_type_gradient_scale([{"type": "phone"}], scales), 2.25)
        self.assertEqual(
            critical_type_gradient_scale([{"type": "unknown"}], scales, default_scale=0.75),
            0.75,
        )

    def test_type_scale_rejects_unbounded_gradient_multiplier(self) -> None:
        with self.assertRaisesRegex(ValueError, "in \\[0, 8\\]"):
            critical_type_gradient_scale([{"type": "phone"}], {"phone": 9.0})

    def test_gated_pairwise_selects_only_safe_best_and_worst(self) -> None:
        advantages, active = compute_gated_pairwise_advantages(
            torch.tensor([0.2, 0.9, 0.7, 0.1]),
            torch.tensor([0.8, 0.3, 0.8, 0.7]),
            torch.tensor([0.9, 0.9, 0.9, 0.9]),
            torch.ones(4),
            torch.full((4,), 1.5),
            4,
        )
        self.assertTrue(torch.equal(advantages, torch.tensor([0.0, 0.0, 1.5, -1.5])))
        self.assertTrue(torch.equal(active, torch.tensor([1.0])))

    def test_gated_pairwise_skips_groups_without_margin_or_safe_positive(self) -> None:
        advantages, active = compute_gated_pairwise_advantages(
            torch.tensor([0.50, 0.52, 0.51, 0.50, 0.1, 0.9, 0.4, 0.2]),
            torch.tensor([0.8, 0.8, 0.8, 0.8, 0.1, 0.2, 0.3, 0.4]),
            torch.ones(8),
            torch.ones(8),
            torch.ones(8),
            4,
            critical_margin=0.05,
            minimum_utterance_reward=0.45,
        )
        self.assertTrue(torch.equal(advantages, torch.zeros(8)))
        self.assertTrue(torch.equal(active, torch.zeros(2)))

    def test_gated_pairwise_requires_strong_positive_and_weak_negative(self) -> None:
        advantages, active = compute_gated_pairwise_advantages(
            torch.tensor([0.91, 0.96, 0.81, 0.74, 0.99, 0.97, 0.93, 0.92]),
            torch.full((8,), 0.85),
            torch.full((8,), 0.95),
            torch.ones(8),
            torch.ones(8),
            4,
            critical_margin=0.15,
            minimum_critical_reward=0.95,
            maximum_negative_critical_reward=0.80,
        )
        self.assertTrue(
            torch.equal(advantages, torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]))
        )
        self.assertTrue(torch.equal(active, torch.tensor([1.0, 0.0])))

    def test_gated_best_only_promotes_verified_positive(self) -> None:
        advantages, active = compute_gated_best_advantages(
            torch.tensor([0.96, 0.99, 0.80, 0.20]),
            torch.tensor([0.90, 0.80, 0.99, 0.99]),
            torch.tensor([0.95, 0.92, 1.00, 1.00]),
            torch.ones(4),
            torch.full((4,), 0.25),
            4,
        )
        self.assertTrue(torch.equal(advantages, torch.tensor([0.0, 0.25, 0.0, 0.0])))
        self.assertTrue(torch.equal(active, torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
