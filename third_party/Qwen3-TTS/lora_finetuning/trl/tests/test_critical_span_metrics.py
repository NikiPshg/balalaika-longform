from __future__ import annotations

import unittest

from lora_finetuning.trl.critical_span_metrics import (
    aggregate_typed_metrics,
    infer_changed_token_span,
    score_typed_text_pair,
    typed_critical_reward,
)


def _fio(**overrides):
    value = {
        "type": "fio",
        "spoken_gold": ["Иван", "Иванович", "Петров"],
        "token_start": 1,
        "token_end": 4,
    }
    value.update(overrides)
    return value


class TypedCriticalSpanMetricTests(unittest.TestCase):
    def test_exact_typed_span_and_stress_normalization(self) -> None:
        result = score_typed_text_pair(
            "Позвон+ил Ив+ан Ив+анович Петр+ов сегодня",
            "позвонил Иван Иванович Петров сегодня",
            [_fio()],
        )
        self.assertTrue(result.spans[0].exact)
        self.assertEqual(result.critical_wer, 0.0)
        self.assertAlmostEqual(typed_critical_reward(result), 1.0)

    def test_span_edit_types(self) -> None:
        cases = (
            ("позвонил Иван Петрович Петров сегодня", "substitutions"),
            ("позвонил Иван Петров сегодня", "deletions"),
            ("позвонил Иван очень Иванович Петров сегодня", "insertions"),
        )
        for hypothesis, field in cases:
            with self.subTest(field=field):
                result = score_typed_text_pair(
                    "позвонил Иван Иванович Петров сегодня", hypothesis, [_fio()]
                )
                self.assertGreaterEqual(getattr(result.spans[0].word_counts, field), 1)
                self.assertFalse(result.spans[0].exact)
                self.assertGreaterEqual(typed_critical_reward(result), 0.0)
                self.assertLess(typed_critical_reward(result), 1.0)

    def test_missing_span_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent"):
            score_typed_text_pair(
                "позвонил Иван сегодня",
                "позвонил Иван сегодня",
                [{"type": "person_name", "spoken_gold": "Пётр"}],
            )

    def test_repeated_span_requires_offsets(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            score_typed_text_pair(
                "код ноль ноль и снова ноль ноль",
                "код ноль ноль и снова ноль ноль",
                [{"type": "identifier", "spoken_gold": "ноль ноль"}],
            )
        result = score_typed_text_pair(
            "код ноль ноль и снова ноль ноль",
            "код ноль ноль и снова один ноль",
            [
                {
                    "type": "identifier",
                    "spoken_gold": "ноль ноль",
                    "token_start": 5,
                    "token_end": 7,
                }
            ],
        )
        self.assertEqual(result.spans[0].word_counts.substitutions, 1)

    def test_multiple_types_and_micro_aggregation(self) -> None:
        spans = [
            {"type": "name_patronymic", "spoken_gold": "Анна Петровна"},
            {"type": "money", "spoken_gold": "десять рублей"},
        ]
        exact = score_typed_text_pair(
            "Анна Петровна перевела десять рублей",
            "Анна Петровна перевела десять рублей",
            spans,
        )
        error = score_typed_text_pair(
            "Анна Петровна перевела десять рублей",
            "Анна Петровна перевела девять рублей",
            spans,
        )
        aggregate = aggregate_typed_metrics([exact, error])
        self.assertEqual(aggregate["row_count"], 2)
        self.assertEqual(
            aggregate["per_type"]["name_patronymic"]["exact_rate"], 1.0
        )
        self.assertEqual(aggregate["per_type"]["money"]["exact_rate"], 0.5)
        self.assertEqual(aggregate["critical"]["span_count"], 4)

    def test_infer_written_numeric_replacement(self) -> None:
        start, end, words = infer_changed_token_span(
            "Код 000-000 уже подтверждён",
            "Код ноль ноль ноль ноль ноль ноль уже подтверждён",
        )
        self.assertEqual((start, end), (1, 7))
        self.assertEqual(words, ("ноль",) * 6)


if __name__ == "__main__":
    unittest.main()
