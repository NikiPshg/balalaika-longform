from __future__ import annotations

import json
import unittest

from lora_finetuning.trl.build_critical_curriculum import initials_record, numeric_record


class CriticalCurriculumTests(unittest.TestCase):
    def test_initials_row_has_typed_offsets(self) -> None:
        row = initials_record(
            {
                "id": "x",
                "text": "Сегодня Иван Петрович ответил",
                "text_stressed": "Сег+одня Ив+ан Петр+ович отв+етил",
                "source_text": "Иван Петрович",
                "task": "name_patronymic",
                "subset": "popular",
                "gender": "m",
                "placement_category": "середина",
                "context_category": "проверка",
            }
        )
        span = row["critical_spans"][0]
        self.assertEqual((span["token_start"], span["token_end"]), (1, 3))
        self.assertEqual(span["type"], "name_patronymic")

    def test_numeric_row_preserves_spoken_gold(self) -> None:
        row = numeric_record(
            {
                "id": "n",
                "text_raw": "Код 000-000 уже принят",
                "text_normalized": "Код ноль ноль ноль ноль ноль ноль уже принят",
                "category": "short_identifier",
                "number_spans_json": json.dumps(
                    [
                        {
                            "surface": "000-000",
                            "normalized_value": "000000",
                            "format": "short_identifier",
                            "pronunciation_mode": "digits",
                        }
                    ]
                ),
            }
        )
        span = row["critical_spans"][0]
        self.assertEqual(span["type"], "identifier")
        self.assertEqual(span["spoken_gold"], ["ноль"] * 6)
        self.assertEqual(row["num_words"], ["ноль"] * 6)


if __name__ == "__main__":
    unittest.main()
