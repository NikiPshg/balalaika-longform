from __future__ import annotations

import unittest

from lora_finetuning.trl.evaluate_critical import (
    compare_aggregates,
    rerank_candidate_rows,
    score_rows,
    subset_manifest,
)


class CriticalEvaluationTests(unittest.TestCase):
    def test_subset_manifest_is_immutable_and_records_parent(self) -> None:
        parent = {
            "schema_version": 1,
            "manifest_hash": "parent",
            "row_count": 2,
            "rows": [{"input_id": "a"}, {"input_id": "b"}],
        }
        subset = subset_manifest(parent, 1, start=1)
        self.assertEqual(subset["row_count"], 1)
        self.assertEqual(subset["parent_manifest_hash"], "parent")
        self.assertEqual(subset["rows"][0]["input_id"], "b")
        self.assertNotEqual(subset["manifest_hash"], "parent")

    def test_reranker_prefers_critical_accuracy_without_slow_motion(self) -> None:
        base = {
            "input_id": "x",
            "status": "ok",
            "normalized_gold": "сумма сто рублей",
            "critical_spans": [
                {
                    "type": "money",
                    "spoken_gold": ["сто", "рублей"],
                    "token_start": 1,
                    "token_end": 3,
                }
            ],
            "generated_duration": 1.2,
        }
        rows = rerank_candidate_rows(
            [
                [{**base, "hypothesis": "сумма сто рублей", "sampling_seed_offset": 1}],
                [{**base, "hypothesis": "сумма сорок рублей", "sampling_seed_offset": 2}],
            ],
            model_label="bon2",
            model_revision="rerank:test",
        )
        self.assertEqual(rows[0]["sampling_seed_offset"], 1)
        self.assertEqual(rows[0]["rerank_candidate_count"], 2)

    def test_score_rows_reports_typed_metrics(self) -> None:
        rows = [
            {
                "model": "baseline",
                "model_revision": "hash",
                "manifest_hash": "manifest",
                "normalized_gold": "Код ноль ноль принят",
                "hypothesis": "Код ноль один принят",
                "critical_spans": [
                    {
                        "type": "identifier",
                        "spoken_gold": ["ноль", "ноль"],
                        "token_start": 1,
                        "token_end": 3,
                    }
                ],
                "status": "ok",
                "quality_status": "ok",
                "mos": 4.0,
                "sim": 0.8,
                "duration_ratio": 1.0,
                "repetition_score": 0.0,
            }
        ]
        scored, aggregate = score_rows(rows)
        self.assertEqual(aggregate["row_count"], 1)
        self.assertGreater(aggregate["micro_critical_wer"], 0.0)
        self.assertEqual(scored[0]["typed_metrics"]["spans"][0]["span_type"], "identifier")

    def test_compare_requires_improvement_and_gates(self) -> None:
        base = {
            "objective": 0.5,
            "mos": 4.0,
            "sim": 0.8,
            "duration_ratio_deviation": 0.1,
            "repetition_score": 0.1,
            "failure_rate": 0.0,
            "quality_coverage": 1.0,
            "per_type": [{"type": "fio", "objective": 0.5}],
        }
        candidate = {
            **base,
            "model": "candidate",
            "objective": 0.4,
            "per_type": [{"type": "fio", "objective": 0.49}],
        }
        result = compare_aggregates(base, candidate)
        self.assertTrue(result["quality_guard_passed"])
        self.assertEqual(result["selected_model"], "candidate")


if __name__ == "__main__":
    unittest.main()
