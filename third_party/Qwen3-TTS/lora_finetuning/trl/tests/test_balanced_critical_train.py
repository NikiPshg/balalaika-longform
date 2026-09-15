from __future__ import annotations

from collections import Counter
import unittest

from lora_finetuning.trl.build_balanced_critical_train import (
    BUCKET_ORDER,
    build_schedule,
    critical_bucket,
)


def _row(name: str, index: int) -> dict:
    spans = (
        [{"type": "fio"}, {"type": "name_patronymic"}]
        if name == "fio"
        else [{"type": name}]
    )
    return {
        "id": f"{name}-{index}",
        "source_id": f"source-{name}-{index}",
        "source_text": "тест",
        "stressed": "т+ест",
        "critical_spans": spans,
    }


class BalancedCriticalTrainTests(unittest.TestCase):
    def test_bucket_priority_maps_initials_to_one_fio_bucket(self) -> None:
        self.assertEqual(
            critical_bucket({"id": "x", "critical_spans": [{"type": "name_patronymic"}]}),
            "fio",
        )
        self.assertEqual(
            critical_bucket(
                {"id": "x", "critical_spans": [{"type": "number"}, {"type": "measurement"}]}
            ),
            "measurement",
        )

    def test_round_robin_is_balanced_deterministic_and_preserves_source(self) -> None:
        rows = [_row(name, index) for name in BUCKET_ORDER for index in range(3)]
        first, report = build_schedule(rows, per_bucket=5, seed=7)
        second, _ = build_schedule(rows, per_bucket=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(
            [row["groups"]["sampling_bucket"] for row in first[: len(BUCKET_ORDER)]],
            list(BUCKET_ORDER),
        )
        counts = Counter(row["groups"]["sampling_bucket"] for row in first)
        self.assertEqual(counts, Counter({name: 5 for name in BUCKET_ORDER}))
        self.assertEqual(report["output_rows"], 35)
        self.assertTrue(all(row["source_id"].startswith("source-") for row in first))
        self.assertEqual(len({row["id"] for row in first}), len(first))


if __name__ == "__main__":
    unittest.main()
