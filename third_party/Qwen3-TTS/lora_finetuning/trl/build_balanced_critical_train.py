#!/usr/bin/env python3
"""Build a deterministic type-balanced schedule from the pinned critical train JSONL."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping


BUCKET_ORDER = (
    "fio",
    "phone",
    "measurement",
    "money",
    "identifier",
    "percent",
    "number",
)


def critical_bucket(row: Mapping[str, Any]) -> str:
    types = {
        str(span.get("type") or "")
        for span in row.get("critical_spans", ())
        if isinstance(span, Mapping)
    }
    if types & {"fio", "name_patronymic", "person_name"}:
        return "fio"
    for name in BUCKET_ORDER[1:]:
        if name in types:
            return name
    raise ValueError(f"row {row.get('id')!r} has no supported critical type: {sorted(types)}")


def build_schedule(
    rows: list[dict[str, Any]], *, per_bucket: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(per_bucket, bool) or per_bucket <= 0:
        raise ValueError("per_bucket must be a positive integer")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in seen_ids:
            raise ValueError(f"input row ids must be non-empty and unique: {row_id!r}")
        seen_ids.add(row_id)
        buckets[critical_bucket(row)].append(row)
    missing = [name for name in BUCKET_ORDER if not buckets[name]]
    if missing:
        raise ValueError(f"balanced schedule is missing buckets: {missing}")

    selected: dict[str, list[dict[str, Any]]] = {}
    for offset, name in enumerate(BUCKET_ORDER):
        source = list(buckets[name])
        random.Random(seed + offset * 1009).shuffle(source)
        values: list[dict[str, Any]] = []
        while len(values) < per_bucket:
            values.extend(source)
        selected[name] = values[:per_bucket]

    schedule: list[dict[str, Any]] = []
    repeat_counts: dict[str, int] = defaultdict(int)
    for index in range(per_bucket):
        for name in BUCKET_ORDER:
            source = selected[name][index]
            original_id = str(source["id"])
            instance = repeat_counts[original_id]
            repeat_counts[original_id] += 1
            row = dict(source)
            row["source_id"] = str(source.get("source_id") or original_id)
            row["id"] = f"balanced-v1:{name}:{index:04d}:{instance}:{original_id}"
            groups = dict(row.get("groups") or {})
            groups["sampling_bucket"] = name
            groups["sampling_schedule"] = "balanced_round_robin_v1"
            row["groups"] = groups
            schedule.append(row)
    report = {
        "schema_version": 1,
        "seed": seed,
        "per_bucket": per_bucket,
        "bucket_order": list(BUCKET_ORDER),
        "input_rows": len(rows),
        "input_bucket_counts": {name: len(buckets[name]) for name in BUCKET_ORDER},
        "output_rows": len(schedule),
        "output_bucket_counts": {name: per_bucket for name in BUCKET_ORDER},
        "unique_source_ids_used": len({row["source_id"] for row in schedule}),
    }
    return schedule, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: row must be an object")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            encoded = json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ) + "\n"
            output.write(encoded)
            digest.update(encoded.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    schedule, report = build_schedule(
        _read_jsonl(args.input), per_bucket=args.per_bucket, seed=args.seed
    )
    report["output_sha256"] = _write_jsonl(args.output, schedule)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
