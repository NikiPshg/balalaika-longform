#!/usr/bin/env python3
"""Append and inspect durable autoresearch experiment events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any


def parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("journal", type=Path)
    record.add_argument("--experiment-id", required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--hypothesis", required=True)
    record.add_argument("--rung", required=True)
    record.add_argument("--seed", type=int)
    record.add_argument("--config")
    record.add_argument("--config-hash")
    record.add_argument("--checkpoint")
    record.add_argument("--gpu")
    record.add_argument("--pid")
    record.add_argument("--metrics")
    record.add_argument("--reason", required=True)
    record.add_argument("--field", action="append", default=[])
    show = sub.add_parser("show")
    show.add_argument("journal", type=Path)
    args = parser.parse_args()
    if args.command == "show":
        print(args.journal.read_text(encoding="utf-8") if args.journal.exists() else "", end="")
        return
    fields: dict[str, Any] = {}
    for assignment in args.field:
        if "=" not in assignment:
            parser.error("--field expects KEY=JSON_OR_TEXT")
        key, value = assignment.split("=", 1)
        if not key:
            parser.error("journal field name must not be empty")
        fields[key] = parse_value(value)
    event = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": args.status,
        "hypothesis": args.hypothesis,
        "rung": args.rung,
        "seed": args.seed,
        "config": args.config,
        "config_hash": args.config_hash,
        "checkpoint": args.checkpoint,
        "gpu": args.gpu,
        "pid": args.pid,
        "metrics": args.metrics,
        "reason": args.reason,
        **fields,
    }
    encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    with args.journal.open("a", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    print(f"recorded {args.experiment_id}: {args.status}")


if __name__ == "__main__":
    main()
