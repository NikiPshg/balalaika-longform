#!/usr/bin/env python3
"""Preserve complete Trainer checkpoints at selected steps using hard links."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, action="append", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def checkpoint_complete(checkpoint: Path) -> bool:
    return (
        (checkpoint / "standalone_complete.json").is_file()
        and (checkpoint / "trainer_state.json").is_file()
        and (checkpoint / "optimizer.pt").is_file()
        and (checkpoint / "scheduler.pt").is_file()
    )


def preserve(source: Path, destination: Path) -> None:
    if destination.exists():
        if not checkpoint_complete(destination):
            raise RuntimeError(f"Incomplete milestone already exists: {destination}")
        print(f"milestone already complete: {destination}", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"Temporary milestone path already exists: {temporary}")
    shutil.copytree(source, temporary, copy_function=os.link)
    if not checkpoint_complete(temporary):
        raise RuntimeError(f"Copied milestone is incomplete: {temporary}")
    temporary.rename(destination)
    print(f"preserved milestone: {source} -> {destination}", flush=True)


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    steps = sorted(set(args.step))
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("Every --step must be positive")

    run_dir = args.run_dir.resolve()
    milestones_dir = run_dir / "milestones"
    pending = set(steps)
    print(
        f"watching {run_dir} for milestones: {', '.join(map(str, steps))}",
        flush=True,
    )
    while pending:
        for step in sorted(pending):
            source = run_dir / f"checkpoint-{step}"
            destination = milestones_dir / f"checkpoint-{step}"
            if checkpoint_complete(destination):
                pending.remove(step)
            elif checkpoint_complete(source):
                preserve(source, destination)
                pending.remove(step)
        if pending:
            time.sleep(args.poll_seconds)
    print("all requested milestones preserved", flush=True)


if __name__ == "__main__":
    main()
