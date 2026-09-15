#!/usr/bin/env python3
"""Wait for a local dataset snapshot and safely resume an active SFT from it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--download-log", type=Path, required=True)
    parser.add_argument("--download-session", required=True)
    parser.add_argument("--train-session", required=True)
    parser.add_argument("--local-train-session", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-tars", type=int, required=True)
    parser.add_argument("--expected-tar-bytes", type=int, required=True)
    parser.add_argument("--active-save-steps", type=int, default=500)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def checkpoint_complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "standalone_complete.json",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
        )
    )


def complete_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if checkpoint_complete(path):
            checkpoints.append((step, path.resolve()))
    return sorted(checkpoints)


def wait_for_download(args: argparse.Namespace) -> None:
    print("waiting for snapshot download", flush=True)
    while True:
        text = (
            args.download_log.read_text(encoding="utf-8", errors="replace")
            if args.download_log.is_file()
            else ""
        )
        if "download_complete" in text:
            print("snapshot downloader reported completion", flush=True)
            return
        if not tmux_alive(args.download_session):
            raise RuntimeError(
                "Snapshot downloader stopped without a download_complete marker"
            )
        time.sleep(args.poll_seconds)


def verify_snapshot(args: argparse.Namespace) -> None:
    tar_paths = sorted((args.dataset_dir / "train").glob("*.tar"))
    tar_bytes = sum(path.stat().st_size for path in tar_paths)
    if len(tar_paths) != args.expected_tars:
        raise RuntimeError(
            f"Expected {args.expected_tars} TARs, found {len(tar_paths)}"
        )
    if tar_bytes != args.expected_tar_bytes:
        raise RuntimeError(
            f"Expected {args.expected_tar_bytes} TAR bytes, found {tar_bytes}"
        )
    for path in (tar_paths[0], tar_paths[len(tar_paths) // 2], tar_paths[-1]):
        with tarfile.open(path, mode="r:") as archive:
            first = next(iter(archive), None)
            if first is None:
                raise RuntimeError(f"Empty TAR shard: {path}")
    print(
        f"snapshot verified: {len(tar_paths)} TARs, {tar_bytes} bytes",
        flush=True,
    )


def wait_for_next_checkpoint(args: argparse.Namespace) -> Path:
    checkpoints = complete_checkpoints(args.run_dir)
    if not checkpoints:
        raise RuntimeError(f"No complete checkpoint below {args.run_dir}")
    latest_step = checkpoints[-1][0]
    target_step = latest_step + args.active_save_steps
    target = args.run_dir / f"checkpoint-{target_step}"
    print(
        f"latest complete checkpoint is {latest_step}; waiting for {target_step}",
        flush=True,
    )
    while tmux_alive(args.train_session):
        if checkpoint_complete(target):
            print(f"handoff checkpoint is complete: {target}", flush=True)
            return target.resolve()
        time.sleep(args.poll_seconds)
    checkpoints = complete_checkpoints(args.run_dir)
    if not checkpoints:
        raise RuntimeError("Training stopped and no complete checkpoint remains")
    print(
        f"training stopped before {target_step}; using {checkpoints[-1][1]}",
        flush=True,
    )
    return checkpoints[-1][1]


def stop_training(args: argparse.Namespace) -> None:
    if not tmux_alive(args.train_session):
        return
    print(f"sending Ctrl-C to {args.train_session}", flush=True)
    subprocess.run(
        ["tmux", "send-keys", "-t", args.train_session, "C-c"], check=True
    )
    deadline = time.monotonic() + 120.0
    while tmux_alive(args.train_session) and time.monotonic() < deadline:
        time.sleep(2.0)
    if tmux_alive(args.train_session):
        raise RuntimeError(
            f"Training session {args.train_session!r} did not stop after Ctrl-C"
        )
    print("remote-stream training stopped cleanly", flush=True)


def launch_local(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    if tmux_alive(args.local_train_session):
        raise RuntimeError(
            f"Local train session already exists: {args.local_train_session}"
        )
    launch = args.repo_root / "lora_finetuning/trl/launch.sh"
    console_log = args.run_dir / "console.log"
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=0,1",
        str(launch),
        "--accelerate-config",
        "multi",
        "--num-processes",
        "2",
        "--console-log",
        str(console_log),
        "sft",
        str(args.config),
        "--resume_from_checkpoint",
        str(checkpoint),
    ]
    shell_command = f"cd {shlex.quote(str(args.repo_root))} && exec {shlex.join(command)}"
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            args.local_train_session,
            shell_command,
        ],
        check=True,
    )
    time.sleep(5.0)
    if not tmux_alive(args.local_train_session):
        raise RuntimeError("Local SFT session exited during startup")
    print(f"local SFT launched in {args.local_train_session}", flush=True)
    return command


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    args.download_log = args.download_log.resolve()
    args.run_dir = args.run_dir.resolve()
    args.config = args.config.resolve()
    if args.expected_tars <= 0 or args.expected_tar_bytes <= 0:
        raise ValueError("Expected snapshot size must be positive")
    if args.active_save_steps <= 0 or args.poll_seconds <= 0:
        raise ValueError("Step and polling intervals must be positive")

    wait_for_download(args)
    verify_snapshot(args)
    checkpoint = wait_for_next_checkpoint(args)
    stop_training(args)
    command = launch_local(args, checkpoint)
    record = {
        "switched_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "expected_tars": args.expected_tars,
        "expected_tar_bytes": args.expected_tar_bytes,
        "resume_checkpoint": str(checkpoint),
        "tmux_session": args.local_train_session,
        "command": command,
    }
    output = args.run_dir / "local_handoff_complete.json"
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"handoff record: {output}", flush=True)


if __name__ == "__main__":
    main()
