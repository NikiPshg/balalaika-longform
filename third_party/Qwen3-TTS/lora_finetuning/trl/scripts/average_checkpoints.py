#!/usr/bin/env python3
"""Build an inference-only exponential average of complete Qwen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file


TRAINING_ONLY_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "standalone_complete.json",
}


def normalized_exponential_weights(count: int, decay: float) -> list[float]:
    """Return oldest-to-newest normalized geometric weights."""

    if count < 1:
        raise ValueError("at least one checkpoint is required")
    if not math.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError("decay must be finite and in (0, 1]")
    raw = [decay ** (count - index - 1) for index in range(count)]
    total = sum(raw)
    return [value / total for value in raw]


def average_state_dicts(
    states: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    if len(states) != len(weights) or not states:
        raise ValueError("states and weights must have the same non-zero length")
    keys = states[0].keys()
    if any(state.keys() != keys for state in states[1:]):
        raise ValueError("checkpoint shard tensor keys differ")
    averaged: dict[str, torch.Tensor] = {}
    for name in keys:
        values = [state[name] for state in states]
        reference = values[-1]
        if any(value.shape != reference.shape or value.dtype != reference.dtype for value in values):
            raise ValueError(f"tensor contract differs for {name}")
        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, dtype=torch.float32)
            for weight, value in zip(weights, values, strict=True):
                accumulator.add_(value.float(), alpha=weight)
            averaged[name] = accumulator.to(reference.dtype).contiguous()
        else:
            if any(not torch.equal(value, reference) for value in values[:-1]):
                raise ValueError(f"non-floating tensor differs for {name}")
            averaged[name] = reference.contiguous()
    return averaged


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_layout(checkpoint: Path) -> tuple[list[str], dict[str, object] | None]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid weight_map in {index_path}")
        return sorted(set(map(str, weight_map.values()))), index
    model_path = checkpoint / "model.safetensors"
    if model_path.is_file():
        return [model_path.name], None
    raise ValueError(f"checkpoint has no safetensors model: {checkpoint}")


def copy_inference_assets(source: Path, output: Path, model_files: Iterable[str]) -> None:
    excluded = set(model_files) | TRAINING_ONLY_FILES | {
        "model.safetensors.index.json",
        "ema.json",
    }
    output.mkdir(parents=True)
    for item in source.iterdir():
        if item.name in excluded:
            continue
        target = output / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="Complete checkpoints ordered oldest to newest; repeat this option.",
    )
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    for checkpoint in checkpoints:
        if not (checkpoint / "standalone_complete.json").is_file():
            raise SystemExit(f"checkpoint is not marked complete: {checkpoint}")

    weights = normalized_exponential_weights(len(checkpoints), args.decay)
    layouts = [model_layout(checkpoint) for checkpoint in checkpoints]
    shard_names, index = layouts[-1]
    if any(layout[0] != shard_names or layout[1] != index for layout in layouts[:-1]):
        raise SystemExit("checkpoint model shard layouts differ")

    copy_inference_assets(checkpoints[-1], output, shard_names)
    if index is not None:
        shutil.copy2(
            checkpoints[-1] / "model.safetensors.index.json",
            output / "model.safetensors.index.json",
        )

    input_hashes: dict[str, dict[str, str]] = {
        str(checkpoint): {} for checkpoint in checkpoints
    }
    output_hashes: dict[str, str] = {}
    for shard_index, shard_name in enumerate(shard_names, 1):
        paths = [checkpoint / shard_name for checkpoint in checkpoints]
        print(f"[{shard_index}/{len(shard_names)}] averaging {shard_name}", flush=True)
        states = [load_file(path, device="cpu") for path in paths]
        averaged = average_state_dicts(states, weights)
        output_path = output / shard_name
        save_file(averaged, output_path, metadata={"format": "pt"})
        del states, averaged
        for checkpoint, path in zip(checkpoints, paths, strict=True):
            input_hashes[str(checkpoint)][shard_name] = sha256_file(path)
        output_hashes[shard_name] = sha256_file(output_path)

    manifest = {
        "schema_version": 1,
        "method": "normalized_exponential_checkpoint_average",
        "inference_only": True,
        "checkpoint_order": "oldest_to_newest",
        "checkpoint_interval_steps": 500,
        "decay_per_checkpoint": args.decay,
        "checkpoints": [
            {"path": str(checkpoint), "weight": weight, "model_sha256": input_hashes[str(checkpoint)]}
            for checkpoint, weight in zip(checkpoints, weights, strict=True)
        ],
        "output_model_sha256": output_hashes,
    }
    (output / "ema.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "standalone_complete.json").write_text(
        json.dumps(
            {
                "format": "qwen3tts-standalone-checkpoint",
                "version": 1,
                "derived": "post-hoc-ema",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
