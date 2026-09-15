#!/usr/bin/env python3
"""Create a local model soup inside the existing Qwen checkpoint format."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file


def interpolate_state_dicts(
    base: dict[str, torch.Tensor],
    tuned: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if base.keys() != tuned.keys():
        missing = sorted(base.keys() - tuned.keys())
        extra = sorted(tuned.keys() - base.keys())
        raise ValueError(f"checkpoint tensor keys differ: missing={missing[:3]}, extra={extra[:3]}")
    merged: dict[str, torch.Tensor] = {}
    for name, base_value in base.items():
        tuned_value = tuned[name]
        if base_value.shape != tuned_value.shape or base_value.dtype != tuned_value.dtype:
            raise ValueError(f"tensor contract differs for {name}")
        if base_value.is_floating_point():
            value = torch.lerp(base_value.float(), tuned_value.float(), alpha).to(base_value.dtype)
        else:
            if not torch.equal(base_value, tuned_value):
                raise ValueError(f"non-floating tensor differs for {name}")
            value = base_value
        merged[name] = value.contiguous()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be finite and in [0, 1]")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    base_model = args.base / "model.safetensors"
    tuned_model = args.tuned / "model.safetensors"
    if not base_model.is_file() or not tuned_model.is_file():
        raise SystemExit("both checkpoints must contain model.safetensors")
    args.output.mkdir(parents=True)
    for source in args.tuned.iterdir():
        if source.is_file() and source.name not in {
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "training_args.bin",
            "trainer_state.json",
            "rng_state_0.pth",
            "rng_state_1.pth",
        }:
            shutil.copy2(source, args.output / source.name)
        elif source.is_dir():
            shutil.copytree(source, args.output / source.name)
    for source in args.base.iterdir():
        if source.is_dir() and not (args.output / source.name).exists():
            shutil.copytree(source, args.output / source.name)

    merged = interpolate_state_dicts(
        load_file(base_model, device="cpu"),
        load_file(tuned_model, device="cpu"),
        args.alpha,
    )
    save_file(merged, args.output / "model.safetensors", metadata={"format": "pt"})
    contract = {
        "schema_version": 1,
        "method": "linear_checkpoint_interpolation",
        "base": str(args.base.resolve()),
        "tuned": str(args.tuned.resolve()),
        "alpha": args.alpha,
    }
    (args.output / "interpolation.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
