"""Full DiT adaptation with one source parent per optimizer update.

Long-SFT processes one continuous parent, Short-SFT accumulates its short windows.
Each window's mean CFM loss is weighted by its actual mel-frame share of that
parent. All processed source seconds and mel frames are recorded explicitly.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import random
import time
import traceback

import torch

from .common import (CFM_KWARGS, DEFAULT_UPSTREAM, EMA_KWARGS, EMA_PACKAGE_VERSION,
                     append_jsonl, atomic_json, capture_rng,
                     checkpoint_ema_history, configure_cuda, create_ema, load_model, mel_for_row, parent_groups,
                     read_manifest, restore_rng, seed_everything, sha256)


def learning_rate(update: int, updates: int, peak: float, warmup_fraction: float) -> float:
    warmup = max(1, round(updates * warmup_fraction))
    if update < warmup:
        return peak * (update + 1) / warmup
    progress = (update - warmup) / max(1, updates - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def parent_at_update(parents: list[str], update: int, seed: int) -> tuple[str, int]:
    if not parents:
        raise ValueError("Empty training parent set")
    epoch, index = divmod(update, len(parents))
    order = sorted(parents)
    random.Random(seed + epoch).shuffle(order)
    return order[index], epoch


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def dev_loss(model, rows: list[dict], device: torch.device, seed: int) -> dict:
    """Fixed corruption/mask/dropout seeds per ordered dev item, preserving train RNG."""
    rng = capture_rng()
    training = model.training
    model.eval()
    total, frames = 0.0, 0
    items = []
    started = time.monotonic()
    try:
        with torch.no_grad():
            for index, row in enumerate(sorted(rows, key=lambda r: r["utt"])):
                seed_everything(seed + index)
                mel = mel_for_row(model, row, device)
                with autocast(device):
                    loss, _, _ = model(inp=mel, text=[row["text"]])
                value = float(loss)
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite dev loss: {row['utt']}")
                count = mel.shape[1]
                total += count * value
                frames += count
                items.append(dict(utt=row["utt"], loss=value, frames=count))
                del mel, loss
    finally:
        model.train(training)
        restore_rng(rng)
    return dict(loss=total / frames, frames=frames, items=items,
                seconds=time.monotonic() - started, fixed_seed=seed)


def save_checkpoint(path: Path, model, optimizer, *, ema, update: int, ledger: dict,
                    protocol: dict, state: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(checkpoint_format_version=2,
                    model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    ema_model_state_dict=ema.state_dict(),
                    rng=capture_rng(), update=update, ledger=ledger, protocol=protocol,
                    state=state), temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-seed", type=int, default=100000)
    parser.add_argument("--dev-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-wall-seconds", type=float, default=0)
    parser.add_argument("--stop-after-update", type=int, default=0,
                        help="Pause at this absolute update cursor without changing the planned learning-rate schedule")
    parser.add_argument("--max-duration-seconds", type=float, default=930.0)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument("--resume")
    parser.add_argument("--skip-initial-dev", action="store_true")
    args = parser.parse_args()
    if args.updates <= 0 or not 0 <= args.warmup_fraction <= 1 or args.stop_after_update < 0:
        parser.error("Invalid updates, warmup fraction, or stop-after-update")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train.jsonl").exists() and not args.resume:
        raise FileExistsError("Existing train.jsonl requires explicit --resume or a new output directory")
    device = configure_cuda()
    seed_everything(args.seed)
    groups = parent_groups(read_manifest(args.manifest))
    dev_rows = read_manifest(args.dev_manifest) if args.dev_manifest else []
    if set(groups).intersection(r["parent"] for r in dev_rows):
        raise ValueError("Training/dev parent overlap")
    protocol = dict(initial_checkpoint=str(Path(args.checkpoint).resolve()),
                    manifest_sha256=sha256(args.manifest),
                    dev_manifest_sha256=sha256(args.dev_manifest) if args.dev_manifest else None,
                    vocab_sha256=sha256(args.vocab), updates=args.updates,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                    warmup_fraction=args.warmup_fraction, clip_grad=args.clip_grad,
                    seed=args.seed, dev_seed=args.dev_seed,
                    max_duration_seconds=args.max_duration_seconds,
                    parent_order="sorted parents shuffled by seed+pass",
                    loss_weight="actual mel frames / parent mel frames",
                    precision="bf16 autocast, float32 weights and AdamW states",
                    ema=dict(package="ema-pytorch", version=EMA_PACKAGE_VERSION, **EMA_KWARGS),
                    ema_counter_policy="reset_for_new_sft_run",
                    ema_initial_state=dict(step=0, initted=False),
                    source_ema_history=checkpoint_ema_history(args.checkpoint),
                    cfm_objective=dict(**CFM_KWARGS,
                        mask_sampling="one contiguous random span covering 70-100% of each item",
                        conditioning_span="remaining unmasked frames; absolute span scales with item length, as upstream"),
                    dev_weights="ema", inference_weights="ema",
                    checkpoint_selection="last planned parent update; no test selection",
                    finite_failure_policy="fail fast; diagnosed path relaunches managed separately")
    model = load_model(args.checkpoint, args.vocab, upstream=args.upstream,
                       max_duration_seconds=args.max_duration_seconds, device=device)
    model.train()
    ema = create_ema(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=False)
    update = 0
    ledger = dict(audio_seconds=0.0, mel_frames=0, windows=0, parent_updates=0)
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "ema_model_state_dict" not in resume:
            raise ValueError("This protocol requires a full EMA checkpoint; restart old non-EMA pilots from original weights")
        if resume["state"] == "failed_optimizer_step":
            raise ValueError("Interrupted AdamW/EMA step cannot be resumed; use a prior committed checkpoint")
        if resume["protocol"] != protocol:
            raise ValueError("Resume protocol differs from saved checkpoint")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        ema.load_state_dict(resume["ema_model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        update, ledger = resume["update"], resume["ledger"]
        if int(ema.step.item()) != update:
            raise ValueError("EMA step must equal committed optimizer update under fresh-SFT counter policy")
        restore_rng(resume["rng"])
        del resume
    atomic_json(output / "protocol.json", dict(**protocol, arguments=vars(args),
                parameters=sum(p.numel() for p in model.parameters()),
                parent_count=len(groups), source_checkpoint_sha256=sha256(args.checkpoint),
                torch_version=torch.__version__, device=torch.cuda.get_device_name(device),
                runtime=dict(physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
                             logical_device=str(device))))
    started = time.monotonic()
    state = "running"
    committed_rng = capture_rng()
    optimizer_inflight = False
    try:
        if dev_rows and not args.skip_initial_dev and update == 0:
            append_jsonl(output / "dev.jsonl", dict(update=update, weights="ema",
                          **dev_loss(ema.ema_model, dev_rows, device, args.dev_seed)))
        while update < args.updates:
            if args.stop_after_update and update >= args.stop_after_update:
                state = "paused_by_update_limit"
                break
            if args.max_wall_seconds and time.monotonic() - started >= args.max_wall_seconds:
                state = "budget_exhausted"
                break
            parent, epoch = parent_at_update(list(groups), update, args.seed)
            rows = groups[parent]
            step_started = time.monotonic()
            # Cache only one parent on CPU. No padding and no training-length filter.
            mels = [mel_for_row(model, row, "cpu") for row in rows]
            frame_counts = [mel.shape[1] for mel in mels]
            total_frames = sum(frame_counts)
            if max(frame_counts) > model.transformer.text_embed.precompute_max_pos:
                raise ValueError(f"Parent {parent} exceeds declared positional buffer; do not silently truncate")
            rate = learning_rate(update, args.updates, args.learning_rate, args.warmup_fraction)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            torch.cuda.reset_peak_memory_stats()
            for row, cpu_mel, frames in zip(rows, mels, frame_counts):
                mel = cpu_mel.to(device)
                with autocast(device):
                    loss, _, _ = model(inp=mel, text=[row["text"]])
                    weighted = loss * (frames / total_frames)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {row['utt']}")
                weighted.backward()
                weighted_loss += float(loss.detach()) * frames / total_frames
                del mel, loss, weighted
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad,
                                                 error_if_nonfinite=True)
            optimizer_inflight = True
            optimizer.step()
            # Exactly once per completed optimizer update, as in upstream Trainer.
            ema.update()
            torch.cuda.synchronize()
            optimizer_inflight = False
            update += 1
            committed_rng = capture_rng()
            audio_seconds = sum(row["_actual_audio_seconds"] for row in rows)
            ledger["audio_seconds"] += audio_seconds
            ledger["mel_frames"] += total_frames
            ledger["windows"] += len(rows)
            ledger["parent_updates"] += 1
            record = dict(update=update, parent=parent, epoch=epoch, loss=weighted_loss,
                          loss_weights="online", ema_step=int(ema.step.item()),
                          gradient_norm=float(norm), learning_rate=rate, windows=len(rows),
                          audio_seconds=audio_seconds, mel_frames=total_frames,
                          nominal_audio_seconds=sum(row["duration"] for row in rows),
                          window_source_samples={row["utt"]: dict(samples=row["_source_samples"],
                              sample_rate=row["_source_sample_rate"],
                              resampled_samples=row["_resampled_samples"],
                              end_overrun_samples=row["_metadata_end_overrun_samples"])
                              for row in rows},
                          window_frames=dict(zip((row["utt"] for row in rows), frame_counts)),
                          cumulative=dict(ledger), step_seconds=time.monotonic() - step_started,
                          elapsed_seconds=time.monotonic() - started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            append_jsonl(output / "train.jsonl", record)
            print({k: record[k] for k in ("update", "parent", "loss", "audio_seconds", "step_seconds")}, flush=True)
            del mels
            if dev_rows and args.dev_every > 0 and (update % args.dev_every == 0 or update == args.updates):
                append_jsonl(output / "dev.jsonl", dict(update=update, weights="ema",
                              **dev_loss(ema.ema_model, dev_rows, device, args.dev_seed)))
            if args.save_every > 0 and update % args.save_every == 0 and update < args.updates:
                save_checkpoint(output / "latest.pt", model, optimizer, ema=ema, update=update,
                                ledger=ledger, protocol=protocol, state="running")
        if update == args.updates:
            state = "complete"
    except BaseException as exc:
        state = "failed_optimizer_step" if optimizer_inflight else "failed"
        restore_rng(committed_rng)
        append_jsonl(output / "failures.jsonl", dict(update=update, error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        # Forward/backward failures preserve the last committed optimizer update.
        # A failure inside AdamW/EMA is marked nonresumable rather than hiding partial state.
        checkpoint_name = ("final.pt" if state == "complete" else
                           "failed_optimizer_step.pt" if state == "failed_optimizer_step" else "latest.pt")
        save_checkpoint(output / checkpoint_name,
                        model, optimizer, ema=ema, update=update, ledger=ledger, protocol=protocol, state=state)
        atomic_json(output / "summary.json", dict(state=state, updates=update, planned_updates=args.updates,
                    ledger=ledger, elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
