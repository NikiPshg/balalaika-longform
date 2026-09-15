"""One isolated synthetic full-model step; root schedules each duration separately."""
from __future__ import annotations

import argparse
import math
import time
import traceback

import torch

from .common import (DEFAULT_UPSTREAM, HOP_LENGTH, MEL_CHANNELS, SAMPLE_RATE,
                     atomic_json, configure_cuda, load_model, seed_everything)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument("--max-duration-seconds", type=float, default=930.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = dict(seconds=args.seconds, seed=args.seed, checkpoint=args.checkpoint,
                  synthetic=True, purpose="resource feasibility, not speech quality",
                  status="starting", torch_version=torch.__version__)
    atomic_json(args.output, result)
    started = time.monotonic()
    try:
        device = configure_cuda()
        seed_everything(args.seed)
        model = load_model(args.checkpoint, args.vocab, upstream=args.upstream,
                           max_duration_seconds=args.max_duration_seconds, device=device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, foreach=False)
        frames = math.floor(args.seconds * SAMPLE_RATE / HOP_LENGTH) + 1
        if frames > model.transformer.text_embed.precompute_max_pos:
            raise ValueError("Probe exceeds declared positional buffer")
        mel = torch.randn(1, frames, MEL_CHANNELS, device=device) - 4
        sentence = "Это проверка памяти и вычислений модели русской речи. "
        text = (sentence * math.ceil(args.seconds * 15 / len(sentence)))[:int(args.seconds * 15)]
        result.update(frames=frames, text_characters=len(text),
                      parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      position_buffer=model.transformer.text_embed.precompute_max_pos,
                      device_name=torch.cuda.get_device_name(device),
                      load_seconds=time.monotonic() - started)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step_started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = model(inp=mel, text=[text])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite forward loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        result.update(status="ok", loss=float(loss.detach()), gradient_norm=float(norm),
                      step_seconds=time.monotonic() - step_started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        result.update(status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                      error=str(exc), traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
    result["wall_seconds"] = time.monotonic() - started
    atomic_json(args.output, result)
    print(result, flush=True)
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
