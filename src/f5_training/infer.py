"""Native F5 flow inference with an explicit optional sentence-chunking control.

No ASR, target recording duration, EOS classification, or quality-based retries.
The public chunk heuristic and duration estimator are reproduced from the pinned
upstream utils_infer.py. Chunks run sequentially on the sole authorized GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import time
import traceback

import numpy as np
import torch

from .common import (DEFAULT_UPSTREAM, HOP_LENGTH, SAMPLE_RATE, atomic_json,
                     configure_cuda, estimated_duration_frames, load_audio,
                     load_model, seed_everything)


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    """Exact upstream sentence/UTF-8 chunk heuristic (not a hard token limit)."""
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text):
        if not sentence:
            continue
        if len(current.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current += sentence + " " if len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " " if len(sentence[-1].encode("utf-8")) == 1 else sentence
    if current:
        chunks.append(current.strip())
    return chunks


def crossfade(waves: list[np.ndarray], seconds: float = 0.15) -> np.ndarray:
    if not waves:
        raise ValueError("No generated waveforms")
    final = waves[0]
    for wave in waves[1:]:
        count = min(int(seconds * SAMPLE_RATE), len(final), len(wave))
        if count <= 0:
            final = np.concatenate((final, wave))
        else:
            fade_out = np.linspace(1, 0, count)
            fade_in = np.linspace(0, 1, count)
            overlap = final[-count:] * fade_out + wave[:count] * fade_in
            final = np.concatenate((final[:-count], overlap, wave[count:]))
    return final.astype(np.float32, copy=False)


class F5Generator:
    def __init__(self, checkpoint: str, vocab: str, vocoder_dir: str, *,
                 upstream: str = DEFAULT_UPSTREAM, max_duration_seconds: float = 930.0,
                 steps: int = 32, cfg_strength: float = 2.0,
                 sway_sampling_coef: float = -1.0, speed: float = 1.0):
        self.device = configure_cuda()
        self.model = load_model(checkpoint, vocab, upstream=upstream,
                                max_duration_seconds=max_duration_seconds,
                                checkpoint_activations=False, device=self.device).eval()
        from vocos import Vocos
        self.vocoder = Vocos.from_hparams(str(Path(vocoder_dir) / "config.yaml"))
        state = torch.load(Path(vocoder_dir) / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.vocoder.load_state_dict(state, strict=True)
        self.vocoder = self.vocoder.float().eval().to(self.device)
        for parameter in self.vocoder.parameters():
            parameter.requires_grad_(False)
        self.protocol = dict(checkpoint=str(Path(checkpoint).resolve()),
                             max_duration_seconds=max_duration_seconds, steps=steps,
                             cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef,
                             speed=speed, precision="bf16 autocast with float32 model weights",
                             duration_estimator="prompt audio frames times UTF-8 text-length ratio",
                             target_duration_oracle=False, crossfade_seconds=0.15,
                             positional_buffer=self.model.transformer.text_embed.precompute_max_pos,
                             vocoder="local pinned vocos-mel-24khz, float32 full-sequence decode",
                             stock_deviations=["extended nonpersistent text positional buffer",
                                               "explicit declared CFM duration ceiling",
                                               "bf16 autocast instead of default fp16 weights",
                                               "sequential chunk execution; per-chunk seed+index",
                                               "fixed supplied prompt audio and transcript; no ASR or trimming"])

    def generate(self, *, prompt_audio: str, prompt_text: str, text: str,
                 output_path: str, seed: int = 0, mode: str = "uninterrupted") -> dict:
        if mode not in {"uninterrupted", "chunked"}:
            raise ValueError("mode must be uninterrupted or chunked")
        output = Path(output_path)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite generated audio: {output}")
        metadata = dict(gen_status="infrastructure_error", stop_reason="error", mode=mode,
                        output_path=str(output.resolve()), seed=seed,
                        sample_rate=SAMPLE_RATE, protocol=self.protocol)
        started = time.monotonic()
        seed_everything(seed)
        torch.cuda.reset_peak_memory_stats()
        waves, actual_frames = [], []
        try:
            if not text.strip() or not prompt_text.strip():
                raise ValueError("Target text and supplied prompt transcript must be nonempty")
            audio = load_audio(dict(audio=prompt_audio, utt="prompt"))
            prompt_seconds = len(audio) / SAMPLE_RATE
            original_rms = float(audio.square().mean().sqrt())
            if original_rms <= 0:
                raise ValueError("Silent prompt")
            if original_rms < 0.1:
                audio = audio * (0.1 / original_rms)
            audio = audio.unsqueeze(0).to(self.device)
            reference_text = prompt_text + " " if len(prompt_text[-1].encode("utf-8")) == 1 else prompt_text
            prompt_frames = audio.shape[-1] // HOP_LENGTH
            if mode == "chunked":
                max_chars = int(len(prompt_text.encode("utf-8")) / prompt_seconds *
                                (22 - prompt_seconds) * self.protocol["speed"])
                chunks = chunk_text(text, max_chars=max_chars)
                metadata["chunk_max_utf8_bytes"] = max_chars
            else:
                chunks = [text]
            durations = [estimated_duration_frames(audio.shape[-1], reference_text, chunk,
                                                   self.protocol["speed"]) for chunk in chunks]
            ceiling = self.model.transformer.text_embed.precompute_max_pos
            metadata.update(prompt_seconds=prompt_seconds, prompt_rms=original_rms,
                            chunk_count=len(chunks), estimated_frames=durations,
                            target_utf8_bytes=len(text.encode("utf-8")))
            with torch.inference_mode():
                # Compute log mel in float32, outside inference autocast.
                conditioning = self.model.mel_spec(audio).transpose(1, 2).contiguous()
                for index, (chunk, duration) in enumerate(zip(chunks, durations)):
                    if max(duration, len(reference_text + chunk) + 1, prompt_frames + 2) > ceiling:
                        metadata.update(gen_status="context_limit", stop_reason="duration_resource_limit")
                        raise ValueError(f"Estimated {duration} frames exceeds declared buffer {ceiling}; no truncation")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        generated, trajectory = self.model.sample(
                            cond=conditioning, text=[reference_text + chunk], duration=duration,
                            steps=self.protocol["steps"], cfg_strength=self.protocol["cfg_strength"],
                            sway_sampling_coef=self.protocol["sway_sampling_coef"],
                            seed=seed + index, max_duration=ceiling)
                    del trajectory
                    actual_frames.append(generated.shape[1])
                    mel = generated[:, prompt_frames:, :].transpose(1, 2).float()
                    wave = self.vocoder.decode(mel)
                    if original_rms < 0.1:
                        wave = wave * (original_rms / 0.1)
                    waves.append(wave.squeeze(0).cpu().numpy())
                    del generated, mel, wave
            final = crossfade(waves, self.protocol["crossfade_seconds"] if mode == "chunked" else 0)
            if not len(final) or not np.isfinite(final).all():
                raise FloatingPointError("Generated waveform is empty or nonfinite")
            import soundfile as sf
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, final, SAMPLE_RATE, subtype="FLOAT")
            metadata.update(gen_status="complete", stop_reason="flow_completed",
                            raw_duration_sec=len(final) / SAMPLE_RATE,
                            actual_frames=actual_frames, samples=len(final))
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                metadata.update(gen_status="oom", stop_reason="cuda_oom")
            metadata.update(error=str(exc), traceback=traceback.format_exc())
            self.model.transformer.clear_cache()
            if waves:
                # Retain every completed chunk when a later chunk fails. The
                # generation status stays a failure, so completion cannot be hidden.
                import soundfile as sf
                partial = crossfade(waves, self.protocol["crossfade_seconds"] if mode == "chunked" else 0)
                output.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output, partial, SAMPLE_RATE, subtype="FLOAT")
                metadata.update(partial_audio=True, completed_chunks=len(waves),
                                raw_duration_sec=len(partial)/SAMPLE_RATE,
                                actual_frames=actual_frames, samples=len(partial))
        torch.cuda.synchronize()
        metadata.update(elapsed=time.monotonic() - started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--vocoder-dir", required=True)
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--mode", choices=("uninterrupted", "chunked"), default="uninterrupted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument("--max-duration-seconds", type=float, default=930.0)
    args = parser.parse_args()
    generator = F5Generator(args.checkpoint, args.vocab, args.vocoder_dir, upstream=args.upstream,
                            max_duration_seconds=args.max_duration_seconds)
    result = generator.generate(prompt_audio=args.prompt_audio, prompt_text=args.prompt_text,
                                text=args.text, output_path=args.output, seed=args.seed, mode=args.mode)
    atomic_json(args.metadata, result)
    print(result, flush=True)
    if result["gen_status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
