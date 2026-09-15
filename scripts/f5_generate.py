#!/usr/bin/env python3
"""Generate a fixed F5 benchmark grid; persist every attempt and resume by run ID."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.f5_training.common import append_jsonl, atomic_json, sha256
from src.f5_training.infer import F5Generator


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def make_grid(bench, references):
    refs = {r['voice_id']: r for r in references}
    if len(refs) != len(references):
        raise ValueError('Duplicate reference voice ID')
    grid = []
    for item in bench:
        ids = [item['voice_id']] if item.get('voice_id') else [
            r['voice_id'] for r in references if r.get('role') == 'primary']
        if not ids:
            raise ValueError(f"No voice for {item['text_id']}")
        for voice in ids:
            reference = refs[voice]
            if not item['text_tts'].strip() or not reference['ref_text'].strip():
                raise ValueError('Missing synthesis/prompt transcript')
            grid.append((item, reference))
    return grid


@contextmanager
def output_directory_lock(directory):
    """Serialize writers; retain the lock inode so queued processes share it."""
    with (Path(directory)/'.generation.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--vocoder-dir', required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--references', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--mode', choices=['uninterrupted', 'chunked'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-duration-seconds', type=float, default=1230)
    parser.add_argument('--max-wall-seconds', type=float, default=0)
    parser.add_argument('--text-id', action='append')
    parser.add_argument('--buckets', nargs='+')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    bench, refs = read_rows(args.benchmark), read_rows(args.references)
    if args.text_id:
        ids = set(args.text_id)
        if ids - {r['text_id'] for r in bench}:
            raise ValueError('Requested text IDs absent from benchmark')
        bench = [r for r in bench if r['text_id'] in ids]
    if args.buckets:
        bench = [r for r in bench if r['bucket'] in args.buckets]
    grid = make_grid(bench, refs)
    if args.limit:
        grid = grid[:args.limit]
    if not grid:
        raise ValueError('Empty evaluation grid')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    # The process may have waited behind another complete generation job. Read
    # every resume artifact only after acquiring ownership of this output grid.
    with output_directory_lock(out):
        generate_locked(args, grid, out)


def generate_locked(args, grid, out):
    runfile = out/'runs.jsonl'
    previous = read_rows(runfile) if runfile.exists() else []
    done = {r['run_id'] for r in previous}
    if len(done) != len(previous):
        raise ValueError('Duplicate existing run IDs; refusing ambiguous resume')
    expected = {f"{args.experiment}__{b['text_id']}__{r['voice_id']}__s{args.seed}"
                for b, r in grid}
    if len(expected) != len(grid):
        raise ValueError('Duplicate run IDs in requested grid')
    if done - expected:
        raise ValueError('Existing run IDs are outside the requested exact grid')
    protocol = dict(checkpoint_sha256=sha256(args.checkpoint), vocab_sha256=sha256(args.vocab),
                    vocoder_sha256=sha256(Path(args.vocoder_dir)/'pytorch_model.bin'),
                    experiment=args.experiment, seed=args.seed, mode=args.mode,
                    max_duration_seconds=args.max_duration_seconds,
                    benchmark_sha256=sha256(args.benchmark), references_sha256=sha256(args.references),
                    grid=[[b['text_id'],r['voice_id']] for b,r in grid],
                    steps=32, cfg_strength=2, sway_sampling_coef=-1, speed=1,
                    target_duration_oracle=False)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Generation protocol differs from existing output')
    else:
        if previous:
            raise ValueError('Existing runs require their original generation protocol')
        atomic_json(out/'protocol.json', protocol)
    if done == expected:
        progress_path = out/'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        progress.update(planned=len(grid), attempted=len(done), completed_grid=True)
        progress.setdefault('elapsed_seconds', 0)
        atomic_json(progress_path, progress)
        print(f'Exact grid already recorded ({len(done)} runs); no model initialization needed', flush=True)
        return
    started = time.monotonic()
    generator = F5Generator(args.checkpoint, args.vocab, args.vocoder_dir,
                            max_duration_seconds=args.max_duration_seconds)
    for item, ref in grid:
        run_id = f"{args.experiment}__{item['text_id']}__{ref['voice_id']}__s{args.seed}"
        if run_id in done:
            continue
        if args.max_wall_seconds and time.monotonic()-started >= args.max_wall_seconds:
            break
        # Filename does not depend on arbitrary text IDs or user text.
        filename = hashlib.sha256(run_id.encode()).hexdigest()[:24]+'.wav'
        result = generator.generate(prompt_audio=ref['wav_path'], prompt_text=ref['ref_text'],
                                    text=item['text_tts'], output_path=str(out/filename),
                                    seed=args.seed, mode=args.mode)
        result.update(run_id=run_id, experiment_id=args.experiment, text_id=item['text_id'],
                      voice_id=ref['voice_id'], bucket=item['bucket'],
                      text_sha256=hashlib.sha256(item['text_tts'].encode()).hexdigest(),
                      reference_wav=ref['wav_path'])
        append_jsonl(runfile, result)
        done.add(run_id)
        print({k:result.get(k) for k in ['text_id','voice_id','gen_status','raw_duration_sec','elapsed','error']}, flush=True)
    atomic_json(out/'progress.json', dict(planned=len(grid), attempted=len(done),
                                         completed_grid=len(done)==len(grid),
                                         elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
