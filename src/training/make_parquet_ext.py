#!/usr/bin/env python3
"""A1-ext / E6 — parquet shards for the `ext_short` corpus (stock CosyVoice layout).

Same columns, same shard layout and the same resume/atomic-rename contract as
src/training/make_parquet_arms.py, whose Tokenizer / flac_bytes / write_shard / verify_shard /
finish_lists / read_dir are IMPORTED here (that file is not modified and nothing is copy-pasted):

    utt, audio_data (FLAC 24 kHz mono 16-bit), wav, text, spk, instruct,
    speech_token (int32 list), n_speech_token, duration, parent, offset_start, offset_end

so `cosyvoice.dataset.processor.parquet_opener` and src/training/processor_ext.py read it
unchanged.

Difference from the arms build: every clip IS its own parent and has exactly one window
[0, dur] (<= 20 s), so there is no long/short pair — one output dir, one row per clip. Each
clip is decoded once: 48 kHz mono 24-bit FLAC -> 24 kHz (storage) and 16 kHz -> whisper
128-mel -> speech_tokenizer_v3.batch.onnx, with the stock `tok[:T//4]` rule inside Tokenizer.

`duration` is the DECODED length of the clip, not the parquet metadata `total_duration`
(they differ by up to ~5 ms); the deviation is measured and reported in parquet_summary.json.

Per row the build asserts |n_speech_token - duration * 25| <= 13; violations are COUNTED and
listed (never dropped silently, PLAN §4.2).

Usage:
    CUDA_VISIBLE_DEVICES=1 python src/training/make_parquet_ext.py --workers 2 --device_id 0
    ... --limit 50            # debug: only the first 50 clips
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
import torch
import torchaudio

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.training.make_parquet_arms import (Tokenizer, flac_bytes, write_shard,   # noqa: E402
                                            verify_shard, finish_lists, read_dir, read_kv)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
torch.set_num_threads(1)

TOKEN_RATE = 25.0
TOKEN_TOL = 13          # |n_speech_token - duration * 25| <= 13


def decode_clip(job):
    """CPU work in a worker process: decode -> 24 kHz FLAC bytes + 16 kHz whisper 128-mel."""
    import whisper  # noqa: imported in the worker so the parent never holds the model state
    utt, path = job
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = torch.from_numpy(x.T).mean(dim=0, keepdim=True)
    x24 = torchaudio.functional.resample(x, sr, 24000)
    x16 = torchaudio.functional.resample(x, sr, 16000)
    assert x16.shape[1] / 16000 <= 30.0 + 1e-3, (utt, x16.shape[1] / 16000)
    feat = whisper.log_mel_spectrogram(x16, n_mels=128)[0].numpy().astype(np.float32)
    return utt, flac_bytes(x24[0].numpy()), x24.shape[1] / 24000, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/train/ext_short')
    ap.add_argument('--clips_per_shard', type=int, default=400)
    ap.add_argument('--tokenize_chunk', type=int, default=64, help='decoded clips held in RAM before a tokenizer call')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--device_id', type=int, default=0, help='index inside CUDA_VISIBLE_DEVICES')
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N clips')
    ap.add_argument('--no_resume', action='store_true')
    args = ap.parse_args()

    D = read_dir(args.dir)
    D['dur_meta'] = read_kv(os.path.join(args.dir, 'utt2dur'))
    utts = list(D['wav'].keys())
    if args.limit:
        utts = utts[:args.limit]
    for u in utts:
        parent, a, b = D['off'][u]
        assert parent == u and a == 0.0, (u, parent, a)
    des = os.path.join(args.dir, 'parquet')
    P = args.clips_per_shard
    n_shards = (len(utts) + P - 1) // P

    plan, n_skip = [], 0
    for k in range(n_shards):
        uk = utts[k * P:(k + 1) * P]
        t = verify_shard(des, k, uk) if not args.no_resume else False
        plan.append((k, uk, t if t is not False else None))
        n_skip += t is not False
    logging.info('clips %d, shards total %d, complete on disk (skipped) %d, to build %d',
                 len(utts), n_shards, n_skip, n_shards - n_skip)

    tokenizer = Tokenizer(args.device_id) if n_skip < n_shards else None
    manifest = open(os.path.join(args.dir, 'token_manifest.jsonl'), 'w', encoding='utf-8')
    t0 = time.time()
    n_done, sec_done, nbytes, n_built = 0, 0.0, 0, 0
    violations, dur_dev = [], []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, uk, done in plan:
            if done is not None:
                for u, n, d in zip(*[done.column(c).to_pylist() for c in ('utt', 'n_speech_token', 'duration')]):
                    manifest.write(json.dumps({'utt': u, 'n_speech_token': int(n), 'duration': float(d),
                                               'from_existing_shard': k}) + '\n')
                    sec_done += float(d)
                    if abs(int(n) - float(d) * TOKEN_RATE) > TOKEN_TOL:
                        violations.append({'utt': u, 'n_speech_token': int(n), 'duration': float(d)})
                n_done += len(uk)
                nbytes += os.path.getsize(os.path.join(des, 'parquet_{:09d}.tar'.format(k)))
                continue
            rows, buf = [], []

            def flush(buf):
                if not buf:
                    return
                toks = tokenizer([b[3] for b in buf])
                for (utt, wbytes, dur, _feat), tk in zip(buf, toks):
                    n = int(len(tk))
                    rows.append(dict(utt=utt, audio_data=wbytes, wav=D['wav'][utt], text=D['text'][utt],
                                     spk=D['spk'][utt], instruct=D['instruct'][utt],
                                     speech_token=tk.tolist(), n_speech_token=n, duration=float(dur),
                                     parent=utt, offset_start=0.0, offset_end=float(dur)))
                    manifest.write(json.dumps({'utt': utt, 'n_speech_token': n, 'duration': float(dur)}) + '\n')
                    if abs(n - dur * TOKEN_RATE) > TOKEN_TOL:
                        violations.append({'utt': utt, 'n_speech_token': n, 'duration': float(dur)})
                buf.clear()

            for utt, wbytes, dur, feat in pool.map(decode_clip, [(u, D['wav'][u]) for u in uk], chunksize=4):
                nbytes += len(wbytes)
                sec_done += dur
                dur_dev.append(dur - float(D['dur_meta'][utt]))
                buf.append((utt, wbytes, dur, feat))
                n_done += 1
                if len(buf) >= args.tokenize_chunk:
                    flush(buf)
            flush(buf)
            write_shard(rows, des, k, D['spk'])
            n_built += len(rows)
            manifest.flush()
            if (k + 1) % 10 == 0 or k == n_shards - 1:
                el = time.time() - t0
                rate = n_done / el if el else 0
                logging.info('shard %d/%d, %d/%d clips (%d built this run), %.2f h audio, %.0f s elapsed, '
                             '%.1f clips/s, eta %.0f s, %.2f GB', k + 1, n_shards, n_done, len(utts), n_built,
                             sec_done / 3600, el, rate, (len(utts) - n_done) / rate if rate else -1, nbytes / 1e9)
    finish_lists(des, n_shards)
    manifest.close()
    dev = np.array(dur_dev) if dur_dev else np.zeros(1)
    summary = dict(clips=n_done, clips_built_this_run=n_built, shards=n_shards, shards_skipped=n_skip,
                   hours=sec_done / 3600, audio_bytes=nbytes, elapsed_sec=time.time() - t0,
                   token_rate_hz=TOKEN_RATE, token_tolerance=TOKEN_TOL,
                   n_token_violations=len(violations), token_violations=violations[:50],
                   decoded_minus_metadata_duration_sec={'max_abs': float(np.abs(dev).max()),
                                                        'mean': float(dev.mean()),
                                                        'n': int(len(dur_dev))},
                   data_list=os.path.join(des, 'data.list'))
    with open(os.path.join(args.dir, 'parquet_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info('done %s', json.dumps({k: v for k, v in summary.items() if k != 'token_violations'}))
    if violations:
        logging.warning('%d rows violate |n_speech_token - duration*25| <= %d (listed in parquet_summary.json)',
                        len(violations), TOKEN_TOL)


if __name__ == '__main__':
    main()
