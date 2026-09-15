#!/usr/bin/env python3
"""Parquet stage for the Long/Short arms (replacement for ~/CosyVoice/tools/make_parquet_list.py).

Differences from the stock tool (which just copies the source file bytes):
  * decodes each PARENT recording once (48 kHz stereo 24-bit FLAC) -> mono -> 24 kHz (storage, the
    training sample rate) and 16 kHz (speech tokenizer input);
  * short windows are sliced by utt2offset and stored as their own FLAC bytes (the local
    hf_dataset/processor changes do not support offsets);
  * speech tokens are extracted OFFLINE with speech_tokenizer_v3.batch.onnx per window (every
    window <= 30 s, the regime the repo's own tools enforce and the regime of inference prompts);
    the long unit's tokens are the concatenation of its windows' tokens in time order, so both
    arms train on IDENTICAL target token sequences (bit-for-bit, verified in the subset run);
  * stores columns utt, audio_data (FLAC 24 kHz mono 16-bit), wav, text, spk, instruct,
    speech_token (int32 list), n_speech_token, duration, parent, offset_start, offset_end.

Output layout is the stock one (parquet_XXXXXXXXX.tar + utt2parquet/spk2parquet json + data.list)
so cosyvoice.dataset.processor.parquet_opener reads it unchanged.

RESUMABLE (A6, 2026-08-28): shard k always covers parents[k*P:(k+1)*P] in wav.scp order, so a
re-run verifies every existing shard pair (both tars parse, row sets == expected utts, json side
files present) and skips complete ones; incomplete/corrupt shards are rebuilt. Shards are written
to a temp name and renamed (atomic), and token_manifest.jsonl is rebuilt from the shards on disk.

Usage (both arms of one parent set in one pass):
  python src/training/make_parquet_arms.py --long_dir data/train/sub2_long --short_dir data/train/sub2_short \
      --parents_per_shard 50 --workers 2
"""
import argparse
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

COSYVOICE_ROOT = os.environ.get('COSYVOICE_ROOT', 'third_party/CosyVoice')
for p in (COSYVOICE_ROOT, os.path.join(COSYVOICE_ROOT, 'third_party', 'Matcha-TTS')):
    if p not in sys.path:
        sys.path.insert(0, p)
MODEL_DIR = os.environ.get('COSYVOICE_MODEL_DIR', 'models/cosyvoice3')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
torch.set_num_threads(1)


def read_kv(path):
    d = {}
    with open(path, encoding='utf-8') as f:
        for l in f:
            l = l.rstrip('\n').split(' ', 1)
            d[l[0]] = l[1] if len(l) > 1 else ''
    return d


def read_dir(d):
    utt2wav = read_kv(os.path.join(d, 'wav.scp'))
    utt2text = read_kv(os.path.join(d, 'text'))
    utt2spk = read_kv(os.path.join(d, 'utt2spk'))
    utt2instruct = read_kv(os.path.join(d, 'instruct')) if os.path.exists(os.path.join(d, 'instruct')) else None
    utt2off = {}
    for k, v in read_kv(os.path.join(d, 'utt2offset')).items():
        parent, a, b = v.split()
        utt2off[k] = (parent, float(a), float(b))
    return dict(wav=utt2wav, text=utt2text, spk=utt2spk, instruct=utt2instruct, off=utt2off)


def flac_bytes(x24, sr=24000):
    buf = io.BytesIO()
    sf.write(buf, x24, sr, format='FLAC', subtype='PCM_16')
    return buf.getvalue()


def decode_parent(job):
    """CPU work: decode + resample + slice + flac-encode + whisper mel. Runs in a worker process."""
    import whisper  # noqa (imported here so workers do not need the main process state)
    parent, path, windows = job   # windows: list of (utt, off0, off1)
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = torch.from_numpy(x.T).mean(dim=0, keepdim=True)
    x24 = torchaudio.functional.resample(x, sr, 24000)
    x16 = torchaudio.functional.resample(x, sr, 16000)
    out_windows = []
    for utt, a, b in sorted(windows, key=lambda w: w[1]):
        s24, e24 = int(round(a * 24000)), int(round(b * 24000))
        s16, e16 = int(round(a * 16000)), int(round(b * 16000))
        seg16 = x16[:, s16:e16]
        assert seg16.shape[1] / 16000 <= 30.0 + 1e-3, (utt, seg16.shape[1] / 16000)
        feat = whisper.log_mel_spectrogram(seg16, n_mels=128)[0].numpy().astype(np.float32)   # [128, T]
        out_windows.append((utt, a, b, flac_bytes(x24[0, s24:e24].numpy()), feat))
    return parent, flac_bytes(x24[0].numpy()), x24.shape[1] / 24000, out_windows


class Tokenizer:
    def __init__(self, device_id=0):
        import onnxruntime
        opt = onnxruntime.SessionOptions()
        opt.intra_op_num_threads = 1
        opt.log_severity_level = 3
        self.sess = onnxruntime.InferenceSession(os.path.join(MODEL_DIR, 'speech_tokenizer_v3.batch.onnx'), sess_options=opt,
                                                 providers=[('CUDAExecutionProvider', {'device_id': device_id})])
        assert 'CUDAExecutionProvider' in self.sess.get_providers(), self.sess.get_providers()

    def __call__(self, feats, max_batch=16):
        """feats: list of [128, T] -> list of int32 arrays of length floor(T/4) (stock SpeechTokenExtractor rule)."""
        out = [None] * len(feats)
        order = sorted(range(len(feats)), key=lambda i: feats[i].shape[1])
        for i0 in range(0, len(order), max_batch):
            idx = order[i0:i0 + max_batch]
            lens = np.array([feats[i].shape[1] for i in idx], dtype=np.int32)
            batch = np.zeros((len(idx), 128, int(lens.max())), dtype=np.float32)
            for j, i in enumerate(idx):
                batch[j, :, :lens[j]] = feats[i]
            tok = self.sess.run(None, {'feats': batch, 'feats_length': lens})[0]
            for j, i in enumerate(idx):
                out[i] = tok[j, :lens[j] // 4].astype(np.int32)
        return out


def write_shard(rows, des_dir, shard_idx, utt2spk):
    os.makedirs(des_dir, exist_ok=True)
    parquet_file = os.path.join(des_dir, 'parquet_{:09d}.tar'.format(shard_idx))
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_file + '.tmp')
    with open(os.path.join(des_dir, 'utt2parquet_{:09d}.json'.format(shard_idx)), 'w') as f:
        json.dump({u: parquet_file for u in df['utt']}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(des_dir, 'spk2parquet_{:09d}.json'.format(shard_idx)), 'w') as f:
        json.dump({utt2spk[u]: parquet_file for u in df['utt']}, f, ensure_ascii=False, indent=2)
    os.replace(parquet_file + '.tmp', parquet_file)   # atomic: a killed run never leaves a truncated .tar
    return parquet_file


def verify_shard(des_dir, shard_idx, expected_utts):
    """True iff the shard exists, parses, has exactly expected_utts and both json side files. Never raises."""
    import pyarrow.parquet as pq
    tar = os.path.join(des_dir, 'parquet_{:09d}.tar'.format(shard_idx))
    j1 = os.path.join(des_dir, 'utt2parquet_{:09d}.json'.format(shard_idx))
    j2 = os.path.join(des_dir, 'spk2parquet_{:09d}.json'.format(shard_idx))
    try:
        if not (os.path.exists(tar) and os.path.exists(j1) and os.path.exists(j2)):
            return False
        t = pq.read_table(tar, columns=['utt', 'n_speech_token', 'duration', 'parent'])
        utts = t.column('utt').to_pylist()
        if set(utts) != set(expected_utts) or len(utts) != len(expected_utts):
            return False
        if set(json.load(open(j1)).keys()) != set(expected_utts):
            return False
        return t
    except Exception as e:  # noqa
        logging.warning('shard %d in %s unreadable (%s) -> rebuild', shard_idx, des_dir, e)
        return False


def finish_lists(des_dir, n_shards):
    with open(os.path.join(des_dir, 'data.list'), 'w') as f1, open(os.path.join(des_dir, 'utt2data.list'), 'w') as f2, \
            open(os.path.join(des_dir, 'spk2data.list'), 'w') as f3:
        for i in range(n_shards):
            f1.write(os.path.join(des_dir, 'parquet_{:09d}.tar'.format(i)) + '\n')
            f2.write(os.path.join(des_dir, 'utt2parquet_{:09d}.json'.format(i)) + '\n')
            f3.write(os.path.join(des_dir, 'spk2parquet_{:09d}.json'.format(i)) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--long_dir', required=True)
    ap.add_argument('--short_dir', required=True)
    ap.add_argument('--parents_per_shard', type=int, default=50)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--device_id', type=int, default=0, help='index inside CUDA_VISIBLE_DEVICES')
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N parents')
    ap.add_argument('--no_resume', action='store_true', help='rebuild every shard even if complete on disk')
    args = ap.parse_args()

    L, S = read_dir(args.long_dir), read_dir(args.short_dir)
    parents = list(L['wav'].keys())
    if args.limit:
        parents = parents[:args.limit]
    # windows_all (written by prepare_arms) lists EVERY window of each kept parent, incl. windows excluded
    # from the short arm (flag 0): the long unit is tokenized over all of them, short rows only for flag 1.
    parent2windows, short_included = {}, set()
    with open(os.path.join(args.long_dir, 'windows_all'), encoding='utf-8') as f:
        for l in f:
            utt, parent, a, b, inc = l.split()
            parent2windows.setdefault(parent, []).append((utt, float(a), float(b)))
            if inc == '1':
                short_included.add(utt)
    assert short_included == set(S['off'].keys()), 'short dir and windows_all disagree'
    assert set(parents) <= set(parent2windows), 'every long unit needs windows'
    long_pq, short_pq = os.path.join(args.long_dir, 'parquet'), os.path.join(args.short_dir, 'parquet')
    P = args.parents_per_shard
    n_shards = (len(parents) + P - 1) // P
    # ---- resume: verify existing shard pairs ----
    plan = []            # (shard_idx, parents_k, complete: bool)
    n_skip = 0
    for k in range(n_shards):
        pk = parents[k * P:(k + 1) * P]
        exp_short = [w[0] for p in pk for w in parent2windows[p] if w[0] in short_included]
        tl = verify_shard(long_pq, k, pk) if not args.no_resume else False
        ts = verify_shard(short_pq, k, exp_short) if not args.no_resume else False
        ok = (tl is not False) and (ts is not False)
        plan.append((k, pk, (tl, ts) if ok else None))
        n_skip += ok
    logging.info('shards total %d, complete on disk (skipped) %d, to build %d', n_shards, n_skip, n_shards - n_skip)
    tokenizer = Tokenizer(args.device_id) if n_skip < n_shards else None
    manifest_long = open(os.path.join(args.long_dir, 'token_manifest.jsonl'), 'w', encoding='utf-8')
    manifest_short = open(os.path.join(args.short_dir, 'token_manifest.jsonl'), 'w', encoding='utf-8')
    t0 = time.time()
    n_done, sec_done, bytes_l, bytes_s, n_built = 0, 0.0, 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, pk, done in plan:
            if done is not None:
                tl, ts = done
                for u, n, d, _ in zip(*[tl.column(c).to_pylist() for c in ('utt', 'n_speech_token', 'duration', 'parent')]):
                    manifest_long.write(json.dumps({'utt': u, 'n_speech_token': int(n), 'duration': float(d),
                                                    'n_windows': len(parent2windows[u]), 'from_existing_shard': k}) + '\n')
                    sec_done += float(d)
                for u, n, d, _ in zip(*[ts.column(c).to_pylist() for c in ('utt', 'n_speech_token', 'duration', 'parent')]):
                    manifest_short.write(json.dumps({'utt': u, 'n_speech_token': int(n), 'duration': float(d)}) + '\n')
                n_done += len(pk)
                bytes_l += os.path.getsize(os.path.join(long_pq, 'parquet_{:09d}.tar'.format(k)))
                bytes_s += os.path.getsize(os.path.join(short_pq, 'parquet_{:09d}.tar'.format(k)))
                continue
            rows_l, rows_s = [], []
            jobs = [(p, L['wav'][p], parent2windows[p]) for p in pk]
            for parent, pbytes, pdur, windows in pool.map(decode_parent, jobs, chunksize=1):
                toks = tokenizer([w[4] for w in windows])
                long_tokens = np.concatenate(toks) if toks else np.zeros(0, dtype=np.int32)
                for (utt, a, b, wbytes, _), tk in zip(windows, toks):
                    if utt not in short_included:
                        continue
                    rows_s.append(dict(utt=utt, audio_data=wbytes, wav=L['wav'][parent], text=S['text'][utt], spk=S['spk'][utt],
                                       instruct=S['instruct'][utt], speech_token=tk.tolist(), n_speech_token=int(len(tk)),
                                       duration=float(b - a), parent=parent, offset_start=float(a), offset_end=float(b)))
                    bytes_s += len(wbytes)
                    manifest_short.write(json.dumps({'utt': utt, 'n_speech_token': int(len(tk)), 'duration': float(b - a)}) + '\n')
                rows_l.append(dict(utt=parent, audio_data=pbytes, wav=L['wav'][parent], text=L['text'][parent], spk=L['spk'][parent],
                                   instruct=L['instruct'][parent], speech_token=long_tokens.tolist(), n_speech_token=int(len(long_tokens)),
                                   duration=float(pdur), parent=parent, offset_start=0.0, offset_end=float(pdur)))
                bytes_l += len(pbytes)
                assert abs(len(long_tokens) - pdur * 25) < 25 * 0.5 + len(windows), (parent, len(long_tokens), pdur)
                manifest_long.write(json.dumps({'utt': parent, 'n_speech_token': int(len(long_tokens)), 'duration': float(pdur),
                                                'n_windows': len(windows)}) + '\n')
                n_done += 1
                sec_done += pdur
                if n_done % 10 == 0 or n_done == len(parents):
                    el = time.time() - t0
                    logging.info('%d/%d parents (%d built this run), %.1f h audio, %.0f s elapsed, long %.2f GB short %.2f GB',
                                 n_done, len(parents), n_built + len(rows_l), sec_done / 3600, el, bytes_l / 1e9, bytes_s / 1e9)
            write_shard(rows_l, long_pq, k, L['spk'])
            write_shard(rows_s, short_pq, k, S['spk'])
            n_built += len(rows_l)
            manifest_long.flush()
            manifest_short.flush()
    shard = n_shards
    finish_lists(long_pq, shard)
    finish_lists(short_pq, shard)
    manifest_long.close()
    manifest_short.close()
    summary = dict(parents=n_done, parents_built_this_run=n_built, shards=shard, shards_skipped=n_skip, hours=sec_done / 3600,
                   long_audio_bytes=bytes_l, short_audio_bytes=bytes_s, elapsed_sec=time.time() - t0)
    with open(os.path.join(args.long_dir, 'parquet_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info('done %s', json.dumps(summary))


if __name__ == '__main__':
    main()
