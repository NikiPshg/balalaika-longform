#!/usr/bin/env python
"""A22-voxcpm (E12 P2): build the VoxCPM training manifests for both arms.

Sources (read-only; same units bit-for-bit as the E7/Q-series arms):
  * long arm  VC-E3P: data/train/long_punct/text (6041 punctuated 15-min-unit texts)
              + data/train/long/{wav.scp,utt2dur,utt2offset} (all offsets are 0..dur,
                i.e. whole source flacs on data/corpus)
  * short arm VC-E2P: data/train/short_punct/text (28427 punctuated window texts)
              + data/train/short/{wav.scp,utt2dur,utt2offset} (windows are
                [offset_start, offset_end) slices of the same parent flacs)
  * dev holdouts: data/train/dev_{long,short}_punct/text + dev_{long,short} kaldi files
                (first N_DEV utts in file order, mirroring the Q-series 32-unit holdout)

Output rows (data/train/voxcpm/{vce3p,vce2p,dev_long,dev_short}.jsonl) carry the
official train_voxcpm_finetune.py columns plus the offset columns our local
data_offsets loader consumes:
    {"utt", "text", "audio", "duration", "offset_start", "offset_end", "parent"}
`duration` keeps compute_sample_lengths() cheap (without it the official filter
decodes every audio file at startup). No ref_audio column: the official plain-TTS row
layout [text, 101, audio, 102] is exactly the inference continuation layout the
Ultimate-Cloning adapter path uses.

The 8192-position packed-length distribution (the official max_batch_tokens filter,
train script lines 138-156) is reported per arm; rows are NOT pre-filtered here --
the official mechanism does that at train start, keeping the manifests bit-complete.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_DEV = 32   # mirror of the Q-series dev holdout size


def read_kaldi_map(path, n_fields=None):
    out = {}
    with open(path, encoding='utf-8') as f:
        for ln in f:
            parts = ln.rstrip('\n').split(' ', 1) if n_fields is None else ln.split()
            out[parts[0]] = parts[1] if n_fields is None else parts[1:]
    return out


def build_arm(kind: str, punct_kind: str, out_path: str, limit: int | None = None):
    base = os.path.join(ROOT, 'data', 'train')
    texts = read_kaldi_map(os.path.join(base, punct_kind, 'text'))
    wav = read_kaldi_map(os.path.join(base, kind, 'wav.scp'), n_fields=1)
    dur = read_kaldi_map(os.path.join(base, kind, 'utt2dur'), n_fields=1)
    off = read_kaldi_map(os.path.join(base, kind, 'utt2offset'), n_fields=1)

    rows = []
    missing = []
    for utt, text in texts.items():
        if utt not in wav or utt not in dur or utt not in off:
            missing.append(utt)
            continue
        parent, o0, o1 = off[utt][0], float(off[utt][1]), float(off[utt][2])
        rows.append({
            'utt': utt,
            'text': text,
            'audio': wav[utt][0],
            'duration': float(dur[utt][0]),
            'offset_start': o0,
            'offset_end': o1,
            'parent': parent,
        })
    if missing:
        raise SystemExit('{}: {} utts in {} text lack kaldi entries, e.g. {}'.format(
            kind, len(missing), punct_kind, missing[:3]))
    rows.sort(key=lambda r: r['utt'])
    if limit:
        rows = rows[:limit]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    total_s = sum(r['duration'] for r in rows)
    print('{} -> {}: rows={} seconds={:.0f} hours={:.2f}'.format(
        punct_kind, out_path, len(rows), total_s, total_s / 3600))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'data', 'train', 'voxcpm'))
    args = ap.parse_args()

    build_arm('long', 'long_punct', os.path.join(args.out_dir, 'vce3p.jsonl'))
    build_arm('short', 'short_punct', os.path.join(args.out_dir, 'vce2p.jsonl'))
    build_arm('dev_long', 'dev_long_punct', os.path.join(args.out_dir, 'dev_long.jsonl'), limit=N_DEV)
    build_arm('dev_short', 'dev_short_punct', os.path.join(args.out_dir, 'dev_short.jsonl'), limit=N_DEV)


if __name__ == '__main__':
    main()
