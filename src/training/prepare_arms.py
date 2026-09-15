#!/usr/bin/env python3
"""Build CosyVoice-style data dirs for the Long (E3) and Short (E2) arms (PLAN §6.3, A6).

Input: A1 manifests (data/manifests/train_long.jsonl, train_short.jsonl, dev.jsonl) and the
dev short windows built with the same cutter (data/train/manifests/dev_short.jsonl).

Output per arm dir (data/train/<name>/): wav.scp, text, utt2spk, spk2utt, instruct,
utt2offset (utt parent offset_start offset_end; for long units the offsets span the whole file),
utt2dur, excluded.jsonl (every row that is NOT used and why), and stats.json.

Rules applied identically to both arms (recorded in reports/decisions.md):
  * training text = `text` (ROVER; A1 proposal confirmed by A6);
  * long rows with empty text are excluded from BOTH arms together with all their windows;
  * short windows with empty text are excluded (their audio stays inside the long unit -> the
    0.34 % asymmetry is reported, not hidden);
  * every row passes the context budget (src/training/context_budget.py); rows failing it are
    listed in excluded.jsonl (none at the moment).

The local hf_dataset/processor changes in ~/CosyVoice do NOT support offsets, so short windows
are sliced to FLAC bytes in the parquet stage (src/training/make_parquet_arms.py), which reads
utt2offset from these dirs.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.training.context_budget import INSTRUCT, check_manifest  # noqa: E402


def stable_frac(key):
    return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % 10_000 / 10_000.0


def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def write_dir(des, rows, kind):
    """rows: list of dicts with utt, wav, text, spk, parent, off0, off1, dur, n_speech_est."""
    os.makedirs(des, exist_ok=True)
    spk2utt = {}
    with open(os.path.join(des, 'wav.scp'), 'w', encoding='utf-8') as fw, \
            open(os.path.join(des, 'text'), 'w', encoding='utf-8') as ft, \
            open(os.path.join(des, 'utt2spk'), 'w', encoding='utf-8') as fs, \
            open(os.path.join(des, 'instruct'), 'w', encoding='utf-8') as fi, \
            open(os.path.join(des, 'utt2offset'), 'w', encoding='utf-8') as fo, \
            open(os.path.join(des, 'utt2dur'), 'w', encoding='utf-8') as fd:
        for r in rows:
            fw.write('{} {}\n'.format(r['utt'], r['wav']))
            ft.write('{} {}\n'.format(r['utt'], r['text']))
            fs.write('{} {}\n'.format(r['utt'], r['spk']))
            fi.write('{} {}\n'.format(r['utt'], INSTRUCT))
            fo.write('{} {} {:.3f} {:.3f}\n'.format(r['utt'], r['parent'], r['off0'], r['off1']))
            fd.write('{} {:.3f}\n'.format(r['utt'], r['dur']))
            spk2utt.setdefault(r['spk'], []).append(r['utt'])
    with open(os.path.join(des, 'spk2utt'), 'w', encoding='utf-8') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    stats = {'kind': kind, 'utts': len(rows), 'speakers': len(spk2utt), 'hours': sum(r['dur'] for r in rows) / 3600,
             'speech_tokens_est': sum(r['n_speech_est'] for r in rows), 'text_tokens': sum(r['n_text'] for r in rows),
             'parents': len({r['parent'] for r in rows})}
    return stats


def build(long_rows, short_rows, des_root, name_long, name_short, subset_frac=None, seed_key='a6'):
    # 1. subset by parent (same parents in both arms)
    if subset_frac is not None:
        keep = {r['sample_id'] for r in long_rows if stable_frac(seed_key + r['sample_id']) < subset_frac}
        long_rows = [r for r in long_rows if r['sample_id'] in keep]
        short_rows = [r for r in short_rows if r['parent_sample_id'] in keep]
    # 2. budgets
    bl = {b['sample_id']: b for b in check_manifest(long_rows, 'duration_sec')}
    bs = {b['sample_id']: b for b in check_manifest(short_rows, 'duration')}
    excluded_long, excluded_short = [], []
    empty_parents = set()
    for r in long_rows:
        b = bl[r['sample_id']]
        if b['empty_text']:
            empty_parents.add(r['sample_id'])
            excluded_long.append({'sample_id': r['sample_id'], 'reason': 'empty_text', 'duration_sec': r['duration_sec']})
        elif not b['fits']:
            excluded_long.append({'sample_id': r['sample_id'], 'reason': 'context_budget', **b})
    out_long, out_short = [], []
    for r in long_rows:
        b = bl[r['sample_id']]
        if b['empty_text'] or not b['fits']:
            continue
        out_long.append({'utt': r['sample_id'], 'wav': r['audio_path'], 'text': r['text'].strip(), 'spk': r['speaker_key'],
                         'parent': r['sample_id'], 'off0': 0.0, 'off1': float(r['duration_sec']), 'dur': float(r['duration_sec']),
                         'n_speech_est': b['n_speech'], 'n_text': b['n_text']})
    for r in short_rows:
        b = bs[r['sample_id']]
        if r['parent_sample_id'] in empty_parents:
            excluded_short.append({'sample_id': r['sample_id'], 'reason': 'parent_empty_text', 'duration': r['duration']})
        elif b['empty_text']:
            excluded_short.append({'sample_id': r['sample_id'], 'reason': 'empty_text', 'duration': r['duration'],
                                   'parent_sample_id': r['parent_sample_id']})
        elif not b['fits']:
            excluded_short.append({'sample_id': r['sample_id'], 'reason': 'context_budget', **b})
        else:
            out_short.append({'utt': r['sample_id'], 'wav': r['audio_path'], 'text': r['text'].strip(), 'spk': r['speaker_key'],
                              'parent': r['parent_sample_id'], 'off0': float(r['offset_start']), 'off1': float(r['offset_end']),
                              'dur': float(r['duration']), 'n_speech_est': b['n_speech'], 'n_text': b['n_text']})
    # parents covered must be identical
    assert {r['parent'] for r in out_short} == {r['parent'] for r in out_long}, 'arms must share the same parent set'
    stats = {}
    included_short = {r['utt'] for r in out_short}
    kept_parents = {r['parent'] for r in out_long}
    for name, rows, exc, kind in ((name_long, out_long, excluded_long, 'long'), (name_short, out_short, excluded_short, 'short')):
        des = os.path.join(des_root, name)
        s = write_dir(des, rows, kind)
        if kind == 'long':
            # ALL windows of every kept parent (excluded empty-text windows included, flag 0): the parquet
            # stage tokenizes the long unit over its full audio and emits short rows only for flag 1.
            with open(os.path.join(des, 'windows_all'), 'w', encoding='utf-8') as f:
                for r in sorted(short_rows, key=lambda r: (r['parent_sample_id'], r['offset_start'])):
                    if r['parent_sample_id'] in kept_parents:
                        f.write('{} {} {:.3f} {:.3f} {}\n'.format(r['sample_id'], r['parent_sample_id'], r['offset_start'],
                                                                 r['offset_end'], int(r['sample_id'] in included_short)))
        s['excluded'] = len(exc)
        s['excluded_hours'] = sum(e.get('duration_sec', e.get('duration', 0.0)) for e in exc) / 3600
        with open(os.path.join(des, 'excluded.jsonl'), 'w', encoding='utf-8') as f:
            for e in exc:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        with open(os.path.join(des, 'stats.json'), 'w', encoding='utf-8') as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        stats[name] = s
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_long', default='data/manifests/train_long.jsonl')
    ap.add_argument('--train_short', default='data/manifests/train_short.jsonl')
    ap.add_argument('--dev_long', default='data/manifests/dev.jsonl')
    ap.add_argument('--dev_short', default='data/train/manifests/dev_short.jsonl')
    ap.add_argument('--des_root', default='data/train')
    ap.add_argument('--subset_frac', type=float, default=None, help='e.g. 0.02 -> deterministic 2 %% of parents')
    ap.add_argument('--dev_subset_frac', type=float, default=None)
    ap.add_argument('--prefix', default='', help='dir name prefix, e.g. sub2_')
    args = ap.parse_args()
    stats = {}
    stats.update(build(load(args.train_long), load(args.train_short), args.des_root,
                       args.prefix + 'long', args.prefix + 'short', args.subset_frac))
    stats.update(build(load(args.dev_long), load(args.dev_short), args.des_root,
                       args.prefix + 'dev_long', args.prefix + 'dev_short', args.dev_subset_frac, seed_key='a6dev'))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
