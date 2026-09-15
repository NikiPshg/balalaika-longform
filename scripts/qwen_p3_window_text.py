#!/usr/bin/env python
"""A10-qwen P3: punctuated (text_e2e-style) texts for the v3.1 SHORT windows (Q-E2P).

The short windows were cut ON `asr_ts['gigaam-v3-e2e-ctc']` word-group boundaries
(src/data/make_short_arm.py:7-9 — DP over inter-group silence + sentence punctuation),
and the group timestamps live in the same coordinate system as data/train/*/utt2offset
(seconds inside the parent segment flac). A window's punctuated text is therefore the
verbatim concatenation of the e2e groups inside its [start, end) — the same recipe A5
used for reference ref_text and A6 used for the E7 long_punct arm.

Usage (CPU, any python3):
  python3 scripts/qwen_p3_window_text.py --set data/train/short --out data/train/short_punct
  python3 scripts/qwen_p3_window_text.py --set data/train/dev_short --out data/train/dev_short_punct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

EPS = 0.05  # boundary tolerance (windows sit exactly on group boundaries)
E2E_KEY = 'gigaam-v3-e2e-ctc'
WORD = re.compile(r"\w+", re.UNICODE)


def parse_ts(ts: str):
    out = []
    for ln in ts.split('\n'):
        p = ln.split('\t')
        if len(p) != 3:
            continue
        try:
            a, b = float(p[0]), float(p[1])
        except ValueError:
            continue
        if p[2].strip():
            out.append((a, b, p[2].strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, help='data/train/short or data/train/dev_short')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-word-ratio', type=float, default=0.8,
                    help='warn when punct words / rover words falls below this')
    args = ap.parse_args()

    wav_scp = {u: p for u, p in (l.split(None, 1) for l in open(os.path.join(args.set, 'wav.scp')))}
    offsets = {}
    for l in open(os.path.join(args.set, 'utt2offset')):
        u, parent, a, b = l.split()
        offsets[u] = (parent, float(a), float(b))
    old_text = {u: t.strip() for u, t in (l.split(None, 1) for l in open(os.path.join(args.set, 'text')))}

    ts_cache: dict[str, list] = {}
    os.makedirs(args.out, exist_ok=True)
    n_ok = n_warn = n_fail = 0
    ratios = []
    with open(os.path.join(args.out, 'text'), 'w', encoding='utf-8') as fo, \
         open(os.path.join(args.out, 'build_warnings.jsonl'), 'w', encoding='utf-8') as fw:
        for u in old_text:
            parent, a, b = offsets[u]
            flac = wav_scp[u].strip()
            jpath = flac[:-5] + '.json'
            if jpath not in ts_cache:
                try:
                    d = json.load(open(jpath))
                    ts_cache[jpath] = parse_ts((d.get('asr_ts') or {}).get(E2E_KEY) or '')
                except Exception:
                    ts_cache[jpath] = []
            lines = ts_cache[jpath]
            groups = [t for (ga, gb, t) in lines if ga >= a - EPS and gb <= b + EPS]
            text = ' '.join(groups).strip()
            n_old = len(WORD.findall(old_text[u]))
            n_new = len(WORD.findall(text))
            ratio = (n_new / n_old) if n_old else 0.0
            ratios.append(ratio)
            if not text:
                n_fail += 1
                fw.write(json.dumps({'utt': u, 'error': 'no e2e groups in window',
                                     'n_rover_words': n_old}, ensure_ascii=False) + '\n')
                continue
            if ratio < args.min_word_ratio or ratio > 1.25:
                n_warn += 1
                fw.write(json.dumps({'utt': u, 'warn': 'word-count mismatch', 'ratio': round(ratio, 3),
                                     'n_rover_words': n_old, 'n_e2e_words': n_new}, ensure_ascii=False) + '\n')
            fo.write(f'{u} {text}\n')
            n_ok += 1
    ratios.sort()
    stats = {'kind': os.path.basename(args.out), 'built_from': args.set,
             'text_source': f"asr_ts['{E2E_KEY}'] word groups inside [start,end) of utt2offset (punctuated, verbatim)",
             'n_written': n_ok, 'n_word_ratio_warn': n_warn, 'n_no_groups': n_fail,
             'word_ratio_median': round(ratios[len(ratios)//2], 4) if ratios else None,
             'word_ratio_p10': round(ratios[int(0.1*len(ratios))], 4) if ratios else None}
    with open(os.path.join(args.out, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))
    if n_fail:
        print(f'WARNING: {n_fail} windows had no groups (listed in build_warnings.jsonl)', file=sys.stderr)


if __name__ == '__main__':
    main()
