#!/usr/bin/env python3
"""Build F5 Short/Long views from existing punctuated, channel-split corpus.

Only parents represented in BOTH views are retained. An optional duration ceiling
applies to the original parent in both arms, never just to Long. No audio is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_map(path):
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1)
        if key in result:
            raise ValueError(f"Duplicate {key} in {path}")
        result[key] = value
    return result


def build_rows(kind, text_kind):
    base = ROOT / 'data/train' / kind
    sources = [ROOT / 'data/train' / text_kind / 'text']
    sources += [base / name for name in ['wav.scp', 'utt2offset', 'utt2dur']]
    texts, audio, offsets, durations = [read_map(p) for p in sources]
    rows = []
    for utt, text in sorted(texts.items()):
        parent, start, end = offsets[utt].split()
        start, end, duration = float(start), float(end), float(durations[utt])
        if start < 0 or end <= start or abs(end - start - duration) > 0.025:
            raise ValueError(f"Invalid duration/offset: {utt}")
        if not Path(audio[utt]).is_file():
            raise ValueError(f"Missing audio: {audio[utt]}")
        if not text.strip():
            raise ValueError(f"Empty text: {utt}")
        rows.append(dict(utt=utt, parent=parent, text=text, audio=audio[utt],
                         duration=duration, offset_start=start, offset_end=end))
    return rows, sources


def paired_views(long, short, ceiling):
    parents = {r['parent']: r for r in long}
    assert len(parents) == len(long)
    windows = defaultdict(list)
    for row in short:
        windows[row['parent']].append(row)
    if set(windows) - set(parents):
        raise ValueError('Short contains parents missing from Long')
    eligible = {p for p in parents if p in windows and parents[p]['duration'] <= ceiling}
    excluded = [dict(parent=p, duration=r['duration'],
                     reason='no_short_text' if p not in windows else 'parent_duration_ceiling')
                for p, r in parents.items() if p not in eligible]
    for parent in eligible:
        previous_end = parents[parent]['offset_start']
        for row in sorted(windows[parent], key=lambda r: r['offset_start']):
            if row['audio'] != parents[parent]['audio']:
                raise ValueError(f'Audio mismatch: {parent}')
            if row['offset_start'] < previous_end - 0.025:
                raise ValueError(f'Overlapping windows: {parent}')
            if row['offset_end'] > parents[parent]['offset_end'] + 0.025:
                raise ValueError(f'Window outside parent: {parent}')
            previous_end = row['offset_end']
    return ([r for r in long if r['parent'] in eligible],
            [r for r in short if r['parent'] in eligible], excluded)


def save_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'data/train/f5')
    parser.add_argument('--max-parent-duration', type=float, default=900.0)
    args = parser.parse_args()
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Refusing to overwrite prepared manifests: {out}')
    out.mkdir(parents=True, exist_ok=True)
    all_sources, splits, stats = [], {}, {}
    for split, prefix in [('train', ''), ('dev', 'dev_')]:
        long, ls = build_rows(prefix+'long', prefix+'long_punct')
        short, ss = build_rows(prefix+'short', prefix+'short_punct')
        long, short, excluded = paired_views(long, short, args.max_parent_duration)
        splits[split] = {r['parent'] for r in long}
        assert splits[split] == {r['parent'] for r in short}
        for arm, rows in [('long', long), ('short', short), ('excluded', excluded)]:
            save_jsonl(out/f'{prefix}{arm}.jsonl', rows)
        long_s, short_s = (sum(r['duration'] for r in rows) for rows in [long, short])
        stats[split] = dict(parents=len(long), short_windows=len(short),
                            long_seconds=long_s, short_seconds=short_s,
                            exposure_gap_percent=100*(long_s-short_s)/long_s,
                            excluded_parents=len(excluded),
                            excluded_seconds=sum(r['duration'] for r in excluded))
        all_sources += ls + ss
        if split == 'dev':
            # Fixed spread across duration, before seeing loss or synthesis.
            ordered = sorted(long, key=lambda r: (r['duration'], r['utt']))
            indices = sorted({round(i*(len(ordered)-1)/11) for i in range(12)})
            chosen = {ordered[i]['parent'] for i in indices}
            save_jsonl(out/'dev_probe_long.jsonl', [r for r in long if r['parent'] in chosen])
            save_jsonl(out/'dev_probe_short.jsonl', [r for r in short if r['parent'] in chosen])
    assert not splits['train'] & splits['dev'], 'Train/dev parent leakage'
    stats['max_parent_duration'] = args.max_parent_duration
    stats['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in all_sources}
    stats['notes'] = ['Existing channel splits retained; no sealed benchmark accessed.',
                      'Missing/empty short transcripts cause a disclosed small audio exposure gap.',
                      'A duration filter, if used, applies to BOTH views by parent.']
    (out/'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in stats.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
