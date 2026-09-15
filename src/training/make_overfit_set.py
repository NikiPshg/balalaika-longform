#!/usr/bin/env python3
"""Build tiny overfit sets (PLAN §11.4 step 2) from already-built parquet shards.

  long : N long units with duration >= --min_long_sec (default 300 s)
  short: N windows of those same parents (first window of each)
Writes data/train/overfit_{long,short}/parquet (train) and data/train/overfit_dev_{long,short}/parquet
(= the same rows, so CV loss shows the same decrease).
"""
import argparse
import json
import os
import pandas as pd


def write(rows, des):
    os.makedirs(des, exist_ok=True)
    pq = os.path.join(des, 'parquet_000000000.tar')
    df = pd.DataFrame(rows)
    df.to_parquet(pq)
    with open(os.path.join(des, 'utt2parquet_000000000.json'), 'w') as f:
        json.dump({u: pq for u in df['utt']}, f)
    with open(os.path.join(des, 'spk2parquet_000000000.json'), 'w') as f:
        json.dump({s: pq for s in df['spk']}, f)
    for name, fn in (('data.list', 'parquet_000000000.tar'), ('utt2data.list', 'utt2parquet_000000000.json'),
                     ('spk2data.list', 'spk2parquet_000000000.json')):
        with open(os.path.join(des, name), 'w') as f:
            f.write(os.path.join(des, fn) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--long_list', default='data/train/sub2_long/parquet/data.list')
    ap.add_argument('--short_list', default='data/train/sub2_short/parquet/data.list')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--min_long_sec', type=float, default=300.0)
    ap.add_argument('--des_root', default='data/train')
    args = ap.parse_args()
    L = pd.concat([pd.read_parquet(l.strip()) for l in open(args.long_list)])
    S = pd.concat([pd.read_parquet(l.strip()) for l in open(args.short_list)])
    L = L[L.duration >= args.min_long_sec].sort_values('utt').head(args.n)
    assert len(L) == args.n, 'not enough long units >= {} s: {}'.format(args.min_long_sec, len(L))
    S = S[S.parent.isin(L.utt)].sort_values(['parent', 'offset_start']).groupby('parent').head(1)
    for name, df in (('overfit_long', L), ('overfit_dev_long', L), ('overfit_short', S), ('overfit_dev_short', S)):
        write(df.to_dict('records'), os.path.join(args.des_root, name, 'parquet'))
    print(json.dumps({'long': [(u, round(d, 1), int(n)) for u, d, n in zip(L.utt, L.duration, L.n_speech_token)],
                      'short': [(u, round(d, 1), int(n)) for u, d, n in zip(S.utt, S.duration, S.n_speech_token)]}, indent=1))


if __name__ == '__main__':
    main()
