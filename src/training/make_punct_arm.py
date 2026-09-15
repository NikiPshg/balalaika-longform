#!/usr/bin/env python3
"""E7 Punct-SFT data builder (A6-mix, pre-registration reports/decisions.md 2026-08-30 «ПРЕРЕГИСТРАЦИЯ E7»).

Builds data/train/long_punct and data/train/dev_long_punct as a pure TRANSFORM of the existing long-arm
shards: for every parquet_XXXXXXXXX.tar under <src>/parquet the row's `text` (ROVER, punct density
~0.013/word) is replaced by the manifest's `text_e2e` (gigaam-v3-e2e-ctc, punctuated + capitalized,
~0.30/word) keyed by utt == manifest sample_id, and the `audio_data` column is DROPPED (the LLM-SFT
pipeline parquet_opener -> tokenize -> budget_check -> sort_by_tokens -> token_batch -> padding_llm never
reads it; budget_check pops it anyway). NO audio re-decode, NO re-tokenization: `speech_token` and every
other column are carried over bit-for-bit, so E7 trains on exactly E3's target sequences.

Fallback: a unit whose manifest row has an EMPTY text_e2e keeps its ROVER text; the pre-registration
expects exactly 2 such train units (dev: 0), and the run FAILS if the count differs from --expected
(anything unexpected must be loud, never silent). A parquet utt missing from the manifest is a hard error.

Output layout = the stock one (same shard numbering, utt2parquet/spk2parquet json, data.list/
utt2data.list/spk2data.list), written with write_shard/finish_lists IMPORTED from
src/training/make_parquet_arms.py (one implementation of the on-disk format). Resumable/atomic like the
other builders: an existing destination shard is kept iff it parses, has the expected utt set AND its
`text` column already equals the expected transform (a stale rover-text shard is rebuilt).

Also writes <dst>/text (kaldi utt->NEW punctuated text), <dst>/text.rover (copy of the source kaldi text,
provenance) and <dst>/stats.json (utts, hours, fallback count+ids, punct density old/new, bytes).

Verification (in-process, after building): every new shard opens through the stock
cosyvoice.dataset.processor.parquet_opener; >= --verify_rows rows sampled across shards have speech_token
BIT-IDENTICAL to the source shard and the expected text; the punct density of the new text is ~0.29/word.
The projected total size is checked against --max_gb (abort if above).

Usage (both pairs, the default):
  PYTHONPATH=. python src/training/make_punct_arm.py
  ... --only train | --only dev     # one pair
"""
import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKDIR = os.path.abspath(os.path.join(_HERE, '..', '..'))
COSYVOICE_ROOT = os.environ.get('COSYVOICE_ROOT', 'third_party/CosyVoice')
for _p in (COSYVOICE_ROOT, os.path.join(COSYVOICE_ROOT, 'third_party', 'Matcha-TTS'), _WORKDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.training.make_parquet_arms import finish_lists, read_kv, write_shard  # noqa: E402  same on-disk format

# Same set as the pre-registration measurement (density 0.252 tts / 0.298 e2e / 0.014 rover)
PUNCT = set('.,!?;:…—–-"\'«»()')


def punct_density(texts):
    words = sum(len(t.split()) for t in texts)
    punct = sum(sum(1 for c in t if c in PUNCT) for t in texts)
    return punct / max(words, 1)


def load_manifest(path):
    man = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            man[d['sample_id']] = d
    return man


def expected_text(utt, man, fallbacks, misses, rover_text):
    if utt not in man:
        misses.append(utt)
        return rover_text
    e2e = man[utt].get('text_e2e')
    if e2e is None or not str(e2e).strip():
        fallbacks.append(utt)
        return rover_text
    return str(e2e)


def shard_paths(src_parquet):
    with open(os.path.join(src_parquet, 'data.list'), encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def dst_shard_ok(dst_parquet, i, exp_utts, exp_text):
    """Keep an existing destination shard iff it parses, has the utt set, no audio_data and the NEW text."""
    tar = os.path.join(dst_parquet, 'parquet_{:09d}.tar'.format(i))
    j1 = os.path.join(dst_parquet, 'utt2parquet_{:09d}.json'.format(i))
    j2 = os.path.join(dst_parquet, 'spk2parquet_{:09d}.json'.format(i))
    try:
        if not (os.path.exists(tar) and os.path.exists(j1) and os.path.exists(j2)):
            return False
        t = pq.read_table(tar)
        if 'audio_data' in t.schema.names:
            return False
        utts = t.column('utt').to_pylist()
        if utts != exp_utts:
            return False
        if t.column('text').to_pylist() != [exp_text[u] for u in utts]:
            return False
        if set(json.load(open(j1)).keys()) != set(exp_utts):
            return False
        return True
    except Exception as e:  # noqa
        logging.warning('dst shard %d in %s unreadable (%s) -> rebuild', i, dst_parquet, e)
        return False


def build_pair(src, dst, manifest_path, expected_fallback, max_gb, verify_rows, rng):
    src_parquet = os.path.join(src, 'parquet')
    dst_parquet = os.path.join(dst, 'parquet')
    os.makedirs(dst_parquet, exist_ok=True)
    man = load_manifest(manifest_path)
    utt2spk = read_kv(os.path.join(src, 'utt2spk'))
    src_shards = shard_paths(src_parquet)
    logging.info('pair %s -> %s: %d shards, manifest %s (%d rows)', src, dst, len(src_shards), manifest_path, len(man))

    fallbacks, misses = [], []
    total_bytes = 0
    n_utts = 0
    hours = 0.0
    dens_old, dens_new = [], []
    built = kept = 0
    kaldi = {}
    for i, src_tar in enumerate(src_shards):
        t = pq.read_table(src_tar)
        cols = [c for c in t.schema.names if c != 'audio_data']
        assert 'audio_data' in t.schema.names and 'speech_token' in cols and 'text' in cols, src_tar
        d = {c: t.column(c).to_pylist() for c in cols}
        utts, rover = d['utt'], d['text']
        exp_text = {}
        sh_fall, sh_miss = [], []
        for u, r in zip(utts, rover):
            exp_text[u] = expected_text(u, man, sh_fall, sh_miss, r)
        fallbacks += sh_fall
        misses += sh_miss
        n_utts += len(utts)
        hours += sum(float(x) for x in d['duration']) / 3600.0
        dens_old.append(punct_density(rover))
        dens_new.append(punct_density([exp_text[u] for u in utts]))
        kaldi.update(exp_text)
        if dst_shard_ok(dst_parquet, i, utts, exp_text):
            kept += 1
        else:
            d['text'] = [exp_text[u] for u in utts]
            rows = [{c: d[c][k] for c in cols} for k in range(len(utts))]
            write_shard(rows, dst_parquet, i, utt2spk)
            built += 1
        total_bytes += os.path.getsize(os.path.join(dst_parquet, 'parquet_{:09d}.tar'.format(i)))
        if total_bytes > max_gb * 1e9:
            raise SystemExit('ABORT: projected size {:.2f} GB exceeds --max_gb {} at shard {}'.format(total_bytes / 1e9, max_gb, i))
        if (i + 1) % 20 == 0 or i + 1 == len(src_shards):
            logging.info('%s: shard %d/%d (built %d, kept %d, %.1f MB, fallbacks %d)',
                         dst, i + 1, len(src_shards), built, kept, total_bytes / 1e6, len(fallbacks))
    if misses:
        raise SystemExit('FAIL: {} parquet utts not in {} (first 5: {}) - the join must be exact'
                         .format(len(misses), manifest_path, misses[:5]))
    if len(fallbacks) != expected_fallback:
        raise SystemExit('FAIL: {} fallback units (empty text_e2e) in {}, pre-registration expects exactly {}: {}'
                         .format(len(fallbacks), dst, expected_fallback, fallbacks[:10]))
    finish_lists(dst_parquet, len(src_shards))

    # provenance: NEW kaldi text + the source ROVER text
    with open(os.path.join(dst, 'text'), 'w', encoding='utf-8') as f:
        for u in sorted(kaldi):
            f.write('{} {}\n'.format(u, kaldi[u]))
    src_text = os.path.join(src, 'text')
    if os.path.exists(src_text):
        with open(src_text, encoding='utf-8') as fi, open(os.path.join(dst, 'text.rover'), 'w', encoding='utf-8') as fo:
            fo.write(fi.read())

    d_old = float(np.mean(dens_old))
    d_new = float(np.mean(dens_new))
    stats = {'src': src, 'manifest': manifest_path, 'shards': len(src_shards), 'built': built, 'kept': kept,
             'utts': n_utts, 'hours': round(hours, 2), 'fallback_count': len(fallbacks), 'fallback_utts': fallbacks,
             'punct_density_rover': round(d_old, 4), 'punct_density_new': round(d_new, 4),
             'bytes': total_bytes, 'audio_data_dropped': True}
    logging.info('%s: %d utts, %.2f h, %d fallbacks, density %.3f -> %.3f, %.1f MB (built %d, kept %d)',
                 dst, n_utts, hours, len(fallbacks), d_old, d_new, total_bytes / 1e6, built, kept)

    # ---- verification: stock reader + bit-identical speech tokens + text ----
    from cosyvoice.dataset.processor import parquet_opener
    dst_shards = shard_paths(dst_parquet)
    assert len(dst_shards) == len(src_shards)
    checked = 0
    for i in rng.sample(range(len(src_shards)), len(src_shards)):
        rows_new = list(parquet_opener(iter([{'src': dst_shards[i]}]), mode='train'))
        assert rows_new, 'parquet_opener produced no rows for {}'.format(dst_shards[i])
        ts = pq.read_table(src_shards[i], columns=['utt', 'speech_token', 'text'])
        src_tok = dict(zip(ts.column('utt').to_pylist(), ts.column('speech_token').to_pylist()))
        src_txt = dict(zip(ts.column('utt').to_pylist(), ts.column('text').to_pylist()))
        for r in rows_new:
            u = r['utt']
            assert 'audio_data' not in r, u
            assert np.array_equal(np.asarray(r['speech_token']), np.asarray(src_tok[u])), \
                'speech_token differs for {} (shard {})'.format(u, i)
            e2e = (man[u].get('text_e2e') or '').strip()
            exp = e2e if e2e else src_txt[u]
            assert r['text'] == exp, 'text mismatch for {}'.format(u)
            if e2e and u not in fallbacks:
                assert r['text'] != src_txt[u] or src_txt[u] == e2e, u
            checked += 1
        if checked >= verify_rows:
            break
    assert checked >= min(verify_rows, n_utts), 'verified only {} rows'.format(checked)
    if not (0.2 <= d_new <= 0.45):
        raise SystemExit('FAIL: new punct density {:.3f} outside the expected ~0.29 band [0.2, 0.45]'.format(d_new))
    stats['verified_rows'] = checked
    with open(os.path.join(dst, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logging.info('%s VERIFIED: %d rows bit-identical speech_token + expected text; stats.json written', dst, checked)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train_src', default='data/train/long')
    ap.add_argument('--train_dst', default='data/train/long_punct')
    ap.add_argument('--train_manifest', default='data/manifests/train_long.jsonl')
    ap.add_argument('--train_expected_fallback', type=int, default=2, help='pre-registered: exactly 2 units lack text_e2e')
    ap.add_argument('--dev_src', default='data/train/dev_long')
    ap.add_argument('--dev_dst', default='data/train/dev_long_punct')
    ap.add_argument('--dev_manifest', default='data/manifests/dev.jsonl')
    ap.add_argument('--dev_expected_fallback', type=int, default=0)
    ap.add_argument('--only', choices=['train', 'dev'], default=None)
    ap.add_argument('--max_gb', type=float, default=5.0, help='abort if the new shards exceed this (expected ~0.1 GB)')
    ap.add_argument('--verify_rows', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    rng = random.Random(args.seed)
    out = {}
    if args.only in (None, 'dev'):
        out['dev'] = build_pair(args.dev_src, args.dev_dst, args.dev_manifest, args.dev_expected_fallback,
                                args.max_gb, args.verify_rows, rng)
    if args.only in (None, 'train'):
        out['train'] = build_pair(args.train_src, args.train_dst, args.train_manifest, args.train_expected_fallback,
                                  args.max_gb, args.verify_rows, rng)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'fallback_utts'} for k, v in out.items()}, indent=2))


if __name__ == '__main__':
    main()
