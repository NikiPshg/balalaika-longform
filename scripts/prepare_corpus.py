#!/usr/bin/env python3
"""Expand frozen manifests, optionally restore corpus audio from the released WebDataset."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'lab260/Balalaika-longform'
REVISION = '96a1abc07caf80703e4dc0578aadb4e43811398d'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', type=Path, help='Local dataset directory containing data/*/*.tar')
    p.add_argument('--download', action='store_true', help='Download pinned dataset; requires huggingface_hub')
    p.add_argument('--cache-dir', type=Path, default=ROOT / '.cache/dataset')
    p.add_argument('--limit', type=int, help='Debug: restore at most this many audio samples')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    out = ROOT / 'data/manifests'
    out.mkdir(parents=True, exist_ok=True)
    for archive in sorted((ROOT / 'data/frozen_manifests').glob('*.jsonl.gz')):
        dest = out / archive.stem
        payload = gzip.decompress(archive.read_bytes())
        if dest.exists() and dest.read_bytes() != payload:
            raise SystemExit(f'Refusing to replace a different manifest: {dest}')
        dest.write_bytes(payload)
    print('Frozen training manifests restored.', flush=True)
    if args.download:
        from huggingface_hub import snapshot_download
        args.shards = Path(snapshot_download(DATASET, repo_type='dataset', revision=REVISION,
                                            local_dir=args.cache_dir))
    if args.shards is None:
        return
    rows = [json.loads(line) for line in (out / 'all.jsonl').read_text().splitlines()]
    mapping = {'balalaika_' + r['sample_id'].replace('.', 'p'): r for r in rows}
    archives = sorted(args.shards.rglob('*.tar'))
    if not archives:
        raise SystemExit(f'No .tar shards under {args.shards}')
    count = 0
    for archive in archives:
        with tarfile.open(archive, 'r:') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.flac'):
                    continue
                key = Path(member.name).stem
                if key not in mapping:
                    raise ValueError(f'Unexpected audio key: {key}')
                row = mapping[key]
                dest = (ROOT / row['audio_path']).resolve()
                if not dest.is_relative_to((ROOT / 'data/corpus').resolve()):
                    raise ValueError('Audio destination outside corpus directory')
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_suffix('.partial')
                with tar.extractfile(member) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target)
                expected = row.get('sha256_audio')
                actual = hashlib.file_digest(temporary.open('rb'), 'sha256').hexdigest()
                if expected and actual != expected:
                    temporary.unlink()
                    raise ValueError(f'Audio hash mismatch: {key}')
                if dest.exists():
                    with dest.open('rb') as f:
                        if hashlib.file_digest(f, 'sha256').hexdigest() != actual:
                            temporary.unlink()
                            raise ValueError(f'Refusing to replace different audio: {dest}')
                    temporary.unlink()
                else:
                    temporary.replace(dest)
                # Qwen preparation reads the original per-parent ASR agreement here.
                # Frozen text files already contain the short-window punctuation.
                sidecar = dest.with_suffix('.json')
                if not sidecar.exists():
                    sidecar.write_text(json.dumps({'sample_id': row['sample_id'],
                        'asr_consistency': row.get('asr_consistency'),
                        'text': row.get('text'), 'text_e2e': row.get('text_e2e'),
                        'note': 'Minimal reconstruction from frozen experiment manifest; no word timestamps.'},
                        ensure_ascii=False) + '\n')
                count += 1
                if args.limit and count >= args.limit:
                    print(f'Restored and hash-checked {count} audio files (limited check).')
                    return
    if count != len(rows):
        raise ValueError(f'Incomplete corpus: {count} audio entries, expected {len(rows)}')
    print(f'Restored and hash-checked {count} audio files.')


if __name__ == '__main__':
    main()
