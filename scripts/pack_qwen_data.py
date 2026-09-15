#!/usr/bin/env python3
"""Assemble prepared Qwen JSONL into eight local shards, front-loading held-out dev rows.

This rebuild helper preserves source order; it is not a byte-level replay of the
historical prepared codec cache, which is not distributed.
"""
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--train', type=Path, required=True)
p.add_argument('--dev', type=Path, required=True)
p.add_argument('--prefix', choices=['qe2p', 'qe3p'], required=True)
p.add_argument('--out', type=Path, default=Path('data/train/qwen'))
a = p.parse_args()
dev = [json.loads(line) for line in a.dev.read_text().splitlines() if line.strip()]
if len(dev) < 32:
    raise SystemExit('At least 32 held-out dev rows are required.')
dev_ids = {r['source_record_id'] for r in dev}
# Validate all IDs before writing any shard; no train/dev overlap is allowed.
seen = set()
with a.train.open() as f:
    for line in f:
        row = json.loads(line)
        key = row['source_record_id']
        if key in seen or key in dev_ids:
            raise SystemExit(f'Duplicate or train/dev overlap: {key}')
        seen.add(key)
a.out.mkdir(parents=True, exist_ok=True)
paths = [a.out / f'{a.prefix}-shard{i:02d}.jsonl' for i in range(8)]
if any(path.exists() for path in paths):
    raise SystemExit('Output shards already exist; choose a new directory.')
from contextlib import ExitStack
with ExitStack() as stack:
    files = [stack.enter_context(path.open('w')) for path in paths]
    for row in dev[:32]:
        files[0].write(json.dumps(row, ensure_ascii=False) + '\n')
    with a.train.open() as f:
        for i, line in enumerate(f):
            files[i % len(files)].write(line)
print(f'Wrote {len(seen)} training and 32 held-out validation rows to eight shards.')
