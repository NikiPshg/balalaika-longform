#!/usr/bin/env python3
"""Score F5 content with the frozen ASR/thresholds and explicit non-AR labels.

The existing AR evaluator remains unchanged. Successful flow generation has
stop_reason=flow_completed, and insufficient end coverage is incomplete_text.
It is never reported as an EOS event. Failed synthesis stays in every denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_evaluation as shared


def flow_config():
    cfg = shared.load_eval_config()
    cfg['final_status']['require_stop_reason_for_complete'] = 'flow_completed'
    return cfg


def validate_flow_runs(rows, benchmark):
    seen = set()
    for row in rows:
        for key in shared.RUN_REQUIRED:
            if key not in row:
                raise ValueError(f'Missing {key} in flow run')
        if row['run_id'] in seen:
            raise ValueError(f"Duplicate run_id: {row['run_id']}")
        seen.add(row['run_id'])
        if row['text_id'] not in benchmark:
            raise ValueError(f"Unknown text_id: {row['text_id']}")
        status = shared.gen_status_of(row)
        if status not in shared.KNOWN_STATUSES or status == 'early_eos':
            raise ValueError(f'Invalid flow generation status: {status}')
        if status == 'complete' and row.get('stop_reason') != 'flow_completed':
            raise ValueError('Flow completion must report flow_completed, never EOS')


def evaluate_flow(rows, benchmark, transcriber, floor, variant):
    validate_flow_runs(rows, benchmark)
    result = shared.evaluate_runs(rows, benchmark, transcriber, floor, variant,
                                  eval_cfg=flow_config())
    for item in result:
        # Reuse exactly the frozen numerical decision, then give it its NAR name.
        if item['status'] == 'early_eos':
            item['status'] = 'incomplete_text'
        item['final_status_reason'] = item['final_status_reason'].replace(
            'eos with ', 'flow completed with ')
        item['generation_family'] = 'non_autoregressive_flow_matching'
        item['status_complete'] = item['status'] == 'complete'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-manifest', required=True, nargs='+')
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--floor-cache', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    out = args.output_dir
    if (out/'per_item.jsonl').exists():
        raise SystemExit(f'Refusing to overwrite scored outputs: {out}')
    benchmark = shared.load_benchmark(args.benchmark)
    runs = [r for p in args.run_manifest for r in shared.read_jsonl(p)]
    validate_flow_runs(runs, benchmark)
    started = time.monotonic()
    tr = shared.GigaAMTranscriber(model_id=shared.DEFAULT_MODEL, use_vad=True,
                                  device=args.device)
    variant = shared.load_spec()['primary_variant']
    floor = shared.compute_floor(list(benchmark.values()), tr,
                                 Path(args.floor_cache), variant, False)
    rows = evaluate_flow(runs, benchmark, tr, floor, variant)
    out.mkdir(parents=True, exist_ok=True)
    shared.write_jsonl(out/'per_item.jsonl', rows)
    summary = shared.build_summary(rows)
    summary['meta'] = dict(asr=tr.describe(), benchmark=args.benchmark,
                           run_manifests=args.run_manifest, normalization=variant,
                           frozen_thresholds=flow_config()['final_status'],
                           frozen_config_sha256=hashlib.sha256((ROOT/'configs/eval.yaml').read_bytes()).hexdigest(),
                           status_semantics='Same content thresholds; incomplete_text replaces early_eos for flow.',
                           elapsed_seconds=time.monotonic()-started)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    lines = ['# F5 content evaluation', '',
             'Same frozen ASR, normalization and content thresholds as the AR results. '
             '`incomplete_text` measures missing tail content; F5 has no EOS event.', '',
             '| Arm | Bucket | N | Complete % | WER % | Coverage % | Statuses |',
             '|---|---|---:|---:|---:|---:|---|']
    for group in summary['rows']:
        selected = [r for r in rows if r['checkpoint']==group['checkpoint'] and
                    (group['bucket']=='ALL' or r['bucket']==group['bucket'])]
        n = len(selected)
        mean = lambda k: sum(float(r[k]) for r in selected)/n
        lines.append(f"| {group['checkpoint']} | {group['bucket']} | {n} | "
                     f"{100*mean('status_complete'):.1f} | {100*mean('wer'):.2f} | "
                     f"{100*mean('source_coverage'):.1f} | {dict(Counter(r['status'] for r in selected))} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(out/'summary.md')


if __name__ == '__main__':
    main()
