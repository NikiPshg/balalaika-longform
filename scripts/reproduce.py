#!/usr/bin/env python3
"""Recompute the submitted results from stored measurements, using CPU only."""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import expanded_eval_limited_report as report
import paper_wer_compact
import paper_window_metrics50


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare_values(actual, expected, location=''):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f'Changed fields: {location}')
        for key in expected:
            compare_values(actual[key], expected[key], location + '/' + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), location
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_values(a, e, f'{location}/{i}')
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-10, equal_nan=True):
            raise AssertionError(f'Changed numerical result: {location}: {actual} vs {expected}')
    elif actual != expected:
        raise AssertionError(f'Changed result: {location}: {actual!r} vs {expected!r}')


def table(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Choose an empty output directory; existing results are never overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'artifacts/published50'
    print('Recomputing the 13 × 50 paired comparison and 10,000-resample intervals...', flush=True)
    report.run('final', source / 'statistics_spec.json', out / 'published50', group='main13')
    actual = json.loads((out / 'published50/statistics/statistics.json').read_text())
    expected = json.loads((source / 'report/statistics/statistics.json').read_text())
    # Paths and source-code hashes differ after relocation; all reported cells must agree.
    compare_values(actual['cells'], expected['cells'], 'published50/cells')
    paper_wer_compact.render(out / 'published50', out / 'figures/fig_wer_limited50_wide')
    print('Rebuilding the windowed MOS and speaker-similarity figures...', flush=True)
    paper_window_metrics50.render(source / 'window_metrics', out / 'figures/fig_quality_limited50')

    splits = []
    for split, name in [('Train', 'train_long'), ('Dev', 'dev'), ('Test', 'test')]:
        with gzip.open(ROOT / 'data/frozen_manifests' / f'{name}.jsonl.gz', 'rt') as f:
            rows = [json.loads(line) for line in f]
        splits.append(dict(split=split, segments=len(rows), hours=sum(r['duration_sec'] for r in rows) / 3600,
                           videos=len({r['video_id'] for r in rows})))
    table(out / 'table1_splits.csv', ['split', 'segments', 'hours', 'videos'], splits)

    pilot = []
    for arm in ['cosy_base', 'cosy_short', 'cosy_long', 'cosy_long_punct']:
        rows = read_rows(ROOT / 'artifacts/two_voice' / f'{arm}.jsonl')
        assert len(rows) == 60
        for bucket in ['B0', 'B1', 'B2', 'B3', 'B4', 'All']:
            cell = rows if bucket == 'All' else [r for r in rows if r['bucket'] == bucket]
            pilot.append(dict(arm=arm, bucket=bucket, attempts=len(cell),
                              complete_pct=100 * sum(r['status'] == 'complete' for r in cell) / len(cell),
                              wer_all_pct=100 * sum(r['wer'] for r in cell) / len(cell)))
    table(out / 'table2_two_voice.csv', ['arm', 'bucket', 'attempts', 'complete_pct', 'wer_all_pct'], pilot)
    assert [round(r['wer_all_pct'], 2) for r in pilot if r['bucket'] == 'All'] == [62.10, 60.88, 20.47, 14.36]
    expected_splits = [(6043, 170.68), (120, 8.50), (193, 9.07)]
    assert [(r['segments'], round(r['hours'], 2)) for r in splits] == expected_splits
    (out / 'VERIFIED.json').write_text(json.dumps({
        'status': 'passed', 'published50_attempts': 650, 'arms': 13,
        'all_reported_cells_and_intervals_match': True, 'bootstrap_resamples': 10000,
        'window_metrics_attempts': 650, 'training_or_synthesis_rerun': False,
        'table1': 'table1_splits.csv', 'table2': 'table2_two_voice.csv',
        'table3': 'published50/main_b4.csv'}, indent=2) + '\n')
    print(f'All reported numbers reproduced. Tables and figures: {out}', flush=True)


if __name__ == '__main__':
    main()
