#!/usr/bin/env python3
"""Summarize complete, matched F5 grids without dropping generation failures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['initial', 'full'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    splits = ['exploratory'] if args.phase=='initial' else ['pilot', 'robust']
    summary = []
    for split in splits:
        common_grid = None
        for mode in ['uninterrupted', 'chunked']:
            for arm in ['base', 'short', 'long']:
                name = f'{args.phase}_{arm}_{mode}_{split}'
                output = ROOT/'outputs/v31_f5'/name
                result = ROOT/'results/v31_f5'/name
                progress = json.loads((output/'progress.json').read_text())
                if not progress['completed_grid']:
                    raise ValueError(f'Incomplete generation grid: {name}')
                rows = read_rows(result/'per_item.jsonl')
                grid = {(r['text_id'],r['voice_id'],r['seed']) for r in rows}
                if len(grid) != len(rows) or len(rows) != progress['planned']:
                    raise ValueError(f'Duplicate or missing evaluation rows: {name}')
                if common_grid is None:
                    common_grid = grid
                elif common_grid != grid:
                    raise ValueError(f'Unmatched comparison grid: {name}')
                voices = [r for r in read_rows(result/'speaker_drift.jsonl') if not r.get('_meta')]
                for bucket in ['ALL', *sorted({r['bucket'] for r in rows})]:
                    selected = [r for r in rows if bucket=='ALL' or r['bucket']==bucket]
                    paths = {r['output_path'] for r in selected}
                    # The existing speaker evaluator uses the output path as its join key.
                    sims = [r['sim_median'] for r in voices if r.get('output_path') in paths
                            and isinstance(r.get('sim_median'), (int,float))]
                    mean = lambda k: statistics.mean(r[k] for r in selected)
                    ratios = [r['duration_ratio'] for r in selected if r.get('duration_ratio') is not None]
                    summary.append(dict(split=split, mode=mode, arm=arm, bucket=bucket,
                        n=len(selected), complete_percent=100*mean('status_complete'),
                        wer_percent=100*mean('wer'), coverage_percent=100*mean('source_coverage'),
                        exact_word_match_percent=100*statistics.mean(r['hits']/r['n_ref_words'] for r in selected),
                        end_coverage_percent=100*mean('end_coverage'),
                        complete_without_floor_wer_ge_30=sum(r['status_complete'] and not r['floor_available']
                                                            and r['wer'] >= .30 for r in selected),
                        speaker_similarity=statistics.median(sims) if sims else None,
                        n_speaker_similarity=len(sims),
                        duration_ratio=statistics.mean(ratios) if ratios else None,
                        empty_transcriptions=sum(r.get('invalid_reason') == 'empty_transcription' for r in selected),
                        asr_errors=sum(bool(r.get('asr_error')) for r in selected),
                        generation_failures=sum(r.get('gen_status') != 'complete' for r in selected),
                        context_limits=sum(r['status']=='context_limit' for r in selected),
                        oom=sum(r['status']=='oom' for r in selected)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'table.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    lines = ['# F5 comparison', '',
        ('Exploratory: both adapted models are measured at the predeclared 1,000-update pause '
         'of the 18,114-update schedule. Six texts, two voices each; this is not the full pilot or Robust-53.'
         if args.phase=='initial' else 'Full predeclared schedules; separate pilot and Robust-53 results.'), '',
        'Every synthesis failure remains in WER/completion denominators. Speaker similarity uses '
        'items with usable voiced windows, with that count reported explicitly. The CSV separately '
        'counts empty transcriptions, ASR execution errors and generator failures.', '',
        '**Status caveat:** the frozen completion rule skips its WER criterion when no human '
        'reference recording exists. It can therefore label high-WER, incomplete readings as complete. '
        'Read WER alongside this status. Aligned coverage includes substitutions; exact word matches '
        '(hits / reference words) are separately recorded in the CSV. The CSV also counts '
        'floorless complete cases with WER >= 30% as a diagnostic, without relabeling them.', '',
        '| Set | Mode | Arm | Bucket | N | Frozen complete % | WER % | Aligned coverage % | Speaker similarity (n) |',
        '|---|---|---|---|---:|---:|---:|---:|---|']
    for r in summary:
        sim = f"{r['speaker_similarity']:.3f}" if r['speaker_similarity'] is not None else 'n/a'
        lines.append(f"| {r['split']} | {r['mode']} | {r['arm']} | {r['bucket']} | {r['n']} | "
                     f"{r['complete_percent']:.1f} | {r['wer_percent']:.2f} | {r['coverage_percent']:.1f} | "
                     f"{sim} ({r['n_speaker_similarity']}) |")
    (args.output_dir/'table.md').write_text('\n'.join(lines)+'\n')
    print(args.output_dir/'table.md')


if __name__ == '__main__':
    main()
