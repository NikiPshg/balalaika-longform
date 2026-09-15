#!/usr/bin/env python3
"""A6: verify the E2/E3 budget match from the trainer's per-step token log.

Lead decision 2026-08-28 (reports/decisions.md), budget-matching variant (b): the arms use different
token caps (long 22500, short 20700 target tokens per micro-batch -- the short cap was corrected from the
provisional 17000 on 2026-08-29, by this very check, after the measured long-arm mean of 20060) so that an
EQUAL number of optimizer steps gives EQUAL mean target tokens per step within +-3 %. The Lead requires this
to be checked on the first 200 steps of the pilot of both arms.

Input: <model_dir>/train_stats.jsonl, one JSON record per OPTIMIZER step, written by
src/training/cosyvoice_train/executor.py::_log_step_stats. Fields used here:
    step              optimizer step (1-based)
    speech_tokens     target speech tokens summed over the micro-batches of that step (accum_grad)
    samples, padded_positions, step_time_sec, total_speech_tokens   (reported as diagnostics)

Usage
    scripts/check_budget_match.py exp/long/pilot exp/short/pilot --steps 200
    scripts/check_budget_match.py exp/long/pilot --steps 200            # one run: report only, no gate
    scripts/check_budget_match.py exp/long/pilot --steps 200 --wait 7200 --json_out out.json

A RUN argument is either a model_dir (train_stats.jsonl is appended to it) or the jsonl file itself.

Exit codes
    0  ok (two runs within tolerance, or a single-run report)
    1  MISMATCH: the mean tokens/step of the two runs differ by more than --tolerance
    2  usage / data problem (no file, fewer than --steps steps and no --allow_partial)

Stdlib only, CPU only: safe to run while training is in progress (the file is flushed per step).
"""
import argparse
import json
import os
import statistics
import sys
import time

TOKEN_KEY = 'speech_tokens'


def stats_path(run):
    if os.path.isdir(run):
        return os.path.join(run, 'train_stats.jsonl')
    return run


def read_steps(path, limit=None):
    """Return records ordered by step, at most `limit` of them, de-duplicated by step.

    A resumed run appends to the same file, so the same step number can appear twice; the LAST
    record for a step wins (it is the one the resumed run actually took).
    """
    by_step, bad = {}, 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if TOKEN_KEY not in rec or 'step' not in rec:
                bad += 1
                continue
            by_step[int(rec['step'])] = rec
    recs = [by_step[s] for s in sorted(by_step)]
    if limit is not None:
        recs = recs[:limit]
    return recs, bad


def wait_for(path, steps, timeout_sec, poll_sec=15.0):
    """Block until `path` holds at least `steps` optimizer steps, or the timeout expires."""
    deadline = time.time() + timeout_sec
    while True:
        if os.path.exists(path):
            recs, _ = read_steps(path)
            if len(recs) >= steps:
                return True
        if time.time() >= deadline:
            return False
        time.sleep(poll_sec)


def summarize(run, steps, allow_partial):
    path = stats_path(run)
    if not os.path.exists(path):
        return None, 'no train_stats.jsonl at {}'.format(path)
    recs, bad = read_steps(path, limit=steps)
    if not recs:
        return None, 'no usable records in {}'.format(path)
    if steps is not None and len(recs) < steps and not allow_partial:
        return None, '{}: only {} steps logged, {} requested (use --allow_partial or --wait)'.format(path, len(recs), steps)
    toks = [int(r[TOKEN_KEY]) for r in recs]
    times = [r['step_time_sec'] for r in recs if r.get('step_time_sec')]
    s = {
        'run': run,
        'train_stats': os.path.abspath(path),
        'steps_used': len(recs),
        'first_step': int(recs[0]['step']),
        'last_step': int(recs[-1]['step']),
        'unparsable_lines': bad,
        'target_tokens_total': sum(toks),
        'target_tokens_mean': statistics.fmean(toks),
        'target_tokens_median': float(statistics.median(toks)),
        'target_tokens_min': min(toks),
        'target_tokens_max': max(toks),
        'target_tokens_stdev': float(statistics.pstdev(toks)) if len(toks) > 1 else 0.0,
        'samples_mean': statistics.fmean([int(r.get('samples', 0)) for r in recs]),
        'padded_positions_max': max(int(r.get('padded_positions', 0)) for r in recs),
        'step_time_sec_median': float(statistics.median(times)) if times else None,
    }
    return s, None


def fmt(s):
    return ('{run}\n'
            '  file            {train_stats}\n'
            '  steps           {steps_used} (step {first_step}..{last_step})\n'
            '  target tokens/step  mean {target_tokens_mean:,.1f}  median {target_tokens_median:,.1f}  '
            'min {target_tokens_min:,}  max {target_tokens_max:,}  sd {target_tokens_stdev:,.1f}\n'
            '  total target tokens {target_tokens_total:,}\n'
            '  samples/step mean {samples_mean:.2f}   padded positions max {padded_positions_max:,}\n'
            '  step time median    {step_time}\n').format(
        step_time='n/a' if s['step_time_sec_median'] is None else '{:.2f} s'.format(s['step_time_sec_median']), **s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', metavar='RUN', help='exp/<arm>/<tag> dir or a train_stats.jsonl (1 or 2)')
    ap.add_argument('--steps', type=int, default=200, help='use the first N optimizer steps (default 200; 0 = all)')
    ap.add_argument('--tolerance', type=float, default=0.03, help='max |mean_a/mean_b - 1| (default 0.03 = Lead +-3 %%)')
    ap.add_argument('--wait', type=float, default=0.0, metavar='SEC', help='wait up to SEC for the runs to log --steps steps')
    ap.add_argument('--allow_partial', action='store_true', help='report even if fewer than --steps steps are logged')
    ap.add_argument('--json_out', default=None, help='also write the report as JSON to this path')
    args = ap.parse_args(argv)

    if len(args.runs) > 2:
        ap.error('give one or two runs')
    steps = args.steps if args.steps and args.steps > 0 else None

    if args.wait > 0 and steps:
        for run in args.runs:
            if not wait_for(stats_path(run), steps, args.wait):
                print('TIMEOUT: {} did not reach {} steps in {:.0f} s'.format(stats_path(run), steps, args.wait))
                if not args.allow_partial:
                    return 2

    summaries = []
    for run in args.runs:
        s, err = summarize(run, steps, args.allow_partial)
        if err:
            print('ERROR: ' + err)
            return 2
        summaries.append(s)

    print('=== budget match (Lead 2026-08-28, variant (b): equal steps => equal mean target tokens/step +-{:.0%}) ==='
          .format(args.tolerance))
    for s in summaries:
        print(fmt(s))

    report = {'tolerance': args.tolerance, 'steps_requested': steps, 'runs': summaries}
    rc = 0
    if len(summaries) == 2:
        a, b = summaries
        if a['steps_used'] != b['steps_used']:
            print('WARNING: unequal step counts ({} vs {}) - the comparison is only valid on equal steps'
                  .format(a['steps_used'], b['steps_used']))
        ratio = a['target_tokens_mean'] / b['target_tokens_mean'] if b['target_tokens_mean'] else float('inf')
        med_ratio = a['target_tokens_median'] / b['target_tokens_median'] if b['target_tokens_median'] else float('inf')
        dev = abs(ratio - 1.0)
        rc = 0 if dev <= args.tolerance else 1
        report.update({'mean_ratio': ratio, 'median_ratio': med_ratio, 'deviation': dev,
                       'within_tolerance': rc == 0, 'total_token_ratio':
                       (a['target_tokens_total'] / b['target_tokens_total']) if b['target_tokens_total'] else None})
        print('mean ratio   {:.4f}  ({:.2f} % from equal, tolerance {:.2f} %)  [{} / {}]'
              .format(ratio, dev * 100, args.tolerance * 100, a['run'], b['run']))
        print('median ratio {:.4f}   total-token ratio {:.4f}'.format(med_ratio, report['total_token_ratio']))
        print('RESULT: {}'.format('PASS - budget matched' if rc == 0 else 'FAIL - budget NOT matched'))
    else:
        report['within_tolerance'] = None
        print('single run: report only (give a second run to gate the +-{:.0%} match)'.format(args.tolerance))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print('json report -> {}'.format(os.path.abspath(args.json_out)))
    return rc


if __name__ == '__main__':
    sys.exit(main())
