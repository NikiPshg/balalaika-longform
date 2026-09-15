#!/usr/bin/env python3
"""A6 E6 Mixed-SFT: full-run budget report of a mixed run, per source, gated against E3's full-run mean.

Pre-registration (reports/decisions.md 2026-08-30, item 5): the mean target tokens per optimizer step over the
WHOLE run must be within +-3 % of E3's (the long-only pilot, exp/long/pilot). E3's mean is COMPUTED here from
its train_stats.jsonl (never hardcoded); the mixed run's records carry `source` ('S'/'L', written by
src/training/cosyvoice_train/executor.py), so the report also gives the per-source means and step counts and
compares the observed source shares with the pattern recorded in <run>/run_info.json (`mix.pattern`).

Usage
    scripts/check_budget_mix.py exp/mix/M31 --ref exp/long/pilot --json_out exp/mix/M31/budget_full.json
    scripts/check_budget_mix.py exp/smoke_mix/smoke --ref exp/long/pilot --steps 20 --allow_partial

A RUN argument is a model_dir (train_stats.jsonl inside) or the jsonl file itself. Records are de-duplicated
by step (last wins), exactly as scripts/check_budget_match.py does; --steps N limits both runs to their first
N steps (0 = the whole file, the pre-registered check).

Exit codes: 0 PASS (within --tolerance), 1 FAIL, 2 usage / data problem.  Stdlib only, CPU only.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_budget_match import TOKEN_KEY, read_steps, stats_path  # noqa: E402  one reader for both checks


def _summ(toks):
    return {'steps': len(toks), 'target_tokens_total': sum(toks), 'target_tokens_mean': statistics.fmean(toks),
            'target_tokens_median': float(statistics.median(toks)), 'target_tokens_min': min(toks),
            'target_tokens_max': max(toks), 'target_tokens_stdev': float(statistics.pstdev(toks)) if len(toks) > 1 else 0.0}


def summarize_mixed(run, steps, allow_partial):
    path = stats_path(run)
    if not os.path.exists(path):
        return None, 'no train_stats.jsonl at {}'.format(path)
    recs, bad = read_steps(path, limit=steps or None)
    if not recs:
        return None, 'no usable records in {}'.format(path)
    if steps and len(recs) < steps and not allow_partial:
        return None, '{}: only {} steps logged, {} requested (use --allow_partial)'.format(path, len(recs), steps)
    toks = [int(r[TOKEN_KEY]) for r in recs]
    times = [r['step_time_sec'] for r in recs if r.get('step_time_sec')]
    out = {'run': run, 'train_stats': os.path.abspath(path), 'first_step': int(recs[0]['step']),
           'last_step': int(recs[-1]['step']), 'unparsable_lines': bad, 'overall': _summ(toks),
           'samples_mean': statistics.fmean([int(r.get('samples', 0)) for r in recs]),
           'padded_positions_max': max(int(r.get('padded_positions', 0)) for r in recs),
           'step_time_sec_median': float(statistics.median(times)) if times else None,
           'peak_vram_reserved_gb_max': max(float(r.get('peak_vram_reserved_gb', 0) or 0) for r in recs),
           'rss_gb_max': max(float(r.get('rss_gb', 0) or 0) for r in recs)}
    by_src = {}
    for r in recs:
        by_src.setdefault(r.get('source'), []).append(int(r[TOKEN_KEY]))
    per = {}
    for s, t in sorted(by_src.items(), key=lambda kv: str(kv[0])):
        per[str(s)] = dict(_summ(t), share=len(t) / len(recs))
    out['per_source'] = per
    out['sequence_head'] = [r.get('source') for r in recs[:16]]
    # expected shares from the pattern in run_info.json (written by train.py in mixed mode)
    ri = os.path.join(os.path.dirname(os.path.abspath(path)), 'run_info.json')
    out['pattern'] = None
    out['expected_share'] = None
    if os.path.exists(ri):
        try:
            with open(ri, encoding='utf-8') as f:
                mix = (json.load(f) or {}).get('mix') or {}
            if mix.get('pattern'):
                pat = mix['pattern'].split(',')
                out['pattern'] = mix['pattern']
                out['expected_share'] = {n: pat.count(n) / len(pat) for n in sorted(set(pat))}
        except (ValueError, OSError) as e:  # a damaged run_info must not hide the token report
            out['run_info_error'] = str(e)
    return out, None


def summarize_ref(run, steps):
    path = stats_path(run)
    if not os.path.exists(path):
        return None, 'reference: no train_stats.jsonl at {}'.format(path)
    recs, bad = read_steps(path, limit=steps or None)
    if not recs:
        return None, 'reference: no usable records in {}'.format(path)
    toks = [int(r[TOKEN_KEY]) for r in recs]
    return dict(_summ(toks), run=run, train_stats=os.path.abspath(path), unparsable_lines=bad,
                first_step=int(recs[0]['step']), last_step=int(recs[-1]['step'])), None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', metavar='RUN', help='exp/mix/<arm> dir or its train_stats.jsonl')
    ap.add_argument('--ref', default='exp/long/pilot', help='E3 reference run dir / jsonl (default exp/long/pilot)')
    ap.add_argument('--steps', type=int, default=0, help='first N steps of both runs (default 0 = whole run)')
    ap.add_argument('--tolerance', type=float, default=0.03, help='max |mean_run/mean_ref - 1| (default 0.03)')
    ap.add_argument('--allow_partial', action='store_true', help='report even if the run has fewer than --steps steps')
    ap.add_argument('--json_out', default=None)
    args = ap.parse_args(argv)

    s, err = summarize_mixed(args.run, args.steps, args.allow_partial)
    if err:
        print('ERROR: ' + err)
        return 2
    ref, err = summarize_ref(args.ref, args.steps)
    if err:
        print('ERROR: ' + err)
        return 2
    ratio = s['overall']['target_tokens_mean'] / ref['target_tokens_mean']
    dev = abs(ratio - 1.0)
    ok = dev <= args.tolerance
    report = {'tolerance': args.tolerance, 'steps_requested': args.steps or None, 'run': s, 'reference': ref,
              'mean_ratio_run_over_ref': ratio, 'deviation': dev, 'within_tolerance': ok,
              'result': 'PASS' if ok else 'FAIL'}

    o = s['overall']
    print('=== E6 budget (pre-registration 2026-08-30 item 5: full-run mean target tokens/step within +-{:.0%} of E3) ==='.format(args.tolerance))
    print('{}\n  file  {}\n  steps {} (step {}..{}), pattern {}'.format(s['run'], s['train_stats'], o['steps'], s['first_step'], s['last_step'], s['pattern']))
    print('  overall  mean {:,.1f}  median {:,.1f}  min {:,}  max {:,}  sd {:,.1f}  total {:,}'.format(
        o['target_tokens_mean'], o['target_tokens_median'], o['target_tokens_min'], o['target_tokens_max'], o['target_tokens_stdev'], o['target_tokens_total']))
    for name, p in s['per_source'].items():
        exp_share = (s['expected_share'] or {}).get(name)
        print('  source {:>4s}  steps {:5d} (share {:.3f}{})  mean {:,.1f}  median {:,.1f}  min {:,}  max {:,}'.format(
            name, p['steps'], p['share'], '' if exp_share is None else ', pattern {:.3f}'.format(exp_share),
            p['target_tokens_mean'], p['target_tokens_median'], p['target_tokens_min'], p['target_tokens_max']))
    print('  sequence head {}'.format(','.join(str(x) for x in s['sequence_head'])))
    print('  samples/step mean {:.2f}  padded positions max {:,}  step time median {}  peak VRAM reserved max {:.2f} GB  RSS max {:.2f} GB'.format(
        s['samples_mean'], s['padded_positions_max'],
        'n/a' if s['step_time_sec_median'] is None else '{:.2f} s'.format(s['step_time_sec_median']),
        s['peak_vram_reserved_gb_max'], s['rss_gb_max']))
    print('reference {}: steps {} (step {}..{})  mean {:,.1f}  median {:,.1f}'.format(
        ref['run'], ref['steps'], ref['first_step'], ref['last_step'], ref['target_tokens_mean'], ref['target_tokens_median']))
    print('mean ratio run/ref {:.4f} ({:.2f} % from equal, tolerance {:.2f} %)'.format(ratio, dev * 100, args.tolerance * 100))
    print('RESULT: {}'.format('PASS - budget matched' if ok else 'FAIL - budget NOT matched'))
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print('json report -> {}'.format(os.path.abspath(args.json_out)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
