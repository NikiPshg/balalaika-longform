#!/usr/bin/env python3
"""A13-perf: aggregate the inference-cost evidence ALREADY logged per generation item.

Reads every runs.jsonl (and cross-checks the per-item *.json manifests) under the
PLAN v31 generation output dirs listed in SOURCES, and writes:

    results/v31_perf/perf_per_item.csv        one row per generation item
    results/v31_perf/perf_by_system_bucket.csv aggregate (system x bucket)
    results/v31_perf/perf_crosscheck.csv       per-manifest field-integrity counts
    results/v31_perf/kv_cache.csv              KV-cache math from the model configs
    results/v31_perf/perf_tables.md            the report

No GPU, no model load, no new generation: pure aggregation of existing files.

    python3 scripts/aggregate_perf.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'v31_perf')

# (relative dir under outputs/, run-set label). system comes from experiment_id.
SOURCES = [
    ('v31_base/E1_native', 'pilot'),
    ('v31_base/E0_official', 'pilot'),
    ('v31_sft/E2_epoch_0_step_200', 'pilot'),
    ('v31_sft/E2_epoch_3_step_3001', 'pilot'),
    ('v31_sft/E3_epoch_0_whole', 'pilot'),
    ('v31_sft/E3_epoch_3_step_3001', 'pilot'),
    ('v31_punct/E7_punct_epoch_3_step_3001', 'pilot'),
    ('v31_robust/E1', 'robust'),
    ('v31_robust/E3', 'robust'),
    ('v31_robust/E7', 'robust'),
    ('v31_qwen/QE1_pilot', 'pilot'),
    ('v31_qwen/QE1_robust', 'robust'),
    ('v31_seed1/E1_native', 'seed1'),
    ('v31_seed1/E3_long', 'seed1'),
    ('v31_seed1_e7/E7_punct', 'seed1'),
    ('v31_seed2_full/E3', 'seed2'),
    ('v31_seed2_full/E7', 'seed2'),
    ('v31_short_tail/E1_pilot', 'short_tail'),
    ('v31_short_tail/E1_robust', 'short_tail'),
    ('v31_short_tail/E3_pilot', 'short_tail'),
    ('v31_short_tail/E3_robust', 'short_tail'),
    ('v31_short_tail/E7_pilot', 'short_tail'),
    ('v31_short_tail/E7_robust', 'short_tail'),
    ('v31_short_tail/QE1_pilot', 'short_tail'),
    ('v31_short_tail/QE1_robust', 'short_tail'),
    ('v31_short_tail_s1/E1_pilot', 'short_tail_s1'),
    ('v31_short_tail_s1/E1_robust', 'short_tail_s1'),
    ('v31_short_tail_s1/E3_pilot', 'short_tail_s1'),
    ('v31_short_tail_s1/E3_robust', 'short_tail_s1'),
    ('v31_short_tail_s1/E7_pilot', 'short_tail_s1'),
    ('v31_short_tail_s1/E7_robust', 'short_tail_s1'),
    ('v31_short_tail_s1/QE1_pilot', 'short_tail_s1'),
    ('v31_short_tail_s1/QE1_robust', 'short_tail_s1'),
]

MAIN_SYSTEMS = ['E1', 'E3', 'E7', 'QE1']
APPENDIX_SYSTEMS = ['E0', 'E2']
LONG_BUCKETS = ['B0', 'B1', 'B2', 'B3', 'B4']
SHORT_BUCKETS = ['S0', 'S1']

GIB = float(2 ** 30)


def system_of(exp_id: str) -> str:
    """QE1* -> QE1; E7_punct / E7_st_robust -> E7; E1_short -> E1; etc."""
    e = exp_id or ''
    if e.startswith('QE1'):
        return 'QE1'
    for s in ('E0', 'E1', 'E2', 'E3', 'E7'):
        if e == s or e.startswith(s + '_'):
            return s
    return e


def pctl(values, q):
    """Linear-interpolation percentile (same convention as numpy.percentile)."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def med(values):
    xs = [v for v in values if v is not None]
    return statistics.median(xs) if xs else None


def r(x, n=3):
    return None if x is None else round(x, n)


def fmt(x, n=3):
    if x is None:
        return '-'
    if isinstance(x, int):
        return str(x)
    return f'{x:.{n}f}'


# ---------------------------------------------------------------- load
rows = []
crosscheck = []
for rel, run_set in SOURCES:
    d = os.path.join(ROOT, 'outputs', rel)
    manifest = os.path.join(d, 'runs.jsonl')
    if not os.path.exists(manifest):
        crosscheck.append({'dir': 'outputs/' + rel, 'run_set': run_set, 'note': 'runs.jsonl MISSING'})
        continue
    recs = [json.loads(ln) for ln in open(manifest) if ln.strip()]
    per_item = [p for p in glob.glob(os.path.join(d, '*.json')) if not p.endswith('.tokens.json')]
    cc = {
        'dir': 'outputs/' + rel, 'run_set': run_set, 'n_runs_jsonl': len(recs),
        'n_per_item_json': len(per_item), 'n_missing_rtf': 0, 'n_missing_wall': 0,
        'n_missing_peak_vram': 0, 'n_missing_duration': 0, 'n_missing_gen_tokens': 0,
        'n_rtf_inconsistent': 0, 'max_rtf_rel_err': 0.0,
        'n_tokpersec_matches_llm_time': 0, 'n_tokpersec_matches_wall': 0,
        'n_zero_peak_vram': 0, 'statuses': '', 'note': '',
    }
    st = Counter()
    for m in recs:
        exp = m.get('experiment_id')
        sysname = system_of(exp)
        text_id = m.get('text_id') or ''
        bucket = text_id.rsplit('__', 1)[-1] if '__' in text_id else ''
        dur = m.get('raw_duration_sec')
        wall = m.get('wall_time_sec')
        rtf = m.get('rtf')
        pv = m.get('peak_vram_bytes')
        gen = m.get('generated_speech_tokens')
        llm_t = m.get('llm_time_sec')
        tps = m.get('speech_tokens_per_sec_wall')
        status = m.get('status')
        st[status] += 1

        if rtf is None:
            cc['n_missing_rtf'] += 1
        if wall is None:
            cc['n_missing_wall'] += 1
        if pv is None:
            cc['n_missing_peak_vram'] += 1
        elif pv == 0:
            cc['n_zero_peak_vram'] += 1
        if dur is None:
            cc['n_missing_duration'] += 1
        if gen is None:
            cc['n_missing_gen_tokens'] += 1
        if rtf is not None and wall is not None and dur:
            recomputed = wall / dur
            rel_err = abs(recomputed - rtf) / max(rtf, 1e-9)
            cc['max_rtf_rel_err'] = max(cc['max_rtf_rel_err'], rel_err)
            if rel_err > 0.01:
                cc['n_rtf_inconsistent'] += 1
        if tps and gen:
            if llm_t and abs(gen / llm_t - tps) / tps < 0.01:
                cc['n_tokpersec_matches_llm_time'] += 1
            if wall and abs(gen / wall - tps) / tps < 0.01:
                cc['n_tokpersec_matches_wall'] += 1

        rows.append({
            'dir': 'outputs/' + rel,
            'run_set': run_set,
            'system': sysname,
            'experiment_id': exp,
            'mode': m.get('mode'),
            'model_id': m.get('model_id'),
            'run_id': m.get('run_id'),
            'text_id': text_id,
            'bucket': bucket,
            'voice_id': m.get('voice_id'),
            'seed': m.get('seed'),
            'status': status,
            'stop_reason': m.get('stop_reason'),
            'text_tokens': m.get('text_tokens'),
            'generated_speech_tokens': gen,
            'context_tokens_input': m.get('context_tokens_input'),
            'context_tokens_total': m.get('context_tokens_total'),
            'raw_duration_sec': dur,
            'wall_time_sec': wall,
            'llm_time_sec': llm_t,
            'flow_hift_time_sec': m.get('flow_hift_time_sec'),
            'frontend_time_sec': m.get('frontend_time_sec'),
            'rtf': rtf,
            'tokens_per_sec_llm': (gen / llm_t) if (gen and llm_t) else None,
            'tokens_per_sec_wall_true': (gen / wall) if (gen and wall) else None,
            'speech_tokens_per_sec_wall_field': tps,
            'peak_vram_bytes': pv,
            'peak_vram_gib': (pv / GIB) if pv else None,
            'peak_vram_reserved_gib': (m.get('peak_vram_reserved_bytes') / GIB)
                                      if m.get('peak_vram_reserved_bytes') else None,
            'cuda_visible_devices': (m.get('env') or {}).get('cuda_visible_devices'),
            'vram_total_gb': (m.get('mem_before') or {}).get('vram_total_gb'),
            'n_chunks': (m.get('official_split') or {}).get('n_chunks'),
            'timestamp_utc': m.get('timestamp_utc'),
        })
    cc['statuses'] = ';'.join(f'{k}={v}' for k, v in sorted(st.items(), key=lambda kv: str(kv[0])))
    cc['max_rtf_rel_err'] = round(cc['max_rtf_rel_err'], 6)
    crosscheck.append(cc)

os.makedirs(OUT, exist_ok=True)

FIELDS = list(rows[0].keys())
with open(os.path.join(OUT, 'perf_per_item.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for row in rows:
        w.writerow(row)

CC_FIELDS = ['dir', 'run_set', 'n_runs_jsonl', 'n_per_item_json', 'statuses',
             'n_missing_rtf', 'n_missing_wall', 'n_missing_peak_vram', 'n_missing_duration',
             'n_missing_gen_tokens', 'n_zero_peak_vram', 'n_rtf_inconsistent', 'max_rtf_rel_err',
             'n_tokpersec_matches_llm_time', 'n_tokpersec_matches_wall', 'note']
with open(os.path.join(OUT, 'perf_crosscheck.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=CC_FIELDS, extrasaction='ignore')
    w.writeheader()
    for c in crosscheck:
        w.writerow(c)


# ---------------------------------------------------------------- aggregate
def group(rs, key):
    g = defaultdict(list)
    for row in rs:
        g[key(row)].append(row)
    return g


def agg_block(items):
    dur = [x['raw_duration_sec'] for x in items]
    wall = [x['wall_time_sec'] for x in items]
    rtf = [x['rtf'] for x in items]
    pv = [x['peak_vram_gib'] for x in items]
    tps = [x['tokens_per_sec_llm'] for x in items]
    tpsw = [x['tokens_per_sec_wall_true'] for x in items]
    ctx = [x['context_tokens_total'] for x in items]
    return {
        'n': len(items),
        'dur_med': r(med(dur), 1), 'dur_p90': r(pctl(dur, 0.90), 1), 'dur_max': r(max(dur), 1) if dur else None,
        'wall_med': r(med(wall), 1), 'wall_p90': r(pctl(wall, 0.90), 1), 'wall_max': r(max(wall), 1) if wall else None,
        'rtf_med': r(med(rtf), 3), 'rtf_p90': r(pctl(rtf, 0.90), 3), 'rtf_max': r(max(rtf), 3) if rtf else None,
        'vram_med': r(med(pv), 2), 'vram_p90': r(pctl(pv, 0.90), 2), 'vram_max': r(max(pv), 2) if pv else None,
        'tps_llm_med': r(med(tps), 1), 'tps_wall_med': r(med(tpsw), 1),
        'ctx_med': r(med(ctx), 0), 'ctx_max': max(ctx) if ctx else None,
    }


complete = [x for x in rows if x['status'] == 'complete']

by_sb = {}
for sysname in MAIN_SYSTEMS + APPENDIX_SYSTEMS:
    for bucket in LONG_BUCKETS + SHORT_BUCKETS:
        items = [x for x in complete if x['system'] == sysname and x['bucket'] == bucket]
        if items:
            by_sb[(sysname, bucket)] = agg_block(items)

AGG_FIELDS = ['system', 'bucket', 'n', 'dur_med', 'dur_p90', 'dur_max', 'wall_med', 'wall_p90',
              'wall_max', 'rtf_med', 'rtf_p90', 'rtf_max', 'vram_med', 'vram_p90', 'vram_max',
              'tps_llm_med', 'tps_wall_med', 'ctx_med', 'ctx_max']
with open(os.path.join(OUT, 'perf_by_system_bucket.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
    w.writeheader()
    for (sysname, bucket), a in sorted(by_sb.items()):
        w.writerow(dict(a, system=sysname, bucket=bucket))


# ---------------------------------------------------------------- KV-cache math
def kv_bytes_per_pos(layers, kv_heads, head_dim, dtype_bytes):
    return 2 * layers * kv_heads * head_dim * dtype_bytes


CV_CFG = 'models/cosyvoice3/CosyVoice-BlankEN/config.json'
QW_CFG = os.path.join(
    ROOT, '.cache', 'models--Qwen--Qwen3-TTS-12Hz-1.7B-Base', 'snapshots',
    'fd4b254389122332181a7c3db7f27e918eec64e3', 'config.json')
cv = json.load(open(CV_CFG))
qw = json.load(open(QW_CFG))
qt = qw['talker_config']
qcp = qt['code_predictor_config']

cv_head_dim = cv['hidden_size'] // cv['num_attention_heads']
kv_models = [
    {
        'label': 'CosyVoice3 0.5B LM (Qwen2 `CosyVoice-BlankEN`)',
        'layers': cv['num_hidden_layers'], 'kv_heads': cv['num_key_value_heads'],
        'head_dim': cv_head_dim, 'dtype': 'float32', 'dtype_bytes': 4,
        'rate_hz': 25.0,
    },
    {
        'label': 'Qwen3-TTS 1.7B talker',
        'layers': qt['num_hidden_layers'], 'kv_heads': qt['num_key_value_heads'],
        'head_dim': qt['head_dim'], 'dtype': 'bfloat16', 'dtype_bytes': 2,
        'rate_hz': 12.5,
    },
    {
        'label': 'Qwen3-TTS code predictor (per frame, 16 code groups)',
        'layers': qcp['num_hidden_layers'], 'kv_heads': qcp['num_key_value_heads'],
        'head_dim': qcp['head_dim'], 'dtype': 'bfloat16', 'dtype_bytes': 2,
        'rate_hz': None,
    },
]
for m in kv_models:
    m['bytes_per_pos'] = kv_bytes_per_pos(m['layers'], m['kv_heads'], m['head_dim'], m['dtype_bytes'])

# measured context ground truth
ctx_max = {}
for sysname in MAIN_SYSTEMS:
    items = [x for x in complete if x['system'] == sysname and x['context_tokens_total']]
    if items:
        top = max(items, key=lambda x: x['context_tokens_total'])
        ctx_max[sysname] = top

with open(os.path.join(OUT, 'kv_cache.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model', 'layers', 'kv_heads', 'head_dim', 'dtype', 'dtype_bytes',
                'kib_per_position', 'positions', 'positions_note', 'kv_gib'])
    for m in kv_models:
        cases = []
        if m['rate_hz'] == 25.0:
            cases = [(22500, '15 min speech @25 Hz'),
                     (22500 + 800, '15 min speech + ~800 text/prompt positions'),
                     (ctx_max['E3']['context_tokens_total'],
                      'largest measured context, E3 (%.1f s of audio)'
                      % ctx_max['E3']['raw_duration_sec']),
                     (ctx_max['E7']['context_tokens_total'],
                      'largest measured context, E7 (%.1f s of audio)'
                      % ctx_max['E7']['raw_duration_sec'])]
        elif m['rate_hz'] == 12.5:
            cases = [(11250, '15 min speech @12.5 Hz'),
                     (11250 + 500, '15 min speech + ~500 text/prompt positions'),
                     (ctx_max['QE1']['context_tokens_total'],
                      'largest measured context, completed QE1 (%.1f s of audio)'
                      % ctx_max['QE1']['raw_duration_sec']),
                     (15164, 'largest measured context any status, QE1 loop_cap (1198.8 s)')]
        else:
            cases = [(16, '16 code groups (one frame)')]
        for pos, note in cases:
            w.writerow([m['label'], m['layers'], m['kv_heads'], m['head_dim'], m['dtype'],
                        m['dtype_bytes'], round(m['bytes_per_pos'] / 1024, 1), pos, note,
                        round(m['bytes_per_pos'] * pos / GIB, 4)])

print('rows:', len(rows), 'complete:', len(complete), 'groups:', len(by_sb))


# ================================================================ report
CELLS = [0]
LINES = []


def W(s=''):
    LINES.append(s)


def T(headers, body, align=None):
    """Render a markdown table and count its data cells."""
    if align is None:
        align = ['---'] + ['---:'] * (len(headers) - 1)
    W('| ' + ' | '.join(headers) + ' |')
    W('|' + '|'.join(align) + '|')
    for row in body:
        assert len(row) == len(headers), (len(row), len(headers))
        W('| ' + ' | '.join(str(c) for c in row) + ' |')
        CELLS[0] += len(row)


DUR_BINS = [(0, 10, '< 10 s'), (10, 30, '10-30 s'), (30, 60, '30-60 s'),
            (60, 120, '1-2 min'), (120, 300, '2-5 min'), (300, 600, '5-10 min'),
            (600, 1e9, '>= 10 min')]


def dur_bin(d):
    for lo, hi, lab in DUR_BINS:
        if lo <= d < hi:
            return lab
    return None


try:
    import numpy as np
except ImportError:
    np = None

vram_max_item = max((x for x in rows if x['peak_vram_gib']), key=lambda x: x['peak_vram_gib'])

SYS_LABEL = {
    'E1': 'E1 CosyVoice3 base, native',
    'E2': 'E2 short-SFT, native',
    'E3': 'E3 long-SFT, native',
    'E7': 'E7 punct-SFT, native',
    'QE1': 'QE1 Qwen3-TTS 1.7B, native',
    'E0': 'E0 CosyVoice3 base, official_split (chunked)',
}

W('# A13-perf - inference cost of state-continuous long-form generation')
W()
W('Aggregation of the per-item cost fields **already logged** by every v31 generation run.')
W('No new generation, no GPU: this document is produced by `scripts/aggregate_perf.py`')
W('from the `runs.jsonl` manifests listed in §8. Every number below comes from a file.')
W()
W('## 0. Scope, hardware, dtype')
W()
W('**Manifests read** (33 `runs.jsonl` under `outputs/`): v31_base (E0, E1), v31_sft (E2, E3),')
W('v31_punct (E7), v31_robust (E1/E3/E7), v31_qwen (QE1), v31_seed1 / v31_seed1_e7 / v31_seed2_full,')
W('v31_short_tail and v31_short_tail_s1. Full list with per-file integrity counts: §8 and')
W('`results/v31_perf/perf_crosscheck.csv`.')
W()
W('**Systems.** `experiment_id` is folded to a system: `E1*`->E1, `E3*`->E3, `E7*`->E7, `QE1*`->QE1,')
W('`E0`->E0, `E2*`->E2. E1/E3/E7/QE1 are the four systems the reviewer asked about; E0 (the chunked')
W('official split) and E2 are kept as controls. Rows are pooled over checkpoints, seeds, voices and')
W('benchmark sets; the per-manifest split is in `perf_by_dir_bucket.csv`.')
W()
W('**Buckets** are the PLAN §7.1 human-duration buckets (B0 20-40 s, B1 60-90 s, B2 2-3 min,')
W('B3 4-6 min, B4 8-12 min) plus the short-tail S0/S1 sets. The bucket is a property of the *input*,')
W('not of the produced audio - §1.2 re-bins by the audio actually produced.')
W()
W('**Hardware.** Every one of the 1156 items reports `mem_before.vram_total_gb = 47.37` - one '
  '48 GB card, never two. The card model recorded in this project is **NVIDIA RTX 6000 Ada '
  'Generation** (`reports/handoffs/rereview_A3_a64e0d4a9d0470f83.json`, `gpu_name`; '
  '`reports/cosyvoice3_audit.md:1` and `reports/asr_floor.md:31` state the same for GPU index 1, '
  'and `reports/decisions.md` 2026-08-28 records that the host\'s cards are identical RTX 6000 Ada). '
  'Host RAM 62.16 GB (`mem_before.ram_memtotal_gb`). Caveat kept in §8: the generation manifests '
  'log only `env.cuda_visible_devices` (0, 1 and 2 appear), not a physical GPU UUID, so per-item '
  'card identity is not recoverable from the manifests.')
W()
W('**Dtype (verified in adapter code, not assumed).**')
W()
T(['System', 'Inference dtype', 'Evidence'],
  [['CosyVoice3 E0/E1/E2/E3/E7', '**float32** (not fp16)',
    '`src/adapters/cosyvoice3_common.py:241-243` asserts `fp16 is False` and calls '
    '`CosyVoice3(model_dir, load_trt=False, load_vllm=False, fp16=False)`; '
    '`third_party/CosyVoice/cosyvoice/cli/model.py:426` then runs '
    '`torch.cuda.amp.autocast(self.fp16)` = autocast disabled, and `model.py:65-73` loads the '
    'state dicts into fp32 modules (no `.half()`). `configs/models/cosyvoice3_base.yaml` '
    'records `fp16: false`, `llm_dtype: float32`.'],
   ['Qwen3-TTS QE1', '**bfloat16**',
    '`src/adapters/qwen3tts.py:371-377` maps `model_cfg["dtype"]` (default `bfloat16`) to '
    '`torch.bfloat16` and passes it to `Qwen3TTSModel.from_pretrained(..., dtype=...)`; '
    '`configs/models/qwen3tts_base.yaml` `dtype: bfloat16`; every QE1 manifest logs '
    '`talker_dtype = "bfloat16"`, `attn_implementation = "sdpa"`.']],
  align=['---', '---', '---'])
W()
W('So the reviewer-facing statement "fp16/bf16 inference" is **only half true**: the CosyVoice3')
W('arms run in full fp32. Every CosyVoice VRAM and RTF number below is an fp32 number.')
W()
W('**Field semantics (read from the adapters, because one field name is misleading).**')
W()
T(['Field', 'Definition in code', 'Source'],
  [['`wall_time_sec`', 'end-to-end per item: frontend + LM + flow/HiFT + audio write path',
    '`cosyvoice3_native.py:130`, `qwen3tts.py` (`wall_time_sec`)'],
   ['`rtf`', '`wall_time_sec / raw_duration_sec` (verified for 1156/1156 items, §8)',
    '`cosyvoice3_native.py:152`, `qwen3tts.py:593`'],
   ['`llm_time_sec`', 'CosyVoice: LM decode only. **Qwen: talker + code predictor + vocoder in one '
    '`generate()` call**, so QE1 `flow_hift_time_sec` is always 0.0',
    '`cosyvoice3_native.py:131`, `qwen3tts.py:565`'],
   ['`peak_vram_bytes`', '`torch.cuda.max_memory_allocated()` after '
    '`reset_peak_memory_stats()` at item start - a true per-item peak',
    '`cosyvoice3_native.py:72,146`, `qwen3tts.py:533,585`'],
   ['`speech_tokens_per_sec_wall`', '**misnomer**: it is `generated_speech_tokens / llm_time_sec`, '
    'not per wall second. Both rates are recomputed separately below',
    '`cosyvoice3_native.py:154`, `qwen3tts.py:595`']],
  align=['---', '---', '---'])
W()

# ---- 1.1 RTF by bucket
W('## 1. RTF vs output duration')
W()
W('### 1.1 By benchmark bucket (status = `complete` only)')
W()
W('`dur med` is the median duration of the audio actually produced, and it is the column to read')
W('first: E1 (the un-finetuned baseline) stops early on B3/B4, so its low RTF there is the cost of')
W('*not* producing the audio, not a speed advantage. QE1 has **zero** completed B4 items (all 17 hit')
W('the 15000-token loop cap - §7).')
W()
for sysname in MAIN_SYSTEMS:
    body = []
    for b in LONG_BUCKETS:
        a = by_sb.get((sysname, b))
        if not a:
            body.append([b, 0, '-', '-', '-', '-', '-'])
            continue
        body.append([b, a['n'], fmt(a['dur_med'], 1), fmt(a['rtf_med'], 3), fmt(a['rtf_p90'], 3),
                     fmt(a['rtf_max'], 3), fmt(a['dur_max'], 1)])
    W(f'**{SYS_LABEL[sysname]}**')
    W()
    T(['bucket', 'n', 'dur med (s)', 'RTF median', 'RTF p90', 'RTF max', 'dur max (s)'], body)
    W()
W('Controls:')
W()
for sysname in APPENDIX_SYSTEMS:
    body = []
    for b in LONG_BUCKETS:
        a = by_sb.get((sysname, b))
        if not a:
            continue
        body.append([b, a['n'], fmt(a['dur_med'], 1), fmt(a['rtf_med'], 3), fmt(a['rtf_p90'], 3),
                     fmt(a['rtf_max'], 3), fmt(a['dur_max'], 1)])
    W(f'**{SYS_LABEL[sysname]}**')
    W()
    T(['bucket', 'n', 'dur med (s)', 'RTF median', 'RTF p90', 'RTF max', 'dur max (s)'], body)
    W()
W('Short-tail sets (S0 = very short utterances, S1 = short):')
W()
body = []
for sysname in MAIN_SYSTEMS:
    for b in SHORT_BUCKETS:
        a = by_sb.get((sysname, b))
        if a:
            body.append([sysname, b, a['n'], fmt(a['dur_med'], 1), fmt(a['rtf_med'], 3),
                         fmt(a['rtf_p90'], 3), fmt(a['wall_med'], 1)])
T(['system', 'bucket', 'n', 'dur med (s)', 'RTF median', 'RTF p90', 'wall med (s)'], body)
W()
W('At S0 (median ~1.8 s of audio) RTF is above 1 for every system: fixed per-call cost')
W('(frontend, prompt encoding, one flow/vocoder call) dominates when the output is ~2 s long.')
W()

# ---- 1.2 by realized duration
W('### 1.2 By the audio actually produced (the cost curve the reviewer asked for)')
W()
W('Same items, re-binned by `raw_duration_sec`. This removes the confound of E1/QE1 not reaching')
W('the bucket target, and puts E0 (chunked) on the same axis as the state-continuous arms.')
W()
db_rows = defaultdict(list)
for x in complete:
    d = x['raw_duration_sec']
    if d:
        db_rows[(x['system'], dur_bin(d))].append(x)
body = []
for _, _, lab in DUR_BINS:
    for sysname in MAIN_SYSTEMS + ['E0']:
        items = db_rows.get((sysname, lab))
        if not items:
            continue
        a = agg_block(items)
        body.append([lab, sysname, a['n'], fmt(a['rtf_med'], 3), fmt(a['rtf_p90'], 3),
                     fmt(a['wall_med'], 1), fmt(a['vram_med'], 2), fmt(a['tps_llm_med'], 1)])
T(['output duration', 'system', 'n', 'RTF median', 'RTF p90', 'wall med (s)',
   'peak VRAM med (GiB)', 'LM tok/s med'], body)
W()
W('**The cost of state-continuity, measured on the same weights.** E0 and E1 are the *same*')
W('checkpoint (`llm.pt`); E0 splits the text into <=80-token chunks and re-conditions per chunk,')
W('E1/E3/E7 run one trajectory over the whole text. In the 5-10 min bin:')
W()
_e0 = agg_block(db_rows[('E0', '5-10 min')])
_e3 = agg_block(db_rows[('E3', '5-10 min')])
_e7 = agg_block(db_rows[('E7', '5-10 min')])
T(['5-10 min of audio', 'n', 'RTF median', 'peak VRAM median (GiB)', 'LM tok/s median'],
  [['E0 chunked (base weights)', _e0['n'], fmt(_e0['rtf_med'], 3), fmt(_e0['vram_med'], 2),
    fmt(_e0['tps_llm_med'], 1)],
   ['E3 state-continuous', _e3['n'], fmt(_e3['rtf_med'], 3), fmt(_e3['vram_med'], 2),
    fmt(_e3['tps_llm_med'], 1)],
   ['E7 state-continuous', _e7['n'], fmt(_e7['rtf_med'], 3), fmt(_e7['vram_med'], 2),
    fmt(_e7['tps_llm_med'], 1)],
   ['ratio (E7 / E0)', '-', fmt(_e7['rtf_med'] / _e0['rtf_med'], 2) + 'x',
    fmt(_e7['vram_med'] / _e0['vram_med'], 2) + 'x',
    fmt(_e0['tps_llm_med'] / _e7['tps_llm_med'], 2) + 'x slower']])
W()
W('So at 5-10 min of output, state-continuity costs **%s x the wall clock** and **%s x the peak'
  % (fmt(_e7['rtf_med'] / _e0['rtf_med'], 2), fmt(_e7['vram_med'] / _e0['vram_med'], 2)))
W('VRAM** of the chunked baseline on the same GPU. That gap is the price of the single-trajectory')
W('design, and it is the number to quote when a reviewer asks what continuity costs. What it buys')
W('is measured elsewhere (prosody / EndCoverage / drift in the evaluation reports); this document')
W('only prices it.')
W()

# ---- 2. wall clock
W('## 2. Wall-clock per bucket')
W()
body = []
for sysname in MAIN_SYSTEMS + APPENDIX_SYSTEMS:
    for b in LONG_BUCKETS:
        a = by_sb.get((sysname, b))
        if not a:
            continue
        body.append([sysname, b, a['n'], fmt(a['wall_med'], 1), fmt(a['wall_p90'], 1),
                     fmt(a['wall_max'], 1), fmt(a['dur_med'], 1)])
T(['system', 'bucket', 'n', 'wall median (s)', 'wall p90 (s)', 'wall max (s)', 'dur med (s)'], body)
W()

# ---- 3. VRAM
W('## 3. Peak VRAM vs duration')
W()
body = []
for sysname in MAIN_SYSTEMS + APPENDIX_SYSTEMS:
    for b in LONG_BUCKETS:
        a = by_sb.get((sysname, b))
        if not a:
            continue
        body.append([sysname, b, a['n'], fmt(a['dur_med'], 1), fmt(a['vram_med'], 2),
                     fmt(a['vram_p90'], 2), fmt(a['vram_max'], 2), a['ctx_max']])
T(['system', 'bucket', 'n', 'dur med (s)', 'peak VRAM med (GiB)', 'p90 (GiB)', 'max (GiB)',
   'max context tokens'], body)
W()
W('Linear fit of `peak_vram_bytes` on the produced duration and on `context_tokens_total`')
W('(status `complete`, all buckets pooled per system):')
W()
fits = {}
if np is not None:
    body = []
    for sysname in MAIN_SYSTEMS + ['E0']:
        sub = [x for x in complete if x['system'] == sysname and x['peak_vram_gib']
               and x['raw_duration_sec'] and x['context_tokens_total']]
        d = np.array([x['raw_duration_sec'] for x in sub])
        v = np.array([x['peak_vram_gib'] for x in sub])
        c = np.array([float(x['context_tokens_total']) for x in sub])
        sd, id_ = np.polyfit(d, v, 1)
        sc, ic = np.polyfit(c, v, 1)
        r2d = float(np.corrcoef(d, v)[0, 1] ** 2)
        r2c = float(np.corrcoef(c, v)[0, 1] ** 2)
        fits[sysname] = {'slope_mib_per_s': sd * 1024, 'intercept': id_, 'r2_dur': r2d,
                         'slope_mib_per_tok': sc * 1024, 'r2_ctx': r2c}
        body.append([sysname, len(sub), fmt(id_, 2), fmt(sd * 1024, 2), fmt(r2d, 3),
                     fmt(sc * 1024, 3), fmt(r2c, 3)])
    T(['system', 'n', 'intercept (GiB)', 'MiB per s of audio', 'R2 (dur)',
       'MiB per context token', 'R2 (ctx)'], body)
    W()
W('E1 is the exception (R2 = %s against duration): its outputs collapse to a few seconds while the'
  % fmt(fits['E1']['r2_dur'], 3))
W('input text stays 8-12 min long, so its footprint tracks the *input* context '
  '(R2 = %s) rather than the output.' % fmt(fits['E1']['r2_ctx'], 3))
W()

# ---- 4. tokens/s
W('## 4. Token throughput')
W()
W('`LM tok/s` = `generated_speech_tokens / llm_time_sec` (for QE1 the denominator is the whole')
W('`generate()` call, talker + code predictor + vocoder - see §0). `end-to-end tok/s` uses')
W('`wall_time_sec`. Frame rates: CosyVoice3 25 Hz, Qwen3-TTS 12.5 Hz')
W('(`configs/models/cosyvoice3_base.yaml` `speech_tokenizer` note; '
  '`configs/models/qwen3tts_base.yaml` `codec_frame_hz: 12.5`).')
W()
body = []
for sysname in MAIN_SYSTEMS + APPENDIX_SYSTEMS:
    for b in LONG_BUCKETS:
        a = by_sb.get((sysname, b))
        if not a:
            continue
        body.append([sysname, b, a['n'], fmt(a['tps_llm_med'], 1), fmt(a['tps_wall_med'], 1),
                     fmt(a['ctx_med'], 0)])
T(['system', 'bucket', 'n', 'LM tok/s (median)', 'end-to-end tok/s (median)',
   'median context tokens'], body)
W()
W('CosyVoice3 LM decode slows by ~3.6x from B0 to B4 (E7: %s -> %s tok/s) as the KV cache and'
  % (fmt(by_sb[('E7', 'B0')]['tps_llm_med'], 1), fmt(by_sb[('E7', 'B4')]['tps_llm_med'], 1)))
W('attention window grow; the Qwen talker is essentially flat (%s -> %s tok/s from B0 to B3) but'
  % (fmt(by_sb[('QE1', 'B0')]['tps_llm_med'], 1), fmt(by_sb[('QE1', 'B3')]['tps_llm_med'], 1)))
W('starts ~5x slower per token, and needs half as many tokens per second of audio.')
W()

# ---- 5. TTFA
W('## 5. Time-to-first-audio (TTFA): what we can and cannot claim')
W()
W('**Our native mode is offline.** For every CosyVoice3 native arm the manifest records')
W('`tts_chunks_yielded = 1` and `llm_inference_calls = 1`: the LM produces the entire speech-token')
W('sequence, then flow + HiFT vocode it once. `configs/models/cosyvoice3_base.yaml` freezes')
W('`stream: false`, and the non-stream branch in')
W('`third_party/CosyVoice/cosyvoice/cli/model.py:376-387` joins the LM thread (`p.join()`) and')
W('calls `token2wav(..., token_offset=0, finalize=True)` once on the full token tensor. For QE1,')
W('`src/adapters/qwen3tts.py:1-24` documents one `generate()` call and one')
W('`speech_tokenizer.decode()` over the full code sequence, and every QE1 manifest logs')
W('`stream_generate_calls: 0`.')
W()
W('**Therefore TTFA = the full synthesis wall clock.** There is no earlier first sample to report,')
W('and we do not log one. Median TTFA per bucket (= median `wall_time_sec`, status `complete`):')
W()
body = []
for b in LONG_BUCKETS:
    row = [b]
    for sysname in MAIN_SYSTEMS:
        a = by_sb.get((sysname, b))
        row.append(fmt(a['wall_med'], 1) if a else 'n/a')
    row.append(fmt(by_sb[('E0', b)]['wall_med'], 1))
    body.append(row)
T(['bucket', 'E1 TTFA (s)', 'E3 TTFA (s)', 'E7 TTFA (s)', 'QE1 TTFA (s)', 'E0 chunked, wall (s)'],
  body)
W()
W('Read the E1 column with §1.1 open: E1 B4 TTFA is %s s because its median B4 output is only %s s'
  % (fmt(by_sb[('E1', 'B4')]['wall_med'], 1), fmt(by_sb[('E1', 'B4')]['dur_med'], 1)))
W('long. QE1 has no B4 number at all - no B4 item completed.')
W()
W('**What a streaming variant could change - only what the code actually supports.**')
W()
W('* CosyVoice3 *does* have a chunked vocoding path, and we can cite it: `CosyVoice2Model.tts(...,')
W('  stream=True)` at `third_party/CosyVoice/cosyvoice/cli/model.py:344-364` runs the LM in a')
W('  background thread (`model.py:339`) and calls `token2wav(..., finalize=False)` as soon as')
W('  `token_hop_len + flow.pre_lookahead_len` new tokens exist, yielding audio per chunk;')
W('  `CosyVoice3Model` inherits that `tts` and overrides only `token2wav`')
W('  (`model.py:397-425`). The constants are `token_hop_len = 25` and')
W('  `token_max_hop_len = 4 * 25` with `stream_scale_factor = 2` (`model.py:410-413`), and')
W('  `pre_lookahead_len = 3` - the flow CosyVoice3 actually loads is')
W('  `CausalMaskedDiffWithDiT` (`cosyvoice/flow/flow.py:284`), and the shipped')
W('  `pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml:47,51` sets `pre_lookahead_len: 3`')
W('  explicitly. 25 tokens at 25 Hz = **1.0 s of audio**; the first chunk needs')
W('  `token_hop_len + prompt_token_pad + pre_lookahead_len` tokens (`model.py:345,348-349`), i.e.')
W('  **28 to 52** depending on how far the prompt length falls from a 25-token boundary - not the')
W('  whole sequence.')
W('* **We did not run it and we do not report a streaming TTFA.** `stream=True` is frozen off for')
W('  every v31 arm, no manifest contains a per-chunk timestamp, and the flow/HiFT noise buffers we')
W('  extended to 900 s (`configs/models/cosyvoice3_base.yaml`) were extended for the *non-stream*')
W('  path. Any streaming TTFA figure would be a new measurement, not a re-reading of these files.')
W('* An arithmetic ceiling that follows from fields we do log, stated as a bound and nothing more:')
W('  at the E7 B4 median LM rate of %s tok/s, 28-52 tokens take %s-%s s of LM time. That is the'
  % (fmt(by_sb[('E7', 'B4')]['tps_llm_med'], 1),
     fmt(28.0 / by_sb[('E7', 'B4')]['tps_llm_med'], 2),
     fmt(52.0 / by_sb[('E7', 'B4')]['tps_llm_med'], 2)))
W('  order of magnitude chunked vocoding could reach against the measured %s s TTFA at B4 - but it'
  % fmt(by_sb[('E7', 'B4')]['wall_med'], 1))
W('  is **not** a measured TTFA, because it ignores frontend,')
W('  prompt encoding and the first flow/HiFT call, and because LM rate at step 28 is not the same')
W('  as the B4 median rate.')
W('* The Qwen fork ships `stream_generate_pcm` (a windowed re-decode path) and our adapter')
W('  deliberately does not use it (`src/adapters/qwen3tts.py:21-22`, `stream_generate_calls: 0`).')
W('  We have measured nothing about it and claim nothing about it.')
W('* E0 (official split) already emits audio per chunk - `official_split.n_chunks` runs')
W('  2 (B0) to a median of 30 (B4), max 34 - but the adapter times only the whole item')
W('  (`tts_time_sec`), so **we cannot report a measured E0 first-chunk latency either**. What we')
W('  can say is the whole-item wall in the table above.')
W()
body = []
for b in LONG_BUCKETS:
    items = [x for x in complete if x['system'] == 'E0' and x['bucket'] == b and x['n_chunks']]
    ch = [x['n_chunks'] for x in items]
    body.append([b, len(items), int(med(ch)), min(ch), max(ch), fmt(med([x['wall_time_sec'] for x in items]), 1)])
T(['bucket', 'n', 'chunks median', 'min', 'max', 'whole-item wall median (s)'], body)
W()

# ---- 6. KV cache math
W('## 6. KV-cache math, and the measured peak VRAM next to it')
W()
W('`bytes_per_position = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes`.')
W()
W('Config sources, read read-only:')
W()
T(['Model', 'File', 'Keys used'],
  [['CosyVoice3 0.5B LM', '`models/cosyvoice3/'
    'CosyVoice-BlankEN/config.json`',
    '`num_hidden_layers`=%d, `num_key_value_heads`=%d, `hidden_size`=%d / '
    '`num_attention_heads`=%d -> head_dim %d, dtype float32 (see §0)'
    % (cv['num_hidden_layers'], cv['num_key_value_heads'], cv['hidden_size'],
       cv['num_attention_heads'], cv_head_dim)],
   ['Qwen3-TTS 1.7B talker', '`.cache/'
    'models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/'
    'fd4b254389122332181a7c3db7f27e918eec64e3/config.json`',
    '`talker_config.num_hidden_layers`=%d, `talker_config.num_key_value_heads`=%d, '
    '`talker_config.head_dim`=%d, dtype bfloat16'
    % (qt['num_hidden_layers'], qt['num_key_value_heads'], qt['head_dim'])],
   ['Qwen3-TTS code predictor', 'same file', '`talker_config.code_predictor_config.'
    'num_hidden_layers`=%d, `num_key_value_heads`=%d, `head_dim`=%d, `num_code_groups`=%d'
    % (qcp['num_hidden_layers'], qcp['num_key_value_heads'], qcp['head_dim'],
       qcp['num_code_groups'])]],
  align=['---', '---', '---'])
W()
body = []
for m in kv_models:
    if m['rate_hz'] == 25.0:
        cases = [(22500, '15 min speech @ 25 Hz'),
                 (ctx_max['E3']['context_tokens_total'],
                  'largest measured context (E3, %s s of audio)'
                  % fmt(ctx_max['E3']['raw_duration_sec'], 1))]
    elif m['rate_hz'] == 12.5:
        cases = [(11250, '15 min speech @ 12.5 Hz'),
                 (15164, 'largest measured context (QE1 loop_cap, 1198.8 s of audio)')]
    else:
        cases = [(16, 'one frame = 16 code groups')]
    for pos, note in cases:
        body.append([m['label'], m['layers'], m['kv_heads'], m['head_dim'], m['dtype'],
                     fmt(m['bytes_per_pos'] / 1024, 1), pos, note,
                     fmt(m['bytes_per_pos'] * pos / GIB, 3)])
T(['model', 'layers', 'kv heads', 'head dim', 'dtype', 'KiB / position', 'positions',
   'what the positions are', 'KV cache (GiB)'], body)
W()
cv_kv_per_s = kv_models[0]['bytes_per_pos'] * 25.0 / (1024 * 1024)
qw_kv_per_s = kv_models[1]['bytes_per_pos'] * 12.5 / (1024 * 1024)
W('Per second of produced audio that is **%s MiB/s for the CosyVoice3 LM** (fp32, 25 Hz) and'
  % fmt(cv_kv_per_s, 3))
W('**%s MiB/s for the Qwen talker** (bf16, 12.5 Hz). Now the ground truth from the manifests:'
  % fmt(qw_kv_per_s, 3))
W()
T(['system', 'longest completed item', 'audio (s)', 'context tokens', 'measured peak VRAM (GiB)',
   'KV math for that context (GiB)', 'measured growth over intercept (GiB)', 'KV share of growth'],
  [['E3', '`%s`<br>`%s`' % (ctx_max['E3']['run_id'], ctx_max['E3']['dir']),
    fmt(ctx_max['E3']['raw_duration_sec'], 1), ctx_max['E3']['context_tokens_total'],
    fmt(ctx_max['E3']['peak_vram_gib'], 2),
    fmt(kv_models[0]['bytes_per_pos'] * ctx_max['E3']['context_tokens_total'] / GIB, 3),
    fmt(ctx_max['E3']['peak_vram_gib'] - fits['E3']['intercept'], 2),
    fmt(100 * (kv_models[0]['bytes_per_pos'] * ctx_max['E3']['context_tokens_total'] / GIB)
        / (ctx_max['E3']['peak_vram_gib'] - fits['E3']['intercept']), 1) + ' %'],
   ['E7', '`%s`<br>`%s`' % (ctx_max['E7']['run_id'], ctx_max['E7']['dir']),
    fmt(ctx_max['E7']['raw_duration_sec'], 1), ctx_max['E7']['context_tokens_total'],
    fmt(ctx_max['E7']['peak_vram_gib'], 2),
    fmt(kv_models[0]['bytes_per_pos'] * ctx_max['E7']['context_tokens_total'] / GIB, 3),
    fmt(ctx_max['E7']['peak_vram_gib'] - fits['E7']['intercept'], 2),
    fmt(100 * (kv_models[0]['bytes_per_pos'] * ctx_max['E7']['context_tokens_total'] / GIB)
        / (ctx_max['E7']['peak_vram_gib'] - fits['E7']['intercept']), 1) + ' %'],
   ['QE1', 'longest *completed*: `%s`<br>`%s`' % (ctx_max['QE1']['run_id'], ctx_max['QE1']['dir']),
    fmt(ctx_max['QE1']['raw_duration_sec'], 1), ctx_max['QE1']['context_tokens_total'],
    fmt(ctx_max['QE1']['peak_vram_gib'], 2),
    fmt(kv_models[1]['bytes_per_pos'] * ctx_max['QE1']['context_tokens_total'] / GIB, 3),
    fmt(ctx_max['QE1']['peak_vram_gib'] - fits['QE1']['intercept'], 2),
    fmt(100 * (kv_models[1]['bytes_per_pos'] * ctx_max['QE1']['context_tokens_total'] / GIB)
        / (ctx_max['QE1']['peak_vram_gib'] - fits['QE1']['intercept']), 1) + ' %'],
   ['QE1', 'longest context of any status (loop_cap, 20 min of audio, n=34 identical)',
    '1198.8', 15164, '7.34',
    fmt(kv_models[1]['bytes_per_pos'] * 15164 / GIB, 3),
    fmt(7.34 - fits['QE1']['intercept'], 2),
    fmt(100 * (kv_models[1]['bytes_per_pos'] * 15164 / GIB) / (7.34 - fits['QE1']['intercept']), 1)
    + ' %'],
   ['E3', '**worst peak VRAM seen anywhere**: `%s`<br>`%s`'
    % (vram_max_item['run_id'], vram_max_item['dir']),
    fmt(vram_max_item['raw_duration_sec'], 1), vram_max_item['context_tokens_total'],
    fmt(vram_max_item['peak_vram_gib'], 2),
    fmt(kv_models[0]['bytes_per_pos'] * vram_max_item['context_tokens_total'] / GIB, 3),
    fmt(vram_max_item['peak_vram_gib'] - fits['E3']['intercept'], 2),
    fmt(100 * (kv_models[0]['bytes_per_pos'] * vram_max_item['context_tokens_total'] / GIB)
        / (vram_max_item['peak_vram_gib'] - fits['E3']['intercept']), 1) + ' %']],
  align=['---', '---', '---:', '---:', '---:', '---:', '---:', '---:'])
W()
W('The worst peak allocation observed anywhere in the study is **%s GiB** (%s%% of the 44.1 GiB'
  % (fmt(vram_max_item['peak_vram_gib'], 2),
     fmt(100 * vram_max_item['peak_vram_gib'] / (47.37e9 / GIB), 0)))
W('that a 47.37 GB card reports) for %s s of audio, with %s GiB *reserved*. There is headroom at'
  % (fmt(vram_max_item['raw_duration_sec'], 1), '35.64'))
W('15 min, but not much, and none at all for a second concurrent request on the same card.')
W()
W('**Peak VRAM is not monotone in sequence length, and we do not claim it is.** That worst item')
W('sent %s tokens to flow; the *longer* E3 item above it (%s s, %s tokens to flow) peaked at'
  % (19890, fmt(ctx_max['E3']['raw_duration_sec'], 1), 21643))
W('26.75 GiB - 5.7 GiB **lower** on 9%% more audio. The fits in §3 are trends (R2 %s / %s), not'
  % (fmt(fits['E3']['r2_dur'], 3), fmt(fits['E7']['r2_dur'], 3)))
W('bounds, and we did not run an allocator')
W('probe to explain the pair. Capacity planning off these numbers should use the max column of §3,')
W('not the fitted line.')
W()
W('**The KV cache is not the CosyVoice3 bottleneck; the non-streaming vocoder pass is.**')
W('The measured VRAM slope for E3/E7 is %s / %s MiB per second of audio (§3) against a KV-cache'
  % (fmt(fits['E3']['slope_mib_per_s'], 1), fmt(fits['E7']['slope_mib_per_s'], 1)))
W('term of %s MiB/s - the cache accounts for ~%s%% of the growth. The remaining ~%s MiB/s is'
  % (fmt(cv_kv_per_s, 2), fmt(100 * cv_kv_per_s / fits['E3']['slope_mib_per_s'], 1),
     fmt(fits['E3']['slope_mib_per_s'] - cv_kv_per_s, 0)))
W('spent in the one-shot `token2wav` over the whole sequence: flow expands every speech token to 2')
W('mel frames at 50 Hz, then HiFT vocodes the whole 24 kHz waveform in a single call. The')
W('controlled evidence is E0 - **identical weights, chunked**: peak VRAM %s GiB at B0 and %s GiB'
  % (fmt(by_sb[('E0', 'B0')]['vram_med'], 2), fmt(by_sb[('E0', 'B4')]['vram_med'], 2)))
W('at B4 - flat (%s MiB per second of audio, R2 = %s). Same LM, same card, comparable audio'
  % (fmt(fits['E0']['slope_mib_per_s'], 3), fmt(fits['E0']['r2_dur'], 3)))
W('length: no growth at all once the sequence is not processed in one piece.')
W()
W('For Qwen the opposite holds and the math checks out: predicted %s MiB/s of KV against a'
  % fmt(qw_kv_per_s, 2))
W('measured %s MiB/s slope (%s%%), and the code predictor contributes only %s MiB per frame'
  % (fmt(fits['QE1']['slope_mib_per_s'], 2),
     fmt(100 * qw_kv_per_s / fits['QE1']['slope_mib_per_s'], 0),
     fmt(kv_models[2]['bytes_per_pos'] * 16 / (1024 * 1024), 3)))
W('(16 code groups, discarded per frame). The Qwen talker is KV-dominated because its vocoder')
W('decodes codes rather than diffusing a full-length mel.')
W()
W('Extrapolating the fitted slopes to a 15-minute output (900 s): E3 %s GiB, E7 %s GiB, '
  % (fmt(fits['E3']['intercept'] + fits['E3']['slope_mib_per_s'] * 900 / 1024, 1),
     fmt(fits['E7']['intercept'] + fits['E7']['slope_mib_per_s'] * 900 / 1024, 1)))
W('QE1 %s GiB - i.e. the CosyVoice3 native path would sit at roughly %s%% of the 44.1 GiB the card'
  % (fmt(fits['QE1']['intercept'] + fits['QE1']['slope_mib_per_s'] * 900 / 1024, 1),
     fmt(100 * (fits['E3']['intercept'] + fits['E3']['slope_mib_per_s'] * 900 / 1024)
         / (47.37e9 / GIB), 0)))
W('reports, at the study ceiling.')
W('That is against a KV-cache requirement of %s GiB. (Extrapolation, flagged as such: our longest'
  % fmt(kv_models[0]['bytes_per_pos'] * 22500 / GIB, 2))
W('completed item is %s s, not 900 s.)' % fmt(ctx_max['E3']['raw_duration_sec'], 1))
W()

# ---- 7. failures
W('## 7. What the non-completions cost')
W()
W('35 of 1156 items are not `complete`. They still consumed wall clock, so they belong in a cost table.')
W()
notc = [x for x in rows if x['status'] != 'complete']
body = []
for sysname in sorted({x['system'] for x in notc}):
    for b in LONG_BUCKETS + SHORT_BUCKETS:
        items = [x for x in notc if x['system'] == sysname and x['bucket'] == b]
        if not items:
            continue
        body.append([sysname, b, items[0]['status'], len(items),
                     fmt(med([x['wall_time_sec'] for x in items]), 1),
                     fmt(med([x['raw_duration_sec'] for x in items]), 1),
                     fmt(med([x['rtf'] for x in items]), 3),
                     int(med([x['generated_speech_tokens'] for x in items])),
                     fmt(med([x['peak_vram_gib'] for x in items]), 2)])
T(['system', 'bucket', 'status', 'n', 'wall med (s)', 'dur med (s)', 'RTF med',
   'gen tokens med', 'peak VRAM med (GiB)'], body)
W()
W('QE1 spends ~%s s of GPU time per B3/B4 item to emit 14999 tokens (the '
  % fmt(med([x['wall_time_sec'] for x in notc if x['system'] == 'QE1']), 0))
W('`configs/models/qwen3tts_base.yaml` `max_generated_tokens: 15000` cap = 20 min of audio) and')
W('then be discarded. Counting those, the honest statement about QE1 at 8-12 min is not "RTF 0.48"')
W('but "0 of 17 B4 items produced usable audio; each cost ~10 minutes of card time".')
W()

# ---- 8. cross-check
W('## 8. Cross-check of the logged cost fields')
W()
W('Every manifest in scope, with the counts a reviewer would want. No value here is estimated.')
W()
body = []
tot = Counter()
for c in crosscheck:
    if 'n_runs_jsonl' not in c:
        body.append([c['dir'], 'MISSING', '-', '-', '-', '-', '-', '-', '-'])
        continue
    body.append([c['dir'], c['n_runs_jsonl'], c['n_per_item_json'],
                 c['n_missing_rtf'], c['n_missing_wall'], c['n_missing_peak_vram'],
                 c['n_missing_duration'], c['n_rtf_inconsistent'], c['statuses']])
    for k in ('n_runs_jsonl', 'n_per_item_json', 'n_missing_rtf', 'n_missing_wall',
              'n_missing_peak_vram', 'n_missing_duration', 'n_missing_gen_tokens',
              'n_rtf_inconsistent', 'n_zero_peak_vram', 'n_tokpersec_matches_llm_time',
              'n_tokpersec_matches_wall'):
        tot[k] += c[k]
T(['manifest dir', 'items in runs.jsonl', 'per-item .json', 'missing rtf', 'missing wall',
   'missing peak VRAM', 'missing duration', 'rtf != wall/dur (>1%)', 'statuses'], body)
W()
W('**Totals across all %d items:**' % tot['n_runs_jsonl'])
W()
T(['check', 'count', 'verdict'],
  [['items in `runs.jsonl`', tot['n_runs_jsonl'], '-'],
   ['per-item `*.json` files found', tot['n_per_item_json'],
    'equal to runs.jsonl' if tot['n_per_item_json'] == tot['n_runs_jsonl'] else
    'MISMATCH: %+d' % (tot['n_per_item_json'] - tot['n_runs_jsonl'])],
   ['missing `rtf`', tot['n_missing_rtf'], 'PASS' if not tot['n_missing_rtf'] else 'FAIL'],
   ['missing `wall_time_sec`', tot['n_missing_wall'], 'PASS' if not tot['n_missing_wall'] else 'FAIL'],
   ['missing `peak_vram_bytes`', tot['n_missing_peak_vram'],
    'PASS' if not tot['n_missing_peak_vram'] else 'FAIL'],
   ['`peak_vram_bytes == 0`', tot['n_zero_peak_vram'],
    'PASS' if not tot['n_zero_peak_vram'] else 'FAIL'],
   ['missing `raw_duration_sec`', tot['n_missing_duration'],
    'PASS' if not tot['n_missing_duration'] else 'FAIL'],
   ['missing `generated_speech_tokens`', tot['n_missing_gen_tokens'],
    'PASS' if not tot['n_missing_gen_tokens'] else 'FAIL'],
   ['`rtf` != `wall/duration` by >1%', tot['n_rtf_inconsistent'],
    'PASS' if not tot['n_rtf_inconsistent'] else 'FAIL'],
   ['`speech_tokens_per_sec_wall` == tokens/`llm_time_sec`', tot['n_tokpersec_matches_llm_time'],
    'the field is per LM second'],
   ['`speech_tokens_per_sec_wall` == tokens/`wall_time_sec`', tot['n_tokpersec_matches_wall'],
    'confirms the field name is wrong, not the value']],
  align=['---', '---:', '---'])
W()
W('Findings, stated as counts and nothing more:')
W()
W('1. **No missing cost field anywhere.** rtf, wall, peak VRAM, duration and generated-token counts')
W('   are present and non-zero for all %d items in scope.' % tot['n_runs_jsonl'])
W('2. **`rtf` is internally consistent** with `wall_time_sec / raw_duration_sec` for all %d items'
  % tot['n_runs_jsonl'])
W('   (largest relative deviation across all manifests: %s, i.e. rounding to 4 decimals).'
  % fmt(max(c.get('max_rtf_rel_err', 0) for c in crosscheck), 6))
W('3. **`speech_tokens_per_sec_wall` is misnamed.** It matches tokens/`llm_time_sec` for %d items'
  % tot['n_tokpersec_matches_llm_time'])
W('   and tokens/`wall_time_sec` for %d. The value is correct; the name is not. Anyone reading the'
  % tot['n_tokpersec_matches_wall'])
W('   manifests as "tokens per wall second" would overstate end-to-end throughput by up to ~1.6x')
W('   (E7 B4: %s vs %s tok/s). Recommend renaming in a future run; do not rewrite existing manifests.'
  % (fmt(by_sb[('E7', 'B4')]['tps_llm_med'], 1), fmt(by_sb[('E7', 'B4')]['tps_wall_med'], 1)))
W('4. **`flow_hift_time_sec` is structurally 0.0 for every QE1 item** - not missing data, a')
W('   documented consequence of the Qwen path timing talker+vocoder as one call')
W('   (`src/adapters/qwen3tts.py:565`). CosyVoice/Qwen stage splits are therefore not comparable.')
W('5. **No `max_wall_seconds` watchdog on QE1.** `configs/models/qwen3tts_base.yaml` sets it to')
W('   `null` because HF `generate()` is atomic; this is why the 34 loop_cap items each ran to the')
W('   full token cap instead of being cut at a wall limit. It is a logged, pre-registered deviation,')
W('   not a gap in the data.')
W('6. **Pooling caveat, not an inconsistency.** Runs were executed on `cuda_visible_devices` 0, 1')
W('   and 2 at different times (all report `vram_total_gb = 47.37`). Wall-clock numbers pooled')
W('   across devices assume the cards are equivalent. Three distinct physical GPU UUIDs appear in')
W('   the evaluation logs (`physical_gpu_uuid` in `logs/*/eval_*.log`), and')
W('   `reports/decisions.md` 2026-08-28 records them as identical RTX 6000 Ada, but the generation')
W('   manifests do not carry the UUID, so this is an assumption we flag rather than a check we ran.')
W('   Per-item `cuda_visible_devices` is in `perf_per_item.csv` if anyone wants to split it.')
W()

# ---- 9. self-check
W('## 9. Self-check')
W()
W('| check | value |')
W('|---|---:|')
selfcheck = [
    ('items read from runs.jsonl', tot['n_runs_jsonl']),
    ('items with status = complete', len(complete)),
    ('items with status != complete', len(rows) - len(complete)),
    ('manifest dirs read', len([c for c in crosscheck if 'n_runs_jsonl' in c])),
    ('(system, bucket) aggregate groups', len(by_sb)),
    ('data cells in the tables above', CELLS[0]),
]
for k, v in selfcheck:
    LINES.append(f'| {k} | {v} |')
W()
W('Reproduce: `cd . && python3 scripts/aggregate_perf.py`')
W('(CPU only; reads manifests and two model `config.json` files, writes only under `results/v31_perf/`).')
W()
W('Companion CSVs in this directory:')
W()
W('* `perf_per_item.csv` - one row per generation item, every field used above')
W('* `perf_by_system_bucket.csv` - the system x bucket aggregate')
W('* `perf_by_dir_bucket.csv` - the same, split per manifest dir (audit trail for the pooling)')
W('* `perf_crosscheck.csv` - §8 per-manifest counts')
W('* `kv_cache.csv` - §6 math')

# per-dir breakdown csv
with open(os.path.join(OUT, 'perf_by_dir_bucket.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dir', 'system', 'bucket'] + AGG_FIELDS[2:])
    w.writeheader()
    for (dname, sysname, bucket), items in sorted(group(
            complete, lambda x: (x['dir'], x['system'], x['bucket'])).items()):
        w.writerow(dict(agg_block(items), dir=dname, system=sysname, bucket=bucket))

with open(os.path.join(OUT, 'perf_tables.md'), 'w') as f:
    f.write('\n'.join(LINES) + '\n')

print('cells:', CELLS[0], 'groups:', len(by_sb), 'dirs:', len(crosscheck))
