#!/usr/bin/env python3
"""Per-sample context-budget check for CosyVoice3 LLM SFT (PLAN §11.2, A6).

How the CosyVoice3 training recipe forms a sequence (cosyvoice/llm/llm.py, CosyVoice3LM.forward ->
Qwen2LM.prepare_lm_input_target, "unistream" branch):

    lm_input  = [sos] + instruct_tokens + text_tokens + [task_id] + speech_tokens
    lm_target = [IGNORE]*(1 + n_instr + n_text) + speech_tokens + [eos]

There is NO separate prompt in SFT: the sample's own speech tokens are the target and the
sample's own text is the condition. Zero-shot prompting at inference is nothing more than a
(prompt_text, prompt_speech) prefix of exactly this layout (audit §4), so
prompt_speech_tokens = 0 in the training budget and the inference budget is checked by the
adapter (A3 watchdog, context_limit).

The "bistream" branch (chosen with p = 0.5 when speech/text ratio > mix_ratio[1]/mix_ratio[0])
interleaves 5 text : 15 speech tokens and inserts one fill token per 5-text group, i.e. up to
ceil((n_text + 1) / 5) extra positions. The budget below reserves those positions explicitly,
so the check is valid whichever branch the sampler picks. Whether the SFT uses bistream at all
is a recipe decision recorded in reports/decisions.md (configs/train/*.yaml, llm.mix_ratio).

Budget rule (frozen here):

    L = 1 (sos) + n_instruct + n_text + 1 (task_id) + n_speech + n_fill(bistream) + 1 (eos)
    L <= CONTEXT (32768) - MARGIN (1024)

n_speech is 25 Hz x duration (measured 25.00 tok/s, audit §5); the exact value from the
tokenizer is written back by prepare/extract stages and re-checked at training time by
processor_ext.budget_check (which raises instead of dropping).
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

COSYVOICE_ROOT = os.environ.get('COSYVOICE_ROOT', 'third_party/CosyVoice')
MODEL_DIR = os.path.join(COSYVOICE_ROOT, 'pretrained_models', 'Fun-CosyVoice3-0.5B')
QWEN_PATH = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')

CONTEXT_LIMIT = 32768        # CosyVoice-BlankEN/config.json max_position_embeddings
SAFETY_MARGIN = 1024         # frozen (A6, 2026-08-28): covers eos/rounding + headroom for the flow/hift-free LM
TOKEN_RATE_HZ = 25.0         # speech tokens per second (audit §5)
INSTRUCT = 'You are a helpful assistant.<|endofprompt|>'   # identical to A3 INSTRUCT_PREFIX and the old recipe
SERVICE_TOKENS = 3           # sos, task_id, eos
MIX_RATIO = (5, 15)          # recipe default; fill tokens reserved even if unistream-only is used

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        for p in (COSYVOICE_ROOT, os.path.join(COSYVOICE_ROOT, 'third_party', 'Matcha-TTS')):
            if p not in sys.path:
                sys.path.insert(0, p)
        from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
        _TOKENIZER = get_qwen_tokenizer(token_path=QWEN_PATH, skip_special_tokens=True, version='cosyvoice3')
    return _TOKENIZER


def n_tokens(text, tokenizer=None):
    tokenizer = tokenizer or get_tokenizer()
    return len(tokenizer.encode(text, allowed_special='all'))


def n_fill_tokens(n_text, mix_ratio=MIX_RATIO):
    # llm.py: for j in range(ceil((text_token_len + 1) / mix_ratio[0])) -> one fill token per full group
    return int(math.ceil((n_text + 1) / mix_ratio[0]))


def budget_row(n_text, n_speech, n_instruct, context_limit=CONTEXT_LIMIT, margin=SAFETY_MARGIN):
    n_fill = n_fill_tokens(n_text)
    seq_unistream = SERVICE_TOKENS + n_instruct + n_text + n_speech
    seq_bistream = seq_unistream + n_fill
    seq_max = seq_bistream
    return {
        'n_instruct': n_instruct,
        'n_text': n_text,
        'n_speech': n_speech,
        'n_fill_bistream': n_fill,
        'seq_unistream': seq_unistream,
        'seq_bistream': seq_bistream,
        'seq_max': seq_max,
        'budget': context_limit - margin,
        'fits': seq_max <= context_limit - margin,
    }


def check_manifest(rows, duration_key, text_key='text', tokenizer=None):
    """Return list of budget dicts (one per row, same order)."""
    tokenizer = tokenizer or get_tokenizer()
    n_instruct = n_tokens(INSTRUCT, tokenizer)
    out = []
    for r in rows:
        text = (r.get(text_key) or '').strip()
        nt = n_tokens(text, tokenizer) if text else 0
        dur = float(r[duration_key])
        ns = int(round(dur * TOKEN_RATE_HZ))
        b = budget_row(nt, ns, n_instruct)
        b.update({'sample_id': r['sample_id'], 'duration_sec': dur, 'empty_text': nt == 0})
        out.append(b)
    return out


def summarize(budgets):
    n = len(budgets)
    fits = [b for b in budgets if b['fits']]
    empty = [b for b in budgets if b['empty_text']]
    seq = sorted(b['seq_max'] for b in budgets)
    nt = sorted(b['n_text'] for b in budgets)
    ns = sorted(b['n_speech'] for b in budgets)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
    return {
        'rows': n,
        'fit': len(fits),
        'not_fit': n - len(fits),
        'not_fit_ids': [b['sample_id'] for b in budgets if not b['fits']],
        'empty_text': len(empty),
        'empty_text_ids': [b['sample_id'] for b in empty],
        'budget': budgets[0]['budget'] if budgets else None,
        'seq_max': {'max': seq[-1] if seq else None, 'p99': pct(seq, .99), 'p50': pct(seq, .5)},
        'n_text': {'max': nt[-1] if nt else None, 'p99': pct(nt, .99), 'p50': pct(nt, .5), 'sum': sum(nt)},
        'n_speech': {'max': ns[-1] if ns else None, 'p99': pct(ns, .99), 'p50': pct(ns, .5), 'sum': sum(ns)},
        'context_occupancy_max': (seq[-1] / CONTEXT_LIMIT) if seq else None,
        'text_tokens_per_speech_token': (sum(nt) / sum(ns)) if ns and sum(ns) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--duration_key', default=None, help='duration_sec (long) or duration (short); auto if omitted')
    ap.add_argument('--text_key', default='text')
    ap.add_argument('--out', required=True, help='per-row budget jsonl')
    ap.add_argument('--summary', required=True, help='summary json')
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8')]
    dk = args.duration_key or ('duration_sec' if 'duration_sec' in rows[0] else 'duration')
    budgets = check_manifest(rows, dk, args.text_key)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for b in budgets:
            f.write(json.dumps(b, ensure_ascii=False) + '\n')
    s = summarize(budgets)
    s.update({'manifest': args.manifest, 'context_limit': CONTEXT_LIMIT, 'margin': SAFETY_MARGIN,
              'instruct': INSTRUCT, 'token_rate_hz': TOKEN_RATE_HZ, 'text_key': args.text_key,
              'duration_buckets_not_fit': dict(Counter(int(b['duration_sec'] // 60) for b in budgets if not b['fits']))})
    with open(args.summary, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in s.items() if k not in ('not_fit_ids', 'empty_text_ids')}, ensure_ascii=False, indent=2))
    if s['not_fit']:
        print('ROWS THAT DO NOT FIT (listed, not dropped silently):', s['not_fit_ids'])


if __name__ == '__main__':
    main()
