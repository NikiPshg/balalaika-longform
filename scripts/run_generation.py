#!/usr/bin/env python
"""Batch generation driver (A3/A7): benchmark texts x voice references -> wav + run manifests.

Example:
  export CUDA_VISIBLE_DEVICES=1
  python scripts/run_generation.py \
      --benchmark data/benchmark/pilot.jsonl --references data/references/references.jsonl \
      --mode native --experiment E1 --out outputs/E1/native --seed 0 \
      --text-filter 'ds3_RTXxGQV7SiA_93.36_816.18__B[01]$' --voice-filter 'ref_(female|male)_01$'

Input schema (PLAN §7.5, canonical fields ONLY; a missing field is a hard error, no synonyms):
  benchmark  : text_id, text_tts
  references : voice_id, wav_path, ref_text        (wav_path relative paths are resolved from the project root)
Resumes by skipping items whose manifest already exists. One model load per process.
OOM -> status oom (empty_cache, continue); other exceptions -> infrastructure_error with traceback.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src', 'adapters'))

BENCHMARK_FIELDS = ('text_id', 'text_tts')
REFERENCE_FIELDS = ('voice_id', 'wav_path', 'ref_text')


class SchemaError(ValueError):
    """A record lacks a canonical PLAN §7.5 field (or it is empty)."""


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for n, ln in enumerate(f, 1):
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError as e:
                    raise SchemaError('{}:{}: invalid JSON: {}'.format(path, n, e)) from None
    return rows


def require_fields(rec: dict, fields, what: str, path: str, lineno: int):
    for k in fields:
        v = rec.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            raise SchemaError('{} record {}:{} lacks canonical field {!r} (PLAN §7.5; present keys: {})'.format(
                what, path, lineno, k, sorted(rec.keys())))


def load_benchmark(path: str, text_filter: str | None = None, with_voice: bool = False):
    """Return [(text_id, text_tts)] — canonical fields only (PLAN §7.5).

    ``with_voice=True`` (--per-item-voices, E8 Robust-20) additionally requires a
    ``voice_id`` on every record and returns [(text_id, text_tts, voice_id)].
    """
    out = []
    for i, rec in enumerate(read_jsonl(path), 1):
        require_fields(rec, BENCHMARK_FIELDS + (('voice_id',) if with_voice else ()),
                       'benchmark', path, i)
        text_id = str(rec['text_id'])
        if text_filter and not re.search(text_filter, text_id):
            continue
        if with_voice:
            out.append((text_id, rec['text_tts'], str(rec['voice_id'])))
        else:
            out.append((text_id, rec['text_tts']))
    ids = [t[0] for t in out]
    if len(set(ids)) != len(ids):
        dup = sorted({t for t in ids if ids.count(t) > 1})
        raise SchemaError('benchmark {}: duplicate text_id {}'.format(path, dup))
    return out


def load_references(path: str, voice_filter: str | None = None, root: str = ROOT):
    """Return [(voice_id, wav_path_abs, ref_text)] — canonical fields only (PLAN §7.5)."""
    out = []
    for i, rec in enumerate(read_jsonl(path), 1):
        require_fields(rec, REFERENCE_FIELDS, 'reference', path, i)
        voice_id = str(rec['voice_id'])
        if voice_filter and not re.search(voice_filter, voice_id):
            continue
        wav = rec['wav_path']
        if not os.path.isabs(wav):
            wav = os.path.join(root, wav)
        if not os.path.exists(wav):
            raise SchemaError('reference {}:{} voice_id={}: wav_path does not exist: {}'.format(path, i, voice_id, wav))
        out.append((voice_id, wav, rec['ref_text']))
    ids = [v for v, _, _ in out]
    if len(set(ids)) != len(ids):
        raise SchemaError('references {}: duplicate voice_id'.format(path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--references', required=True)
    ap.add_argument('--mode', choices=['native', 'official'], required=True)
    ap.add_argument('--experiment', required=True, help='E0|E1|E2|E3|... (recorded in manifests)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, nargs='+', default=None, help='overrides config seeds list')
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'models', 'cosyvoice3_base.yaml'))
    ap.add_argument('--weights', default=None, help='override model.weights (e.g. llm.rl.pt or an SFT checkpoint path)')
    ap.add_argument('--limit', type=int, default=None, help='debug: stop after N items')
    ap.add_argument('--text-filter', default=None, help="regex (re.search) on text_id, e.g. '<root_id>__B[01]$'")
    ap.add_argument('--voice-filter', default=None, help="regex (re.search) on voice_id, e.g. 'ref_(female|male)_01$'")
    ap.add_argument('--per-item-voices', action='store_true',
                    help='each benchmark record carries its own voice_id (E8 Robust-20): the text is '
                         'generated ONLY with that voice instead of the texts x references cross '
                         'product. A voice_id missing from --references is a startup error.')
    ap.add_argument('--watchdog-max-tokens', type=int, default=None)
    ap.add_argument('--watchdog-max-seconds', type=float, default=None)
    ap.add_argument('--no-token-dump', action='store_true')
    args = ap.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        print('WARNING: CUDA_VISIBLE_DEVICES={!r}; PLAN §0.1 requires GPU index 1'.format(os.environ.get('CUDA_VISIBLE_DEVICES')), file=sys.stderr)

    # schema is validated BEFORE the model is loaded so a bad file fails in < 1 s
    texts = load_benchmark(args.benchmark, args.text_filter, with_voice=args.per_item_voices)
    refs = load_references(args.references, args.voice_filter)
    if not texts:
        sys.exit('no benchmark records left after --text-filter {!r}'.format(args.text_filter))
    if not refs:
        sys.exit('no reference records left after --voice-filter {!r}'.format(args.voice_filter))
    if args.per_item_voices:
        ref_by_id = {v: (v, w, t) for v, w, t in refs}
        missing = sorted({vid for _t, _x, vid in texts if vid not in ref_by_id})
        if missing:
            sys.exit('--per-item-voices: benchmark voice_id(s) {} absent from {} '
                     '(after --voice-filter {!r})'.format(missing, args.references, args.voice_filter))
        work = [((tid, tx), ref_by_id[vid]) for tid, tx, vid in texts]
    else:
        work = [((tid, tx), ref) for tid, tx in texts for ref in refs]

    import cosyvoice3_common as C  # noqa: E402  (imports torch + the CosyVoice repo)
    cfg = C.load_yaml(args.config)
    model_cfg = dict(cfg['model'])
    if args.weights:
        model_cfg['weights'] = args.weights
    gen_cfg = dict(cfg['generation'])
    # E0 official keeps the STOCK noise buffers (Lead decision, reports/decisions.md 2026-08-28): it splits the
    # text, so no chunk approaches 300 s, and the ~0.9 GB of extra CPU RAM buys nothing. Native (E1/E2/E3) uses
    # the 900 s buffers from the yaml. Whatever ends up in effect is MEASURED into every manifest.
    if args.mode == 'official':
        for k, v in (cfg.get('official_mode_overrides') or {}).items():
            model_cfg[k] = v
    wd_over = {}
    if args.watchdog_max_tokens is not None:
        wd_over['max_generated_tokens'] = args.watchdog_max_tokens
    if args.watchdog_max_seconds is not None:
        wd_over['max_wall_seconds'] = args.watchdog_max_seconds
    seeds = args.seed if args.seed is not None else list(cfg.get('seeds', [0]))

    if args.mode == 'native':
        import cosyvoice3_native as adapter
    else:
        import cosyvoice3_official as adapter

    os.makedirs(args.out, exist_ok=True)
    global_jsonl = os.path.join(args.out, 'runs.jsonl')
    log_path = os.path.join(args.out, 'run_generation.log')

    def log(msg):
        line = '[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log('mode={} experiment={} texts={} refs={} seeds={} config={} weights={} benchmark={} references={}'.format(
        args.mode, args.experiment, len(texts), len(refs), seeds, args.config, model_cfg.get('weights'), args.benchmark, args.references))
    log('noise buffers requested: flow={}s hift={}s'.format(model_cfg.get('flow_noise_buffer_sec'), model_cfg.get('hift_noise_buffer_sec')))
    log('mem at start: {}'.format(C.free_mem_info()))
    t_load = time.time()
    cv = C.load_cosyvoice3(model_cfg, gen_cfg=gen_cfg, output_cfg=cfg.get('output'))
    log('model loaded in {:.1f}s; repo_revision={}; config check: {}; mem: {}'.format(
        time.time() - t_load, C.repo_revision(), C.last_config_check_summary(), C.free_mem_info()))

    n_done = n_skip = n_fail = 0
    for (text_id, text), (voice_id, ref_wav, ref_text) in work:
        for seed in seeds:
                stem = '{}__{}__s{}'.format(text_id, voice_id, seed)
                out_wav = os.path.join(args.out, stem + '.wav')
                out_json = os.path.join(args.out, stem + '.json')
                if os.path.exists(out_json):
                    n_skip += 1
                    continue
                if args.limit is not None and n_done >= args.limit:
                    log('limit reached')
                    return
                log('START {} words={} chars={} ref={} mem={}'.format(stem, len(text.split()), len(text), os.path.basename(ref_wav), C.free_mem_info()))
                try:
                    m = adapter.synthesize(text, ref_wav, ref_text, seed, gen_cfg, out_wav, watchdog=wd_over or None,
                                           model_cfg=model_cfg, text_id=text_id, voice_id=voice_id,
                                           experiment_id=args.experiment, run_id=stem,
                                           cv=cv, keep_tokens=not args.no_token_dump)
                except Exception as e:  # noqa: BLE001 - adapter-level failure (e.g. FileExistsError race)
                    import torch
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                    m = {'run_id': stem, 'experiment_id': args.experiment, 'mode': adapter.MODE, 'text_id': text_id, 'voice_id': voice_id,
                         'seed': seed, 'status': 'oom' if is_oom else 'infrastructure_error',
                         'gen_status': 'oom' if is_oom else 'infrastructure_error',
                         'status_source': 'generator (provisional); final PLAN §3.4 status assigned by scripts/run_evaluation.py',
                         'stop_reason': 'exception',
                         'exception_type': type(e).__name__, 'exception_message': str(e)[:2000],
                         'exception_traceback': traceback.format_exc()[-4000:], 'output_path': None}
                    if not os.path.exists(out_json):
                        C.write_json_no_overwrite(out_json, m)
                    if is_oom:
                        torch.cuda.empty_cache()
                    n_fail += 1
                with open(global_jsonl, 'a', encoding='utf-8') as f:
                    slim = {k: v for k, v in m.items() if k not in ('llm_calls', 'chunks', 'exception_traceback', 'generation_config')}
                    f.write(json.dumps(slim, ensure_ascii=False) + '\n')
                n_done += 1
                log('END   {} status={} stop={} gen_tokens={} dur={}s wall={}s rtf={} peakVRAM={}GB mem={}'.format(
                    stem, m.get('status'), m.get('stop_reason'), m.get('generated_speech_tokens'), m.get('raw_duration_sec'),
                    m.get('wall_time_sec'), m.get('rtf'), round((m.get('peak_vram_bytes') or 0) / 2**30, 2), C.free_mem_info()))
    log('DONE done={} skipped={} adapter_failures={}'.format(n_done, n_skip, n_fail))


if __name__ == '__main__':
    main()
