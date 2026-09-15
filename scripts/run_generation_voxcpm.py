#!/usr/bin/env python
"""Batch generation driver for the VoxCPM2 arms (E12): benchmark texts x voice
references -> wav + run manifests, mirroring scripts/run_generation.py EXACTLY
(same input schema, same resume rule, same runs.jsonl fields).

Example:
  export CUDA_VISIBLE_DEVICES=2
  python scripts/run_generation_voxcpm.py \
      --benchmark data/benchmark/pilot.jsonl --references data/references/references.jsonl \
      --experiment VCE1 --out outputs/v31_voxcpm/VCE1_pilot --seed 0 \
      --voice-filter 'ref_(female|male)_01$'

The script re-executes itself under the upstream repo's pinned venv
(third_party/VoxCPM/.venv/bin/python) when started from any other
interpreter, so callers do not need to know about that environment.

Input schema (PLAN §7.5, canonical fields ONLY; identical to run_generation.py -- the
loaders are IMPORTED from it, not copied):
  benchmark  : text_id, text_tts        (+ voice_id with --per-item-voices)
  references : voice_id, wav_path, ref_text
Resumes by skipping items whose manifest already exists. One model load per process.
OOM -> status oom (empty_cache, continue); other exceptions -> infrastructure_error.

NB: the adapter file src/adapters/voxcpm.py shares its name with the upstream package;
it is loaded here via importlib.spec_from_file_location under the module name
'voxcpm_adapter' so the upstream `voxcpm` package keeps its import name.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


def _load_adapter():
    """Load src/adapters/voxcpm.py as module 'voxcpm_adapter' (name-clash-safe)."""
    if 'voxcpm_adapter' in sys.modules:
        return sys.modules['voxcpm_adapter']
    path = os.path.join(ROOT, 'src', 'adapters', 'voxcpm.py')
    spec = importlib.util.spec_from_file_location('voxcpm_adapter', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['voxcpm_adapter'] = mod
    spec.loader.exec_module(mod)
    return mod


adapter = _load_adapter()          # torch-free at import
import run_generation as RG  # noqa: E402  (torch-free: schema loaders + SchemaError)


def reexec_under_voxcpm_venv():
    """Restart this script with the upstream venv python (idempotent).

    The venv's bin/python is executed by its SYMLINK path on purpose: CPython finds
    pyvenv.cfg (and therefore the venv's site-packages) relative to the executable path,
    so resolving the symlink would silently run the bare uv base interpreter.
    """
    venv_py = adapter.VENV_PYTHON
    venv_root = os.path.abspath(os.path.dirname(os.path.dirname(venv_py)))
    if os.path.abspath(sys.prefix) == venv_root or os.environ.get('VOXCPM_ADAPTER_NO_REEXEC'):
        return
    if not os.path.exists(venv_py):
        sys.exit('voxcpm venv python not found: {} (uv sync --frozen in {})'.format(
            venv_py, adapter.UPSTREAM_ROOT))
    env = dict(os.environ)
    env['PYTHONPATH'] = adapter.UPSTREAM_SRC + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.setdefault('HF_HUB_CACHE', os.path.join(ROOT, '.cache'))
    os.execve(venv_py, [venv_py, os.path.abspath(__file__)] + sys.argv[1:], env)


def load_yaml(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--references', required=True)
    ap.add_argument('--mode', choices=['native'], default='native',
                    help='only native single-trajectory generation exists for VoxCPM2 (E12 prereg)')
    ap.add_argument('--experiment', required=True, help='VCE1|VCE2P|VCE3P|... (recorded in manifests)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, nargs='+', default=None, help='overrides config seeds list')
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'models', 'voxcpm2_base.yaml'))
    ap.add_argument('--weights', default=None,
                    help='override model.model_dir (an SFT checkpoint directory for VC-E2P/VC-E3P)')
    ap.add_argument('--limit', type=int, default=None, help='debug: stop after N items')
    ap.add_argument('--text-filter', default=None, help="regex (re.search) on text_id")
    ap.add_argument('--voice-filter', default=None, help="regex (re.search) on voice_id")
    ap.add_argument('--per-item-voices', action='store_true',
                    help='each benchmark record carries its own voice_id (Robust-20): the text is '
                         'generated ONLY with that voice instead of the texts x references cross product')
    ap.add_argument('--watchdog-max-tokens', type=int, default=None)
    ap.add_argument('--watchdog-max-seconds', type=float, default=None)
    ap.add_argument('--no-token-dump', action='store_true',
                    help='accepted for driver parity; VoxCPM has no discrete tokens to dump')
    args = ap.parse_args()

    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        print('WARNING: CUDA_VISIBLE_DEVICES is not set; E12 runs on an explicitly '
              'granted free card out of 0/1/2 (card 3 is foreign, always)', file=sys.stderr)

    # schema is validated BEFORE anything heavy so a bad file fails in < 1 s
    texts = RG.load_benchmark(args.benchmark, args.text_filter, with_voice=args.per_item_voices)
    refs = RG.load_references(args.references, args.voice_filter, root=ROOT)
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

    cfg = load_yaml(args.config)
    model_cfg = dict(cfg['model'])
    if args.weights:
        model_cfg['model_dir'] = args.weights
        model_cfg['weights'] = args.weights
    gen_cfg = dict(cfg['generation'])
    wd_over = {}
    if args.watchdog_max_tokens is not None:
        wd_over['max_generated_tokens'] = args.watchdog_max_tokens
    if args.watchdog_max_seconds is not None:
        wd_over['max_wall_seconds'] = args.watchdog_max_seconds
    seeds = args.seed if args.seed is not None else list(cfg.get('seeds', [0]))

    reexec_under_voxcpm_venv()   # everything below needs torch + the upstream voxcpm package

    os.makedirs(args.out, exist_ok=True)
    global_jsonl = os.path.join(args.out, 'runs.jsonl')
    log_path = os.path.join(args.out, 'run_generation.log')

    def log(msg):
        line = '[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log('mode=native experiment={} texts={} refs={} seeds={} config={} model_dir={} benchmark={} references={}'.format(
        args.experiment, len(texts), len(refs), seeds, args.config,
        model_cfg.get('model_dir') or adapter.DEFAULT_MODEL_DIR, args.benchmark, args.references))
    log('mem at start: {}'.format(adapter.free_mem_info()))
    t_load = time.time()
    tts = adapter.load_voxcpm(model_cfg)
    log('model loaded in {:.1f}s; repo_revision={}; config check: {}; mem: {}'.format(
        time.time() - t_load, adapter.repo_revision(model_cfg.get('upstream_root', adapter.UPSTREAM_ROOT)),
        adapter.check_generate_defaults(tts.tts_model), adapter.free_mem_info()))

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
                log('START {} words={} chars={} ref={} mem={}'.format(
                    stem, len(text.split()), len(text), os.path.basename(ref_wav), adapter.free_mem_info()))
                try:
                    m = adapter.synthesize(text, ref_wav, ref_text, seed, gen_cfg, out_wav,
                                           watchdog=wd_over or None, model_cfg=model_cfg,
                                           text_id=text_id, voice_id=voice_id,
                                           experiment_id=args.experiment, run_id=stem,
                                           cv=tts, keep_tokens=not args.no_token_dump)
                except Exception as e:  # noqa: BLE001 - adapter-level failure (e.g. FileExistsError race)
                    import torch
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                    m = {'run_id': stem, 'experiment_id': args.experiment, 'mode': adapter.MODE,
                         'text_id': text_id, 'voice_id': voice_id, 'seed': seed,
                         'status': 'oom' if is_oom else 'infrastructure_error',
                         'gen_status': 'oom' if is_oom else 'infrastructure_error',
                         'status_source': 'generator (provisional); final PLAN §3.4 status assigned by scripts/run_evaluation.py',
                         'stop_reason': 'exception',
                         'exception_type': type(e).__name__, 'exception_message': str(e)[:2000],
                         'exception_traceback': traceback.format_exc()[-4000:], 'output_path': None,
                         'raw_duration_sec': 0.0}
                    if not os.path.exists(out_json):
                        adapter.write_json_no_overwrite(out_json, m)
                    if is_oom:
                        torch.cuda.empty_cache()
                    n_fail += 1
                with open(global_jsonl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(adapter.slim_run_row(m), ensure_ascii=False) + '\n')
                n_done += 1
                log('END   {} status={} stop={} gen_tokens={} dur={}s wall={}s rtf={} peakVRAM={}GB mem={}'.format(
                    stem, m.get('status'), m.get('stop_reason'), m.get('generated_speech_tokens'),
                    m.get('raw_duration_sec'), m.get('wall_time_sec'), m.get('rtf'),
                    round((m.get('peak_vram_bytes') or 0) / 2**30, 2), adapter.free_mem_info()))
    log('DONE done={} skipped={} adapter_failures={}'.format(n_done, n_skip, n_fail))


if __name__ == '__main__':
    main()
