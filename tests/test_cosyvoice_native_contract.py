"""Native-contract test for the CosyVoice3 adapters (A3). Runs on GPU 1 (~2-4 min incl. model load).

Run:  export CUDA_VISIBLE_DEVICES=1
      python tests/test_cosyvoice_native_contract.py
(pytest is not installed in the cosyvoice env; the file works both as a pytest module and as a script.)
Fixtures: data/smoke/ (built deterministically from the parquet by scripts/make_smoke_inputs.py; built here if absent).
Scratch : tmp/a3/contract_* (deleted at the end unless KEEP_CONTRACT_TMP=1).

Asserts:
  * native synthesize() of a 3-sentence Russian text -> exactly ONE frontend_zero_shot call,
    ONE model.tts call, ONE llm.inference trajectory, ONE yielded wav chunk;
  * official synthesize() of the same text -> >1 frontend_zero_shot / llm.inference calls;
  * §10 manifest completeness; wav exists, duration > 0, audio finite; never-overwrite guard;
  * token watchdog -> partial wav saved, status loop_cap / stop watchdog;
  * the frozen yaml equals the repo's effective defaults (config check ran, 0 mismatches) and a
    tampered yaml raises ConfigDivergence; the model cache is keyed by (model_dir, weights).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src', 'adapters'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

try:
    import cosyvoice3_common as C  # noqa: E402
except ImportError as _e:  # torch / CosyVoice repo not importable (e.g. pytest in .venv-eval): skip, do not error
    if 'pytest' in sys.modules:
        import pytest
        pytest.skip('needs the cosyvoice env: {}'.format(_e), allow_module_level=True)
    raise
from make_smoke_inputs import ensure_smoke_inputs  # noqa: E402
from run_generation import load_references  # noqa: E402

# three short sentences -> the stock splitter (token_max_n=80, token_min_n=60) keeps them as one chunk,
# so we use a slightly longer 3-sentence text that exceeds 80 tokens in total to force >1 chunk in official mode.
TEXT_3S = ('Сегодня мы поговорим о том, как устроены большие языковые модели и почему они иногда ошибаются в длинных текстах. '
           'Во второй части разберём, как измерять качество синтеза речи на русском языке и какие метрики действительно важны. '
           'В конце подведём итоги и обсудим, что можно улучшить в следующих экспериментах.')

REQUIRED_MANIFEST_KEYS = ['run_id', 'experiment_id', 'model_id', 'model_revision', 'repo_revision', 'weights', 'mode', 'text_id',
                          'voice_id', 'reference_audio_sha256', 'seed', 'generation_config', 'text_chars', 'text_words',
                          'text_tokens', 'prompt_speech_tokens', 'generated_speech_tokens', 'context_tokens_total', 'status',
                          'stop_reason', 'output_path', 'raw_duration_sec', 'voiced_duration_sec', 'wall_time_sec', 'rtf',
                          'peak_vram_bytes', 'exception_type', 'exception_message']

_STATE = {}


def _setup():
    if 'cv' not in _STATE:
        prompt_wav, prompt_txt, refs_path, _ = ensure_smoke_inputs()
        voice_id, wav_abs, ref_text = load_references(refs_path)[0]
        assert voice_id == 'smoke_v' and os.path.samefile(wav_abs, prompt_wav)
        cfg = C.load_yaml(os.path.join(ROOT, 'configs', 'models', 'cosyvoice3_base.yaml'))
        _STATE['cfg'] = cfg
        _STATE['cv'] = C.load_cosyvoice3(cfg['model'], gen_cfg=cfg['generation'], output_cfg=cfg.get('output'))
        _STATE['config_check'] = dict(C._LAST_CONFIG_CHECK)
        os.makedirs(os.path.join(ROOT, 'tmp', 'a3'), exist_ok=True)
        _STATE['tmp'] = tempfile.mkdtemp(prefix='contract_', dir=os.path.join(ROOT, 'tmp', 'a3'))
        _STATE['prompt_wav'] = wav_abs
        _STATE['ref_text'] = ref_text
    return _STATE


def _cleanup():
    d = _STATE.get('tmp')
    if d and os.path.isdir(d) and not os.environ.get('KEEP_CONTRACT_TMP'):
        shutil.rmtree(d, ignore_errors=True)


def _check_manifest(m, mode):
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in m]
    assert not missing, 'manifest missing keys: {}'.format(missing)
    assert m['status'] in C.STATUSES, m['status']
    assert m['stop_reason'] in C.STOP_REASONS, m['stop_reason']
    assert m['mode'] == mode
    assert os.path.exists(m['manifest_path'])
    with open(m['manifest_path'], encoding='utf-8') as f:
        json.load(f)


def test_config_matches_repo_defaults():
    st = _setup()
    r = st['config_check']
    assert r, 'config check did not run at load time'
    assert r['mismatches'] == [], r['mismatches']
    assert len(r['checked']) >= 14, r
    assert r['unreadable'] == [], 'every frozen value must be readable from the repo objects: {}'.format(r['unreadable'])
    # a tampered value must fail loudly
    bad = dict(st['cfg']['generation'])
    bad['top_p'] = 0.95
    try:
        C.load_cosyvoice3(st['cfg']['model'], gen_cfg=bad)
        raise AssertionError('expected ConfigDivergence for top_p=0.95')
    except C.ConfigDivergence as e:
        assert 'top_p' in str(e)
    finally:
        C.load_cosyvoice3(st['cfg']['model'], gen_cfg=st['cfg']['generation'], output_cfg=st['cfg'].get('output'))


def test_model_cache_key():
    st = _setup()
    keys = list(C._MODEL_CACHE.keys())
    assert len(keys) == 1 and isinstance(keys[0], tuple) and len(keys[0]) == 2, keys
    model_dir, weights_path = keys[0]
    assert os.path.samefile(model_dir, st['cfg']['model']['model_dir'])
    assert weights_path == os.path.join(os.path.abspath(model_dir), st['cfg']['model']['weights'])
    assert C.load_cosyvoice3(st['cfg']['model']) is st['cv']  # same (model_dir, weights) -> same object, no reload


def test_native_single_trajectory():
    st = _setup()
    out = os.path.join(st['tmp'], 'native.wav')
    import cosyvoice3_native as native
    m = native.synthesize(TEXT_3S, st['prompt_wav'], st['ref_text'], 0, st['cfg']['generation'], out, model_cfg=st['cfg']['model'],
                          text_id='t3s', voice_id='smoke_v', experiment_id='TEST', cv=st['cv'])
    _check_manifest(m, 'native')
    nc = m['native_contract']
    assert nc['frontend_zero_shot_calls'] == 1, nc
    assert nc['tts_calls'] == 1, nc
    assert nc['llm_inference_calls'] == 1, nc
    assert nc['tts_chunks_yielded'] == 1, nc
    assert len(m['llm_calls']) == 1
    assert m['text_tokens'] > 80, 'text must be long enough to force the official splitter (>80 tokens)'
    import soundfile as sf
    exp_prompt_tok = int(sf.info(st['prompt_wav']).duration * 25)  # 25 Hz speech tokenizer; frontend.py:174-178 trims to min(mel/2, tokens)
    assert abs(m['prompt_speech_tokens'] - exp_prompt_tok) <= 1, (m['prompt_speech_tokens'], exp_prompt_tok)
    assert m['context_tokens_total'] == 1 + m['prompt_text_tokens'] + m['text_tokens'] + 1 + m['prompt_speech_tokens'] + m['generated_speech_tokens']
    assert os.path.exists(out)
    assert m['raw_duration_sec'] > 0 and m['audio']['valid'] and not m['audio']['has_nan']
    assert m['generated_speech_tokens'] > 0
    assert m['status'] in ('complete', 'loop_cap'), m['status']  # loop_cap would itself be a valid (bad) result, not a test bug
    assert m['peak_vram_bytes'] > 0
    assert m['watchdog']['expected_basis'].endswith('x text_tokens'), m['watchdog']  # no human duration is read (PLAN §7.5)
    # never-overwrite guard
    try:
        native.synthesize(TEXT_3S, st['prompt_wav'], st['ref_text'], 0, st['cfg']['generation'], out, model_cfg=st['cfg']['model'], cv=st['cv'])
        raise AssertionError('expected FileExistsError')
    except FileExistsError:
        pass
    _STATE['native_manifest'] = m


def test_official_is_split():
    st = _setup()
    out = os.path.join(st['tmp'], 'official.wav')
    import cosyvoice3_official as official
    m = official.synthesize(TEXT_3S, st['prompt_wav'], st['ref_text'], 0, st['cfg']['generation'], out, model_cfg=st['cfg']['model'],
                            text_id='t3s', voice_id='smoke_v', experiment_id='TEST', cv=st['cv'])
    _check_manifest(m, 'official_split')
    nc = m['native_contract']
    assert m['official_split']['n_chunks'] > 1, m['official_split']
    assert m['official_split']['text_frontend_backend'] == st['cfg']['model']['text_frontend_backend']
    assert nc['frontend_zero_shot_calls'] > 1, nc
    assert nc['llm_inference_calls'] > 1, nc
    assert nc['tts_calls'] == nc['frontend_zero_shot_calls'] == nc['llm_inference_calls'] == m['official_split']['n_chunks']
    assert os.path.exists(out) and m['raw_duration_sec'] > 0 and m['audio']['valid']
    _STATE['official_manifest'] = m


def test_watchdog_partial_output():
    """Force the token watchdog at 40 tokens: partial wav must be saved with status loop_cap / stop watchdog."""
    st = _setup()
    out = os.path.join(st['tmp'], 'watchdog.wav')
    import cosyvoice3_native as native
    m = native.synthesize(TEXT_3S, st['prompt_wav'], st['ref_text'], 0, st['cfg']['generation'], out, watchdog={'max_generated_tokens': 40},
                          model_cfg=st['cfg']['model'], text_id='t3s', voice_id='smoke_v', experiment_id='TEST', cv=st['cv'])
    assert m['status'] == 'loop_cap' and m['stop_reason'] == 'watchdog', (m['status'], m['stop_reason'])
    assert m['generated_speech_tokens'] == 40, m['generated_speech_tokens']
    assert os.path.exists(out) and m['raw_duration_sec'] > 0
    assert m['native_contract']['llm_inference_calls'] == 1


def teardown_module(module=None):
    _cleanup()


TESTS = (test_config_matches_repo_defaults, test_model_cache_key, test_native_single_trajectory, test_official_is_split,
         test_watchdog_partial_output)

if __name__ == '__main__':
    import time
    results = {}
    try:
        for fn in TESTS:
            t0 = time.time()
            try:
                fn()
                results[fn.__name__] = 'PASS'
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                results[fn.__name__] = 'FAIL: {}: {}'.format(type(e).__name__, e)
            print('{} -> {} ({:.1f}s)'.format(fn.__name__, results[fn.__name__], time.time() - t0), flush=True)
        for k in ('native_manifest', 'official_manifest'):
            if k in _STATE:
                m = _STATE[k]
                print(k, json.dumps({x: m.get(x) for x in ('status', 'stop_reason', 'text_tokens', 'prompt_text_tokens', 'prompt_speech_tokens',
                                                            'generated_speech_tokens', 'context_tokens_total', 'raw_duration_sec', 'wall_time_sec',
                                                            'rtf', 'peak_vram_bytes', 'native_contract')}, ensure_ascii=False))
        print('CONFIG CHECK', json.dumps(_STATE.get('config_check')))
        print('TMP DIR', _STATE.get('tmp'), '(removed unless KEEP_CONTRACT_TMP=1)')
    finally:
        _cleanup()
    sys.exit(0 if all(v == 'PASS' for v in results.values()) else 1)
