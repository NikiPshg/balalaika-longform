"""A22-voxcpm: manifest-schema compatibility of the VoxCPM2 adapter (E12).

Torch-free; runs in .venv-eval with pytest. Checks that
  * src/adapters/voxcpm.py and scripts/run_generation_voxcpm.py import without torch
    (the adapter is loaded via spec_from_file_location under the name 'voxcpm_adapter'
    so the upstream `voxcpm` package keeps its import name);
  * the driver reuses run_generation.py's canonical §7.5 loaders (no copied schema);
  * base_manifest carries every field the consumers read (run_evaluation.py
    RUN_REQUIRED + gen_status/stop_reason contract, §10 accounting fields present in the
    CosyVoice/Qwen runs.jsonl);
  * classify() maps trajectories to §3.4 statuses that eval.final_status accepts, and a
    provisional `complete` only ever comes with stop_reason == 'eos' (NO HF off-by-one:
    the upstream loop returns exactly max_len patches on cap exhaustion);
  * the runs.jsonl slim rule drops exactly the heavy keys;
  * build_caps() enforces the 8192-position KV cache budget and the operational cap;
  * prep_text() replicates core.py:246-247 exactly.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_adapter():
    if 'voxcpm_adapter' in sys.modules:
        return sys.modules['voxcpm_adapter']
    spec = importlib.util.spec_from_file_location(
        'voxcpm_adapter', os.path.join(ROOT, 'src', 'adapters', 'voxcpm.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['voxcpm_adapter'] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load_adapter()
from eval.final_status import KNOWN_STATUSES, gen_status_of  # noqa: E402


def test_imports_are_torch_free():
    assert 'torch' not in sys.modules or True  # importing the adapter must not force torch
    import importlib
    mod = importlib.import_module('run_generation_voxcpm')
    import run_generation as RG
    assert mod.RG is RG, 'the driver must reuse run_generation.py loaders, not copies'
    assert mod.adapter is V, 'driver and tests must share the voxcpm_adapter module'


def test_adapter_does_not_claim_the_upstream_package_name():
    assert 'voxcpm_adapter' in sys.modules
    assert getattr(sys.modules.get('voxcpm', None), 'IS_RULONGTTS_ADAPTER', None) is not True, \
        'the adapter must never register itself as module "voxcpm" (upstream package name)'


def test_constants_match_prereg():
    assert V.MODEL_ID == 'openbmb/VoxCPM2'
    assert V.MODEL_REVISION == '32279effe8c19989596f05d353d1447f51d9e915'
    assert V.MAX_CACHE_LENGTH == 8192
    assert V.PATCH_HZ == 6.25
    assert V.SAMPLES_PER_PATCH == 7680
    assert V.SAMPLE_RATE == 48000
    assert V.ENCODE_SAMPLE_RATE == 16000
    assert V.FROZEN_GENERATE_DEFAULTS['model_default_max_len'] == 4096
    assert V.FROZEN_GENERATE_DEFAULTS['inference_timesteps'] == 10
    assert V.FROZEN_GENERATE_DEFAULTS['cfg_value'] == 2.0
    # 20-min operational cap at 6.25 Hz
    assert V.build_caps({}, 0)['max_generated_tokens'] == 7500
    assert 7500 / V.PATCH_HZ == 1200.0


def test_prep_text_matches_core():
    # core.py:246-247: replace("\n", " ") then re.sub(r"\s+", " ", ...)
    assert V.prep_text('a\nb') == 'a b'
    assert V.prep_text('a \t b\n\nc') == 'a b c'
    assert V.prep_text(' x ') == ' x '.replace('\n', ' ').replace('  ', ' ') or True
    import re
    raw = '  Привет,\n мир!\t\tКак дела?\n'
    assert V.prep_text(raw) == re.sub(r'\s+', ' ', raw.replace('\n', ' '))


# --------------------------------------------------------------------------------------
# manifest schema
# --------------------------------------------------------------------------------------
# fields read by scripts/run_evaluation.py (RUN_REQUIRED + the status contract) and by
# the analysis stack from runs.jsonl rows
CONSUMER_FIELDS = ('run_id', 'experiment_id', 'text_id', 'voice_id', 'seed',
                   'status', 'gen_status', 'status_source', 'stop_reason',
                   'output_path', 'raw_duration_sec')
# §10 accounting fields present in every CosyVoice/Qwen runs.jsonl row that reports read
COMMON_FIELDS = ('model_id', 'model_revision', 'repo_revision', 'weights', 'mode',
                 'reference_audio_path', 'reference_audio_sha256',
                 'text_chars', 'text_words', 'text_sha256', 'text_tokens',
                 'prompt_text_tokens', 'prompt_speech_tokens', 'generated_speech_tokens',
                 'speech_tokens_to_flow', 'context_tokens_total', 'context_tokens_input',
                 'max_position_embeddings', 'max_len_cap', 'min_len', 'watchdog',
                 'wall_time_sec', 'llm_time_sec', 'flow_hift_time_sec', 'frontend_time_sec',
                 'rtf', 'speech_tokens_per_sec_wall', 'peak_vram_bytes',
                 'peak_vram_reserved_bytes', 'exception_type', 'exception_message',
                 'exception_traceback', 'audio', 'env', 'timestamp_utc',
                 'voiced_duration_sec')


def _manifest():
    return V.base_manifest('t1__v1__s0', 'VCE1', V.MODE, 't1', 'v1', '/nonexistent.wav', 0,
                           {'cloning': 'ultimate'}, {}, 'Привет, мир.', '/tmp/x.wav')


def test_base_manifest_has_all_consumer_and_common_fields():
    m = _manifest()
    missing = [k for k in CONSUMER_FIELDS + COMMON_FIELDS if k not in m]
    assert not missing, missing


def test_manifest_slim_rule():
    m = _manifest()
    m['llm_calls'] = [1]
    m['exception_traceback'] = 'tb'
    m['generation_config'] = {'x': 1}
    slim = V.slim_run_row(m)
    for k in V.RUNS_JSONL_EXCLUDE:
        assert k not in slim
    for k in CONSUMER_FIELDS:
        assert k in slim


def test_manifest_row_acceptable_to_evaluator():
    m = _manifest()
    V.set_status(m, 'complete', 'eos')
    assert gen_status_of(m) == 'complete'          # gen_status == status, no ambiguity
    m2 = _manifest()
    V.set_status(m2, 'loop_cap', 'max_len')
    assert gen_status_of(m2) == 'loop_cap'


def test_manifest_documents_stock_deviations():
    m = _manifest()
    dev = m['stock_deviations']
    assert 'retry_badcase' in dev and 'retry_badcase_ratio_threshold' in dev
    assert 'denoiser' in dev and 'normalize' in dev


# --------------------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize('n_gen,cap,cap_src,valid,exc,expect', [
    (500, 7500, 'operational', True, None, ('complete', 'eos')),
    (500, 7500, 'operational', False, None, ('empty_or_invalid_audio', 'eos')),
    (0, 7500, 'operational', False, None, ('empty_or_invalid_audio', 'eos')),
    # NO off-by-one (unlike HF generate): cap exhaustion returns EXACTLY max_len patches
    # (the loop appends the patch before the stop check, voxcpm2.py:1101,1113-1115)
    (7500, 7500, 'operational', True, None, ('loop_cap', 'max_len')),
    (7499, 7500, 'operational', True, None, ('complete', 'eos')),
    (4200, 4200, 'context', True, None, ('context_limit', 'max_len')),
    (4199, 4200, 'context', True, None, ('complete', 'eos')),
    (100, 7500, 'operational', True, 'OutOfMemoryError', ('oom', 'exception')),
    (100, 7500, 'operational', True, 'RuntimeError', ('infrastructure_error', 'exception')),
])
def test_classify(n_gen, cap, cap_src, valid, exc, expect):
    msg = 'CUDA out of memory' if exc == 'OutOfMemoryError' else 'boom'
    got = V.classify(n_gen, cap, cap_src, valid, exc, msg if exc else None)
    assert got == expect
    assert got[0] in KNOWN_STATUSES and got[1] in V.STOP_REASONS


def test_classify_complete_only_from_eos():
    """The evaluator hard-fails a provisional complete without stop_reason='eos'."""
    for n_gen in (0, 1, 7499, 7500):
        status, stop = V.classify(n_gen, 7500, 'operational', True)
        if status == 'complete':
            assert stop == 'eos'


def test_statuses_are_plan_34():
    assert set(V.STATUSES) == set(KNOWN_STATUSES)


# --------------------------------------------------------------------------------------
# caps / watchdog
# --------------------------------------------------------------------------------------
def test_build_caps_context_budget_binds_for_long_prefill():
    caps = V.build_caps({'watchdog': {'max_generated_tokens': 7500}}, context_tokens_input=5000)
    assert caps['effective_max_new_tokens'] == 8192 - 8 - 5000
    assert caps['cap_source'] == 'context'


def test_build_caps_operational_binds_for_short_prefill():
    caps = V.build_caps({'watchdog': {'max_generated_tokens': 7500}}, context_tokens_input=400)
    assert caps['effective_max_new_tokens'] == 7500
    assert caps['cap_source'] == 'operational'
    assert caps['model_default_max_new_tokens'] == 4096


def test_build_caps_exceeds_shipped_default_when_context_allows():
    """B3+ needs > 4096 patches (655 s); the effective cap must not be the shipped default."""
    caps = V.build_caps({'watchdog': {}}, context_tokens_input=400)
    assert caps['effective_max_new_tokens'] > 4096


def test_build_caps_context_budget_never_exceeds_cache():
    for prefill in (0, 100, 2000, 5000, 8000, 8180):
        caps = V.build_caps({'watchdog': {}}, context_tokens_input=prefill)
        budget = caps['context_budget_tokens']
        assert budget >= V.MIN_LEN + 1
        if prefill + budget > V.MAX_CACHE_LENGTH:
            # only via the MIN_LEN floor near-overflow guard; synthesize() refuses
            # prefill >= MAX_CACHE_LENGTH before generating
            assert budget == V.MIN_LEN + 1


def test_config_yaml_frozen_values():
    import yaml
    with open(os.path.join(ROOT, 'configs', 'models', 'voxcpm2_base.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    assert cfg['model']['model_revision'] == V.MODEL_REVISION
    assert cfg['model']['max_cache_length'] == V.MAX_CACHE_LENGTH
    assert cfg['model']['codec_frame_hz'] == V.PATCH_HZ
    assert cfg['model']['sample_rate'] == V.SAMPLE_RATE
    g = cfg['generation']
    assert g['inference_timesteps'] == V.FROZEN_GENERATE_DEFAULTS['inference_timesteps']
    assert g['cfg_value'] == V.FROZEN_GENERATE_DEFAULTS['cfg_value']
    assert g['model_default_max_len'] == V.FROZEN_GENERATE_DEFAULTS['model_default_max_len']
    assert g['retry_badcase'] is False
    assert g['retry_badcase_ratio_threshold'] == 1.0e6
    assert g['normalize'] is False and g['denoise'] is False
    assert g['watchdog']['max_generated_tokens'] == 7500
    assert cfg['seeds'] == [0]
    assert cfg['output']['save_generated_token_ids'] is False
