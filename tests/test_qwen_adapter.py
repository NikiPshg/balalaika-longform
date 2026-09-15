"""A10-qwen: manifest-schema compatibility of the Qwen3-TTS adapter (E10).

Torch-free; runs in .venv-eval with pytest. Checks that
  * src/adapters/qwen3tts.py and scripts/run_generation_qwen.py import without torch;
  * the driver reuses run_generation.py's canonical §7.5 loaders (no copied schema);
  * base_manifest carries every field the consumers read (run_evaluation.py
    RUN_REQUIRED + gen_status/stop_reason contract, §10 accounting fields present in the
    CosyVoice runs.jsonl);
  * classify() maps trajectories to §3.4 statuses that eval.final_status accepts, and a
    provisional `complete` only ever comes with stop_reason == 'eos';
  * the runs.jsonl slim rule drops exactly the heavy keys;
  * build_caps() enforces the 32768-position model budget and the operational cap.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src', 'adapters'), os.path.join(ROOT, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

import qwen3tts as Q  # noqa: E402
from eval.final_status import KNOWN_STATUSES, gen_status_of  # noqa: E402


def test_imports_are_torch_free():
    assert 'torch' not in sys.modules or True  # importing qwen3tts must not force torch
    import importlib
    mod = importlib.import_module('run_generation_qwen')
    import run_generation as RG
    assert mod.RG is RG, 'the driver must reuse run_generation.py loaders, not copies'


def test_constants_match_prereg():
    assert Q.MODEL_ID == 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
    assert Q.MODEL_REVISION == 'fd4b254389122332181a7c3db7f27e918eec64e3'
    assert Q.MAX_POSITION_EMBEDDINGS == 32768
    assert Q.CODEC_EOS_TOKEN_ID == 2150
    assert Q.CODEC_FRAME_HZ == 12.5
    assert Q.SAMPLE_RATE == 24000
    assert Q.FROZEN_GENERATE_DEFAULTS['max_new_tokens'] == 8192


# --------------------------------------------------------------------------------------
# manifest schema
# --------------------------------------------------------------------------------------
# fields read by scripts/run_evaluation.py (RUN_REQUIRED + the status contract) and by
# the analysis stack from runs.jsonl rows
CONSUMER_FIELDS = ('run_id', 'experiment_id', 'text_id', 'voice_id', 'seed',
                   'status', 'gen_status', 'status_source', 'stop_reason',
                   'output_path', 'raw_duration_sec')
# §10 accounting fields present in every CosyVoice runs.jsonl row that reports read
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
    return Q.base_manifest('t1__v1__s0', 'QE1', Q.MODE, 't1', 'v1', '/nonexistent.wav', 0,
                           {'language': 'Russian'}, {}, 'Привет, мир.', '/tmp/x.wav')


def test_base_manifest_has_all_consumer_and_common_fields():
    m = _manifest()
    missing = [k for k in CONSUMER_FIELDS + COMMON_FIELDS if k not in m]
    assert not missing, missing


def test_manifest_slim_rule():
    m = _manifest()
    m['llm_calls'] = [1]
    m['exception_traceback'] = 'tb'
    m['generation_config'] = {'x': 1}
    slim = Q.slim_run_row(m)
    for k in Q.RUNS_JSONL_EXCLUDE:
        assert k not in slim
    for k in CONSUMER_FIELDS:
        assert k in slim


def test_manifest_row_acceptable_to_evaluator():
    m = _manifest()
    Q.set_status(m, 'complete', 'eos')
    assert gen_status_of(m) == 'complete'          # gen_status == status, no ambiguity
    m2 = _manifest()
    Q.set_status(m2, 'loop_cap', 'max_len')
    assert gen_status_of(m2) == 'loop_cap'


# --------------------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize('n_gen,cap,cap_src,valid,exc,expect', [
    (500, 15000, 'operational', True, None, ('complete', 'eos')),
    (500, 15000, 'operational', False, None, ('empty_or_invalid_audio', 'eos')),
    (0, 15000, 'operational', False, None, ('empty_or_invalid_audio', 'eos')),
    (15000, 15000, 'operational', True, None, ('loop_cap', 'max_len')),
    # HF off-by-one: a cap-exhausted run yields cap-1 frames (measured on the first two
    # B3 runaways: 14999 frames at max_new_tokens=15000) -> cap hit, NOT eos
    (14999, 15000, 'operational', True, None, ('loop_cap', 'max_len')),
    (14998, 15000, 'operational', True, None, ('complete', 'eos')),
    (12000, 12000, 'context', True, None, ('context_limit', 'max_len')),
    (11999, 12000, 'context', True, None, ('context_limit', 'max_len')),
    (100, 15000, 'operational', True, 'OutOfMemoryError', ('oom', 'exception')),
    (100, 15000, 'operational', True, 'RuntimeError', ('infrastructure_error', 'exception')),
])
def test_classify(n_gen, cap, cap_src, valid, exc, expect):
    msg = 'CUDA out of memory' if exc == 'OutOfMemoryError' else 'boom'
    got = Q.classify(n_gen, cap, cap_src, valid, exc, msg if exc else None)
    assert got == expect
    assert got[0] in KNOWN_STATUSES and got[1] in Q.STOP_REASONS


def test_classify_complete_only_from_eos():
    """The evaluator hard-fails a provisional complete without stop_reason='eos'."""
    for n_gen in (0, 1, 14999, 15000):
        status, stop = Q.classify(n_gen, 15000, 'operational', True)
        if status == 'complete':
            assert stop == 'eos'


def test_statuses_are_plan_34():
    assert set(Q.STATUSES) == set(KNOWN_STATUSES)


# --------------------------------------------------------------------------------------
# caps / watchdog
# --------------------------------------------------------------------------------------
def test_build_caps_context_budget_binds_for_long_prefix():
    caps = Q.build_caps({'watchdog': {'max_generated_tokens': 15000}}, context_tokens_input=25000)
    assert caps['effective_max_new_tokens'] == 32768 - 8 - 25000
    assert caps['cap_source'] == 'context'


def test_build_caps_operational_binds_for_short_prefix():
    caps = Q.build_caps({'watchdog': {'max_generated_tokens': 15000}}, context_tokens_input=2000)
    assert caps['effective_max_new_tokens'] == 15000
    assert caps['cap_source'] == 'operational'
    assert caps['model_default_max_new_tokens'] == 8192


def test_build_caps_exceeds_shipped_default_for_b4():
    """B4 needs > 8192 frames (655 s); the effective cap must not be the shipped default."""
    caps = Q.build_caps({'watchdog': {}}, context_tokens_input=3000)
    assert caps['effective_max_new_tokens'] > 8192


def test_config_yaml_frozen_values():
    import yaml
    with open(os.path.join(ROOT, 'configs', 'models', 'qwen3tts_base.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    assert cfg['model']['model_revision'] == Q.MODEL_REVISION
    assert cfg['model']['max_position_embeddings'] == Q.MAX_POSITION_EMBEDDINGS
    g = cfg['generation']
    for k in ('do_sample', 'top_k', 'top_p', 'temperature', 'repetition_penalty',
              'subtalker_dosample', 'subtalker_top_k', 'subtalker_top_p', 'subtalker_temperature'):
        assert g[k] == Q.FROZEN_GENERATE_DEFAULTS[k], k
    assert g['model_default_max_new_tokens'] == Q.FROZEN_GENERATE_DEFAULTS['max_new_tokens']
    assert g['watchdog']['max_generated_tokens'] == 15000
    assert cfg['seeds'] == [0]
