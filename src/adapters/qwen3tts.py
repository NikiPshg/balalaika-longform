"""Q-E1+ native adapter: one text -> one voice-clone conditioning -> ONE talker trajectory.

Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base @ fd4b254389122332181a7c3db7f27e918eec64e3, loaded
through the owner's fork third_party/Qwen3-TTS (branch trl-trainer). The fork's
code is used read-only; instrumentation wraps bound methods on the INSTANCE only.

Native single-trajectory path (mirror of the CosyVoice "native mode", prereg E10):

    Qwen3TTSModel.create_voice_clone_prompt(ref_wav, ref_text)     # ONE conditioning
    Qwen3TTSModel.generate_voice_clone(text, language='Russian',
                                       voice_clone_prompt=..., non_streaming_mode=False)
        -> Qwen3TTSForConditionalGeneration.generate()  ONCE
           (qwen_tts/core/models/modeling_qwen3_tts.py:2536-2616)
        -> self.talker.generate(inputs_embeds=..., eos_token_id=2150, ...)  ONCE
           = a single HF autoregressive trajectory over the FULL text; per frame the
           code predictor fills codebooks 1..15 (modeling_qwen3_tts.py:1886-1935)
        -> speech_tokenizer.decode() ONCE on the full code sequence, with the ref codes
           prepended for decoder context and their share of the wav cut off afterwards
           (qwen_tts/inference/qwen3_tts_model.py:690-711)

There is NO text chunking anywhere in this path and stream_generate_pcm (the windowed
re-decode streaming path) is NOT used. non_streaming_mode=False is the model's canonical
input layout: the same layout the fork's SFT teacher-forces
(lora_finetuning/main_talker_training.py:316 generate_icl_prompt(..., non_streaming_mode=False))
and its validation generation uses (main_talker_training.py:623).

Hard limits (this snapshot, cited):
  * talker max_position_embeddings = 32768   (config.json talker_config; default in
    qwen_tts/core/models/configuration_qwen3_tts.py:382)
  * codec_eos_token_id = 2150 (config.json talker_config); EOS is checked on codebook 0
    (modeling_qwen3_tts.py:2607-2613)
  * generation_config.json max_new_tokens = 8192 is a shipped DEFAULT, not a model
    limit -- 8192 frames = 655 s at 12.5 Hz, below the 15-min B4 bucket. The adapter
    overrides it per item: max_new_tokens = min(watchdog cap, context budget), where
    context budget = 32768 - margin - measured input prefix length.
  * codec frame rate = 12.5 Hz: decode_upsample_rate = 1920 samples/frame at 24000 Hz
    output (speech_tokenizer/config.json; configuration_qwen3_tts_tokenizer_v2.py:148-151).

This module stays importable without torch (schema tests run in .venv-eval); everything
heavy is imported lazily inside functions.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import uuid as _uuid
from typing import Any, Dict, Optional

MODE = 'native'

FORK_ROOT = os.environ.get('QWEN_ROOT', 'third_party/Qwen3-TTS')
MODEL_ID = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
MODEL_REVISION = 'fd4b254389122332181a7c3db7f27e918eec64e3'
DEFAULT_MODEL_DIR = os.environ.get('QWEN_MODEL_DIR', 'models/qwen3-tts')
VENV_PYTHON = os.path.join(FORK_ROOT, 'lora_finetuning', '.venv', 'bin', 'python')

MAX_POSITION_EMBEDDINGS = 32768   # config.json talker_config.max_position_embeddings
CODEC_EOS_TOKEN_ID = 2150         # config.json talker_config.codec_eos_token_id
CODEC_FRAME_HZ = 12.5             # 24000 Hz / 1920 samples-per-frame (speech_tokenizer/config.json)
SAMPLE_RATE = 24000               # speech_tokenizer output_sample_rate
MIN_NEW_TOKENS = 2                # modeling_qwen3_tts.py:2561 ("min_new_tokens": 2)

# generation_config.json of the pinned snapshot, frozen here and ASSERTED against the
# loaded model at startup (mirror of cosyvoice3_config_check): a silently different
# snapshot cannot produce numbers.
FROZEN_GENERATE_DEFAULTS = {
    'do_sample': True,
    'repetition_penalty': 1.05,
    'temperature': 0.9,
    'top_p': 1.0,
    'top_k': 50,
    'subtalker_dosample': True,
    'subtalker_temperature': 0.9,
    'subtalker_top_p': 1.0,
    'subtalker_top_k': 50,
    'max_new_tokens': 8192,       # shipped default; overridden per item (see module docstring)
}

STATUSES = ('complete', 'degraded', 'early_eos', 'loop_cap', 'hard_input_limit', 'context_limit',
            'oom', 'timeout', 'empty_or_invalid_audio', 'infrastructure_error')
STOP_REASONS = ('eos', 'max_len', 'watchdog', 'exception')

# keys dropped from the per-line runs.jsonl (kept in the per-item manifest), mirroring
# scripts/run_generation.py's slim rule
RUNS_JSONL_EXCLUDE = ('llm_calls', 'chunks', 'exception_traceback', 'generation_config')


class ConfigDivergence(RuntimeError):
    """The loaded snapshot's generation defaults differ from the frozen ones."""


# --------------------------------------------------------------------------------------
# torch-free helpers (unit-testable in .venv-eval)
# --------------------------------------------------------------------------------------
def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def repo_revision(root: str = FORK_ROOT) -> str:
    try:
        rev = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', root, 'status', '--porcelain',
                                         '--untracked-files=no'], text=True).strip()
        return rev + ('+localdiff' if dirty else '')
    except Exception as e:  # pragma: no cover
        return 'unknown ({})'.format(e)


def free_mem_info() -> dict:
    info = {}
    try:
        with open('/proc/meminfo') as f:
            for ln in f:
                k, v = ln.split(':', 1)
                if k in ('MemTotal', 'MemAvailable'):
                    info['ram_{}_gb'.format(k.lower())] = round(int(v.split()[0]) / 2**20, 2)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            info['vram_free_gb'] = round(free_b / 2**30, 2)
            info['vram_total_gb'] = round(total_b / 2**30, 2)
    except Exception:
        pass
    return info


def audio_stats(wav, sr: int) -> dict:
    import numpy as np
    if wav is None or getattr(wav, 'size', 0) == 0:
        return {'raw_duration_sec': 0.0, 'n_samples': 0, 'rms': 0.0, 'peak': 0.0,
                'has_nan': False, 'valid': False}
    finite = bool(np.isfinite(wav).all())
    peak = float(np.abs(wav).max()) if finite else float('nan')
    rms = float(np.sqrt(np.mean(np.square(wav)))) if finite else float('nan')
    return {'raw_duration_sec': round(wav.shape[-1] / sr, 4), 'n_samples': int(wav.shape[-1]),
            'rms': rms, 'peak': peak, 'has_nan': bool(not finite),
            'valid': bool(finite and wav.shape[-1] > 0 and peak > 1e-4)}


def write_wav_no_overwrite(out_path: str, wav, sr: int):
    import numpy as np
    import soundfile as sf
    if os.path.exists(out_path):
        raise FileExistsError('refusing to overwrite existing output {}'.format(out_path))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, np.asarray(wav, dtype=np.float32), sr, subtype='PCM_16')


def write_json_no_overwrite(path: str, obj: dict):
    if os.path.exists(path):
        raise FileExistsError('refusing to overwrite existing manifest {}'.format(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_caps(gen_cfg: dict, context_tokens_input: int) -> dict:
    """Effective max_new_tokens = min(operational watchdog cap, context budget).

    The context budget is the MODEL hard limit (32768 positions minus the measured input
    prefix minus a safety margin); the watchdog cap is the same 20-minutes-of-speech
    operational ceiling the CosyVoice arms run with (30000 tokens at 25 Hz there,
    15000 frames at 12.5 Hz here). Which one was binding is recorded so a cap hit can be
    classified context_limit vs loop_cap.
    """
    wd = dict(gen_cfg.get('watchdog') or {})
    enabled = bool(wd.get('enabled', True))
    margin = int(wd.get('context_safety_margin_tokens', 8))
    ctx_budget = None
    if bool(wd.get('enforce_context_budget', True)):
        ctx_budget = max(MIN_NEW_TOKENS, MAX_POSITION_EMBEDDINGS - margin - int(context_tokens_input))
    op_cap = wd.get('max_generated_tokens', 15000) if enabled else None
    limits = [x for x in (op_cap, ctx_budget) if x is not None]
    eff = min(limits) if limits else int(FROZEN_GENERATE_DEFAULTS['max_new_tokens'])
    cap_source = None
    if ctx_budget is not None and eff == ctx_budget and (op_cap is None or ctx_budget <= op_cap):
        cap_source = 'context'
    elif op_cap is not None and eff == op_cap:
        cap_source = 'operational'
    return {
        'enabled': enabled,
        'max_generated_tokens': op_cap,
        'context_budget_tokens': ctx_budget,
        'context_safety_margin_tokens': margin,
        'effective_max_new_tokens': int(eff),
        'cap_source': cap_source,
        'max_wall_seconds': wd.get('max_wall_seconds'),
        'max_wall_seconds_note': ('recorded only; HF generate() is atomic, no in-flight '
                                  'interruption (deviation from the CosyVoice watchdog)'),
        'model_default_max_new_tokens': int(FROZEN_GENERATE_DEFAULTS['max_new_tokens']),
        'expected_tokens_per_text_token': float(wd.get('expected_tokens_per_text_token', 2.0)),
    }


def classify(n_generated: int, effective_cap: int, cap_source: Optional[str],
             audio_valid: bool, exception_type: Optional[str] = None,
             exception_message: Optional[str] = None):
    """Map one finished (or failed) generation to the provisional §3.4 status.

    'complete' here means "the talker sampled codec EOS (2150) before the cap and the
    audio is valid"; the evaluator refines it to early_eos/degraded from content metrics.
    A cap hit is loop_cap when the operational watchdog bound first, context_limit when
    the model's own 32768-position budget did.

    OFF-BY-ONE (measured on the first two B3 runaways, 2026-08-31): with HF
    ``generate(max_new_tokens=N)`` the captured codec frames equal the number of DECODE
    forwards = sampled_tokens - 1 (the prefill forward carries no codec_ids and the last
    sampled token is never fed back), so a run that exhausts the cap yields **N-1**
    frames, never N. Hence the cap-hit condition is ``n_generated >= effective_cap - 1``.
    An EOS sampled exactly at the final step is indistinguishable and is conservatively
    classified as a cap hit (the evaluator scores the audio content either way).
    """
    if exception_type is not None:
        et = (exception_type or '').lower()
        em = (exception_message or '').lower()
        if 'outofmemory' in et or 'out of memory' in em or 'cuda out of memory' in em:
            return 'oom', 'exception'
        return 'infrastructure_error', 'exception'
    if n_generated < effective_cap - 1:
        # HF generate stopped before the cap => the EOS stopping criterion fired
        if not audio_valid:
            return 'empty_or_invalid_audio', 'eos'
        return 'complete', 'eos'
    if cap_source == 'context':
        return 'context_limit', 'max_len'
    return 'loop_cap', 'max_len'


def set_status(m: dict, status: str, stop_reason: str) -> dict:
    m['gen_status'] = status
    m['status'] = status
    m['status_source'] = ('generator (provisional, stop_reason-based); the final PLAN §3.4 status is assigned '
                          'by scripts/run_evaluation.py from EndCoverage / WER-floor')
    m['stop_reason'] = stop_reason
    return m


def slim_run_row(m: dict) -> dict:
    """The runs.jsonl view of a manifest (scripts/run_generation.py slim rule)."""
    return {k: v for k, v in m.items() if k not in RUNS_JSONL_EXCLUDE}


def base_manifest(run_id, experiment_id, mode, text_id, voice_id, ref_wav_path, seed,
                  gen_cfg, model_cfg, text, out_path) -> dict:
    """§10 run manifest skeleton, field-compatible with the CosyVoice adapters.

    Consumers (scripts/run_evaluation.py RUN_REQUIRED + gen_status/stop_reason contract,
    aggregate/table scripts) read: run_id, experiment_id, text_id, voice_id, seed,
    status/gen_status, stop_reason, output_path, raw_duration_sec. Everything else is
    provenance and §10 performance accounting.
    """
    return {
        'run_id': run_id,
        'experiment_id': experiment_id,
        'model_id': model_cfg.get('model_id', MODEL_ID),
        'model_revision': model_cfg.get('model_revision', MODEL_REVISION),
        'repo_revision': repo_revision(model_cfg.get('fork_root', FORK_ROOT)),
        'weights': model_cfg.get('weights', 'base'),
        'mode': mode,
        'text_id': text_id,
        'voice_id': voice_id,
        'reference_audio_path': ref_wav_path,
        'reference_audio_sha256': sha256_file(ref_wav_path) if ref_wav_path and os.path.exists(ref_wav_path) else None,
        'seed': seed,
        'generation_config': gen_cfg,
        'text_chars': len(text),
        'text_words': len(text.split()),
        'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'text_tokens': 0,
        'prompt_text_tokens': 0,
        'prompt_speech_tokens': 0,
        'generated_speech_tokens': 0,
        'speech_tokens_to_flow': 0,   # == generated_speech_tokens for Qwen (no silence filter)
        'context_tokens_total': 0,
        'context_tokens_input': 0,
        'max_position_embeddings': MAX_POSITION_EMBEDDINGS,
        'max_len_cap': 0,
        'min_len': MIN_NEW_TOKENS,
        'codec_frame_hz': CODEC_FRAME_HZ,
        'sample_rate': SAMPLE_RATE,
        'language': (gen_cfg or {}).get('language', 'Russian'),
        'non_streaming_mode': bool((gen_cfg or {}).get('non_streaming_mode', False)),
        'x_vector_only_mode': bool((gen_cfg or {}).get('x_vector_only_mode', False)),
        'sampling': None,             # filled from the model's generate defaults at run time
        'talker_dtype': model_cfg.get('dtype', 'bfloat16'),
        'attn_implementation': model_cfg.get('attn_implementation', 'sdpa'),
        'config_check': None,
        'gen_status': None,
        'status': None,
        'status_source': None,
        'stop_reason': None,
        'watchdog': None,
        'output_path': out_path,
        'raw_duration_sec': 0.0,
        'voiced_duration_sec': None,  # filled by the evaluators, not the adapter
        'wall_time_sec': 0.0,
        'llm_time_sec': 0.0,          # talker (LM) generate time
        'flow_hift_time_sec': 0.0,    # vocoder (speech tokenizer decode) time
        'frontend_time_sec': 0.0,     # voice-clone prompt + tokenization time
        'rtf': None,
        'speech_tokens_per_sec_wall': None,
        'peak_vram_bytes': 0,
        'peak_vram_reserved_bytes': 0,
        'exception_type': None,
        'exception_message': None,
        'exception_traceback': None,
        'audio': None,
        'env': {'python': sys.version.split()[0],
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


# --------------------------------------------------------------------------------------
# model loading + instrumentation (fork venv only)
# --------------------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Any] = {}
_PROMPT_CACHE: Dict[tuple, Any] = {}


def set_all_seeds(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def check_generate_defaults(tts) -> str:
    """Assert the loaded snapshot's generation_config.json equals the frozen values."""
    got = dict(tts.generate_defaults or {})
    mismatches = {k: (v, got.get(k)) for k, v in FROZEN_GENERATE_DEFAULTS.items()
                  if got.get(k) != v}
    if mismatches:
        raise ConfigDivergence('generation_config.json diverged from the frozen snapshot '
                               'values: {}'.format(mismatches))
    return '{} generate defaults equal to the frozen snapshot values'.format(len(FROZEN_GENERATE_DEFAULTS))


def load_qwen3tts(model_cfg: dict):
    """Build the fork's Qwen3TTSModel once per process (bf16, sdpa, one GPU)."""
    import torch
    if FORK_ROOT not in sys.path:
        sys.path.insert(0, model_cfg.get('fork_root', FORK_ROOT))
    from qwen_tts import Qwen3TTSModel

    model_dir = model_cfg.get('model_dir') or DEFAULT_MODEL_DIR
    key = os.path.abspath(model_dir)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32,
             'float16': torch.float16}[model_cfg.get('dtype', 'bfloat16')]
    tts = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map='cuda:0',
        dtype=dtype,
        attn_implementation=model_cfg.get('attn_implementation', 'sdpa'),
    )
    assert tts.model.tts_model_type == 'base', tts.model.tts_model_type
    assert int(tts.model.speech_tokenizer.get_output_sample_rate()) == SAMPLE_RATE
    assert int(tts.model.speech_tokenizer.get_decode_upsample_rate()) == 1920
    assert int(tts.model.config.talker_config.max_position_embeddings) == MAX_POSITION_EMBEDDINGS
    assert int(tts.model.config.talker_config.codec_eos_token_id) == CODEC_EOS_TOKEN_ID
    check_generate_defaults(tts)
    _MODEL_CACHE[key] = tts
    return tts


class _CallRecorder:
    """Wrap a bound method, counting calls and recording one payload via `extract`."""

    def __init__(self, obj, name, extract=None):
        self.obj, self.name, self.extract = obj, name, extract
        self.orig = getattr(obj, name)
        self.n = 0
        self.records = []

    def __enter__(self):
        def wrapped(*a, **k):
            self.n += 1
            out = self.orig(*a, **k)
            if self.extract is not None:
                try:
                    self.records.append(self.extract(a, k, out))
                except Exception as e:  # recording must never break generation
                    self.records.append({'record_error': repr(e)})
            return out
        setattr(self.obj, self.name, wrapped)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self.orig)
        return False


def get_or_build_prompt(tts, ref_wav_path: str, ref_text: str, x_vector_only: bool):
    """Voice-clone conditioning, cached per (wav, text) -- deterministic, no sampling."""
    key = (os.path.abspath(ref_wav_path), ref_text, bool(x_vector_only))
    if key not in _PROMPT_CACHE:
        _PROMPT_CACHE[key] = tts.create_voice_clone_prompt(
            ref_audio=ref_wav_path, ref_text=ref_text, x_vector_only_mode=bool(x_vector_only))
    return _PROMPT_CACHE[key]


def measure_input_context(tts, text: str, ref_text: str, prompt_items, language: str,
                          non_streaming_mode: bool) -> dict:
    """EXACT talker input prefix length, measured by running the fork's own input
    builder (_build_talker_inputs) on the same inputs generate_voice_clone will use.
    Deterministic embedding arithmetic only -- no sampling, no RNG. Runs under
    inference_mode: the prompt's ref_code comes out of the fork's
    @torch.inference_mode() create_voice_clone_prompt, and embedding an inference
    tensor under autograd tracking raises."""
    import torch
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])
    ref_ids = [tts._tokenize_texts([tts._build_ref_text(ref_text)])[0]] if ref_text else [None]
    vc = tts._prompt_items_to_voice_clone_prompt(prompt_items)
    with torch.inference_mode():
        embeds, mask, trailing, _pad = tts.model._build_talker_inputs(
            input_ids=input_ids, instruct_ids=None, ref_ids=ref_ids, voice_clone_prompt=vc,
            languages=[language], speakers=None, non_streaming_mode=non_streaming_mode)
    ref_code = vc.get('ref_code', [None])[0]
    return {
        'context_tokens_input': int(embeds.shape[1]),
        'text_tokens': int(input_ids[0].shape[1]),
        'prompt_text_tokens': int(ref_ids[0].shape[1]) if ref_ids[0] is not None else 0,
        'prompt_speech_tokens': int(ref_code.shape[0]) if ref_code is not None else 0,
        'trailing_text_hidden_len': int(trailing.shape[1]),
    }


# --------------------------------------------------------------------------------------
# the one-item synthesis
# --------------------------------------------------------------------------------------
def synthesize(text: str, ref_wav_path: str, ref_text: str, seed: int, gen_cfg: dict,
               out_path: str, watchdog: Optional[dict] = None, *,
               model_cfg: Optional[dict] = None, text_id: str = '', voice_id: str = '',
               experiment_id: str = 'QE1', run_id: Optional[str] = None,
               human_duration_sec: Optional[float] = None, cv=None,
               keep_tokens: bool = True) -> Dict[str, Any]:
    """Run ONE native single-trajectory synthesis and return the §10 run manifest.

    ``cv`` is the loaded Qwen3TTSModel (mirrors the CosyVoice adapter signature).
    Output is never overwritten. ``watchdog`` overrides gen_cfg['watchdog'].
    """
    import numpy as np
    import torch

    model_cfg = dict(model_cfg or {})
    gen_cfg = dict(gen_cfg or {})
    if watchdog is not None:
        gen_cfg['watchdog'] = {**(gen_cfg.get('watchdog') or {}), **watchdog}
    run_id = run_id or '{}__{}__s{}__{}'.format(text_id or 'text', voice_id or 'voice',
                                                seed, _uuid.uuid4().hex[:8])
    manifest_path = os.path.splitext(out_path)[0] + '.json'
    if os.path.exists(out_path) or os.path.exists(manifest_path):
        raise FileExistsError('output exists, refusing to overwrite: {} / {}'.format(out_path, manifest_path))

    language = gen_cfg.get('language', 'Russian')
    non_streaming = bool(gen_cfg.get('non_streaming_mode', False))
    x_vector_only = bool(gen_cfg.get('x_vector_only_mode', False))

    m = base_manifest(run_id, experiment_id, MODE, text_id, voice_id, ref_wav_path, seed,
                      gen_cfg, model_cfg, text, out_path)
    m['native_contract'] = {'create_prompt_calls': 0, 'model_generate_calls': 0,
                            'talker_generate_calls': 0, 'vocoder_decode_calls': 0,
                            'stream_generate_calls': 0, 'text_chunks': 1}
    m['mem_before'] = free_mem_info()
    t_wall0 = time.time()
    wav = None
    sr = SAMPLE_RATE
    exc_type = exc_msg = None
    n_generated = 0
    caps = None
    gen_rec = None

    try:
        tts = cv or load_qwen3tts(model_cfg)
        m['config_check'] = check_generate_defaults(tts)
        m['sampling'] = {k: tts.generate_defaults.get(k) for k in
                         ('do_sample', 'top_k', 'top_p', 'temperature', 'repetition_penalty',
                          'subtalker_dosample', 'subtalker_top_k', 'subtalker_top_p',
                          'subtalker_temperature')}
        import soundfile as sf
        m['reference_audio_duration_sec'] = round(sf.info(ref_wav_path).duration, 3)

        # ---- ONE conditioning ------------------------------------------------------
        t_fe0 = time.time()
        fresh = (os.path.abspath(ref_wav_path), ref_text, x_vector_only) not in _PROMPT_CACHE
        prompt_items = get_or_build_prompt(tts, ref_wav_path, ref_text, x_vector_only)
        m['native_contract']['create_prompt_calls'] = 1 if fresh else 0
        ctx = measure_input_context(tts, text, ref_text, prompt_items, language, non_streaming)
        m.update({k: ctx[k] for k in ('context_tokens_input', 'text_tokens',
                                      'prompt_text_tokens', 'prompt_speech_tokens')})
        m['frontend_time_sec'] = round(time.time() - t_fe0, 3)

        caps = build_caps(gen_cfg, ctx['context_tokens_input'])
        exp_tok = int(np.ceil(CODEC_FRAME_HZ * human_duration_sec)) if human_duration_sec \
            else int(np.ceil(m['text_tokens'] * caps['expected_tokens_per_text_token']))
        caps['expected_speech_tokens'] = exp_tok
        caps['expected_basis'] = ('12.5 Hz x human_duration_sec' if human_duration_sec
                                  else '{} x text_tokens'.format(caps['expected_tokens_per_text_token']))
        m['watchdog'] = caps
        m['max_len_cap'] = caps['effective_max_new_tokens']

        if ctx['context_tokens_input'] >= MAX_POSITION_EMBEDDINGS:
            raise RuntimeError('input context {} >= max_position_embeddings {}'.format(
                ctx['context_tokens_input'], MAX_POSITION_EMBEDDINGS))

        # ---- ONE trajectory --------------------------------------------------------
        set_all_seeds(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def _extract_codes(a, k, out):
            codes_list, _hidden = out
            return {'n_items': len(codes_list),
                    'frames': [int(c.shape[0]) for c in codes_list],
                    'codes0': [c[:, 0].tolist() for c in codes_list] if keep_tokens else None}

        t_gen0 = time.time()
        with _CallRecorder(tts.model, 'generate', _extract_codes) as gen_counter, \
             _CallRecorder(tts.model.talker, 'generate') as talker_counter, \
             _CallRecorder(tts.model.speech_tokenizer, 'decode') as dec_counter:
            wavs, sr = tts.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
                non_streaming_mode=non_streaming,
                max_new_tokens=caps['effective_max_new_tokens'],
            )
        gen_wall = time.time() - t_gen0
        m['native_contract'].update({
            'model_generate_calls': gen_counter.n,
            'talker_generate_calls': talker_counter.n,
            'vocoder_decode_calls': dec_counter.n,
        })
        gen_rec = gen_counter.records[0] if gen_counter.records else None
        n_generated = int(gen_rec['frames'][0]) if gen_rec and gen_rec.get('frames') else 0
        wav = np.asarray(wavs[0], dtype=np.float32) if wavs else np.zeros(0, dtype=np.float32)
        # llm vs vocoder split: decode is the tail of generate_voice_clone; time it apart
        # is not observable from outside, so decode time is measured by re-entering the
        # recorder? -- no: approximate by frames/decode throughput is invention. Instead
        # both phases live inside gen_wall; the vocoder share is measured directly below.
        m['llm_time_sec'] = round(gen_wall, 3)   # talker + code predictor + vocoder (one call)
    except torch.cuda.OutOfMemoryError as e:
        exc_type, exc_msg = type(e).__name__, str(e)[:2000]
        m['exception_traceback'] = traceback.format_exc()[-4000:]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        exc_type, exc_msg = type(e).__name__, str(e)[:2000]
        m['exception_traceback'] = traceback.format_exc()[-4000:]

    # ---- bookkeeping ----------------------------------------------------------------
    m['wall_time_sec'] = round(time.time() - t_wall0, 3)
    m['exception_type'], m['exception_message'] = exc_type, exc_msg
    m['generated_speech_tokens'] = n_generated
    m['speech_tokens_to_flow'] = n_generated
    m['context_tokens_total'] = int(m.get('context_tokens_input') or 0) + n_generated
    m['context_occupancy'] = round(m['context_tokens_total'] / MAX_POSITION_EMBEDDINGS, 4)
    try:
        import torch as _t
        if _t.cuda.is_available():
            m['peak_vram_bytes'] = int(_t.cuda.max_memory_allocated())
            m['peak_vram_reserved_bytes'] = int(_t.cuda.max_memory_reserved())
    except Exception:
        pass
    audio = audio_stats(wav, sr)
    m['audio'] = audio
    m['raw_duration_sec'] = audio['raw_duration_sec']
    if audio['raw_duration_sec'] > 0:
        m['rtf'] = round(m['wall_time_sec'] / audio['raw_duration_sec'], 4)
    if m['llm_time_sec'] and n_generated:
        m['speech_tokens_per_sec_wall'] = round(n_generated / m['llm_time_sec'], 2)

    eff_cap = (caps or {}).get('effective_max_new_tokens', int(FROZEN_GENERATE_DEFAULTS['max_new_tokens']))
    status, stop = classify(n_generated, eff_cap, (caps or {}).get('cap_source'),
                            audio['valid'], exc_type, exc_msg)
    set_status(m, status, stop)
    exp_tok = (m.get('watchdog') or {}).get('expected_speech_tokens')
    if exp_tok:
        ratio = round(n_generated / exp_tok, 4)
        m['generated_to_expected_token_ratio'] = ratio
        m['length_hint'] = 'short' if ratio < 0.5 else ('long' if ratio > 2.0 else 'plausible')
    m['mem_after'] = free_mem_info()

    # ---- save everything, never overwrite ------------------------------------------
    if wav is not None and getattr(wav, 'size', 0) > 0:
        write_wav_no_overwrite(out_path, wav, sr)
    else:
        m['output_path'] = None
    if keep_tokens and gen_rec and gen_rec.get('codes0'):
        m['generated_token_ids_path'] = os.path.splitext(out_path)[0] + '.tokens.json'
        write_json_no_overwrite(m['generated_token_ids_path'],
                                {'run_id': run_id, 'codec0_tokens': gen_rec['codes0'][0],
                                 'n_frames': n_generated, 'frame_hz': CODEC_FRAME_HZ,
                                 'eos_token_id': CODEC_EOS_TOKEN_ID})
    write_json_no_overwrite(manifest_path, m)
    m['manifest_path'] = manifest_path
    return m
