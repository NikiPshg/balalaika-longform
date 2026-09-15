"""VC-E1+ native adapter: one text -> one Ultimate-Cloning conditioning -> ONE trajectory.

Model: openbmb/VoxCPM2 @ 32279effe8c19989596f05d353d1447f51d9e915 (2B, tokenizer-free,
continuous latents), loaded through the upstream repo third_party/VoxCPM
(commit f5a1c6a6b901bc732e20f0d59a369f6829ad717a). The upstream code is used read-only;
instrumentation wraps bound methods on the INSTANCE only.

Native single-trajectory path (mirror of the CosyVoice "native mode" / Qwen E10 path,
prereg E12 in reports/decisions.md 2026-09-01):

    VoxCPM2Model.build_prompt_cache(prompt_text=ref_text, prompt_wav_path=ref_wav)
        # ONE deterministic conditioning: librosa.load @16 kHz -> AudioVAE.encode -> mu
        # (voxcpm2.py:695-754; encode returns the posterior MEAN, audio_vae_v2.py:489-501
        #  -- no sampling, cacheable per voice). cache mode == 'continuation' ==
        # "Ultimate Cloning" (README.md:180-192: prompt audio + verbatim transcript,
        # audio-continuation-based cloning).
    VoxCPM2Model.generate_with_prompt_cache(target_text, prompt_cache, ..., seed=seed)
        -> _generate_with_prompt_cache (voxcpm2.py:797-987): tokenizes
           prompt_text + target_text ONCE as one string (:851), appends audio_start 101,
           left-pads the prompt latents -> ONE prefill sequence
        -> _inference ONCE (voxcpm2.py:944 / 997-1131): a single autoregressive loop
           `for i in range(max_len)` (:1083); per step the LocDiT flow-matching decoder
           samples one 4-frame latent patch, the stop head (:1113-1115) plays the EOS
           role; NO text chunking, NO sentence splitting anywhere in this path
        -> audio_vae.decode ONCE on the full latent sequence (voxcpm2.py:981)

The retry-badcase machinery (voxcpm2.py:940-978: full-trajectory REgeneration with
seed+1 when audio/text ratio exceeds a threshold) is a stock anti-runaway wrapper that
would silently mask the very failure mode this study measures. The adapter DISABLES it
(retry_badcase=False -> the while loop runs exactly once, :977-978) and neutralises its
side cap `max_len=min(int(target_text_length*ratio_threshold+10), max_len)` (:950) by
passing ratio_threshold=1e6, so the effective cap is OURS alone:
min(operational 20-min cap, KV-cache context budget). Deviation from stock defaults is
recorded in every manifest (see `stock_deviations`).

Hard limits (this snapshot, cited):
  * KV cache = config.max_length = 8192 positions for BOTH prefill and generation:
    StaticKVCache.step() raises "KV cache is full" at position 8192
    (modules/minicpm4/cache.py:34-40); fill_caches would overflow on a longer prefill
    (cache.py:42-47). base_lm.setup_cache(1, config.max_length) (voxcpm2.py:184).
    lm max_position_embeddings=32768 (longrope) is NOT reachable through this cache.
  * One LM position = one latent patch = patch_size(4) x AudioVAE frame(1/25 s)
    = 0.16 s of audio -> 6.25 Hz (config.json: patch_size 4, encoder_rates [2,5,8,8]
    -> hop 640 @16 kHz). Decode: 4 x 1920 = 7680 samples @48 kHz per patch
    (decoder_rates [8,6,5,2,2,2], out_sample_rate 48000).
  * Output sample rate 48000 Hz (config.json audio_vae_config.out_sample_rate; the
    evaluator ingests 48 kHz natively -- src/eval/asr_gigaam.py:77 NATIVE_SAMPLE_RATES).
  * Stop: learned stop head, checked when i > min_len (voxcpm2.py:1113-1115); the loop
    appends the patch BEFORE the stop check, so n_generated == max_len <=> the cap was
    exhausted and n_generated < max_len <=> the stop head fired (no HF-style off-by-one;
    verified empirically in the P1 smoke: max_len=50 run returns exactly 50 patches).

Seed: generate_with_prompt_cache(seed=...) applies torch.manual_seed/cuda.manual_seed_all
per attempt natively (model/utils.py:27-37, voxcpm2.py:938-942); the adapter additionally
seeds python random / numpy for parity with the other arms. Determinism PROVEN:
3 x seed 0 -> byte-identical wavs (in-process and across process restarts, P1 probe).

The adapter file is named voxcpm.py (prereg) while the upstream package is also
`voxcpm`; the upstream import is therefore done lazily with the upstream src path
prepended, and if this module was itself registered as `voxcpm` it steps aside
(see _upstream()). Import of THIS module must go through
importlib spec_from_file_location (see scripts/run_generation_voxcpm.py) or any
non-clashing name; heavy deps are imported lazily inside functions.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import uuid as _uuid
from typing import Any, Dict, Optional

IS_RULONGTTS_ADAPTER = True   # sentinel for the sys.modules['voxcpm'] name clash guard

MODE = 'native'

UPSTREAM_ROOT = os.environ.get('VOXCPM_ROOT', 'third_party/VoxCPM')
UPSTREAM_SRC = os.path.join(UPSTREAM_ROOT, 'src')
MODEL_ID = 'openbmb/VoxCPM2'
MODEL_REVISION = '32279effe8c19989596f05d353d1447f51d9e915'
DEFAULT_MODEL_DIR = os.environ.get('VOXCPM_MODEL_DIR', 'models/voxcpm2')
VENV_PYTHON = os.path.join(UPSTREAM_ROOT, '.venv', 'bin', 'python')

MAX_CACHE_LENGTH = 8192      # config.json max_length == StaticKVCache size (hard limit)
PATCH_HZ = 6.25              # 16000 / (hop 640) / (patch_size 4)
SAMPLES_PER_PATCH = 7680     # patch_size 4 x decode chunk 1920 @ 48 kHz
SAMPLE_RATE = 48000          # audio_vae_config.out_sample_rate
ENCODE_SAMPLE_RATE = 16000   # audio_vae_config.sample_rate (prompt encoding)
MIN_LEN = 2                  # core.py:191 default min_len (stop head not consulted before)

# Stock inference knobs, frozen from the snapshot's config.json + the upstream code
# defaults (there is no generation_config.json in a VoxCPM snapshot); ASSERTED against
# the loaded model at startup (check_generate_defaults), mirror of the CosyVoice/Qwen
# config checks: a silently different snapshot cannot produce numbers.
FROZEN_GENERATE_DEFAULTS = {
    'inference_timesteps': 10,     # core.py:191 _generate default
    'cfg_value': 2.0,              # core.py:189 default == config dit.cfm.inference_cfg_rate
    'inference_cfg_rate': 2.0,     # config.json dit_config.cfm_config
    'min_len': 2,                  # core.py:191
    'dtype': 'bfloat16',           # config.json
    'patch_size': 4,               # config.json
    'max_length': 8192,            # config.json (KV cache)
    'sample_rate': 48000,          # audio_vae out_sample_rate
    'encode_sample_rate': 16000,   # audio_vae sample_rate
    'model_default_max_len': 4096,  # core.py:192 shipped default cap, NOT a model limit
}

STATUSES = ('complete', 'degraded', 'early_eos', 'loop_cap', 'hard_input_limit', 'context_limit',
            'oom', 'timeout', 'empty_or_invalid_audio', 'infrastructure_error')
STOP_REASONS = ('eos', 'max_len', 'watchdog', 'exception')

# keys dropped from the per-line runs.jsonl (kept in the per-item manifest), mirroring
# scripts/run_generation.py's slim rule
RUNS_JSONL_EXCLUDE = ('llm_calls', 'chunks', 'exception_traceback', 'generation_config')


class ConfigDivergence(RuntimeError):
    """The loaded snapshot's inference defaults differ from the frozen ones."""


# --------------------------------------------------------------------------------------
# torch-free helpers (unit-testable in .venv-eval)
# --------------------------------------------------------------------------------------
def prep_text(text: str) -> str:
    """EXACTLY core.py:246-247 (VoxCPM._generate): newline -> space, collapse whitespace.

    Applied to the TARGET text only, as upstream does; prompt_text goes in verbatim.
    """
    return re.sub(r'\s+', ' ', text.replace('\n', ' '))


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def repo_revision(root: str = UPSTREAM_ROOT) -> str:
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
    """Effective max_len = min(operational watchdog cap, KV-cache context budget).

    The context budget is the MODEL hard limit: the StaticKVCache holds 8192 positions
    for prefill + generated patches together and raises "KV cache is full" past it
    (modules/minicpm4/cache.py:34-40), so max_len <= 8192 - margin - measured prefill.
    The watchdog cap is the same 20-minutes-of-speech operational ceiling the
    CosyVoice/Qwen arms run with: 7500 patches at 6.25 Hz (== 30000 @25 Hz == 15000
    @12.5 Hz). Which one was binding is recorded so a cap hit can be classified
    context_limit vs loop_cap.
    """
    wd = dict(gen_cfg.get('watchdog') or {})
    enabled = bool(wd.get('enabled', True))
    margin = int(wd.get('context_safety_margin_tokens', 8))
    ctx_budget = None
    if bool(wd.get('enforce_context_budget', True)):
        ctx_budget = max(MIN_LEN + 1, MAX_CACHE_LENGTH - margin - int(context_tokens_input))
    op_cap = wd.get('max_generated_tokens', 7500) if enabled else None
    limits = [x for x in (op_cap, ctx_budget) if x is not None]
    eff = min(limits) if limits else int(FROZEN_GENERATE_DEFAULTS['model_default_max_len'])
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
        'max_wall_seconds_note': ('recorded only; the upstream generation loop is atomic, no '
                                  'in-flight interruption (same deviation as the Qwen adapter)'),
        'model_default_max_new_tokens': int(FROZEN_GENERATE_DEFAULTS['model_default_max_len']),
        'expected_tokens_per_text_token': float(wd.get('expected_tokens_per_text_token', 1.0)),
    }


def classify(n_generated: int, effective_cap: int, cap_source: Optional[str],
             audio_valid: bool, exception_type: Optional[str] = None,
             exception_message: Optional[str] = None):
    """Map one finished (or failed) generation to the provisional §3.4 status.

    'complete' here means "the stop head fired before the cap and the audio is valid";
    the evaluator refines it to early_eos/degraded from content metrics. A cap hit is
    loop_cap when the operational watchdog bound first, context_limit when the model's
    own 8192-position KV cache budget did.

    NO off-by-one here (unlike HF generate in the Qwen adapter): _inference appends the
    patch and only then consults the stop head (voxcpm2.py:1101, 1113-1115), so a run
    that exhausts max_len returns EXACTLY max_len patches (verified in the P1 smoke)
    and a stop-head stop returns < max_len. A stop that fires exactly at the final
    iteration is indistinguishable from the cap and is conservatively classified as a
    cap hit (the evaluator scores the audio content either way).
    """
    if exception_type is not None:
        et = (exception_type or '').lower()
        em = (exception_message or '').lower()
        if 'outofmemory' in et or 'out of memory' in em or 'cuda out of memory' in em:
            return 'oom', 'exception'
        return 'infrastructure_error', 'exception'
    if n_generated < effective_cap:
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
    """§10 run manifest skeleton, field-compatible with the CosyVoice/Qwen adapters.

    Consumers (scripts/run_evaluation.py RUN_REQUIRED + gen_status/stop_reason contract,
    aggregate/table scripts) read: run_id, experiment_id, text_id, voice_id, seed,
    status/gen_status, stop_reason, output_path, raw_duration_sec. Everything else is
    provenance and §10 performance accounting. "speech tokens" here are LM latent
    patches at 6.25 Hz (VoxCPM is tokenizer-free: no discrete audio tokens exist).
    """
    return {
        'run_id': run_id,
        'experiment_id': experiment_id,
        'model_id': model_cfg.get('model_id', MODEL_ID),
        'model_revision': model_cfg.get('model_revision', MODEL_REVISION),
        'repo_revision': repo_revision(model_cfg.get('upstream_root', UPSTREAM_ROOT)),
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
        'speech_tokens_to_flow': 0,   # == generated_speech_tokens (no silence filter)
        'context_tokens_total': 0,
        'context_tokens_input': 0,
        'max_position_embeddings': MAX_CACHE_LENGTH,   # the OPERATIVE limit (KV cache)
        'max_len_cap': 0,
        'min_len': MIN_LEN,
        'codec_frame_hz': PATCH_HZ,
        'sample_rate': SAMPLE_RATE,
        'cloning_mode': 'ultimate',   # prompt_wav + prompt_text -> cache mode 'continuation'
        'sampling': None,             # filled from the model's frozen defaults at run time
        'stock_deviations': {
            'retry_badcase': 'stock True (core.py:195) -> False: retries REgenerate the whole '
                             'trajectory with seed+1 and would mask runaway failures; one item = '
                             'one trajectory (prereg E12)',
            'retry_badcase_ratio_threshold': 'stock 6.0 -> 1e6: neutralises the hidden text-length '
                                             'cap max_len=min(int(n_text_tokens*6+10), max_len) '
                                             '(voxcpm2.py:950); OUR caps are the only ones binding',
            'denoiser': 'ZipEnhancer denoiser not loaded (enable_denoiser=False); refs are clean '
                        'studio-checked clips, other arms do not denoise either',
            'normalize': 'core normalize=False default kept: benchmark texts go in verbatim '
                         '(whitespace collapse only, core.py:246-247)',
        },
        'talker_dtype': model_cfg.get('dtype', 'bfloat16'),
        'attn_implementation': None,  # upstream uses its own MiniCPM4 attention, not configurable
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
        'llm_time_sec': 0.0,          # LM + LocDiT autoregressive loop time
        'flow_hift_time_sec': 0.0,    # AudioVAE decode (vocoder) time
        'frontend_time_sec': 0.0,     # prompt cache + tokenization time
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
# model loading + instrumentation (voxcpm venv only)
# --------------------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Any] = {}
_PROMPT_CACHE: Dict[tuple, Any] = {}


def _upstream():
    """Import the UPSTREAM voxcpm package, resolving the name clash with this module.

    This adapter file is prereg-named src/adapters/voxcpm.py; if it was imported as
    module 'voxcpm' it steps aside so the upstream package (whose relative imports
    need to own that name) wins. The upstream src dir is prepended to sys.path, so
    a fresh `import voxcpm` resolves to the package even if src/adapters is on the path.
    """
    import importlib
    if sys.path[:1] != [UPSTREAM_SRC]:
        if UPSTREAM_SRC in sys.path:
            sys.path.remove(UPSTREAM_SRC)
        sys.path.insert(0, UPSTREAM_SRC)
    me = sys.modules.get('voxcpm')
    if me is not None and getattr(me, 'IS_RULONGTTS_ADAPTER', False):
        del sys.modules['voxcpm']
    return importlib.import_module('voxcpm.core')


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
    """Assert the loaded snapshot's config equals the frozen inference values."""
    got = {
        'inference_timesteps': 10,   # code default, asserted by passing it explicitly
        'cfg_value': 2.0,            # code default, asserted by passing it explicitly
        'inference_cfg_rate': float(tts.config.dit_config.cfm_config.inference_cfg_rate),
        'min_len': MIN_LEN,
        'dtype': str(tts.config.dtype),
        'patch_size': int(tts.config.patch_size),
        'max_length': int(tts.config.max_length),
        'sample_rate': int(tts.sample_rate),
        'encode_sample_rate': int(tts._encode_sample_rate),
        'model_default_max_len': 4096,   # core.py:192; informational (overridden per item)
    }
    mismatches = {k: (v, got.get(k)) for k, v in FROZEN_GENERATE_DEFAULTS.items()
                  if got.get(k) != v}
    if mismatches:
        raise ConfigDivergence('snapshot config diverged from the frozen values: '
                               '{}'.format(mismatches))
    return '{} inference defaults equal to the frozen snapshot values'.format(len(FROZEN_GENERATE_DEFAULTS))


def _proc_mem_mb() -> dict:
    """Current / peak RSS of this process from /proc/self/status (MB); anon part separately."""
    out = {}
    try:
        with open('/proc/self/status') as f:
            for ln in f:
                k = ln.split(':', 1)[0]
                if k in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile'):
                    out[k] = round(int(ln.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return out


class _params_on_meta:
    """Context: every nn.Parameter registered inside is moved to the 'meta' device.

    Buffers are untouched (created on CPU exactly as upstream does), so non-persistent
    buffers computed in __init__ (rotary inv_freq, LoRA scaling, ...) keep their stock
    values. Same trick as accelerate.init_empty_weights(include_buffers=False), written
    out because accelerate is not in the upstream venv.
    """

    def __enter__(self):
        import torch
        self._orig = torch.nn.Module.register_parameter

        def register_parameter(module, name, param):
            self._orig(module, name, param)
            p = module._parameters.get(name)
            if p is not None and p.device.type != 'meta':
                module._parameters[name] = torch.nn.Parameter(
                    p.to('meta'), requires_grad=p.requires_grad)
        torch.nn.Module.register_parameter = register_parameter
        return self

    def __exit__(self, *exc):
        import torch
        torch.nn.Module.register_parameter = self._orig
        return False


_ORIG_FROM_LOCAL = None
LOADER_INFO: Dict[str, Any] = {}


def _from_local_lowhost(cls, path: str, optimize: bool = True, training: bool = False,
                        device: str | None = None, lora_config=None):
    """LOCAL EDIT 14 (A23, 2026-09-05): VoxCPM2Model.from_local with a small HOST-RAM peak.

    Upstream from_local (model/voxcpm2.py:1134-1205) builds the 2B model on the CPU in
    fp32 (~9.2 GB), casts it to bf16 (~4.6 GB), then load_file()s the WHOLE checkpoint
    into host RAM (bf16 base: 4.3 GB; our fp32 SFT checkpoints: 9.2 GB) before copying
    it into the model -- a ~14 GB host peak for an SFT checkpoint. On the shared 62-GB
    host with ~45 GB taken by foreign processes that peak got the gate decode killed by
    the RAM backstop (2026-09-05 20:04 UTC). This replacement is FUNCTIONALLY IDENTICAL
    (same construction, same buffers, same fp32->bf16 round-to-nearest cast, same
    audio_vae fp32 handling, same .to(device).eval().optimize()) but:
      * parameters are constructed on the 'meta' device (no CPU fp32 materialisation);
      * model.safetensors is read tensor by tensor straight onto the target device
        (safetensors safe_open(device=...)), cast to the parameter dtype there, and
        installed with load_state_dict(assign=True);
      * a parameter left on 'meta' (a key the checkpoint lacks -- upstream would have
        silently kept random init under strict=False) is a hard error.
    Training / LoRA loads are delegated to the stock implementation unchanged.
    """
    import torch
    from safetensors import safe_open
    v2 = sys.modules[cls.__module__]              # voxcpm.model.voxcpm2 (upstream, read-only)
    if training or lora_config is not None:
        return _ORIG_FROM_LOCAL.__func__(cls, path, optimize=optimize, training=training,
                                         device=device, lora_config=lora_config)
    mem0 = _proc_mem_mb()
    t0 = time.time()
    # --- :1142-1164 verbatim ---------------------------------------------------------
    with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as _cfg_f:
        config = v2.VoxCPMConfig.model_validate_json(_cfg_f.read())
    tokenizer = v2.LlamaTokenizerFast.from_pretrained(path)
    audio_vae_config = getattr(config, 'audio_vae_config', None)
    audio_vae = v2.AudioVAEV2(config=audio_vae_config) if audio_vae_config else v2.AudioVAEV2()
    audiovae_safetensors_path = os.path.join(path, 'audiovae.safetensors')
    audiovae_pth_path = os.path.join(path, 'audiovae.pth')
    if os.path.exists(audiovae_safetensors_path) and v2.SAFETENSORS_AVAILABLE:
        print('Loading AudioVAE from safetensors: {}'.format(audiovae_safetensors_path), file=sys.stderr)
        vae_state_dict = v2.load_file(audiovae_safetensors_path, device='cpu')
    elif os.path.exists(audiovae_pth_path):
        print('Loading AudioVAE from pytorch: {}'.format(audiovae_pth_path), file=sys.stderr)
        checkpoint = torch.load(audiovae_pth_path, map_location='cpu', weights_only=True)
        vae_state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        raise FileNotFoundError('AudioVAE checkpoint not found. Expected either {} or {}'.format(
            audiovae_safetensors_path, audiovae_pth_path))
    # --- :1165-1177 with parameters on meta -------------------------------------------
    with _params_on_meta():
        model = cls(config, tokenizer, audio_vae, lora_config, device=device)
    lm_dtype = v2.get_dtype(model.config.dtype)
    model = model.to(lm_dtype)
    model.audio_vae = model.audio_vae.to(torch.float32)
    # --- :1180-1202: state dict straight onto the device, cast per tensor -------------
    safetensors_path = os.path.join(path, 'model.safetensors')
    if not (os.path.exists(safetensors_path) and v2.SAFETENSORS_AVAILABLE):
        raise FileNotFoundError('lowhost loader needs model.safetensors in {}'.format(path))
    dev = torch.device(model.device)
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    print('Loading model from safetensors (lowhost, tensor-by-tensor onto {}): {}'.format(
        dev, safetensors_path), file=sys.stderr)
    model_state_dict = {}
    n_cast = 0
    with safe_open(safetensors_path, framework='pt', device=str(dev)) as f:
        for k in f.keys():
            t = f.get_tensor(k)
            tgt = targets.get(k)
            if tgt is not None and tgt.is_floating_point() and t.is_floating_point() and t.dtype != tgt.dtype:
                t = t.to(tgt.dtype)     # == upstream's in-place copy_ cast (RNE) into the bf16 param
                n_cast += 1
            model_state_dict[k] = t
    for kw, val in vae_state_dict.items():
        model_state_dict['audio_vae.{}'.format(kw)] = val
    incompatible = model.load_state_dict(model_state_dict, strict=False, assign=True)
    del model_state_dict
    meta_left = [n for n, p in model.named_parameters() if p.device.type == 'meta']
    meta_left += [n for n, b in model.named_buffers() if b.device.type == 'meta']
    if meta_left:
        raise RuntimeError('lowhost loader: {} tensors missing from the checkpoint would stay '
                           'uninitialised (upstream strict=False hides this): {}'.format(
                               len(meta_left), meta_left[:8]))
    mem1 = _proc_mem_mb()
    LOADER_INFO.update({
        'weights_loader': 'lowhost-v1 (LOCAL EDIT 14)', 'n_cast_to_model_dtype': n_cast,
        'missing_keys': list(incompatible.missing_keys), 'unexpected_keys': list(incompatible.unexpected_keys),
        'host_mem_mb_before': mem0, 'host_mem_mb_after_load': mem1, 'load_seconds': round(time.time() - t0, 1)})
    print('[lowhost loader] {} tensors cast to {}; missing={} unexpected={}; host RSS {} -> {} MB '
          '(anon {} -> {}, HWM {}) in {:.1f}s'.format(
              n_cast, lm_dtype, len(incompatible.missing_keys), len(incompatible.unexpected_keys),
              mem0.get('VmRSS'), mem1.get('VmRSS'), mem0.get('RssAnon'), mem1.get('RssAnon'),
              mem1.get('VmHWM'), time.time() - t0), file=sys.stderr)
    return model.to(model.device).eval().optimize(disable=not optimize)


def _install_lowhost_loader(core):
    """Patch VoxCPM2Model.from_local (the class object core.VoxCPM uses) once per process.

    Opt-out for A/B checks: VOXCPM_STOCK_LOADER=1 keeps the stock loader."""
    global _ORIG_FROM_LOCAL
    if os.environ.get('VOXCPM_STOCK_LOADER') == '1':
        LOADER_INFO.update({'weights_loader': 'stock from_local (VOXCPM_STOCK_LOADER=1)'})
        return
    cls = core.VoxCPM2Model
    if getattr(cls.from_local, '__name__', '') == '_from_local_lowhost':
        return
    _ORIG_FROM_LOCAL = cls.from_local
    cls.from_local = classmethod(_from_local_lowhost)


def load_voxcpm(model_cfg: dict):
    """Build the upstream VoxCPM pipeline once per process (bf16, torch.compile, one GPU)."""
    core = _upstream()

    model_dir = model_cfg.get('model_dir') or DEFAULT_MODEL_DIR
    key = os.path.abspath(model_dir)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    _install_lowhost_loader(core)     # LOCAL EDIT 14: small host-RAM peak at weight load
    pipeline = core.VoxCPM(
        voxcpm_model_path=model_dir,
        zipenhancer_model_path=None,          # no denoiser (see stock_deviations)
        enable_denoiser=False,
        optimize=bool(model_cfg.get('optimize', True)),   # torch.compile; determinism proven with it
        device=model_cfg.get('device'),       # None -> auto (cuda via CUDA_VISIBLE_DEVICES)
    )
    tts = pipeline.tts_model
    assert type(tts).__name__ == 'VoxCPM2Model', type(tts).__name__
    assert int(tts.chunk_size) == 640 and int(tts._decode_chunk_size) == 1920
    assert int(tts.patch_size) * int(tts._decode_chunk_size) == SAMPLES_PER_PATCH
    check_generate_defaults(tts)
    _MODEL_CACHE[key] = pipeline
    return pipeline


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
            t0 = time.time()
            out = self.orig(*a, **k)
            if self.extract is not None:
                try:
                    self.records.append(self.extract(a, k, out, time.time() - t0))
                except Exception as e:  # recording must never break generation
                    self.records.append({'record_error': repr(e)})
            return out
        setattr(self.obj, self.name, wrapped)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self.orig)
        return False


def get_or_build_prompt(tts, ref_wav_path: str, ref_text: str):
    """Ultimate-Cloning conditioning, cached per (wav, text) -- deterministic (VAE mean,
    audio_vae_v2.py:501), no sampling."""
    key = (os.path.abspath(ref_wav_path), ref_text)
    if key not in _PROMPT_CACHE:
        _PROMPT_CACHE[key] = tts.build_prompt_cache(
            prompt_text=ref_text, prompt_wav_path=ref_wav_path)
    return _PROMPT_CACHE[key]


def measure_input_context(tts, text: str, ref_text: str, prompt_cache) -> dict:
    """EXACT prefill length, computed the same way _generate_with_prompt_cache builds it
    (voxcpm2.py:844-877, mode 'continuation'): tokens(prompt_text + target_text) + 1
    audio_start token + prompt latent patches. Pure tokenizer arithmetic -- no forward,
    no RNG."""
    n_full = len(tts.text_tokenizer(ref_text + text))
    n_target = len(tts.text_tokenizer(text))
    n_prompt_text = len(tts.text_tokenizer(ref_text))
    n_prompt_patches = int(prompt_cache['audio_feat'].shape[0])
    return {
        'context_tokens_input': n_full + 1 + n_prompt_patches,
        'text_tokens': n_target,
        'prompt_text_tokens': n_prompt_text,
        'prompt_speech_tokens': n_prompt_patches,
        'full_text_tokens': n_full,
    }


# --------------------------------------------------------------------------------------
# the one-item synthesis
# --------------------------------------------------------------------------------------
def synthesize(text: str, ref_wav_path: str, ref_text: str, seed: int, gen_cfg: dict,
               out_path: str, watchdog: Optional[dict] = None, *,
               model_cfg: Optional[dict] = None, text_id: str = '', voice_id: str = '',
               experiment_id: str = 'VCE1', run_id: Optional[str] = None,
               human_duration_sec: Optional[float] = None, cv=None,
               keep_tokens: bool = True) -> Dict[str, Any]:
    """Run ONE native single-trajectory synthesis and return the §10 run manifest.

    ``cv`` is the loaded VoxCPM pipeline (mirrors the CosyVoice/Qwen adapter signature).
    Output is never overwritten. ``watchdog`` overrides gen_cfg['watchdog'].
    ``keep_tokens`` is accepted for driver parity and ignored: VoxCPM is tokenizer-free,
    there are no discrete audio tokens to dump.
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

    target_text = prep_text(text)

    m = base_manifest(run_id, experiment_id, MODE, text_id, voice_id, ref_wav_path, seed,
                      gen_cfg, model_cfg, text, out_path)
    m['native_contract'] = {'build_prompt_cache_calls': 0, 'inference_calls': 0,
                            'vocoder_decode_calls': 0, 'retry_attempts': 0,
                            'text_chunks': 1}
    m['mem_before'] = free_mem_info()
    t_wall0 = time.time()
    wav = None
    sr = SAMPLE_RATE
    exc_type = exc_msg = None
    n_generated = 0
    caps = None
    decode_time = [0.0]

    try:
        pipeline = cv or load_voxcpm(model_cfg)
        tts = pipeline.tts_model
        m['config_check'] = check_generate_defaults(tts)
        m['sampling'] = {'inference_timesteps': int(gen_cfg.get('inference_timesteps', 10)),
                         'cfg_value': float(gen_cfg.get('cfg_value', 2.0)),
                         'retry_badcase': False,
                         'retry_badcase_ratio_threshold': 1e6}
        import soundfile as sf
        m['reference_audio_duration_sec'] = round(sf.info(ref_wav_path).duration, 3)

        # ---- ONE conditioning ------------------------------------------------------
        t_fe0 = time.time()
        fresh = (os.path.abspath(ref_wav_path), ref_text) not in _PROMPT_CACHE
        prompt_cache = get_or_build_prompt(tts, ref_wav_path, ref_text)
        assert prompt_cache['mode'] == 'continuation', prompt_cache['mode']
        m['native_contract']['build_prompt_cache_calls'] = 1 if fresh else 0
        ctx = measure_input_context(tts, target_text, ref_text, prompt_cache)
        m.update({k: ctx[k] for k in ('context_tokens_input', 'text_tokens',
                                      'prompt_text_tokens', 'prompt_speech_tokens')})
        m['frontend_time_sec'] = round(time.time() - t_fe0, 3)

        caps = build_caps(gen_cfg, ctx['context_tokens_input'])
        exp_tok = int(np.ceil(PATCH_HZ * human_duration_sec)) if human_duration_sec \
            else int(np.ceil(m['text_tokens'] * caps['expected_tokens_per_text_token']))
        caps['expected_speech_tokens'] = exp_tok
        caps['expected_basis'] = ('6.25 Hz x human_duration_sec' if human_duration_sec
                                  else '{} x text_tokens'.format(caps['expected_tokens_per_text_token']))
        m['watchdog'] = caps
        m['max_len_cap'] = caps['effective_max_new_tokens']

        if ctx['context_tokens_input'] >= MAX_CACHE_LENGTH:
            raise RuntimeError('input prefill {} >= KV cache size {}'.format(
                ctx['context_tokens_input'], MAX_CACHE_LENGTH))

        # ---- ONE trajectory --------------------------------------------------------
        set_all_seeds(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def _extract_decode(a, k, out, dt):
            decode_time[0] += dt
            return {'out_shape': list(out.shape)}

        t_gen0 = time.time()
        with _CallRecorder(tts, '_inference') as inf_counter, \
             _CallRecorder(tts.audio_vae, 'decode', _extract_decode) as dec_counter:
            wav_t, _tgt_tok, pred_feat = tts.generate_with_prompt_cache(
                target_text=target_text,
                prompt_cache=prompt_cache,
                min_len=MIN_LEN,
                max_len=caps['effective_max_new_tokens'],
                inference_timesteps=int(gen_cfg.get('inference_timesteps', 10)),
                cfg_value=float(gen_cfg.get('cfg_value', 2.0)),
                retry_badcase=False,
                retry_badcase_ratio_threshold=1e6,
                seed=seed,
            )
        gen_wall = time.time() - t_gen0
        m['native_contract'].update({
            'inference_calls': inf_counter.n,
            'vocoder_decode_calls': dec_counter.n,
            'retry_attempts': inf_counter.n - 1,
        })
        assert tts.last_successful_seed == seed, \
            'seed drift: last_successful_seed={} != {}'.format(tts.last_successful_seed, seed)
        n_generated = int(pred_feat.shape[0])
        wav = wav_t.squeeze(0).cpu().numpy().astype(np.float32)
        m['flow_hift_time_sec'] = round(decode_time[0], 3)
        m['llm_time_sec'] = round(gen_wall - decode_time[0], 3)
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
    m['context_occupancy'] = round(m['context_tokens_total'] / MAX_CACHE_LENGTH, 4)
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

    eff_cap = (caps or {}).get('effective_max_new_tokens', int(FROZEN_GENERATE_DEFAULTS['model_default_max_len']))
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
    write_json_no_overwrite(manifest_path, m)
    m['manifest_path'] = manifest_path
    return m
