"""Shared machinery for the CosyVoice3 adapters (A3).

Everything here works WITHOUT modifying third_party/CosyVoice: the repo is put on
sys.path, the stock ``CosyVoice3`` object is built once, and instrumentation is done by
wrapping bound methods on the *instances* (``model.llm.inference``,
``frontend.frontend_zero_shot``) for the duration of one ``synthesize`` call.

Key repo facts this module relies on (see reports/cosyvoice3_audit.md for line refs):

* ``CosyVoice3Model.tts`` (cosyvoice/cli/model.py:328-394) starts ONE thread that calls
  ``self.llm.inference`` exactly once (model.py:113-120), collects every yielded token in
  ``tts_speech_token_dict[uuid]`` (model.py:121-128, dropping >5 consecutive silent tokens),
  and in non-stream mode waits for the thread and runs flow+hift ONCE on the full token
  sequence (model.py:376-387).
* ``CosyVoice3LM.inference`` (cosyvoice/llm/llm.py:458-502) computes
  ``max_len = int(text_len * max_token_text_ratio)`` and ``inference_wrapper`` (llm.py:536-549)
  loops ``for i in range(max_len)`` and ``break``s on a stop token. Hence
  ``n_yielded == max_len``  <=>  the loop cap was hit (no EOS was ever sampled).
* Exceptions inside the LLM thread are NOT propagated by the repo (the thread just dies,
  ``llm_end_dict`` stays False, and the non-stream path proceeds with the partial tokens).
  The wrapper below captures them so the manifest can classify oom / infrastructure_error.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

COSYVOICE_ROOT = os.environ.get('COSYVOICE_ROOT', 'third_party/CosyVoice')
MATCHA_ROOT = os.path.join(COSYVOICE_ROOT, 'third_party', 'Matcha-TTS')
for _p in (COSYVOICE_ROOT, MATCHA_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Device selection belongs to the caller; CUDA_VISIBLE_DEVICES is respected.
# PLAN §0.1: write nothing outside the working folder. deepspeed (imported by the CosyVoice repo) keeps a
# Triton autotune table at ~/.triton/autotune unless TRITON_CACHE_DIR is set (deepspeed/.../matmul_ext.py:53,86).
# Integration smoke 2026-08-28 found the pickles + locks being written to $HOME; redirect them into tmp/.
os.environ.setdefault('TRITON_CACHE_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tmp', 'triton'))
os.makedirs(os.environ['TRITON_CACHE_DIR'], exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402

MODEL_ID = 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512'
MODEL_REVISION = '29e01c4e8d000f4bcd70751be16fa94bf3d85a18'  # PLAN §4.1 audited snapshot
DEFAULT_MODEL_DIR = os.environ.get('COSYVOICE_MODEL_DIR', 'models/cosyvoice3')
INSTRUCT_PREFIX = 'You are a helpful assistant.<|endofprompt|>'  # required: llm.py:479 asserts 151646 in text
ENDOFPROMPT_ID = 151646
SPEECH_TOKEN_RATE_HZ = 25  # measured: 249 tokens / 9.96 s and 280 / 11.199 s (reports/cosyvoice3_audit.md)
MAX_POSITION_EMBEDDINGS = 32768  # CosyVoice-BlankEN/config.json
MAX_PROMPT_WAV_SEC = 30  # frontend.py:97 assert

STATUSES = ('complete', 'degraded', 'early_eos', 'loop_cap', 'hard_input_limit', 'context_limit',
            'oom', 'timeout', 'empty_or_invalid_audio', 'infrastructure_error')
STOP_REASONS = ('eos', 'max_len', 'watchdog', 'exception')


# --------------------------------------------------------------------------------------
# config / provenance helpers
# --------------------------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def repo_revision() -> str:
    try:
        rev = subprocess.check_output(['git', '-C', COSYVOICE_ROOT, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', COSYVOICE_ROOT, 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        return rev + ('+localdiff(reports/cosyvoice_local_diff.patch)' if dirty else '')
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
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        info['vram_free_gb'] = round(free_b / 2**30, 2)
        info['vram_total_gb'] = round(total_b / 2**30, 2)
    return info


# --------------------------------------------------------------------------------------
# model loading (one per process)
# --------------------------------------------------------------------------------------
_MODEL_CACHE: Dict[tuple, Any] = {}
_LAST_CONFIG_CHECK: Dict[str, Any] = {}

from cosyvoice3_config_check import ConfigDivergence, check_generation_config  # noqa: E402,F401


def _resolve_weights(model_dir: str, weights: str) -> str:
    return weights if os.path.isabs(weights) else os.path.join(model_dir, weights)


FLOW_MEL_HZ = 50  # cosyvoice3.yaml: token_frame_rate 25 x token_mel_ratio 2


def extend_flow_noise_buffer(cv, seconds: Optional[float], seed: int = 0):
    """Grow ``cv.model.flow.decoder.rand_noise`` to ``50 * seconds`` frames (prefix preserved)."""
    if not seconds:
        return
    dec = cv.model.flow.decoder
    cur = dec.rand_noise
    need = int(round(FLOW_MEL_HZ * float(seconds)))
    if need <= cur.shape[2]:
        return
    g = torch.Generator().manual_seed(seed)
    extra = torch.randn([cur.shape[0], cur.shape[1], need - cur.shape[2]], generator=g, dtype=cur.dtype)
    dec.rand_noise = torch.cat([cur, extra.to(cur.device)], dim=2)


HIFT_SR = 24000  # cosyvoice3.yaml sample_rate; SineGen2.sine_waves / SourceModuleHnNSF.uv are sized 300 * 24000 (generator.py:226, :356)


def extend_hift_noise_buffers(cv, seconds: Optional[float], seed: int = 0):
    """Grow the causal HiFT noise buffers to ``24000 * seconds`` samples (prefix preserved, uniform noise like stock).

    Stock: ``SineGen2.sine_waves = torch.rand(1, 300*24000, 9)`` (generator.py:226) and
    ``SourceModuleHnNSF.uv = torch.rand(1, 300*24000, 1)`` (:356); ``:310`` / ``:372`` slice them to the
    output length, so a > 300 s waveform raises ``RuntimeError: The size of tensor a (N) must match ... (7200000)``
    (measured on GPU 1, tmp/a3/fixpass_gpu1/memprobe2.log). Extended on the INSTANCE only; the repo is untouched.
    """
    if not seconds:
        return
    need = int(round(HIFT_SR * float(seconds)))
    src = cv.model.hift.m_source
    g = torch.Generator().manual_seed(seed)
    for owner, attr in ((src.l_sin_gen, 'sine_waves'), (src, 'uv')):
        cur = getattr(owner, attr, None)
        if cur is None or need <= cur.shape[1]:
            continue
        extra = torch.rand([cur.shape[0], need - cur.shape[1], cur.shape[2]], generator=g, dtype=cur.dtype)
        setattr(owner, attr, torch.cat([cur, extra.to(cur.device)], dim=1))


STOCK_NOISE_BUFFER_SEC = 300.0  # pristine snapshot: flow_matching.py:200, generator.py:226 / :356


def live_noise_buffers(cv) -> Dict[str, Optional[float]]:
    """MEASURED noise-buffer sizes of the loaded instance (not the configured ones).

    PLAN §10 + reports/decisions.md 2026-08-28 (Lead): every run manifest MUST record which buffer
    configuration produced the wav. The stock 300 s buffers are what makes B3/B4 impossible, and the
    native arm runs with them extended to 900 s, so a wav without this field is unattributable.
    """
    out: Dict[str, Optional[float]] = {'flow_noise_buffer_sec': None, 'hift_noise_buffer_sec': None}
    try:
        out['flow_noise_buffer_sec'] = round(cv.model.flow.decoder.rand_noise.shape[2] / FLOW_MEL_HZ, 3)
    except Exception:  # pragma: no cover - a repo change shows up as null, never as a wrong number
        pass
    try:
        sw = cv.model.hift.m_source.l_sin_gen.sine_waves.shape[1]
        uv = cv.model.hift.m_source.uv.shape[1]
        out['hift_noise_buffer_sec'] = round(min(sw, uv) / HIFT_SR, 3)  # the shorter buffer raises first
    except Exception:  # pragma: no cover
        pass
    return out


def record_noise_buffers(m: dict, cv) -> dict:
    """Fill the §10 buffer fields of manifest ``m`` from the LIVE objects + the config-check summary."""
    b = live_noise_buffers(cv)
    m.update(b)
    m['noise_buffers_stock_sec'] = STOCK_NOISE_BUFFER_SEC
    m['noise_buffers_extended'] = bool(max([v for v in b.values() if v is not None], default=0.0) > STOCK_NOISE_BUFFER_SEC)
    m['config_check'] = last_config_check_summary()
    return m


def set_status(m: dict, status: str, stop_reason: str) -> dict:
    """Write the PROVISIONAL generator status (reports/decisions.md 2026-08-28, Lead).

    ``gen_status`` is the generator's own label and depends only on stop_reason + audio validity.
    ``status`` is written as a copy of it (so every existing consumer keeps working), but the FINAL
    PLAN §3.4 status is assigned by ``scripts/run_evaluation.py``: complete / early_eos / degraded
    need ASR EndCoverage and WER-floor, which no adapter can know. The evaluator overwrites
    ``status`` in its own result rows; the manifest on disk is immutable (PLAN §10).
    """
    m['gen_status'] = status
    m['status'] = status
    m['status_source'] = ('generator (provisional, stop_reason-based); the final PLAN §3.4 status is assigned '
                          'by scripts/run_evaluation.py from EndCoverage / WER-floor')
    m['stop_reason'] = stop_reason
    return m


def last_config_check_summary() -> str:
    r = _LAST_CONFIG_CHECK
    if not r:
        return 'not run'
    return '{} values equal to repo defaults, {} mismatches, {} unreadable{}'.format(
        len(r.get('checked', [])), len(r.get('mismatches', [])), len(r.get('unreadable', [])),
        (' (' + '; '.join(r['unreadable']) + ')') if r.get('unreadable') else '')


def load_cosyvoice3(model_cfg: dict, gen_cfg: Optional[dict] = None, output_cfg: Optional[dict] = None):
    """Build the stock CosyVoice3 object once per process per (model_dir, weights) (fp16=False, no TRT/vLLM).

    When ``gen_cfg`` is given, the frozen yaml values are asserted against the repo's effective
    defaults read from the loaded objects (``cosyvoice3_config_check``); divergence raises
    ``ConfigDivergence`` so a silently outdated config cannot produce numbers.
    """
    model_dir = model_cfg.get('model_dir', DEFAULT_MODEL_DIR)
    weights = model_cfg.get('weights', 'llm.pt')
    key = (os.path.abspath(model_dir), os.path.abspath(_resolve_weights(model_dir, weights)))
    if key in _MODEL_CACHE:
        cv = _MODEL_CACHE[key]
    else:
        from cosyvoice.cli.cosyvoice import CosyVoice3
        fp16 = bool(model_cfg.get('fp16', False))
        assert fp16 is False, 'PLAN §4 / configs/models/cosyvoice3_base.yaml freeze fp16=false'
        cv = CosyVoice3(model_dir, load_trt=False, load_vllm=False, fp16=False)
        if weights != 'llm.pt':
            # e.g. llm.rl.pt (E5) or an SFT checkpoint (E2/E3): strict load into the same module
            path = _resolve_weights(model_dir, weights)
            state = torch.load(path, map_location=cv.model.device, weights_only=True)
            state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
            cv.model.llm.load_state_dict(state, strict=True)
            cv.model.llm.to(cv.model.device).eval()
        # Freeze the text frontend backend explicitly (see audit §2): on this machine wetext's
        # modelscope download fails so the repo falls back to text_frontend=''; pin it so the
        # official path never silently switches to wetext English TN when the network is up.
        cv.frontend.text_frontend = model_cfg.get('text_frontend_backend', '')
        # Non-stream flow renders at most rand_noise.shape[2] mel frames = 50 Hz x 300 s in the stock
        # snapshot (flow_matching.py:200, sliced at :222 -> mask_in[:] = mask fails at :104 for longer
        # outputs). Extend the buffer on the INSTANCE (repo untouched) by appending seeded noise, so
        # every output <= 300 s stays bit-identical to stock and longer ones become possible.
        extend_flow_noise_buffer(cv, model_cfg.get('flow_noise_buffer_sec'))
        extend_hift_noise_buffers(cv, model_cfg.get('hift_noise_buffer_sec'))
        cv._a3_weights_path = key[1]
        _MODEL_CACHE[key] = cv
    if gen_cfg is not None:
        _LAST_CONFIG_CHECK.clear()
        _LAST_CONFIG_CHECK.update(check_generation_config({'model': model_cfg, 'generation': gen_cfg, 'output': output_cfg or {}}, cv))
    return cv


# --------------------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------------------
@dataclass
class LLMTrace:
    """Per-trajectory record filled by the wrapped ``llm.inference`` generator."""
    n_calls: int = 0
    n_yielded: int = 0
    max_len: int = -1
    min_len: int = -1
    stop_reason: Optional[str] = None      # eos | max_len | watchdog | exception
    watchdog_kind: Optional[str] = None    # tokens | seconds | context
    llm_seconds: float = 0.0
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None
    text_tokens_seen: int = -1
    prompt_text_tokens_seen: int = -1
    prompt_speech_tokens_seen: int = -1
    tokens: List[int] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)  # one record per llm.inference call


class Watchdog:
    def __init__(self, max_tokens: Optional[int], max_seconds: Optional[float], context_budget_tokens: Optional[int]):
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.context_budget_tokens = context_budget_tokens
        self.t0 = None

    def check(self, n_tokens: int) -> Optional[str]:
        if self.context_budget_tokens is not None and n_tokens >= self.context_budget_tokens:
            return 'context'
        if self.max_tokens is not None and n_tokens >= self.max_tokens:
            return 'tokens'
        if self.max_seconds is not None and self.t0 is not None and (time.time() - self.t0) >= self.max_seconds:
            return 'seconds'
        return None


def wrap_llm_inference(llm, trace: LLMTrace, watchdog: Optional[Watchdog], keep_tokens: bool = True):
    """Return a replacement for ``llm.inference`` that records the trajectory.

    The replacement calls the ORIGINAL bound method, so sampling, max_len, min_len and EOS
    handling are exactly the repo's. It only observes the yielded tokens and stops early on
    the watchdog. The caller must restore the original attribute afterwards.
    """
    orig = llm.inference
    ratio_max = 20.0
    ratio_min = 2.0

    def instrumented(text, text_len, prompt_text, prompt_text_len, prompt_speech_token, prompt_speech_token_len,
                     embedding, sampling=25, max_token_text_ratio=ratio_max, min_token_text_ratio=ratio_min, uuid=''):
        trace.n_calls += 1
        n_text = int(text_len.item())
        trace.text_tokens_seen = n_text
        trace.prompt_text_tokens_seen = int(prompt_text_len.item())
        trace.prompt_speech_tokens_seen = int(prompt_speech_token_len.item())
        # identical formulas to llm.py:497-498 (text_len there is tts text only after subtraction)
        trace.max_len = int(n_text * max_token_text_ratio)
        trace.min_len = int(n_text * min_token_text_ratio)
        gen = orig(text, text_len, prompt_text, prompt_text_len, prompt_speech_token, prompt_speech_token_len,
                   embedding, sampling=sampling, max_token_text_ratio=max_token_text_ratio,
                   min_token_text_ratio=min_token_text_ratio, uuid=uuid)
        t0 = time.time()
        if watchdog is not None and watchdog.t0 is None:
            watchdog.t0 = t0  # wall clock starts at the FIRST llm call and spans all chunks
        n = 0
        try:
            for tok in gen:
                n += 1
                if keep_tokens:
                    trace.tokens.append(int(tok))
                yield tok
                if watchdog is not None:
                    kind = watchdog.check(trace.n_yielded + n)  # cumulative over calls (official mode has >1)
                    if kind is not None:
                        trace.stop_reason = 'watchdog'
                        trace.watchdog_kind = kind
                        try:
                            gen.close()
                        except Exception:
                            pass
                        break
            else:
                # generator exhausted by the repo itself: EOS (break) or loop cap (range end)
                trace.stop_reason = 'max_len' if n >= trace.max_len else 'eos'
        except BaseException as e:  # noqa: BLE001 - we must classify OOM etc.
            trace.stop_reason = 'exception'
            trace.exception_type = type(e).__name__
            trace.exception_message = str(e)[:2000]
            trace.exception_traceback = traceback.format_exc()[-4000:]
            # do not re-raise: the repo's llm_job thread would swallow it anyway; we surface it via the trace
        finally:
            trace.n_yielded += n
            trace.llm_seconds += time.time() - t0
            trace.calls.append({'n_yielded': n, 'text_tokens': n_text, 'max_len': trace.max_len, 'min_len': trace.min_len,
                                'stop_reason': trace.stop_reason, 'watchdog_kind': trace.watchdog_kind,
                                'seconds': round(time.time() - t0, 3)})

    return instrumented


class CallCounter:
    """Wrap a bound method and count invocations (used for frontend_zero_shot / tts)."""

    def __init__(self, obj, name):
        self.obj, self.name = obj, name
        self.orig = getattr(obj, name)
        self.calls: List[Dict[str, Any]] = []

    def __enter__(self):
        def wrapped(*a, **k):
            self.calls.append({'args': [repr(x)[:80] for x in a], 'kwargs': list(k.keys())})
            return self.orig(*a, **k)
        setattr(self.obj, self.name, wrapped)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self.orig)
        return False

    @property
    def n(self):
        return len(self.calls)


# --------------------------------------------------------------------------------------
# audio + manifest helpers
# --------------------------------------------------------------------------------------
def audio_stats(wav: np.ndarray, sr: int) -> dict:
    if wav is None or wav.size == 0:
        return {'raw_duration_sec': 0.0, 'n_samples': 0, 'rms': 0.0, 'peak': 0.0, 'has_nan': False, 'valid': False}
    finite = np.isfinite(wav).all()
    peak = float(np.abs(wav).max()) if finite else float('nan')
    rms = float(np.sqrt(np.mean(np.square(wav)))) if finite else float('nan')
    return {'raw_duration_sec': round(wav.shape[-1] / sr, 4), 'n_samples': int(wav.shape[-1]), 'rms': rms, 'peak': peak,
            'has_nan': bool(not finite), 'valid': bool(finite and wav.shape[-1] > 0 and peak > 1e-4)}


def write_wav_no_overwrite(out_path: str, wav: np.ndarray, sr: int):
    if os.path.exists(out_path):
        raise FileExistsError('refusing to overwrite existing output {}'.format(out_path))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, wav.astype(np.float32), sr, subtype='PCM_16')


def write_json_no_overwrite(path: str, obj: dict):
    if os.path.exists(path):
        raise FileExistsError('refusing to overwrite existing manifest {}'.format(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def expected_speech_tokens(text_tokens: int, human_duration_sec: Optional[float], tokens_per_text_token: float) -> int:
    if human_duration_sec is not None and human_duration_sec > 0:
        return int(math.ceil(human_duration_sec * SPEECH_TOKEN_RATE_HZ))
    return int(math.ceil(text_tokens * tokens_per_text_token))


def build_watchdog(gen_cfg: dict, text_tokens: int, input_context_tokens: int, human_duration_sec: Optional[float]) -> (Optional[Watchdog], dict):
    wd = gen_cfg.get('watchdog', {}) or {}
    if not wd.get('enabled', True):
        return None, {'enabled': False}
    exp = expected_speech_tokens(text_tokens, human_duration_sec, float(wd.get('expected_tokens_per_text_token', 5.0)))
    exp_basis = '25 Hz x human_duration_sec' if (human_duration_sec is not None and human_duration_sec > 0) \
        else '{} x text_tokens'.format(wd.get('expected_tokens_per_text_token', 5.0))
    factor = wd.get('expected_factor', None)
    abs_cap = wd.get('max_generated_tokens', None)
    limits = [x for x in (abs_cap, int(math.ceil(exp * factor)) if factor else None) if x]
    max_tokens = min(limits) if limits else None
    max_seconds = wd.get('max_wall_seconds', None)
    margin = int(wd.get('context_safety_margin_tokens', 0))
    ctx_budget = MAX_POSITION_EMBEDDINGS - margin - input_context_tokens if wd.get('enforce_context_budget', True) else None
    if ctx_budget is not None and ctx_budget < 0:
        ctx_budget = 0
    return Watchdog(max_tokens, max_seconds, ctx_budget), {
        'enabled': True, 'expected_speech_tokens': exp, 'expected_basis': exp_basis, 'expected_factor': factor, 'max_generated_tokens': max_tokens,
        'max_wall_seconds': max_seconds, 'context_budget_tokens': ctx_budget, 'context_safety_margin_tokens': margin}


def classify(trace: LLMTrace, wd_info: dict, audio: dict, pre_error: Optional[dict] = None,
             context_tokens_total: int = 0) -> (str, str):
    """Map instrumentation to the §3.4 status and §10 stop_reason.

    'complete' here means "stopped by EOS with valid audio"; the evaluator (A4) refines it to
    early_eos / degraded using ASR coverage and duration ratio — the adapter cannot know.
    """
    if pre_error is not None:
        return pre_error['status'], 'exception'
    if trace.stop_reason == 'exception':
        et = (trace.exception_type or '').lower()
        em = (trace.exception_message or '').lower()
        if 'outofmemory' in et or 'out of memory' in em or 'cuda out of memory' in em:
            return 'oom', 'exception'
        return 'infrastructure_error', 'exception'
    if trace.stop_reason == 'watchdog':
        if trace.watchdog_kind == 'seconds':
            return 'timeout', 'watchdog'
        if trace.watchdog_kind == 'context':
            return 'context_limit', 'watchdog'
        return 'loop_cap', 'watchdog'
    if trace.stop_reason == 'max_len':
        return 'loop_cap', 'max_len'
    if trace.stop_reason == 'eos':
        if not audio.get('valid', False):
            return 'empty_or_invalid_audio', 'eos'
        if context_tokens_total > MAX_POSITION_EMBEDDINGS:
            return 'context_limit', 'eos'
        return 'complete', 'eos'
    if trace.n_calls == 0:
        return 'infrastructure_error', 'exception'
    return 'infrastructure_error', 'exception'


def base_manifest(run_id, experiment_id, mode, text_id, voice_id, ref_wav_path, seed, gen_cfg, model_cfg, text, out_path) -> dict:
    return {
        'run_id': run_id,
        'experiment_id': experiment_id,
        'model_id': MODEL_ID,
        'model_revision': MODEL_REVISION,
        'repo_revision': repo_revision(),
        'weights': model_cfg.get('weights', 'llm.pt'),
        'mode': mode,
        'text_id': text_id,
        'voice_id': voice_id,
        'reference_audio_path': ref_wav_path,
        'reference_audio_sha256': sha256_file(ref_wav_path) if ref_wav_path and os.path.exists(ref_wav_path) else None,
        'seed': seed,
        'generation_config': gen_cfg,
        'text_chars': len(text),
        'text_words': len(text.split()),
        'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),  # detects a silent benchmark rebuild
        'text_tokens': 0,
        'prompt_text_tokens': 0,
        'prompt_speech_tokens': 0,
        'generated_speech_tokens': 0,
        'speech_tokens_to_flow': 0,
        'context_tokens_total': 0,
        'context_tokens_input': 0,
        'max_position_embeddings': MAX_POSITION_EMBEDDINGS,
        'max_len_cap': 0,
        'min_len': 0,
        # noise buffers: measured from the loaded instance by record_noise_buffers() (PLAN §10, decisions.md 2026-08-28)
        'flow_noise_buffer_sec': None,
        'hift_noise_buffer_sec': None,
        'noise_buffers_stock_sec': STOCK_NOISE_BUFFER_SEC,
        'noise_buffers_extended': None,
        'config_check': None,
        'gen_status': None,     # provisional, written by set_status()
        'status': None,         # copy of gen_status; the EVALUATOR assigns the final PLAN §3.4 status
        'status_source': None,
        'stop_reason': None,
        'watchdog': None,
        'output_path': out_path,
        'raw_duration_sec': 0.0,
        'voiced_duration_sec': None,  # filled by A4/A5 evaluators (VAD), not by the adapter
        'wall_time_sec': 0.0,
        'llm_time_sec': 0.0,
        'flow_hift_time_sec': 0.0,
        'frontend_time_sec': 0.0,
        'rtf': None,
        'speech_tokens_per_sec_wall': None,
        'peak_vram_bytes': 0,
        'peak_vram_reserved_bytes': 0,
        'exception_type': None,
        'exception_message': None,
        'exception_traceback': None,
        'audio': None,
        'env': {'torch': torch.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'python': sys.version.split()[0]},
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


def set_seed(seed: int):
    from cosyvoice.utils.common import set_all_random_seed
    set_all_random_seed(int(seed))


def tokens_to_flow_after_silence_filter(tokens: List[int], silent_tokens: List[int], max_silent: int = 5) -> int:
    """Reproduce model.py:121-128 to report how many LM tokens reach flow/hift."""
    cur, kept = 0, 0
    for t in tokens:
        if t in silent_tokens:
            cur += 1
            if cur > max_silent:
                continue
        else:
            cur = 0
        kept += 1
    return kept
