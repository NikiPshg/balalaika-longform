"""Assert that configs/models/cosyvoice3_base.yaml equals the repo's EFFECTIVE defaults (A3, torch-free).

The frozen yaml is documentation of what the stock CosyVoice3 does; nothing reads the sampling
values from it at inference time (the repo uses its own defaults). So if the repo or the model
snapshot ever changes, the yaml would silently lie. ``check_generation_config`` reads the effective
values from the loaded ``CosyVoice3`` object (``cv``) and raises ``ConfigDivergence`` listing every
mismatch. It is called once at load time by ``cosyvoice3_common.load_cosyvoice3``.

Kept free of torch imports so it can be unit-tested with a stub object in the eval venv.
"""
from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Optional


class ConfigDivergence(AssertionError):
    pass


def _get(obj, path: str, default=None):
    cur = obj
    for part in path.split('.'):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return default if cur is None else cur


def _partial_kw(fn, name: str):
    """Keyword bound in a functools.partial (hyperpyyaml ``!name:`` produces partials)."""
    kw = getattr(fn, 'keywords', None) or {}
    return kw.get(name)


def _qualname(fn) -> Optional[str]:
    f = getattr(fn, 'func', fn)
    mod = getattr(f, '__module__', None)
    name = getattr(f, '__name__', None)
    return '{}.{}'.format(mod, name) if mod and name else None


def _source(fn) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return ''


def _sig_default(fn, param: str):
    try:
        p = inspect.signature(fn).parameters[param]
        return p.default
    except (KeyError, TypeError, ValueError):
        return None


def effective_values(cv) -> Dict[str, Any]:
    """Read the effective generation defaults from a loaded CosyVoice3 object (or a stub with the same attributes)."""
    llm = _get(cv, 'model.llm')
    sampling_fn = getattr(llm, 'sampling', None)
    eff: Dict[str, Any] = {
        'sampling_fn': _qualname(sampling_fn),
        'top_p': _partial_kw(sampling_fn, 'top_p'),
        'top_k': _partial_kw(sampling_fn, 'top_k'),
        'win_size': _partial_kw(sampling_fn, 'win_size'),
        'tau_r': _partial_kw(sampling_fn, 'tau_r'),
        'sampling': _sig_default(getattr(llm, 'inference', None), 'sampling'),
        'max_token_text_ratio': _sig_default(getattr(llm, 'inference', None), 'max_token_text_ratio'),
        'min_token_text_ratio': _sig_default(getattr(llm, 'inference', None), 'min_token_text_ratio'),
        'flow_inference_cfg_rate': _get(cv, 'model.flow.decoder.inference_cfg_rate'),
        'speech_token_size': getattr(llm, 'speech_token_size', None),
        'fp16': _get(cv, 'model.fp16'),
        'text_frontend_backend': _get(cv, 'frontend.text_frontend'),
        'sample_rate': getattr(cv, 'sample_rate', None),
        'max_position_embeddings': _get(cv, 'model.llm.llm.model.config.max_position_embeddings'),
    }
    shape = getattr(_get(cv, 'model.flow.decoder.rand_noise'), 'shape', None)
    eff['flow_noise_buffer_sec'] = (shape[2] / 50.0) if shape is not None and len(shape) == 3 else None
    sw = getattr(_get(cv, 'model.hift.m_source.l_sin_gen.sine_waves'), 'shape', None)
    uv = getattr(_get(cv, 'model.hift.m_source.uv'), 'shape', None)
    if sw is not None and uv is not None and len(sw) == 3 and len(uv) == 3:
        # both HiFT buffers must agree; report the shorter one (the one that would raise first)
        eff['hift_noise_buffer_sec'] = min(sw[1], uv[1]) / 24000.0
    else:
        eff['hift_noise_buffer_sec'] = None
    m = re.search(r'n_timesteps\s*=\s*(\d+)', _source(_get(cv, 'model.flow.inference')))
    eff['flow_n_timesteps'] = int(m.group(1)) if m else None
    m = re.search(r'max_silent_token_num\s*=\s*\d+\s*,\s*(\d+)|max_silent_token_num\s*=\s*(\d+)', _source(_get(cv, 'model.llm_job')))
    eff['silent_token_filter'] = int(m.group(1) or m.group(2)) if m else None
    return eff


def check_generation_config(cfg: dict, cv, raise_on_divergence: bool = True) -> Dict[str, Any]:
    """Compare yaml ``model``/``generation``/``output`` sections with ``effective_values(cv)``.

    Returns {'checked': [...], 'mismatches': [...], 'unreadable': [...]}; raises ConfigDivergence on mismatch.
    A value that cannot be read from the object (None) is reported as 'unreadable', not as a mismatch —
    except the ones every CosyVoice3 object must expose (sampling_fn, top_p, ratios, speech_token_size).
    """
    gen = cfg.get('generation') or {}
    mdl = cfg.get('model') or {}
    out = cfg.get('output') or {}
    expected = {
        'sampling_fn': gen.get('sampling_fn'), 'top_p': gen.get('top_p'), 'top_k': gen.get('top_k'),
        'win_size': gen.get('win_size'), 'tau_r': gen.get('tau_r'), 'sampling': gen.get('sampling'),
        'max_token_text_ratio': gen.get('max_token_text_ratio'), 'min_token_text_ratio': gen.get('min_token_text_ratio'),
        'flow_n_timesteps': gen.get('flow_n_timesteps'), 'flow_inference_cfg_rate': gen.get('flow_inference_cfg_rate'),
        'silent_token_filter': gen.get('silent_token_filter'),
        'speech_token_size': mdl.get('speech_token_size'), 'fp16': mdl.get('fp16'),
        'text_frontend_backend': mdl.get('text_frontend_backend'),
        'max_position_embeddings': mdl.get('max_position_embeddings'),
        'flow_noise_buffer_sec': mdl.get('flow_noise_buffer_sec'),
        'hift_noise_buffer_sec': mdl.get('hift_noise_buffer_sec'),
        'sample_rate': out.get('sample_rate'),
    }
    must_read = ('sampling_fn', 'top_p', 'top_k', 'win_size', 'tau_r', 'max_token_text_ratio', 'min_token_text_ratio', 'speech_token_size')
    eff = effective_values(cv)
    checked: List[str] = []
    mismatches: List[str] = []
    unreadable: List[str] = []
    for k, want in expected.items():
        if want is None:
            continue  # not frozen in the yaml -> nothing to assert
        got = eff.get(k)
        if got is None:
            (mismatches if k in must_read else unreadable).append('{}: yaml={!r} effective=<unreadable>'.format(k, want))
            continue
        same = (abs(float(got) - float(want)) < 1e-9) if isinstance(want, (int, float)) and not isinstance(want, bool) \
            and isinstance(got, (int, float)) else (got == want)
        (checked if same else mismatches).append('{}: yaml={!r} effective={!r}'.format(k, want, got))
    res = {'checked': checked, 'mismatches': mismatches, 'unreadable': unreadable}
    if mismatches and raise_on_divergence:
        raise ConfigDivergence('configs/models/cosyvoice3_base.yaml diverges from the repo effective defaults:\n  ' + '\n  '.join(mismatches))
    return res
