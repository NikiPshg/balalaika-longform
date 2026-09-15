"""E0 production control: the STOCK ``CosyVoice3.inference_zero_shot`` path (NOT native).

What the stock path does (cosyvoice/cli/cosyvoice.py:91-103):
  * prompt_text -> text_normalize(split=False, text_frontend=True)   (cosyvoice.py:92)
  * tts_text    -> text_normalize(split=True,  text_frontend=True)   (cosyvoice.py:93)
        -> non-Chinese branch (frontend.py:153-158): spell_out_number (English inflect!) +
           split_paragraph(lang="en", token_max_n=80, token_min_n=60, merge_len=20)
  * for every chunk: frontend_zero_shot(...) + model.tts(...)         (cosyvoice.py:96-102)
        => N independent conditioning contexts and N independent LM trajectories,
           N wavs yielded and concatenated here with plain np.concatenate (no crossfade).
Every chunk re-uses the same prompt audio; the KV cache is reset between chunks. This is the
production control, explicitly labelled non-native (PLAN §5: E0 cannot prove native ability).
"""
from __future__ import annotations

import os
import sys
import time
import uuid as _uuid
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosyvoice3_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

MODE = 'official_split'


def synthesize(text: str, ref_wav_path: str, ref_text: str, seed: int, gen_cfg: dict, out_path: str,
               watchdog: Optional[dict] = None, *, model_cfg: Optional[dict] = None, text_id: str = '',
               voice_id: str = '', experiment_id: str = 'E0', run_id: Optional[str] = None,
               human_duration_sec: Optional[float] = None, cv=None, keep_tokens: bool = True) -> Dict[str, Any]:
    model_cfg = model_cfg or {}
    gen_cfg = dict(gen_cfg or {})
    if watchdog is not None:
        gen_cfg['watchdog'] = {**(gen_cfg.get('watchdog') or {}), **watchdog}
    run_id = run_id or '{}__{}__s{}__{}'.format(text_id or 'text', voice_id or 'voice', seed, _uuid.uuid4().hex[:8])
    manifest_path = os.path.splitext(out_path)[0] + '.json'
    if os.path.exists(out_path) or os.path.exists(manifest_path):
        raise FileExistsError('output exists, refusing to overwrite: {} / {}'.format(out_path, manifest_path))

    m = C.base_manifest(run_id, experiment_id, MODE, text_id, voice_id, ref_wav_path, seed, gen_cfg, model_cfg, text, out_path)
    m['native'] = False
    m['native_contract'] = {'frontend_zero_shot_calls': 0, 'tts_calls': 0, 'llm_inference_calls': 0, 'tts_chunks_yielded': 0}
    m['mem_before'] = C.free_mem_info()
    t_wall0 = time.time()
    trace = C.LLMTrace()
    wav = None
    pre_error = None
    sr = 24000
    chunk_records = []

    try:
        cv = cv or C.load_cosyvoice3(model_cfg, gen_cfg=gen_cfg)
        sr = cv.sample_rate
        C.record_noise_buffers(m, cv)   # MEASURED flow/hift noise buffers of this instance (PLAN §10)
        import soundfile as sf
        info = sf.info(ref_wav_path)
        m['reference_audio_duration_sec'] = round(info.duration, 3)
        if info.duration > C.MAX_PROMPT_WAV_SEC:
            pre_error = {'status': 'hard_input_limit', 'message': 'prompt audio {:.2f}s > {}s (frontend.py:97)'.format(info.duration, C.MAX_PROMPT_WAV_SEC)}
            raise RuntimeError(pre_error['message'])

        C.set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        prompt_text = C.INSTRUCT_PREFIX + ref_text
        text_frontend = bool(gen_cfg.get('official_text_frontend', True))  # stock default True
        # report what the stock normaliser/splitter does to the text (same calls as cosyvoice.py:92-93)
        norm_prompt = cv.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        chunks_text = cv.frontend.text_normalize(text, split=True, text_frontend=text_frontend)
        m['official_split'] = {'text_frontend': text_frontend, 'text_frontend_backend': cv.frontend.text_frontend,
                               'n_chunks': len(chunks_text), 'prompt_text_changed': norm_prompt != prompt_text,
                               'joined_text_changed': ''.join(chunks_text).replace(' ', '') != text.replace(' ', ''),
                               'chunk_chars': [len(c) for c in chunks_text]}
        tk = cv.frontend.tokenizer
        m['text_tokens'] = sum(len(tk.encode(c, allowed_special=cv.frontend.allowed_special)) for c in chunks_text)
        m['prompt_text_tokens'] = len(tk.encode(norm_prompt, allowed_special=cv.frontend.allowed_special))
        m['max_len_cap'] = int(m['text_tokens'] * float(gen_cfg.get('max_token_text_ratio', 20)))  # sum over chunks
        m['min_len'] = int(m['text_tokens'] * float(gen_cfg.get('min_token_text_ratio', 2)))
        # per-chunk contexts are independent => no global context budget; only the token/time watchdog
        gen_cfg_wd = dict(gen_cfg.get('watchdog') or {})
        gen_cfg_wd['enforce_context_budget'] = False
        wd, wd_info = C.build_watchdog({**gen_cfg, 'watchdog': gen_cfg_wd}, m['text_tokens'], 0, human_duration_sec)
        m['watchdog'] = wd_info

        llm = cv.model.llm
        orig_inf = llm.inference
        llm.inference = C.wrap_llm_inference(llm, trace, wd, keep_tokens=keep_tokens)
        try:
            with C.CallCounter(cv.frontend, 'frontend_zero_shot') as fe_counter, C.CallCounter(cv.model, 'tts') as tts_counter:
                pieces = []
                t_tts0 = time.time()
                gen = cv.inference_zero_shot(text, prompt_text, ref_wav_path, zero_shot_spk_id='', stream=False,
                                             speed=float(gen_cfg.get('speed', 1.0)), text_frontend=text_frontend)
                for out in gen:
                    pieces.append(out['tts_speech'].squeeze(0).cpu().numpy())
                    last = trace.calls[-1] if trace.calls else {}
                    chunk_records.append({'chunk_index': len(pieces) - 1, 'samples': int(pieces[-1].shape[-1]),
                                          'duration_sec': round(pieces[-1].shape[-1] / sr, 3), **last})
                    if trace.stop_reason in ('watchdog', 'exception'):
                        gen.close()  # skip the remaining chunks: partial output is kept
                        break
                m['tts_time_sec'] = round(time.time() - t_tts0, 3)
            m['native_contract']['frontend_zero_shot_calls'] = fe_counter.n
            m['native_contract']['tts_calls'] = tts_counter.n
            m['native_contract']['tts_chunks_yielded'] = len(pieces)
            wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        finally:
            llm.inference = orig_inf
        # prompt speech tokens are re-extracted per chunk; record the per-chunk value
        m['prompt_speech_tokens'] = trace.prompt_speech_tokens_seen if trace.prompt_speech_tokens_seen >= 0 else 0
        m['context_tokens_input'] = 1 + m['prompt_text_tokens'] + m['text_tokens'] + 1 + m['prompt_speech_tokens']  # nominal sum
    except torch.cuda.OutOfMemoryError as e:
        if pre_error is None:
            pre_error = {'status': 'oom', 'message': str(e)[:2000]}
        m['exception_type'], m['exception_message'] = type(e).__name__, str(e)[:2000]
        import traceback
        m['exception_traceback'] = traceback.format_exc()[-4000:]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        if pre_error is None:
            pre_error = {'status': 'infrastructure_error', 'message': str(e)[:2000]}
        m['exception_type'], m['exception_message'] = type(e).__name__, str(e)[:2000]
        import traceback
        m['exception_traceback'] = traceback.format_exc()[-4000:]

    m['wall_time_sec'] = round(time.time() - t_wall0, 3)
    m['llm_time_sec'] = round(trace.llm_seconds, 3)
    m['flow_hift_time_sec'] = round(max(0.0, m.get('tts_time_sec', 0.0) - trace.llm_seconds), 3) if 'tts_time_sec' in m else None
    m['native_contract']['llm_inference_calls'] = trace.n_calls
    m['generated_speech_tokens'] = trace.n_yielded
    m['context_tokens_total'] = m['context_tokens_input'] + trace.n_yielded  # NOT one context: sum over chunks
    m['max_chunk_context_tokens'] = max([1 + m['prompt_text_tokens'] + c['text_tokens'] + 1 + m['prompt_speech_tokens'] + c['n_yielded'] for c in trace.calls], default=0)
    m['chunks'] = chunk_records
    if trace.tokens and cv is not None:
        m['speech_tokens_to_flow'] = C.tokens_to_flow_after_silence_filter(trace.tokens, cv.model.silent_tokens)
        if keep_tokens:
            m['generated_token_ids_path'] = os.path.splitext(out_path)[0] + '.tokens.json'
    if trace.exception_type and not m['exception_type']:
        m['exception_type'], m['exception_message'], m['exception_traceback'] = trace.exception_type, trace.exception_message, trace.exception_traceback
    if torch.cuda.is_available():
        m['peak_vram_bytes'] = int(torch.cuda.max_memory_allocated())
        m['peak_vram_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
    audio = C.audio_stats(wav, sr)
    m['audio'] = audio
    m['raw_duration_sec'] = audio['raw_duration_sec']
    if audio['raw_duration_sec'] > 0:
        m['rtf'] = round(m['wall_time_sec'] / audio['raw_duration_sec'], 4)
    if trace.llm_seconds > 0:
        m['speech_tokens_per_sec_wall'] = round(trace.n_yielded / trace.llm_seconds, 2)
    # a chunk that hit its own max_len makes the whole official run loop_cap (documented in audit)
    any_max_len = any(c.get('stop_reason') == 'max_len' for c in trace.calls)
    if trace.stop_reason == 'eos' and any_max_len:
        trace.stop_reason = 'max_len'
    status, stop_reason = C.classify(trace, m['watchdog'] or {}, audio, pre_error, 0)
    C.set_status(m, status, stop_reason)   # gen_status = provisional; the evaluator assigns the final §3.4 status
    # provisional length hint (NOT a status): A4 decides early_eos/degraded from ASR coverage + duration ratio
    exp_tok = (m['watchdog'] or {}).get('expected_speech_tokens')
    if exp_tok:
        ratio = round(trace.n_yielded / exp_tok, 4)
        m['generated_to_expected_token_ratio'] = ratio
        m['length_hint'] = 'short' if ratio < 0.5 else ('long' if ratio > 2.0 else 'plausible')
    m['mem_after'] = C.free_mem_info()

    if wav is not None and wav.size > 0:
        C.write_wav_no_overwrite(out_path, wav, sr)
    else:
        m['output_path'] = None
    if keep_tokens and trace.tokens:
        C.write_json_no_overwrite(m['generated_token_ids_path'], {'run_id': run_id, 'tokens': trace.tokens, 'llm_calls': trace.calls})
    m['llm_calls'] = trace.calls
    C.write_json_no_overwrite(manifest_path, m)
    m['manifest_path'] = manifest_path
    return m
