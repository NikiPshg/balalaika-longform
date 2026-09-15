"""E1/E2/E3 native adapter: one text -> one conditioning context -> one LM trajectory.

Path (all repo functions, nothing in third_party/CosyVoice is modified):

    frontend.frontend_zero_shot(text, INSTRUCT + ref_text, ref_wav, 24000, '')   # ONE call, no text_normalize
    model.tts(**model_input, stream=False, speed=1.0)                            # ONE call -> ONE llm.inference
        -> llm_job thread: llm.inference(...) once (model.py:113-128)
        -> p.join(); token2wav(all tokens, finalize=True) once (model.py:376-387)

The text is passed to the tokenizer verbatim: ``text_normalize`` is NOT called, which is
byte-identical to the repo's own ``text_normalize(text, split=False, text_frontend=False)``
(frontend.py:134-135 returns the input unchanged). Russian normalisation is therefore A2's
frozen canonical spoken text, decoupled from the repo's zh/en splitter.
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

MODE = 'native'


def synthesize(text: str, ref_wav_path: str, ref_text: str, seed: int, gen_cfg: dict, out_path: str,
               watchdog: Optional[dict] = None, *, model_cfg: Optional[dict] = None, text_id: str = '',
               voice_id: str = '', experiment_id: str = 'E1', run_id: Optional[str] = None,
               human_duration_sec: Optional[float] = None, cv=None, keep_tokens: bool = True) -> Dict[str, Any]:
    """Run ONE native long-form synthesis and return the §10 run manifest (also written next to the wav).

    ``watchdog`` (dict) overrides ``gen_cfg['watchdog']``. Output is never overwritten.
    """
    model_cfg = model_cfg or {}
    gen_cfg = dict(gen_cfg or {})
    if watchdog is not None:
        gen_cfg['watchdog'] = {**(gen_cfg.get('watchdog') or {}), **watchdog}
    run_id = run_id or '{}__{}__s{}__{}'.format(text_id or 'text', voice_id or 'voice', seed, _uuid.uuid4().hex[:8])
    manifest_path = os.path.splitext(out_path)[0] + '.json'
    if os.path.exists(out_path) or os.path.exists(manifest_path):
        raise FileExistsError('output exists, refusing to overwrite: {} / {}'.format(out_path, manifest_path))

    m = C.base_manifest(run_id, experiment_id, MODE, text_id, voice_id, ref_wav_path, seed, gen_cfg, model_cfg, text, out_path)
    m['native_contract'] = {'frontend_zero_shot_calls': 0, 'tts_calls': 0, 'llm_inference_calls': 0, 'tts_chunks_yielded': 0}
    m['mem_before'] = C.free_mem_info()
    t_wall0 = time.time()
    trace = C.LLMTrace()
    wav = None
    pre_error = None
    sr = 24000

    try:
        cv = cv or C.load_cosyvoice3(model_cfg, gen_cfg=gen_cfg)
        sr = cv.sample_rate
        C.record_noise_buffers(m, cv)   # MEASURED flow/hift noise buffers of this instance (PLAN §10)
        # ---- hard input limits (frontend.py:97: prompt wav must be <= 30 s at 16 kHz) ----
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

        prompt_text = C.INSTRUCT_PREFIX + ref_text  # NO text_normalize (see module docstring)
        # ---- ONE frontend call -----------------------------------------------------------
        t_fe0 = time.time()
        with C.CallCounter(cv.frontend, 'frontend_zero_shot') as fe_counter:
            model_input = cv.frontend.frontend_zero_shot(text, prompt_text, ref_wav_path, sr, '')
        m['frontend_time_sec'] = round(time.time() - t_fe0, 3)
        m['native_contract']['frontend_zero_shot_calls'] = fe_counter.n
        m['text_tokens'] = int(model_input['text_len'].item())
        m['prompt_text_tokens'] = int(model_input['prompt_text_len'].item())
        m['prompt_speech_tokens'] = int(model_input['llm_prompt_speech_token_len'].item())
        m['prompt_mel_frames'] = int(model_input['prompt_speech_feat_len'].item())
        assert C.ENDOFPROMPT_ID in model_input['prompt_text'].flatten().tolist(), '<|endofprompt|> missing from prompt_text'
        # llm.py:494: lm_input = [sos] + emb(prompt_text + text) + [task_id] + prompt_speech
        ctx_in = 1 + m['prompt_text_tokens'] + m['text_tokens'] + 1 + m['prompt_speech_tokens']
        m['context_tokens_input'] = ctx_in
        m['max_len_cap'] = int(m['text_tokens'] * float(gen_cfg.get('max_token_text_ratio', 20)))
        m['min_len'] = int(m['text_tokens'] * float(gen_cfg.get('min_token_text_ratio', 2)))
        if ctx_in >= C.MAX_POSITION_EMBEDDINGS:
            pre_error = {'status': 'context_limit', 'message': 'input context {} >= max_position_embeddings {}'.format(ctx_in, C.MAX_POSITION_EMBEDDINGS)}
            raise RuntimeError(pre_error['message'])

        wd, wd_info = C.build_watchdog(gen_cfg, m['text_tokens'], ctx_in, human_duration_sec)
        m['watchdog'] = wd_info

        # ---- ONE tts call == ONE llm.inference trajectory ----------------------------------
        llm = cv.model.llm
        orig_inf = llm.inference
        llm.inference = C.wrap_llm_inference(llm, trace, wd, keep_tokens=keep_tokens)
        try:
            with C.CallCounter(cv.model, 'tts') as tts_counter:
                chunks = []
                t_tts0 = time.time()
                for out in cv.model.tts(**model_input, stream=False, speed=float(gen_cfg.get('speed', 1.0))):
                    chunks.append(out['tts_speech'].squeeze(0).cpu().numpy())
                m['tts_time_sec'] = round(time.time() - t_tts0, 3)
            m['native_contract']['tts_calls'] = tts_counter.n
            m['native_contract']['tts_chunks_yielded'] = len(chunks)
            wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        finally:
            llm.inference = orig_inf
    except torch.cuda.OutOfMemoryError as e:  # main-thread OOM (flow/hift/frontend)
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

    # ---- bookkeeping ----------------------------------------------------------------------
    m['wall_time_sec'] = round(time.time() - t_wall0, 3)
    m['llm_time_sec'] = round(trace.llm_seconds, 3)
    m['flow_hift_time_sec'] = round(max(0.0, m.get('tts_time_sec', 0.0) - trace.llm_seconds), 3) if 'tts_time_sec' in m else None
    m['native_contract']['llm_inference_calls'] = trace.n_calls
    m['generated_speech_tokens'] = trace.n_yielded
    m['llm_max_len_seen'] = trace.max_len
    m['llm_min_len_seen'] = trace.min_len
    m['context_tokens_total'] = m['context_tokens_input'] + trace.n_yielded
    m['context_occupancy'] = round(m['context_tokens_total'] / C.MAX_POSITION_EMBEDDINGS, 4)
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
    status, stop_reason = C.classify(trace, m['watchdog'] or {}, audio, pre_error, m['context_tokens_total'])
    C.set_status(m, status, stop_reason)   # gen_status = provisional; the evaluator assigns the final §3.4 status
    # provisional length hint (NOT a status): A4 decides early_eos/degraded from ASR coverage + duration ratio
    exp_tok = (m['watchdog'] or {}).get('expected_speech_tokens')
    if exp_tok:
        ratio = round(trace.n_yielded / exp_tok, 4)
        m['generated_to_expected_token_ratio'] = ratio
        m['length_hint'] = 'short' if ratio < 0.5 else ('long' if ratio > 2.0 else 'plausible')
    m['mem_after'] = C.free_mem_info()

    # ---- save everything, never overwrite ---------------------------------------------------
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
