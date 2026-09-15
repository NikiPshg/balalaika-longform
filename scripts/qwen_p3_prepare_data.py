#!/usr/bin/env python
"""A10-qwen P3: build the prepared-Qwen JSONL (qwen_prepared_full_utterance rows) from a
v3.1 kaldi set (wav.scp + utt2offset + a punctuated text file).

Row schema = what PreparedQwenFullUtteranceMapper consumes (trainers/qwen_tts_sft.py:582-763;
field-by-field table in reports/qwen_p3_plan.md §2.1). Storage ref/target split is the
first REF_SEC seconds of codec frames (the mapper re-draws its own training prefix from
the concatenated codes, so this split is only a container layout). ref_text is left
empty: the mapper concatenates ref_text+text into one transcript anyway.

Audio: flac crop per utt2offset (44.1 kHz stereo PCM_16 in v3.1) -> mono -> 24 kHz ->
frozen 12.5 Hz speech tokenizer encode ([T,16] int codes) + speaker embedding (2048-d)
from the first REF_SEC crop. asr_agreement_mean = parent segment json `asr_consistency`
/ 100 (real v3.1 number, cached per parent).

Run INSIDE the fork venv on ONE granted GPU (resumable, appends):
  CUDA_VISIBLE_DEVICES=2 third_party/Qwen3-TTS/lora_finetuning/.venv/bin/python \
    scripts/qwen_p3_prepare_data.py --wav-scp data/train/long/wav.scp \
    --utt2offset data/train/long/utt2offset --text data/train/long_punct/text \
    --out data/train/qwen/train_long.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get('QWEN_ROOT', 'third_party/Qwen3-TTS'))

SNAP = os.environ.get('QWEN_MODEL_DIR', 'models/qwen3-tts')
SR = 24000
CODEC_HZ = 12.5
REF_SEC = 8.0
REF_FRAMES = int(REF_SEC * CODEC_HZ)  # 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-scp', required=True)
    ap.add_argument('--utt2offset', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--utts-file', default=None, help='optional: only these utt ids, in this order')
    ap.add_argument('--min-frames', type=int, default=REF_FRAMES + 30,
                    help='skip utts shorter than this many codec frames (ref 100 + continuation)')
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    from qwen_tts import Qwen3TTSModel

    wav_scp = {u: p.strip() for u, p in (l.split(None, 1) for l in open(args.wav_scp))}
    offsets = {}
    for l in open(args.utt2offset):
        u, parent, a, b = l.split()
        offsets[u] = (parent, float(a), float(b))
    texts = {u: t.strip() for u, t in (l.split(None, 1) for l in open(args.text))}
    order = [l.strip() for l in open(args.utts_file)] if args.utts_file else list(texts)

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)['source_record_id'])
            except Exception:
                pass
    print(f'{len(order)} utts requested, {len(done)} already in {args.out}', flush=True)

    tts = Qwen3TTSModel.from_pretrained(SNAP, device_map='cuda:0', dtype=torch.bfloat16,
                                        attn_implementation='sdpa')
    st = tts.model.speech_tokenizer
    assert int(st.get_output_sample_rate()) == SR and int(st.get_encode_downsample_rate()) == 1920

    cons_cache: dict[str, float] = {}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_new = n_skip = 0
    t0 = time.time()
    with open(args.out, 'a', encoding='utf-8') as fo:
        for u in order:
            if u in done:
                continue
            if args.limit is not None and n_new >= args.limit:
                break
            if u not in texts or u not in offsets:
                print(f'SKIP {u}: missing text/offset', flush=True)
                n_skip += 1
                continue
            parent, a, b = offsets[u]
            flac = wav_scp.get(u) or wav_scp[parent]
            jpath = flac[:-5] + '.json'
            if parent not in cons_cache:
                try:
                    cons_cache[parent] = float(json.load(open(jpath)).get('asr_consistency') or 0.0) / 100.0
                except Exception:
                    cons_cache[parent] = 0.0
            info = sf.info(flac)
            fr0, fr1 = int(a * info.samplerate), min(int(b * info.samplerate), info.frames)
            wav, fsr = sf.read(flac, start=fr0, stop=fr1, dtype='float32', always_2d=True)
            wav = wav.mean(axis=1)
            if fsr != SR:
                wav = librosa.resample(y=wav, orig_sr=fsr, target_sr=SR)
            dur = len(wav) / SR
            with torch.inference_mode():
                codes = st.encode(wav, sr=SR).audio_codes[0]          # [T,16] on GPU
                ref_wav = wav[:int(REF_SEC * SR)]
                spk = tts.model.extract_speaker_embedding(ref_wav.astype(np.float32), SR)
            codes = codes.detach().to('cpu', torch.int64).tolist()
            if len(codes) < args.min_frames:
                print(f'SKIP {u}: only {len(codes)} frames', flush=True)
                n_skip += 1
                continue
            row = {
                'source_record_id': u,
                'agreement_bucket': 'agreement_ge_0_95',
                'text_source': 'rover_punctuated_accented',
                'text_provenance': 'gigaam-v3-e2e-ctc punctuated (E7 text discipline); stress re-applied by the mapper',
                'orientation': 'prefix_reference',
                'profile': 'balanced',
                'boundary_type': 'punctuation',
                'boundary_tier': 'primary',
                'asr_agreement_mean': round(cons_cache[parent], 6),
                'ref_duration': round(REF_FRAMES / CODEC_HZ, 3),
                'duration': round(dur - REF_FRAMES / CODEC_HZ, 3),
                'ref_audio_codes': codes[:REF_FRAMES],
                'audio_codes': codes[REF_FRAMES:],
                'ref_spk_embedding': [round(float(x), 6) for x in spk.float().cpu().tolist()],
                'ref_text': '',
                'text': texts[u],
            }
            fo.write(json.dumps(row, ensure_ascii=False) + '\n')
            n_new += 1
            if n_new % 50 == 0:
                el = time.time() - t0
                print(f'[{n_new}] {u} dur={dur:.1f}s frames={len(codes)} '
                      f'({n_new/el:.2f} utt/s)', flush=True)
    print(f'DONE new={n_new} skipped={n_skip} total_in_file={len(done)+n_new}', flush=True)


if __name__ == '__main__':
    main()
