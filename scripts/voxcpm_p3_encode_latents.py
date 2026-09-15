#!/usr/bin/env python
"""A22-voxcpm (E12 P2): offline AudioVAE latents for the LONG arm (+ dev holdout).

Why: the VAE encoder's first conv over a 568-s unit is one 4.33 GiB fp32 allocation;
inside the training process (2B fp32 weights + grads + AdamW ~= 37 GB) that OOMs
(P2 probe). Standalone, the encode fits easily; the training then reads latents.

Math is IDENTICAL to the training-time packer path:
    soundfile slice read (data_offsets._read_slice_16k_mono: mono mean + librosa
    resample to 16 kHz)  ->  right-pad to a multiple of patch_len 2560 samples
    (packers.py:61-63)  ->  AudioVAE.encode(...)["mu"] fp32 (audio_vae_v2.py:489-501,
    deterministic)  ->  [T_latent, 64] float32 .npy

Output: data/train/voxcpm/latents/<arm>/<utt>.npy + the manifest rewritten with a
'latents' column ( *_lat.jsonl ). Runs in the voxcpm venv on one granted GPU.

Usage:
  CUDA_VISIBLE_DEVICES=2 third_party/VoxCPM/.venv/bin/python \
      scripts/voxcpm_p3_encode_latents.py --manifest data/train/voxcpm/vce3p.jsonl \
      --arm long --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, 'third_party/VoxCPM/src')

SNAP = os.environ.get('VOXCPM_MODEL_DIR', os.path.join(ROOT, 'models/voxcpm2'))
PATCH_LEN = 2560   # patch_size 4 x hop 640 @16 kHz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--arm', required=True, help='subdir under data/train/voxcpm/latents/')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-manifest', default=None,
                    help="default: manifest with suffix _lat.jsonl")
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    from voxcpm.modules.audiovae import AudioVAEV2, AudioVAEConfigV2
    from voxcpm_train.data_offsets import _read_slice_16k_mono

    lat_dir = os.path.join(ROOT, 'data', 'train', 'voxcpm', 'latents', args.arm)
    os.makedirs(lat_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8')]

    cfg = json.load(open(os.path.join(SNAP, 'config.json')))
    vae = AudioVAEV2(config=AudioVAEConfigV2(**cfg['audio_vae_config']))
    try:
        from safetensors.torch import load_file
        st = os.path.join(SNAP, 'audiovae.safetensors')
        sd = load_file(st) if os.path.exists(st) else None
    except Exception:
        sd = None
    if sd is None:
        ckpt = torch.load(os.path.join(SNAP, 'audiovae.pth'), map_location='cpu', weights_only=True)
        sd = ckpt.get('state_dict', ckpt)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not [k for k in missing if 'encoder' in k], missing
    vae = vae.to('cuda').to(torch.float32).eval()

    class RowDS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            out = os.path.join(lat_dir, r['utt'] + '.npy')
            if os.path.exists(out):
                return r['utt'], None, out
            wave = _read_slice_16k_mono(r['audio'], r['offset_start'], r['offset_end'], 16000)
            if wave.shape[0] % PATCH_LEN != 0:
                wave = np.pad(wave, (0, PATCH_LEN - wave.shape[0] % PATCH_LEN))
            return r['utt'], wave, out

    def collate(b):
        return b

    dl = DataLoader(RowDS(), batch_size=1, num_workers=args.workers, collate_fn=collate)
    n_done = n_skip = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            utt, wave, out = batch[0]
            if wave is None:
                n_skip += 1
                continue
            w = torch.from_numpy(wave).unsqueeze(0).to('cuda')
            z = vae.encode(w, 16000)                    # [1, D, T']
            lat = z.squeeze(0).transpose(0, 1).contiguous().cpu().numpy().astype('float32')
            tmp = out + '.tmp.npy'
            np.save(tmp, lat)
            os.replace(tmp, out)
            n_done += 1
            if n_done % 200 == 0:
                print('[{} s] {} encoded, {} skipped'.format(round(time.time() - t0), n_done, n_skip), flush=True)

    out_manifest = args.out_manifest or args.manifest.replace('.jsonl', '_lat.jsonl')
    with open(out_manifest, 'w', encoding='utf-8') as f:
        for r in rows:
            r2 = dict(r)
            r2['latents'] = os.path.join(lat_dir, r['utt'] + '.npy')
            assert os.path.exists(r2['latents']), r2['latents']
            f.write(json.dumps(r2, ensure_ascii=False) + '\n')
    print('DONE encoded={} skipped={} -> {} ({} rows)'.format(n_done, n_skip, out_manifest, len(rows)))


if __name__ == '__main__':
    main()
