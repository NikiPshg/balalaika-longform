# Training source and recipes

The CPU review setup does not install the model training stacks. Use a separate environment for each backbone. Training used GPUs with 48 GB memory; the long conditions can approach that limit. Small smoke runs and different hardware do not imply reproduction of the paper's training budget.

The source code and configs are included. Base models, fine-tuned checkpoints, prepared codec arrays and VAE latent caches are not included. Full GPU preparation and training have not been repeated for this code package. Commands below document the starting points for rebuilding them. Run from the repository root unless stated otherwise.

## Upstream source and base weights

| Component | Source revision | Base model revision |
| --- | --- | --- |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` @ `29e01c4e8d000f4bcd70751be16fa94bf3d85a18` |
| Qwen source snapshot, included | `8c5a685be79b39dce931bdef76ef7e88c99d013d` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` @ `fd4b254389122332181a7c3db7f27e918eec64e3` |
| [VoxCPM](https://github.com/OpenBMB/VoxCPM) | `f5a1c6a6b901bc732e20f0d59a369f6829ad717a` | `openbmb/VoxCPM2` @ `32279effe8c19989596f05d353d1447f51d9e915` |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | `9c614e9657089213efc6a7421b30630be138a3f5` | `Misha24-10/F5-TTS_RUSSIAN` @ `ea166adeae4c80ec5ee423a671e2bdb83906cf84` |

Clone the three external source repositories into `third_party/CosyVoice`, `third_party/VoxCPM`, and `third_party/F5-TTS`, check out the revisions above and initialize required submodules. Install their dependencies in separate environments. Place the model snapshots at `models/cosyvoice3`, `models/voxcpm2`, `models/qwen3-tts`, and `models/f5/russian`. The F5 recipe uses `F5TTS_v1_Base/model_240000_inference.safetensors`, its `vocab.txt`, and the Vocos 24 kHz vocoder for synthesis.

The retained runtime versions are recorded in `artifacts/runtime_versions.json`. For orientation: CosyVoice uses Torch 2.3.1 and Transformers 4.51.3; Qwen uses Transformers 4.57.3, Accelerate 1.12.0, TRL 0.27.0 and Silero Stress 1.4; VoxCPM uses its own model/training dependencies; F5 uses `ema-pytorch==0.8.3`. Consult upstream installation instructions for the CUDA build appropriate to the machine. These are separate environments, not a single combined requirements file.

Model source roots can be overridden with `COSYVOICE_ROOT`, `QWEN_ROOT`, `VOXCPM_ROOT`, and `F5_ROOT`; base model directories use `COSYVOICE_MODEL_DIR`, `QWEN_MODEL_DIR`, and `VOXCPM_MODEL_DIR`. Set `CUDA_VISIBLE_DEVICES` in the shell before each GPU command.

## Data

First restore the audio and frozen manifests as described in [data.md](data.md). The published comparisons use the frozen `data/train/` text/crop views. Do not replace them with newly segmented or newly transcribed versions when comparing with the paper.

## CosyVoice3

`src/training/cosyvoice_train/` contains the local LM trainer and executor. The flow model, vocoder and tokenizer remain frozen. Offline tokenization splits long parents into the same windows used for the short view; long token sequences concatenate those window encodings.

```bash
export COSYVOICE_ROOT=third_party/CosyVoice
export COSYVOICE_MODEL_DIR=models/cosyvoice3
export PYTHONPATH=".:$COSYVOICE_ROOT:$COSYVOICE_ROOT/third_party/Matcha-TTS"
python src/training/make_parquet_arms.py --long_dir data/train/long --short_dir data/train/short
python src/training/make_parquet_arms.py --long_dir data/train/dev_long --short_dir data/train/dev_short
python src/training/make_punct_arm.py
bash scripts/train_cosyvoice.sh short
bash scripts/train_cosyvoice.sh long
bash scripts/train_cosyvoice.sh long_punct
```

The trainer uses 3,000 optimizer updates, learning rate 1e-5, 50 warmup steps and gradient clipping at 5. The target-speech-token batch caps differ between the long and short conditions to bring observed exposure closer. The checkpoint filenames in the recorded runs are `epoch_3_step_3001.pt`; the filename includes the trainer's step numbering convention. `scripts/check_budget_match.py` reads the training ledgers to compare exposure.

## Qwen3-TTS

The included source snapshot contains the `all_talker` trainer used here, including the talker and code-predictor losses. The `lora_finetuning` directory name is inherited upstream; these experiments use the full `all_talker` scope specified in the YAML, not a LoRA-only adaptation.

Install `third_party/Qwen3-TTS` and its training dependencies in the Qwen environment. The retained `lora_finetuning/pyproject.toml` and `uv.lock` document its dependency setup; TRL 0.27.0 and Trackio 0.20.2 are additionally required by the trainer.

Prepare each of the four views (`long`, `short`, `dev_long`, `dev_short`) by substituting its name below:

```bash
mkdir -p data/train/qwen
python scripts/qwen_p3_prepare_data.py \
  --wav-scp data/train/long/wav.scp \
  --utt2offset data/train/long/utt2offset \
  --text data/train/long_punct/text \
  --out data/train/qwen/train_long.jsonl
```

Use `short_punct/text`, `dev_long_punct/text` and `dev_short_punct/text` for the other views, with output names `train_short.jsonl`, `dev_long.jsonl`, `dev_short.jsonl`. This stage re-encodes audio with the Qwen codec; CosyVoice speech tokens are not interchangeable with it.

```bash
python scripts/pack_qwen_data.py --train data/train/qwen/train_long.jsonl --dev data/train/qwen/dev_long.jsonl --prefix qe3p
python scripts/pack_qwen_data.py --train data/train/qwen/train_short.jsonl --dev data/train/qwen/dev_short.jsonl --prefix qe2p
```

The assembler places 32 held-out dev rows first and checks for overlap with training IDs. It rebuilds a usable eight-shard cache; the historical encoded cache and its exact shard ordering are not distributed, so this step does not promise bit-for-bit repetition of the old training stream.

Use `configs/train/qwen/qe3p.yaml` (long) or `qe2p.yaml` (short). Because the upstream launcher changes directory, make `model_path`, `dataset_data_file`, `output_dir`, and `logging_dir` absolute in a local copy of the YAML before launching:

```bash
bash third_party/Qwen3-TTS/lora_finetuning/trl/launch.sh sft /absolute/path/to/local-config.yaml --python /absolute/path/to/qwen-environment/bin/python
```

The long recipe uses 1,133 updates with accumulation 16; the short recipe uses 1,185 updates with accumulation 12. Both use punctuated text and the same base initialization. Silero Stress is applied in the training mapper. Generation uses the full supplied reference prompt and the same conditioning layout as training.

## VoxCPM2

The local trainer is `scripts/train_voxcpm_finetune_local.py`. It adds the long-form data path, checkpointing and exposure logging around the upstream trainer.

```bash
export PYTHONPATH="third_party/VoxCPM/src:."
python scripts/voxcpm_p3_prepare_data.py
python scripts/voxcpm_p3_encode_latents.py --manifest data/train/voxcpm/vce3p.jsonl --arm vce3p
python scripts/voxcpm_p3_encode_latents.py --manifest data/train/voxcpm/dev_long.jsonl --arm dev_long
export VOXCPM_GRAD_CKPT=1
export VOXCPM_SAVE_OPTIMIZER=0
export VOXCPM_SAVE_TOTAL_LIMIT=1
python scripts/train_voxcpm_finetune_local.py --config_path configs/train/voxcpm/vce3p.yaml
python scripts/train_voxcpm_finetune_local.py --config_path configs/train/voxcpm/vce2p.yaml
```

The long recipe uses 1,139 updates, batch size 1, accumulation 19. The short recipe uses 1,185 updates, batch size 4, accumulation 18. Both use learning rate 1e-5 and an 8,192-position limit. The latter excludes 144 long units (~18.2% of long-view audio seconds), an important limitation of that comparison. The original budget audit is included under `artifacts/training/`.

## F5-TTS

`src/f5_training/train.py` updates the full flow-matching model. One optimizer update corresponds to one source parent: a long sample or gradient accumulation over its short windows, weighted by their mel-frame shares. This provides an exposure-matched comparison without pretending that update counts mean the same thing across architectures.

```bash
python scripts/f5_prepare_data.py
python scripts/train_f5.py configs/train/f5/long.json --dry-run
python scripts/train_f5.py configs/train/f5/long.json
python scripts/train_f5.py configs/train/f5/short.json
```

Both recipes use 18,114 updates, or three passes over 6,038 common parents. The published runs use EMA weights; their protocols and summaries are included in `artifacts/training/`. Continuous synthesis uses 32 steps, CFG 2.0 and sway coefficient -1.0, with duration estimated from the prompt and text, not from a target recording. This setting remains poor on the long evaluation passages; the artifact retains those outcomes.
