#!/usr/bin/env bash
# Launch one submitted CosyVoice3 condition. Run from repository root.
set -euo pipefail
arm=${1:?usage: bash scripts/train_cosyvoice.sh long|short|long_punct}
case "$arm" in
 long) config=configs/train/cv3_long_sft.yaml;;
 short) config=configs/train/cv3_short_sft.yaml;;
 long_punct) config=configs/train/cv3_punct_sft.yaml;;
 *) echo 'Expected long, short, or long_punct' >&2; exit 2;;
esac
export COSYVOICE_ROOT=${COSYVOICE_ROOT:-third_party/CosyVoice}
export COSYVOICE_MODEL_DIR=${COSYVOICE_MODEL_DIR:-models/cosyvoice3}
export PYTHONPATH="$(pwd):$COSYVOICE_ROOT:$COSYVOICE_ROOT/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
model_dir="checkpoints/cosyvoice/$arm"
if [[ -e "$model_dir" ]]; then echo "Choose a fresh checkpoint directory: $model_dir exists" >&2; exit 2; fi
for path in "$COSYVOICE_MODEL_DIR/llm.pt" "data/train/$arm/parquet/data.list" "data/train/dev_$arm/parquet/data.list"; do
 [[ -f "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
"${PYTHON:-python}" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone \
 src/training/cosyvoice_train/train.py --train_engine torch_ddp --config "$config" \
 --train_data "data/train/$arm/parquet/data.list" --cv_data "data/train/dev_$arm/parquet/data.list" \
 --qwen_pretrain_path "$COSYVOICE_MODEL_DIR/CosyVoice-BlankEN" --onnx_path "$COSYVOICE_MODEL_DIR" \
 --model llm --checkpoint "$COSYVOICE_MODEL_DIR/llm.pt" --model_dir "$model_dir" \
 --tensorboard_dir "$model_dir/tensorboard" --ddp.dist_backend nccl \
 --num_workers 2 --prefetch 2 --pin_memory --use_amp --timeout 600 --gradient_checkpointing
