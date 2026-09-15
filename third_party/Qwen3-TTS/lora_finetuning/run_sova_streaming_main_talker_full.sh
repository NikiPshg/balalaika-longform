#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "${REPO_ROOT}/.venv-lora/bin/python" \
    "${SCRIPT_DIR}/.venv/bin/python" \
    "${REPO_ROOT}/.venv/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="/opt/omni/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/hf-cache}"
export PYTHONUNBUFFERED=1

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/artifacts/models/Qwen3TTS-RL-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/checkpoints/sova_streaming_main_talker_full}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${REPO_ROOT}/artifacts/runs/sova_streaming_main_talker_full}"
NUM_GPUS="${NUM_GPUS:-1}"
if (( NUM_GPUS == 1 )) && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}"

case "${MIXED_PRECISION:-bf16}" in
  bf16) bf16=true; fp16=false ;;
  fp16) bf16=false; fp16=true ;;
  no|fp32) bf16=false; fp16=false ;;
  *) echo "MIXED_PRECISION must be bf16, fp16, or no" >&2; exit 2 ;;
esac

eval_steps="${EVAL_STEPS:-${VALIDATION_EVERY_STEPS:-100}}"
save_steps="${SAVE_STEPS:-${CHECKPOINT_EVERY_STEPS:-100}}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-${PREFETCH_WORKERS:-1}}"
eval_strategy=steps
save_strategy=steps
if (( eval_steps <= 0 )); then eval_strategy=no; fi
if (( save_steps <= 0 )); then save_strategy=no; fi

train_args=(
  --model_path "${MODEL_PATH}"
  --dataset_name "${DATASET_NAME:-lab260/sova_balalaika}"
  --dataset_split "${DATASET_SPLIT:-train}"
  --dataset_revision "${DATASET_REVISION:-be89f9bbc9908afd34b28e05441bbb9f655c0183}"
  --streaming true
  --output_dir "${OUTPUT_DIR}"
  --logging_dir "${TENSORBOARD_DIR}"
  --report_to tensorboard
  --do_train "${DO_TRAIN:-true}"
  --do_eval "${DO_EVAL:-true}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE:-${BATCH_SIZE:-8}}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --dataloader_num_workers "${dataloader_num_workers}"
  --max_e2e_rover_wer "${MAX_E2E_ROVER_WER:-0.0}"
  --max_steps "${MAX_STEPS:-10000}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --warmup_steps "${WARMUP_STEPS:-200}"
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
  --bf16 "${bf16}"
  --fp16 "${fp16}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
  --logging_strategy steps
  --logging_steps "${LOGGING_STEPS:-10}"
  --eval_strategy "${eval_strategy}"
  --eval_steps "${eval_steps}"
  --eval_on_start "${EVAL_ON_START:-true}"
  --save_strategy "${save_strategy}"
  --save_steps "${save_steps}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --save_final_model "${SAVE_FINAL_MODEL:-${SAVE_FINAL:-true}}"
  --validation_samples "${VALIDATION_SAMPLES:-32}"
  --validation_encode_batch_size "${VALIDATION_ENCODE_BATCH_SIZE:-8}"
  --validation_references "${VALIDATION_REFERENCES:-8}"
  --validation_texts "${VALIDATION_TEXTS:-26}"
  --validation_generation_batch_size "${VALIDATION_GENERATION_BATCH_SIZE:-4}"
  --generate_audio "${GENERATE_AUDIO:-true}"
  --asr_enabled "${ASR_ENABLED:-true}"
  --hf_upload_samples "${HF_UPLOAD_SAMPLES:-true}"
  --hf_samples_repo "${HF_SAMPLES_REPO:-bitmanagerai/etc}"
  --hf_samples_path "${HF_SAMPLES_PATH:-qwen3tts-training}"
  --run_name "${RUN_NAME:-full-main-talker-streaming-rl2}"
  --asr_model_path "${ASR_MODEL_PATH:-${REPO_ROOT}/artifacts/models/gigaam-v3-e2e-ctc-onnx}"
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS:-false}"
  --ddp_broadcast_buffers "${DDP_BROADCAST_BUFFERS:-false}"
  --ddp_timeout "${DDP_TIMEOUT:-14400}"
  --remove_unused_columns false
  --seed "${SEED:-42}"
)

if (( dataloader_num_workers > 0 )); then
  train_args+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR:-1}")
elif [[ -n "${DATALOADER_PREFETCH_FACTOR:-}" ]]; then
  echo "DATALOADER_PREFETCH_FACTOR requires DATALOADER_NUM_WORKERS > 0" >&2
  exit 2
fi

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  train_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

launcher=("${PYTHON_BIN}")
if (( NUM_GPUS > 1 )); then
  launcher=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node "${NUM_GPUS}"
  )
fi

exec "${launcher[@]}" "${SCRIPT_DIR}/sft_streaming_main_talker_full.py" "${train_args[@]}"
