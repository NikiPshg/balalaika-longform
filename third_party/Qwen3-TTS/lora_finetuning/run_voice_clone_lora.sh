#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR=""
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/DO_NOT_REMOVE}"

PREP_JSONL=""
TRAIN_JSONL="/home/tachka5090-1/kirill_workspace/full_train.jsonl"
VAL_JSONL="/home/tachka5090-1/kirill_workspace/full_val.jsonl"

DEVICE="${DEVICE:-cuda:0}"
MIXED_PRECISION="bf16"
PREP_DTYPE="${PREP_DTYPE:-${MIXED_PRECISION}}"
if [[ "${PREP_DTYPE}" == "no" ]]; then
  PREP_DTYPE="fp32"
fi
ATTN_IMPLEMENTATION="eager"

BATCH_SIZE="6"
EVAL_BATCH_SIZE="8"
DATALOADER_NUM_WORKERS="12"
GRADIENT_ACCUMULATION_STEPS="12"
LR="2e-5"
NUM_EPOCHS="10"
START_EPOCH="${START_EPOCH:-0}"
SAVE_EVERY="${SAVE_EVERY:-1}"
EVAL_EVERY="${EVAL_EVERY:-1}"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_BIAS="${LORA_BIAS:-none}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,codec_head,lm_head.0,lm_head.1,lm_head.2,lm_head.3,lm_head.4,lm_head.5,lm_head.6,lm_head.7,lm_head.8,lm_head.9,lm_head.10,lm_head.11,lm_head.12,lm_head.13,lm_head.14}"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"

SINGLE_BATCH_TEST="0"
SINGLE_BATCH_MAX_STEPS="500"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MIN_REF_SEC="${MIN_REF_SEC:-3.0}"
MIN_TARGET_SEC="${MIN_TARGET_SEC:-2.0}"
LANGUAGE="${LANGUAGE:-Auto}"
SKIP_PREP="true"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="1024"
VALIDATION_AUDIO_EVERY="1"
VALIDATION_AUDIO_SAMPLES="${VALIDATION_AUDIO_SAMPLES:-0}"
VALIDATION_AT_STEP="10000"
VALIDATION_DECODE_MODE="${VALIDATION_DECODE_MODE:-generate}"
NON_STREAMING_MODE="${NON_STREAMING_MODE:-true}"
DO_SAMPLE="${DO_SAMPLE:-true}"
SUBTALKER_DOSAMPLE="${SUBTALKER_DOSAMPLE:-true}"

WANDB_ENABLED="1"
WANDB_PROJECT="qwen"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="train_with_accents_1"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-voice-clone,lora,qwen3-tts}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

export WANDB_PROJECT WANDB_ENTITY WANDB_NAME WANDB_GROUP WANDB_TAGS
export WANDB_ID WANDB_RESUME WANDB_MODE WANDB_DIR
if [[ -n "${WANDB_API_KEY}" ]]; then
  export WANDB_API_KEY
fi

mkdir -p "$(dirname "${PREP_JSONL}")" "${OUTPUT_DIR}"

if [[ "${SKIP_PREP}" != "1" && "${SKIP_PREP}" != "true" ]]; then
  prep_args=(
    --data_dir "${DATA_DIR}"
    --base_model_path "${BASE_MODEL_PATH}"
    --output_jsonl "${PREP_JSONL}"
    --device "${DEVICE}"
    --dtype "${PREP_DTYPE}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --batch_size "${BATCH_SIZE}"
    --min_ref_sec "${MIN_REF_SEC}"
    --min_target_sec "${MIN_TARGET_SEC}"
    --language "${LANGUAGE}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    prep_args+=(--max_samples "${MAX_SAMPLES}")
  fi
  python "${SCRIPT_DIR}/prepare_voice_clone_data.py" "${prep_args[@]}"
fi

train_args=(
  --model_path "${BASE_MODEL_PATH}"
  --output_model_path "${OUTPUT_DIR}"
  --train_jsonl "${TRAIN_JSONL}"
  --val_jsonl "${VAL_JSONL}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --lr "${LR}"
  --epochs "${NUM_EPOCHS}"
  --start_epoch "${START_EPOCH}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --save_every "${SAVE_EVERY}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_bias "${LORA_BIAS}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --seed "${SEED}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --validation_audio_every "${VALIDATION_AUDIO_EVERY}"
  --validation_audio_samples "${VALIDATION_AUDIO_SAMPLES}"
  --validation_decode_mode "${VALIDATION_DECODE_MODE}"
  --non_streaming_mode "${NON_STREAMING_MODE}"
  --do_sample "${DO_SAMPLE}"
  --subtalker_dosample "${SUBTALKER_DOSAMPLE}"
  --wandb_enabled "${WANDB_ENABLED}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_tags "${WANDB_TAGS}"
  --wandb_resume "${WANDB_RESUME}"
  --wandb_mode "${WANDB_MODE}"
  --wandb_dir "${WANDB_DIR}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  train_args+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_NAME}" ]]; then
  train_args+=(--wandb_name "${WANDB_NAME}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  train_args+=(--wandb_group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_ID}" ]]; then
  train_args+=(--wandb_id "${WANDB_ID}")
fi

if [[ -n "${VALIDATION_AT_STEP}" ]]; then
  train_args+=(--validation_at_step "${VALIDATION_AT_STEP}")
fi

if [[ -n "${RESUME_ADAPTER}" ]]; then
  train_args+=(--resume_adapter "${RESUME_ADAPTER}")
fi

if [[ "${SINGLE_BATCH_TEST}" == "1" || "${SINGLE_BATCH_TEST}" == "true" ]]; then
  train_args+=(--single_batch_test --max_steps "${SINGLE_BATCH_MAX_STEPS}")
fi

python "${SCRIPT_DIR}/sft_voice_clone_lora.py" "${train_args[@]}"
