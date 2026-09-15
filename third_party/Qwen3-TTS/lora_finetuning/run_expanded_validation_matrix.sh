#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/artifacts/models/Qwen3TTS-RL-2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/artifacts/checkpoints/sova_streaming_main_talker_full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/expanded_validation/full-main-talker-8x26}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-${REPO_ROOT}/artifacts/runs/expanded-validation-8x26}"
RUN_NAME="${RUN_NAME:-full-main-talker-streaming-rl2-expanded-8x26}"

mkdir -p "${OUTPUT_ROOT}" "${TENSORBOARD_ROOT}"

for step in 0 100 200 300 400; do
  if [[ "${step}" == "0" ]]; then
    checkpoint="${BASE_MODEL}"
  else
    checkpoint="${CHECKPOINT_ROOT}/checkpoint-${step}"
    legacy_checkpoint="${CHECKPOINT_ROOT}/checkpoint-step-${step}"
    if [[ ! -s "${checkpoint}/config.json" && -s "${legacy_checkpoint}/config.json" ]]; then
      checkpoint="${legacy_checkpoint}"
    fi
  fi
  if [[ ! -s "${checkpoint}/config.json" ]]; then
    echo "missing checkpoint config: ${checkpoint}/config.json" >&2
    exit 1
  fi

  MODEL_PATH="${checkpoint}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/model-step-${step}" \
  TENSORBOARD_DIR="${TENSORBOARD_ROOT}/model-step-${step}" \
  RUN_NAME="${RUN_NAME}" \
  DO_TRAIN=false \
  DO_EVAL=true \
  EVAL_ON_START=false \
  EVAL_STEPS=0 \
  GENERATE_AUDIO=true \
  VALIDATION_SAMPLES=32 \
  VALIDATION_REFERENCES=8 \
  VALIDATION_TEXTS=26 \
  VALIDATION_GENERATION_BATCH_SIZE=4 \
  ASR_ENABLED=true \
  HF_UPLOAD_SAMPLES=true \
  SAVE_FINAL_MODEL=false \
  SAVE_STEPS=0 \
  bash "${SCRIPT_DIR}/run_sova_streaming_main_talker_full.sh"
done
