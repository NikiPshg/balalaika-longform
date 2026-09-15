#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$TRL_DIR/../.." && pwd)"

STAGE1_CONFIG="$TRL_DIR/configs/sft_proprietary_v2_stage1_main_from_youtube20k_10k.yaml"
STAGE2_CONFIG="$TRL_DIR/configs/sft_proprietary_v2_stage2_all_from_stage1_10k.yaml"
STAGE1_OUTPUT="$TRL_DIR/outputs/sft-proprietary-v2-stage1-main-from-youtube20k-10k"
STAGE2_OUTPUT="$TRL_DIR/outputs/sft-proprietary-v2-stage2-all-from-stage1-10k"
LOCAL_DATA="$TRL_DIR/datasets/balalaika_proprietary_v2_91a5696c/tokenizations/qwen/agreement_ge_0_95.train.jsonl"
EXPECTED_DATA_SIZE=78406430554
VISIBLE_GPUS="${SFT_GPUS:-0,1}"

log() {
  printf '[proprietary-v2-two-stage] %s | %s\n' "$(date -u +%FT%TZ)" "$*"
}

latest_complete_checkpoint() {
  local output="$1" candidate step best_step=-1 best=""
  shopt -s nullglob
  for candidate in "$output"/checkpoint-*; do
    [[ -f "$candidate/standalone_complete.json" ]] || continue
    step="${candidate##*-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if (( step > best_step )); then
      best_step="$step"
      best="$candidate"
    fi
  done
  shopt -u nullglob
  printf '%s' "$best"
}

run_stage() {
  local name="$1" config="$2" output="$3" required_checkpoint="$4"
  local resume=""

  if [[ -f "$output/standalone_complete.json" && -f "$required_checkpoint/standalone_complete.json" ]]; then
    log "$name already complete at $required_checkpoint"
    return
  fi

  resume="$(latest_complete_checkpoint "$output")"
  mkdir -p "$output"
  log "starting $name on CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS"
  if [[ -n "$resume" ]]; then
    log "resuming $name from $resume"
    CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" "$TRL_DIR/launch.sh" \
      --accelerate-config multi --num-processes 2 \
      --console-log "$output/console.log" \
      sft "$config" --resume_from_checkpoint "$resume"
  else
    CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" "$TRL_DIR/launch.sh" \
      --accelerate-config multi --num-processes 2 \
      --console-log "$output/console.log" \
      sft "$config"
  fi

  if [[ ! -f "$required_checkpoint/standalone_complete.json" ]]; then
    log "$name exited without a complete required checkpoint: $required_checkpoint"
    return 1
  fi
  log "$name complete at $required_checkpoint"
}

cd "$REPO_ROOT"

IFS=',' read -r -a gpu_ids <<< "$VISIBLE_GPUS"
if [[ "${#gpu_ids[@]}" -ne 2 ]]; then
  log "SFT_GPUS must contain exactly two comma-separated GPU ids"
  exit 2
fi
if [[ ! -f "$LOCAL_DATA" ]]; then
  log "local prepared dataset is missing: $LOCAL_DATA"
  exit 2
fi
actual_size="$(stat -c %s "$LOCAL_DATA")"
if [[ "$actual_size" -ne "$EXPECTED_DATA_SIZE" ]]; then
  log "local prepared dataset size mismatch: expected $EXPECTED_DATA_SIZE, got $actual_size"
  exit 2
fi

run_stage \
  stage1-main-talker \
  "$STAGE1_CONFIG" \
  "$STAGE1_OUTPUT" \
  "$STAGE1_OUTPUT/checkpoint-10000"

run_stage \
  stage2-all-talker \
  "$STAGE2_CONFIG" \
  "$STAGE2_OUTPUT" \
  "$STAGE2_OUTPUT/checkpoint-10000"

log "both stages complete (20,000 fresh optimizer steps total)"
