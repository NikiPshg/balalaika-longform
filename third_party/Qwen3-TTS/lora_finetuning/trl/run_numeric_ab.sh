#!/usr/bin/env bash
set -euo pipefail

TRL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TRL_DIR/../.." && pwd)"
PYTHON_BIN="$TRL_DIR/../.venv/bin/python"
OUTPUT_ROOT="${1:-$TRL_DIR/outputs/numeric-probes/checkpoint-6700-ab}"
SFT_CHECKPOINT="${2:-$TRL_DIR/outputs/sft-main-talker-full-base/checkpoint-6700}"
MAX_NEW_TOKENS="${3:-512}"
SELECTED_ROWS="$OUTPUT_ROOT/selected_rows.json"

BASE_REVISION="fd4b254389122332181a7c3db7f27e918eec64e3"
VOICES_REVISION="f57528db42342b79201cc0d360355e2532b088b7"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Project Python is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$SELECTED_ROWS" ]]; then
  echo "Selected rows are missing: $SELECTED_ROWS" >&2
  exit 2
fi
if [[ ! -f "$SFT_CHECKPOINT/standalone_complete.json" ]]; then
  echo "SFT checkpoint is not a complete standalone export: $SFT_CHECKPOINT" >&2
  exit 2
fi
if [[ ! "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "max_new_tokens must be a positive integer: $MAX_NEW_TOKENS" >&2
  exit 2
fi

mapfile -t PROBE_TEXTS < <(jq -er '.rows[].text' "$SELECTED_ROWS")
if [[ "${#PROBE_TEXTS[@]}" -ne 10 ]]; then
  echo "Expected exactly 10 probe texts, got ${#PROBE_TEXTS[@]}" >&2
  exit 2
fi

TEXT_ARGS=()
for probe_text in "${PROBE_TEXTS[@]}"; do
  TEXT_ARGS+=(--text "$probe_text")
done

COMMON_ARGS=(
  --voices-revision "$VOICES_REVISION"
  --voice anastasia-1
  --seed 2608
  --max-new-tokens "$MAX_NEW_TOKENS"
)

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"

echo "[numeric-ab] upstream Base: unstressed target and reference"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$TRL_DIR/probe_voice_clone.py" \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --model-label upstream-base \
  --model-revision "$BASE_REVISION" \
  --target-stress-mode none \
  --reference-stress-mode strip \
  --output-dir "$OUTPUT_ROOT/base" \
  "${COMMON_ARGS[@]}" \
  "${TEXT_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/base.log"

echo "[numeric-ab] SFT: Silero-stressed target and manually stressed reference"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$TRL_DIR/probe_voice_clone.py" \
  --model-path "$SFT_CHECKPOINT" \
  --model-label "$(basename "$SFT_CHECKPOINT")" \
  --target-stress-mode silero \
  --reference-stress-mode as-is \
  --output-dir "$OUTPUT_ROOT/sft" \
  "${COMMON_ARGS[@]}" \
  "${TEXT_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/sft.log"

echo "[numeric-ab] complete: $OUTPUT_ROOT"
