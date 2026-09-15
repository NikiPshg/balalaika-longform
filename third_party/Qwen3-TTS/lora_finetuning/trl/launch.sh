#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_PYTHON="${REPO_ROOT}/lora_finetuning/.venv/bin/python"
export TRACKIO_DIR="${TRACKIO_DIR:-${SCRIPT_DIR}/trackio}"

usage() {
  cat <<'EOF'
Usage:
  launch.sh [launcher options] {sft|grpo} CONFIG [TRAINER OVERRIDES...]

Launcher options (must come before METHOD):
  --accelerate-config {single|multi|PATH}
  --num-processes N
  --python PATH              Override TRL_PYTHON/the project venv.
  --console-log PATH         Append stdout/stderr from the real run to PATH.
  --validate                 Check the entrypoint and config, then exit.
  --dry-run                  Print the exact command without running it.
  -h, --help

Examples:
  lora_finetuning/trl/launch.sh sft lora_finetuning/trl/configs/sft_full.yaml
  lora_finetuning/trl/launch.sh --accelerate-config multi --num-processes 2 \
    grpo lora_finetuning/trl/configs/grpo.yaml --max_steps 20
EOF
}

die() {
  printf 'launch.sh: %s\n' "$*" >&2
  exit 2
}

accelerate_choice="single"
num_processes=""
python_bin="${TRL_PYTHON:-${DEFAULT_PYTHON}}"
console_log=""
validate_only=0
dry_run=0

while (($#)); do
  case "$1" in
    --accelerate-config)
      (($# >= 2)) || die "--accelerate-config requires a value"
      accelerate_choice="$2"
      shift 2
      ;;
    --num-processes)
      (($# >= 2)) || die "--num-processes requires a value"
      num_processes="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python_bin="$2"
      shift 2
      ;;
    --console-log)
      (($# >= 2)) || die "--console-log requires a value"
      console_log="$2"
      shift 2
      ;;
    --validate)
      validate_only=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      die "unknown launcher option '$1' (trainer overrides go after METHOD and CONFIG)"
      ;;
    *)
      break
      ;;
  esac
done

(($# >= 2)) || { usage >&2; exit 2; }
method="$1"
config_input="$2"
shift 2
trainer_overrides=("$@")

case "${method}" in
  sft) entrypoint="${REPO_ROOT}/lora_finetuning/trl/sft.py" ;;
  grpo) entrypoint="${REPO_ROOT}/lora_finetuning/trl/grpo.py" ;;
  *) die "METHOD must be 'sft' or 'grpo', got '${method}'" ;;
esac

case "${accelerate_choice}" in
  single) accelerate_config="${SCRIPT_DIR}/configs/accelerate/single_gpu.yaml" ;;
  multi) accelerate_config="${SCRIPT_DIR}/configs/accelerate/multi_gpu.yaml" ;;
  /*) accelerate_config="${accelerate_choice}" ;;
  *) accelerate_config="${REPO_ROOT}/${accelerate_choice}" ;;
esac

case "${config_input}" in
  /*) config_path="${config_input}" ;;
  *) config_path="${REPO_ROOT}/${config_input}" ;;
esac

[[ -x "${python_bin}" ]] || die "Python is not executable: ${python_bin}"
[[ -f "${entrypoint}" ]] || die "entrypoint is missing: ${entrypoint}"
[[ -r "${config_path}" ]] || die "config is not readable: ${config_path}"
[[ -r "${accelerate_config}" ]] || die "Accelerate config is not readable: ${accelerate_config}"
if [[ -n "${num_processes}" && ! "${num_processes}" =~ ^[1-9][0-9]*$ ]]; then
  die "--num-processes must be a positive integer"
fi

# The repository root deliberately stays on sys.path instead of
# lora_finetuning/: that directory contains this folder named `trl`, which must
# never take precedence over the installed Hugging Face `trl` package.
cd "${REPO_ROOT}"

if ((validate_only)); then
  "${python_bin}" "${entrypoint}" --help >/dev/null
  "${python_bin}" - "${config_path}" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit(f"{path}: YAML root must be a mapping")
print(f"validated: {path}")
PY
  exit 0
fi

command=(
  "${python_bin}" -m accelerate.commands.launch
  --config_file "${accelerate_config}"
)
if [[ -n "${num_processes}" ]]; then
  command+=(--num_processes "${num_processes}")
fi
command+=("${entrypoint}" --config "${config_path}")
command+=("${trainer_overrides[@]}")

if ((dry_run)); then
  printf 'cd %q\n' "${REPO_ROOT}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ -n "${console_log}" ]]; then
  case "${console_log}" in
    /*) console_log_path="${console_log}" ;;
    *) console_log_path="${REPO_ROOT}/${console_log}" ;;
  esac
  mkdir -p -- "$(dirname -- "${console_log_path}")"
  printf 'console log: %s\n' "${console_log_path}"
  exec > >(tee -a -- "${console_log_path}") 2>&1
fi

exec "${command[@]}"
