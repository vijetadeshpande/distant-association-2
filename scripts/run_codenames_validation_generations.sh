#!/usr/bin/env bash
# Run Codenames validation generation sequentially for a list of checkpoints.
#
# Execute this inside the existing/default VERL image. Each CHECKPOINTS entry
# is only a Hugging Face ID, local merged-HF path, or rclone remote. Output
# labels are derived automatically with scripts/checkpoint_utils.py.
#
# The Python evaluator resumes an existing output by prompt index, so rerunning
# this launcher does not regenerate completed prompt groups.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/codenames_validation_rollouts}"
EVALUATOR="${SCRIPT_DIR}/generate_codenames_validation_rollouts.py"
export RCLONE_CHECKPOINT_DIR="${RCLONE_CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/rclone_downloads}"
# 0 means auto-detect and use every GPU visible through CUDA_VISIBLE_DEVICES
# (or every GPU reported by nvidia-smi when CUDA_VISIBLE_DEVICES is unset).
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-0}"

# Add tuned checkpoints as plain paths, for example:
#   "/checkpoints/qwen3-8b/step230/actor/merged_hf"
#   "gdrive:Distant-Association/qwen3-14b/step230/actor/merged_hf"
CHECKPOINTS=(
  #"gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-1.7b-20260521-225347/global_step_230/actor/merged_hf"
  #"Qwen/Qwen3-1.7B"

  # 4B
  #"gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-4b-20260522-174546/global_step_230/actor/merged_hf"
  #"Qwen/Qwen3-4B"

  # 8B
  "gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-8b-20260521-220104/global_step_230/actor/merged_hf"
  "gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-8b-20260521-220104/global_step_56/actor/merged_hf"
  "gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-8b-20260521-220104/global_step_112/actor/merged_hf"
  "gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-8b-20260521-220104/global_step_168/actor/merged_hf"
  "gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-8b-20260521-220104/global_step_224/actor/merged_hf"
  "Qwen/Qwen3-8B"

  # 14B
  #"gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-14b-20260904-023756/global_step_230/actor/merged_hf"
  #"Qwen/Qwen3-14B"
)

if [[ ! -f "${EVALUATOR}" ]]; then
  echo "ERROR: evaluator not found: ${EVALUATOR}" >&2
  exit 1
fi
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "ERROR: CHECKPOINTS is empty" >&2
  exit 1
fi
if [[ ! "${TENSOR_PARALLEL_SIZE}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "ERROR: TENSOR_PARALLEL_SIZE must be 0 (auto) or a positive integer" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Running ${#CHECKPOINTS[@]} checkpoints sequentially"
echo "Output directory: ${OUTPUT_DIR}"

for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ -z "${checkpoint}" ]]; then
    echo "ERROR: CHECKPOINTS contains an empty path" >&2
    exit 1
  fi
  label="$("${PYTHON_BIN}" -m scripts.checkpoint_utils short-name "${checkpoint}")"
  if [[ ! "${label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: could not derive a safe output label from: ${checkpoint}" >&2
    exit 1
  fi

  output="${OUTPUT_DIR}/${label}.jsonl"
  echo
  echo "======================================================================"
  echo "Checkpoint: ${label}"
  echo "Model:      ${checkpoint}"
  if [[ "${TENSOR_PARALLEL_SIZE}" == "0" ]]; then
    echo "TP size:    auto (all visible GPUs)"
  else
    echo "TP size:    ${TENSOR_PARALLEL_SIZE}"
  fi
  echo "Output:     ${output}"
  echo "======================================================================"

  "${PYTHON_BIN}" "${EVALUATOR}" \
    --model "${checkpoint}" \
    --model-label "${label}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --output "${output}" \
    "$@"
done

echo
echo "Completed all ${#CHECKPOINTS[@]} checkpoints"
