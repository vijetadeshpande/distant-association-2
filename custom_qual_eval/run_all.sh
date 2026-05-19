#!/usr/bin/env bash
# Run qualitative inference on a fixed list of models against the v5 val parquet.
# Output: custom_qual_eval/outputs/<slug>.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-5/codenames_rlvr_val.parquet}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PER_GROUP="${PER_GROUP:-3}"   # 8 (task x difficulty) combos x 3 = 24 examples
SAMPLE_PARQUET="${SAMPLE_PARQUET:-${OUT_DIR}/_sample_seed${SAMPLE_SEED}.parquet}"
ALL="${ALL:-0}"               # ALL=1 -> run inference on the FULL val parquet

mkdir -p "${OUT_DIR}"

if [ "${ALL}" = "1" ]; then
  # Skip pre-sampling — point every model at the full parquet.
  EVAL_PARQUET="${VAL_PARQUET}"
  SLUG_SUFFIX="-all"
  echo "=== ALL=1: running on every row of ${EVAL_PARQUET} ==="
else
  # Pre-sample once: PER_GROUP rows per (task, difficulty_rule). Every model
  # run below reads this exact file, so all checkpoints are evaluated on
  # identical prompts. Delete the file (or bump SAMPLE_SEED) to redraw.
  if [ ! -f "${SAMPLE_PARQUET}" ]; then
    echo "=== materialising fixed sample -> ${SAMPLE_PARQUET} ==="
    python3 "${SCRIPT_DIR}/run_eval.py" \
        --data "${VAL_PARQUET}" \
        --data-source parquet \
        --per-group "${PER_GROUP}" \
        --seed "${SAMPLE_SEED}" \
        --build-sample-to "${SAMPLE_PARQUET}"
  else
    echo "=== reusing fixed sample at ${SAMPLE_PARQUET} ==="
  fi
  EVAL_PARQUET="${SAMPLE_PARQUET}"
  SLUG_SUFFIX=""
fi

# For an HF dataset, build the fixed sample the same way once and point the
# loop below at it. Example:
#   python3 run_eval.py --data <hf-dataset-name> --data-source hf \
#       --hf-split validation --hf-n 24 --seed 42 \
#       --build-sample-to outputs/_hf_sample_seed42.parquet
# (Use --hf-n 0 to take ALL rows of the HF split.)

# (slug, model spec) pairs. Add more checkpoints here later.
MODELS=(
  "qwen3-8b|Qwen/Qwen3-8B"
  "codenames-dapo-qwen3-8b-step112|gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-8b-20260517-052154/global_step_112/actor/merged_hf"
  "codenames-dapo-qwen3-8b-step56|gdrive-distant-association:distant-association-drive/checkpoints/Distant-Association/codenames-dapo-qwen3-8b-20260517-052154/global_step_56/actor/merged_hf"
)

for entry in "${MODELS[@]}"; do
  slug="${entry%%|*}"
  model="${entry#*|}"
  out="${OUT_DIR}/${slug}${SLUG_SUFFIX}.jsonl"
  echo "=== ${slug} -> ${out} ==="
  python3 "${SCRIPT_DIR}/run_eval.py" \
      --model "${model}" \
      --data "${EVAL_PARQUET}" \
      --data-source parquet \
      --use-all-rows \
      --output "${out}" \
      "$@"
done
