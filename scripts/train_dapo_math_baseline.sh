#!/usr/bin/env bash
# Train Qwen3-8B with DAPO on the curated DAPO-Math baseline.
#
# Fixed training hyperparameters are intentionally shared with
# train_codenames_dapo.sh through codenames_dapo.yaml.  This launcher changes
# only the model requested for this run, the train/validation data, and the
# reward implementation.  DAPO-Math uses VeRL's built-in data-source dispatcher:
#
#   data_source=math_dapo -> verl.utils.reward_score.math_dapo.compute_score
#
# Therefore reward.custom_reward_function.path must remain unset.  In
# particular, do not point it at custom_reward_functions/codenames_reward.py.
set -euo pipefail

DEBUG="${DEBUG:-0}"
if [ "${1:-}" = "debug" ] || [ "${1:-}" = "--debug" ]; then
  DEBUG=1
  shift
fi

# Keep these values identical to train_codenames_dapo.sh.
MICRO_BSZ_PER_GPU=4
N_RESP_PER_PROMPT=32
TRAIN_PROMPT_BSZ=32

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_YAML="${SCRIPT_DIR}/codenames_dapo.yaml"
DATA_DIR="${REPO_ROOT}/custom_data/training_prompts/baseline-dapo-math"
cd "${REPO_ROOT}"

TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/dapo_math_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/dapo_math_val.parquet}"
TRAINEE_MODEL_ID="${TRAINEE_MODEL_ID:-Qwen/Qwen3-8B}"
TRAINEE_MODEL_REVISION="${TRAINEE_MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"

for required_file in "${CONFIG_YAML}" "${TRAIN_PARQUET}" "${VAL_PARQUET}"; do
  if [ ! -f "${required_file}" ]; then
    echo "[setup] required file not found: ${required_file}" >&2
    exit 1
  fi
done

python3 - "${TRAIN_PARQUET}" "${VAL_PARQUET}" <<'PY'
import re
import sys

import pyarrow.parquet as pq

required_columns = {"prompt", "data_source", "reward_model", "extra_info"}
for path in sys.argv[1:]:
    table = pq.read_table(path)
    missing = required_columns - set(table.column_names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for index, row in enumerate(table.to_pylist()):
        if row["data_source"] != "math_dapo":
            raise ValueError(f"{path}: row {index} has data_source={row['data_source']!r}")
        ground_truth = row["reward_model"].get("ground_truth")
        if not isinstance(ground_truth, str) or re.fullmatch(r"[+-]?\d+", ground_truth) is None:
            raise ValueError(f"{path}: row {index} has invalid integer ground truth {ground_truth!r}")
        if [message["role"] for message in row["prompt"]] != ["system", "user"]:
            raise ValueError(f"{path}: row {index} does not contain one system and one user message")
    print(f"[data] validated {table.num_rows} math_dapo rows in {path}")
PY

if [ -z "${TRAINEE_MODEL_PATH:-}" ]; then
  echo "[model] resolving ${TRAINEE_MODEL_ID} at revision ${TRAINEE_MODEL_REVISION}"
  TRAINEE_MODEL_PATH="$(
    python3 - "${TRAINEE_MODEL_ID}" "${TRAINEE_MODEL_REVISION}" <<'PY'
import sys

from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
  )"
fi

# A debug run is one real optimizer step on a 32-prompt slice.  It is useful
# after the inference-only smoke check, but is not run automatically.
if [ "${DEBUG}" = "1" ]; then
  DEBUG_TRAIN_PARQUET="${DEBUG_TRAIN_PARQUET:-/tmp/dapo_math_debug_train.parquet}"
  echo "[debug] subsampling ${TRAIN_PROMPT_BSZ} rows from ${TRAIN_PARQUET}"
  python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('${TRAIN_PARQUET}').slice(0, ${TRAIN_PROMPT_BSZ})
pq.write_table(t, '${DEBUG_TRAIN_PARQUET}')
print(f'[debug] wrote {t.num_rows} rows -> ${DEBUG_TRAIN_PARQUET}')
"
  TRAIN_PARQUET="${DEBUG_TRAIN_PARQUET}"
fi

TOTAL_GPUS="$(nvidia-smi -L | wc -l)"
[ "${TOTAL_GPUS}" -lt 1 ] && { echo "[gpu] no GPUs detected via nvidia-smi -L" >&2; exit 1; }
N_TRAIN_GPUS="${TOTAL_GPUS}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
echo "[layout] all ${N_TRAIN_GPUS} GPUs -> training"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

total_trajectories=$((TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT))
if (( total_trajectories % N_TRAIN_GPUS != 0 )); then
  echo "[batch] ERROR: TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT (${total_trajectories})" >&2
  echo "[batch]        is not divisible by N_TRAIN_GPUS (${N_TRAIN_GPUS})." >&2
  exit 1
fi
PPO_MINI_BSZ=${TRAIN_PROMPT_BSZ}
GEN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}
echo "[batch] prompts/step=${TRAIN_PROMPT_BSZ}  trajectories/step=${total_trajectories}  micro/gpu=${MICRO_BSZ_PER_GPU}"

TOTAL_EPOCHS=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_YAML}'))['trainer']['total_epochs'])")
[ "${DEBUG}" = "1" ] && TOTAL_EPOCHS=1
dataset_rows=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${TRAIN_PARQUET}').num_rows)")
total_steps=$(( (dataset_rows * TOTAL_EPOCHS + GEN_PROMPT_BSZ - 1) / GEN_PROMPT_BSZ ))
save_freq=$(( total_steps / 8 )); [ "${save_freq}" -lt 1 ] && save_freq=1
echo "[ckpt] epochs=${TOTAL_EPOCHS}  rows=${dataset_rows}  steps~${total_steps}  save_freq=${save_freq}"

EXP_NAME="${EXP_NAME_PREFIX:-dapo-math-qwen3-8b}-$(date +%Y%m%d-%H%M%S)"
[ "${DEBUG}" = "1" ] && EXP_NAME="debug-${EXP_NAME}"

debug_overrides=()
if [ "${DEBUG}" = "1" ]; then
  DEBUG_DUMP_DIR="${REPO_ROOT}/debug_dumps/${EXP_NAME}"
  mkdir -p "${DEBUG_DUMP_DIR}/rollout" "${DEBUG_DUMP_DIR}/validation"
  echo "[debug] dump dir=${DEBUG_DUMP_DIR}"

  DEBUG_VAL_PARQUET="${DEBUG_VAL_PARQUET:-/tmp/dapo_math_debug_val.parquet}"
  echo "[debug] taking last ${N_TRAIN_GPUS} rows from ${VAL_PARQUET}"
  python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('${VAL_PARQUET}')
t = t.slice(max(t.num_rows - ${N_TRAIN_GPUS}, 0), ${N_TRAIN_GPUS})
pq.write_table(t, '${DEBUG_VAL_PARQUET}')
print(f'[debug] wrote {t.num_rows} rows -> ${DEBUG_VAL_PARQUET}')
"
  VAL_PARQUET="${DEBUG_VAL_PARQUET}"
  debug_overrides+=(
    "trainer.total_epochs=1"
    "trainer.val_before_train=True"
    "trainer.test_freq=1"
    "trainer.rollout_data_dir=${DEBUG_DUMP_DIR}/rollout"
    "trainer.validation_data_dir=${DEBUG_DUMP_DIR}/validation"
    "trainer.log_val_generations=8"
    "algorithm.filter_groups.enable=False"
  )
fi

# Match the Codenames launcher's best-effort checkpoint mirroring behavior.
SYNC_SCRIPT="${SCRIPT_DIR}/sync_checkpoints_to_gdrive.sh"
SYNC_CHECKPOINTS="${SYNC_CHECKPOINTS:-1}"
SYNC_STARTED=0

start_checkpoint_sync() {
  if [ "${SYNC_CHECKPOINTS}" != "1" ]; then
    echo "[ckpt-sync] SYNC_CHECKPOINTS=${SYNC_CHECKPOINTS} - Drive sync disabled"
    return 0
  fi
  if [ "${DEBUG}" = "1" ]; then
    echo "[ckpt-sync] debug run - skipping Drive sync"
    return 0
  fi
  if bash "${SYNC_SCRIPT}" start; then
    SYNC_STARTED=1
  else
    echo "[ckpt-sync] WARN: sync daemon failed to start - training continues" >&2
  fi
}

stop_checkpoint_sync() {
  [ "${SYNC_STARTED}" = "1" ] || return 0
  echo "[ckpt-sync] stopping sync daemon (it runs one final upload first)"
  bash "${SYNC_SCRIPT}" stop || true
}
trap stop_checkpoint_sync EXIT

start_checkpoint_sync

python3 -m verl.trainer.main_ppo \
    --config-path "${REPO_ROOT}/scripts" \
    --config-name codenames_dapo \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    +data.gen_batch_size=${GEN_PROMPT_BSZ} \
    actor_rollout_ref.model.path="${TRAINEE_MODEL_PATH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    "${debug_overrides[@]}" \
    trainer.n_gpus_per_node=${N_TRAIN_GPUS} \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_freq=${save_freq} \
    "$@"
