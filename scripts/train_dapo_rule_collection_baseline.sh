#!/usr/bin/env bash
# Train Qwen3-8B with DAPO on the curated RuleCollection baseline.
#
# Fixed training hyperparameters are intentionally shared with
# train_codenames_dapo.sh and train_dapo_math_baseline.sh through
# codenames_dapo.yaml. This launcher changes only the required model, data,
# experiment name, and reward implementation:
#
#   custom_reward_functions/rule_collection_reward.py:compute_score
set -euo pipefail

DEBUG="${DEBUG:-0}"
if [ "${1:-}" = "debug" ] || [ "${1:-}" = "--debug" ]; then
  DEBUG=1
  shift
fi

# Keep these values identical to the two reference launchers.
MICRO_BSZ_PER_GPU=8
N_RESP_PER_PROMPT=32
TRAIN_PROMPT_BSZ=32

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_YAML="${SCRIPT_DIR}/codenames_dapo.yaml"
DATA_DIR="${REPO_ROOT}/custom_data/training_prompts/baseline-rule-collection"
REWARD_FN_PATH="${REPO_ROOT}/custom_reward_functions/rule_collection_reward.py"
cd "${REPO_ROOT}"

TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/rule_collection_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/rule_collection_val.parquet}"

# The experiment is intentionally restricted to the same pinned Qwen3-8B
# revision used to curate both baseline datasets.
TRAINEE_MODEL_ID="Qwen/Qwen3-8B"
TRAINEE_MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"

for required_file in "${CONFIG_YAML}" "${TRAIN_PARQUET}" "${VAL_PARQUET}" "${REWARD_FN_PATH}"; do
  if [ ! -f "${required_file}" ]; then
    echo "[setup] required file not found: ${required_file}" >&2
    exit 1
  fi
done

# Fail before allocating GPUs if the parquet-to-reward contract is broken.
python3 - "${TRAIN_PARQUET}" "${VAL_PARQUET}" <<'PY'
import sys

import pyarrow.parquet as pq

from custom_reward_functions.rule_collection_reward import compute_score, parse_answer

domain_labels = {
    "AR-LSAT": {"a", "b", "c", "d", "e"},
    "Folio": {"true", "false", "unknown"},
    "Logic NLI": {"contradiction", "self_contradiction", "neutral", "entailment"},
    "Logical Deduction": {"a", "b", "c", "d", "e", "f", "g"},
    "ProntoQA": {"true", "false", "unknown"},
    "ProofWriter": {"true", "false", "unknown"},
}
required_columns = {"prompt", "data_source", "reward_model", "extra_info"}

for path in sys.argv[1:]:
    table = pq.read_table(path)
    if set(table.column_names) != required_columns:
        raise ValueError(
            f"{path}: expected exactly {sorted(required_columns)}, got {sorted(table.column_names)}"
        )
    for index, row in enumerate(table.to_pylist()):
        source = row["data_source"]
        if source not in domain_labels:
            raise ValueError(f"{path}: row {index} has unsupported data_source={source!r}")
        if [message.get("role") for message in row["prompt"]] != ["system", "user"]:
            raise ValueError(f"{path}: row {index} does not contain one system and one user message")

        extra_info = row["extra_info"] or {}
        legal_labels = {
            str(label).strip().casefold() for label in (extra_info.get("legal_labels") or [])
        }
        if not legal_labels or not legal_labels.issubset(domain_labels[source]):
            raise ValueError(f"{path}: row {index} has invalid legal_labels={sorted(legal_labels)}")

        ground_truth = (row["reward_model"] or {}).get("ground_truth")
        gold, error = parse_answer(ground_truth)
        if error is not None or gold not in legal_labels:
            raise ValueError(f"{path}: row {index} has invalid ground truth {ground_truth!r}")
        result = compute_score(source, ground_truth, ground_truth, extra_info)
        if result["score"] != 1.0:
            raise ValueError(f"{path}: row {index} fails its gold reward check: {result}")
    print(f"[data] validated {table.num_rows} RuleCollection rows in {path}")
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

# A debug run is one real optimizer step on a 32-prompt slice. Run the
# inference-only smoke test first when validating a new environment.
if [ "${DEBUG}" = "1" ]; then
  DEBUG_TRAIN_PARQUET="${DEBUG_TRAIN_PARQUET:-/tmp/rule_collection_debug_train.parquet}"
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

# Resolve the two trainer values used by the shell-side checkpoint estimate.
# Hydra applies repeated CLI overrides last-wins, but parsing them here keeps
# this status line in sync and lets the final launch pass each value once.
TOTAL_EPOCHS=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_YAML}'))['trainer']['total_epochs'])")
[ "${DEBUG}" = "1" ] && TOTAL_EPOCHS=1
CLI_SAVE_FREQ=""
passthrough_overrides=()
for override in "$@"; do
  case "${override}" in
    trainer.total_epochs=*|+trainer.total_epochs=*|++trainer.total_epochs=*)
      TOTAL_EPOCHS="${override#*=}"
      ;;
    trainer.save_freq=*|+trainer.save_freq=*|++trainer.save_freq=*)
      CLI_SAVE_FREQ="${override#*=}"
      ;;
    *)
      passthrough_overrides+=("${override}")
      ;;
  esac
done
if ! [[ "${TOTAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ckpt] ERROR: trainer.total_epochs must be a positive integer, got ${TOTAL_EPOCHS}" >&2
  exit 1
fi
if [ -n "${CLI_SAVE_FREQ}" ] && ! [[ "${CLI_SAVE_FREQ}" =~ ^-?[0-9]+$ ]]; then
  echo "[ckpt] ERROR: trainer.save_freq must be an integer, got ${CLI_SAVE_FREQ}" >&2
  exit 1
fi

dataset_rows=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${TRAIN_PARQUET}').num_rows)")
total_steps=$(( (dataset_rows * TOTAL_EPOCHS + GEN_PROMPT_BSZ - 1) / GEN_PROMPT_BSZ ))
if [ -n "${CLI_SAVE_FREQ}" ]; then
  save_freq="${CLI_SAVE_FREQ}"
else
  save_freq=$(( total_steps / 8 )); [ "${save_freq}" -lt 1 ] && save_freq=1
fi
echo "[ckpt] epochs=${TOTAL_EPOCHS}  rows=${dataset_rows}  steps~${total_steps}  save_freq=${save_freq}"

EXP_NAME="${EXP_NAME_PREFIX:-rule-collection-qwen3-8b}-$(date +%Y%m%d-%H%M%S)"
[ "${DEBUG}" = "1" ] && EXP_NAME="debug-${EXP_NAME}"

debug_overrides=()
if [ "${DEBUG}" = "1" ]; then
  DEBUG_DUMP_DIR="${REPO_ROOT}/debug_dumps/${EXP_NAME}"
  mkdir -p "${DEBUG_DUMP_DIR}/rollout" "${DEBUG_DUMP_DIR}/validation"
  echo "[debug] dump dir=${DEBUG_DUMP_DIR}"

  DEBUG_VAL_PARQUET="${DEBUG_VAL_PARQUET:-/tmp/rule_collection_debug_val.parquet}"
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

# Match the reference launchers' best-effort checkpoint mirroring behavior.
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
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    "${debug_overrides[@]}" \
    trainer.n_gpus_per_node=${N_TRAIN_GPUS} \
    trainer.experiment_name="${EXP_NAME}" \
    "${passthrough_overrides[@]}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${save_freq}
