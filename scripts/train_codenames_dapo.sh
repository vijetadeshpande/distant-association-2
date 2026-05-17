#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Train an LLM with DAPO on Codenames.
#
# Three clue-scoring modes (guess rows always use rule-based scoring):
#
#   JUDGE_MODEL_ID=""  (default)   → GloVe cosine similarity, no external calls
#   JUDGE_MODEL_ID=<id>
#     JUDGE_BACKEND=openrouter     → judge via OpenRouter API  (default)
#     JUDGE_BACKEND=local          → judge via local vLLM server on reserved GPUs
#
# GPU layout for local judge:
#
#     +---------------------------+        HTTP         +--------------------+
#     |  VeRL driver + Ray head   |  ----------------> |  vLLM judge server |
#     |  Qwen3 FSDP actor         |   OpenAI /v1/chat  |  (TP=JUDGE_TP)     |
#     |  + vLLM rollout (TP=N)    |                    |  GPUs N_TRAIN..N-1 |
#     |  GPUs 0..N_TRAIN_GPUS-1   |                    +--------------------+
#     +---------------------------+
#
# Fixed training hyperparameters live in codenames_dapo.yaml (same folder).
# This script handles only environment setup and values that depend on
# GPU count, file paths, or are computed at launch time.
# -----------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# Fixed hyperparameters (held constant across runs; do not override via env)
# -----------------------------------------------------------------------------
MICRO_BSZ_PER_GPU=1        # micro-batch trajectories per GPU
N_RESP_PER_PROMPT=32       # rollout group size (responses per prompt)
TRAIN_PROMPT_BSZ=32        # global prompts per training step
                           # → 32 * N_RESP_PER_PROMPT = 1024 trajectories/step
                           # constraint: (TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT)
                           #             must be divisible by N_TRAIN_GPUS

# -----------------------------------------------------------------------------
# 0. Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_YAML="${SCRIPT_DIR}/codenames_dapo.yaml"
cd "${REPO_ROOT}"

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-5/codenames_rlvr_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-5/codenames_rlvr_val.parquet}"
REWARD_FN_PATH="${REPO_ROOT}/custom_reward_functions/codenames_reward.py"

TRAINEE_MODEL_ID="${TRAINEE_MODEL_ID:-Qwen/Qwen3-8B}"
TRAINEE_MODEL_PATH="${TRAINEE_MODEL_PATH:-${TRAINEE_MODEL_ID}}"

# Judge settings
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-google/gemini-2.5-flash-lite}"          # empty = cosine mode
JUDGE_BACKEND="${JUDGE_BACKEND:-openrouter}"  # "openrouter" or "local"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-qwen3-judge}"  # --served-model-name for local vLLM

case "${JUDGE_MODEL_ID,,}" in
  ""|none|null) USE_JUDGE=0 ;;
  *)            USE_JUDGE=1 ;;
esac

# GPU reservation is only needed when the judge runs on local GPUs.
[ "${USE_JUDGE}" = "1" ] && [ "${JUDGE_BACKEND}" = "local" ] \
  && USE_LOCAL_JUDGE=1 || USE_LOCAL_JUDGE=0

# -----------------------------------------------------------------------------
# 1. GPU layout
# -----------------------------------------------------------------------------
TOTAL_GPUS="$(nvidia-smi -L | wc -l)"
[ "${TOTAL_GPUS}" -lt 1 ] && { echo "[gpu] no GPUs detected via nvidia-smi -L" >&2; exit 1; }
ROLLOUT_TP="${ROLLOUT_TP:-2}"

if [ "${USE_LOCAL_JUDGE}" = "1" ]; then
  JUDGE_TP="${JUDGE_TP:-2}"
  N_TRAIN_GPUS="${N_TRAIN_GPUS:-6}"
  N_JUDGE_GPUS="${N_JUDGE_GPUS:-2}"
  TRAIN_GPU_IDS=$(seq -s, 0 $((N_TRAIN_GPUS - 1)))
  JUDGE_GPU_IDS=$(seq -s, ${N_TRAIN_GPUS} $((TOTAL_GPUS - 1)))
  echo "[layout] backend=local       train=${TRAIN_GPU_IDS}   judge=${JUDGE_GPU_IDS}"
else
  N_TRAIN_GPUS="${TOTAL_GPUS}"
  JUDGE_GPU_IDS=""
  mode=$( [ "${USE_JUDGE}" = "1" ] && echo "openrouter" || echo "cosine" )
  echo "[layout] mode=${mode}   all ${N_TRAIN_GPUS} GPUs → training"
fi

# -----------------------------------------------------------------------------
# 2. Local judge server  (skipped for openrouter and cosine modes)
# -----------------------------------------------------------------------------
JUDGE_LOG="${JUDGE_LOG:-/tmp/codenames_judge.log}"
JUDGE_PIDFILE="${JUDGE_PIDFILE:-/tmp/codenames_judge.pid}"

launch_judge() {
  echo "[judge] starting vLLM for ${JUDGE_MODEL_ID} on GPUs ${JUDGE_GPU_IDS}"
  CUDA_VISIBLE_DEVICES="${JUDGE_GPU_IDS}" \
  nohup vllm serve "${JUDGE_MODEL_ID}" \
      --quantization bitsandbytes \
      --load-format bitsandbytes \
      --tensor-parallel-size "${JUDGE_TP}" \
      --dtype bfloat16 \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.80 \
      --enable-chunked-prefill \
      --max-num-seqs 32 \
      --served-model-name "${JUDGE_SERVED_NAME}" \
      --host "${JUDGE_HOST}" \
      --port "${JUDGE_PORT}" \
      > "${JUDGE_LOG}" 2>&1 &
  echo $! > "${JUDGE_PIDFILE}"
  echo "[judge] pid=$(cat "${JUDGE_PIDFILE}"), log=${JUDGE_LOG}"
}

wait_for_judge() {
  local url="http://${JUDGE_HOST}:${JUDGE_PORT}/v1/models"
  echo "[judge] waiting for healthcheck at ${url}"
  for i in $(seq 1 180); do
    if curl -fsS "${url}" >/dev/null 2>&1; then echo "[judge] ready after ${i}s"; return 0; fi
    sleep 2
  done
  echo "[judge] healthcheck timed out — see ${JUDGE_LOG}" >&2; return 1
}

stop_judge() {
  if [ -f "${JUDGE_PIDFILE}" ]; then
    local pid; pid=$(cat "${JUDGE_PIDFILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[judge] stopping pid=${pid}"
      kill "${pid}" || true; wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${JUDGE_PIDFILE}"
  fi
}

if [ "${USE_LOCAL_JUDGE}" = "1" ]; then
  trap stop_judge EXIT
  if [ "${SKIP_JUDGE:-0}" != "1" ]; then
    launch_judge
    wait_for_judge
  else
    echo "[judge] SKIP_JUDGE=1 — assuming judge is already up at ${JUDGE_HOST}:${JUDGE_PORT}"
  fi
  # Override the default URL so the reward function reaches this server.
  export JUDGE_URL="http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions"
fi

# -----------------------------------------------------------------------------
# 3. GloVe artifacts  (cosine mode only)
# -----------------------------------------------------------------------------
if [ "${USE_JUDGE}" = "0" ]; then
  GLOVE_TXT="${REPO_ROOT}/custom_data/glove_vectors/dolma_300_2024_1.2M.100_combined.txt"
  GLOVE_ZIP="${REPO_ROOT}/custom_data/glove_vectors/glove.2024.dolma.300d.zip"
  if [ -z "${GLOVE_SRC:-}" ]; then
    if [ -f "${GLOVE_TXT}" ]; then GLOVE_SRC="${GLOVE_TXT}"; else GLOVE_SRC="${GLOVE_ZIP}"; fi
  fi

  GLOVE_STEM="${REPO_ROOT}/custom_data/glove_vectors/dolma_300_2024_1.2M.100_combined"
  export GLOVE_NPY_PATH="${GLOVE_NPY_PATH:-${GLOVE_STEM}.npy}"
  export GLOVE_VOCAB_PATH="${GLOVE_VOCAB_PATH:-${GLOVE_STEM}_vocab.pkl}"
  echo "[cosine] GLOVE_NPY_PATH=${GLOVE_NPY_PATH}"

  if [ ! -f "${GLOVE_NPY_PATH}" ] || [ ! -f "${GLOVE_VOCAB_PATH}" ]; then
    if [ -f "${GLOVE_SRC}" ]; then
      echo "[cosine] artifacts missing — building from ${GLOVE_SRC}"
      python3 "${REPO_ROOT}/scripts/build_glove_lookup.py" \
          "${GLOVE_SRC}" --output-stem "${GLOVE_STEM}"
    else
      echo "[cosine] ERROR: GloVe artifacts missing AND source not found." >&2
      echo "[cosine]   expected: ${GLOVE_TXT}" >&2
      echo "[cosine]         or: ${GLOVE_ZIP}" >&2
      echo "[cosine]   Override GLOVE_SRC=<path> or set JUDGE_MODEL_ID to use a judge." >&2
      exit 1
    fi
  fi
  if [ ! -f "${GLOVE_NPY_PATH}" ] || [ ! -f "${GLOVE_VOCAB_PATH}" ]; then
    echo "[cosine] ERROR: build finished but artifacts are still missing:" >&2
    echo "[cosine]   ${GLOVE_NPY_PATH}" >&2; echo "[cosine]   ${GLOVE_VOCAB_PATH}" >&2
    exit 1
  fi
fi

# Make custom_reward_functions importable when VeRL loads the reward file.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# NCCL 2.27+: disable VMM path that segfaults on vast.ai containers (RTX 5090 / driver 590).
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# -----------------------------------------------------------------------------
# 4. Batch sizes  (TRAIN_PROMPT_BSZ, MICRO_BSZ_PER_GPU, N_RESP_PER_PROMPT
#                 are fixed at the top of this script)
# -----------------------------------------------------------------------------
total_trajectories=$((TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT))
if (( total_trajectories % N_TRAIN_GPUS != 0 )); then
  echo "[batch] ERROR: TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT (${total_trajectories})" >&2
  echo "[batch]        is not divisible by N_TRAIN_GPUS (${N_TRAIN_GPUS})." >&2
  exit 1
fi
PPO_MINI_BSZ=${TRAIN_PROMPT_BSZ}
GEN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}
echo "[batch] prompts/step=${TRAIN_PROMPT_BSZ}  trajectories/step=${total_trajectories}  micro/gpu=${MICRO_BSZ_PER_GPU}"

# Checkpoint frequency: ~16 saves per run. Read epoch count from yaml so the
# two sources of truth stay in sync.
TOTAL_EPOCHS=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_YAML}'))['trainer']['total_epochs'])")
dataset_rows=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${TRAIN_PARQUET}').num_rows)")
total_steps=$(( (dataset_rows * TOTAL_EPOCHS + GEN_PROMPT_BSZ - 1) / GEN_PROMPT_BSZ ))
save_freq=$(( total_steps / 16 )); [ "${save_freq}" -lt 1 ] && save_freq=1
echo "[ckpt] epochs=${TOTAL_EPOCHS}  rows=${dataset_rows}  steps≈${total_steps}  save_freq=${save_freq}"

EXP_NAME="${EXP_NAME_PREFIX:-codenames-dapo-$(basename "${TRAINEE_MODEL_ID,,}")}-$(date +%Y%m%d-%H%M%S)"

# -----------------------------------------------------------------------------
# 5. Judge reward kwargs  (empty array = cosine mode, no overrides needed)
# -----------------------------------------------------------------------------
judge_overrides=()
if [ "${USE_JUDGE}" = "1" ]; then
  # Tune these for OpenRouter rate limits or local GPU capacity.
  export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-128}"
  export JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-0}"
  export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-256}"
  export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"

  # Model name sent in the API payload differs by backend:
  #   local     → the --served-model-name vLLM was launched with
  #   openrouter → the full OpenRouter model id (e.g. "openai/gpt-4o")
  if [ "${JUDGE_BACKEND}" = "local" ]; then
    JUDGE_PAYLOAD_MODEL="${JUDGE_SERVED_NAME}"
  else
    JUDGE_PAYLOAD_MODEL="${JUDGE_MODEL_ID}"
  fi
  judge_overrides+=("+reward.reward_kwargs.judge_model=${JUDGE_PAYLOAD_MODEL}")
  judge_overrides+=("+reward.reward_kwargs.judge_backend=${JUDGE_BACKEND}")
fi

# -----------------------------------------------------------------------------
# 6. Launch DAPO training
#    Fixed hyperparameters come from codenames_dapo.yaml via --config-name.
#    Only dynamic / environment-specific values are passed as CLI overrides.
# -----------------------------------------------------------------------------
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
    "${judge_overrides[@]}" \
    trainer.n_gpus_per_node=${N_TRAIN_GPUS} \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_freq=${save_freq} \
    "$@"
