#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Train an LLM with DAPO on Codenames. The reward function dispatches
# per-row based on extra_info["task"]:
#   - clue task: judge-LLM scoring OR GloVe-cosine fallback
#   - guess task: pure rule-based scoring, no external resources
#
# Three modes selected by JUDGE_MODEL_ID:
#   - JUDGE_MODEL_ID="" / "none" (DEFAULT)  -> no judge
#       * clue rows score via GloVe cosine against extra_info["clue"]
#       * guess rows score by rules; GloVe artifacts are not consulted
#       * all GPUs go to training
#   - JUDGE_MODEL_ID=<hf id>                 -> launch judge vLLM server
#       * clue rows score via judge HTTP; guess rows still skip the judge
#       * GPU split N_TRAIN_GPUS / N_JUDGE_GPUS
#
# Pipeline diagram (judge mode)
#
#     +---------------------------+        HTTP         +--------------------+
#     |  VeRL driver + Ray head   |  ---------------->  |  vLLM judge server |
#     |  Qwen3 FSDP actor         |   OpenAI /v1/chat   |  judge model       |
#     |  + vLLM rollout (TP=N)    |                     |  (TP=JUDGE_TP)     |
#     |  GPUs 0..N_TRAIN_GPUS-1   |                     |  GPUs N_TRAIN..N-1 |
#     +---------------------------+                     +--------------------+
#
# Key design points (see .claude/rules/model-based-reward.md):
#   1. Judge lives OUTSIDE Ray, on its own GPUs, its own Python env.
#   2. Custom async compute_score is Phase-2 concurrent because VeRL's
#      experimental reward_loop fans out run_single via asyncio.gather.
#   3. reward_model.enable=False — we do NOT use VeRL's discriminative
#      RewardModelWorker; the generative judge is reached purely via HTTP.
# -----------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Paths and identities — edit for your environment
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-5/codenames_rlvr_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-5/codenames_rlvr_val.parquet}"
REWARD_FN_PATH="${REPO_ROOT}/custom_reward_functions/codenames_reward.py"

TRAINEE_MODEL_ID="${TRAINEE_MODEL_ID:-Qwen/Qwen3-1.7B}"
TRAINEE_MODEL_PATH="${TRAINEE_MODEL_PATH:-${TRAINEE_MODEL_ID}}"

# Default: no judge. Clue rows score via GloVe-cosine, guess rows score
# by rules. Override with JUDGE_MODEL_ID=<hf id> to launch the judge.
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-qwen3-judge}"

case "${JUDGE_MODEL_ID,,}" in
  ""|none|null) USE_JUDGE=0 ;;
  *)            USE_JUDGE=1 ;;
esac

# -----------------------------------------------------------------------------
# 1. GPU split — training pool vs. judge pool
#    With a judge: start Ray with ALL GPUs visible, but only advertise the
#    training ones to VeRL via trainer.n_gpus_per_node. Pin the judge to the
#    rest via CUDA_VISIBLE_DEVICES when we launch it.
#    Without a judge (cosine mode): give all GPUs to training.
# -----------------------------------------------------------------------------
TOTAL_GPUS="${TOTAL_GPUS:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
JUDGE_TP="${JUDGE_TP:-2}"

if [ "${USE_JUDGE}" = "1" ]; then
  N_TRAIN_GPUS="${N_TRAIN_GPUS:-6}"        # VeRL sees only these
  N_JUDGE_GPUS="${N_JUDGE_GPUS:-2}"        # reserved for the judge
  TRAIN_GPU_IDS=$(seq -s, 0 $((N_TRAIN_GPUS - 1)))
  JUDGE_GPU_IDS=$(seq -s, ${N_TRAIN_GPUS} $((TOTAL_GPUS - 1)))
  echo "[layout] mode=judge   train GPUs=${TRAIN_GPU_IDS}   judge GPUs=${JUDGE_GPU_IDS}"
else
  N_TRAIN_GPUS="${TOTAL_GPUS}"
  N_JUDGE_GPUS=0
  TRAIN_GPU_IDS=$(seq -s, 0 $((N_TRAIN_GPUS - 1)))
  JUDGE_GPU_IDS=""
  echo "[layout] mode=cosine  train GPUs=${TRAIN_GPU_IDS}   (no judge)"
fi

# -----------------------------------------------------------------------------
# 2. Launch the judge vLLM server (Approach A).
#    IMPORTANT: this is an independent process — a separate conda/venv is
#    strongly recommended so its vLLM version (>=0.9 for Qwen3 MoE) does not
#    collide with the VeRL-pinned vLLM used by the rollout HybridEngine.
# -----------------------------------------------------------------------------
JUDGE_LOG="${JUDGE_LOG:-/tmp/codenames_judge.log}"
JUDGE_PIDFILE="${JUDGE_PIDFILE:-/tmp/codenames_judge.pid}"

launch_judge() {
  echo "[judge] starting vLLM server for ${JUDGE_MODEL_ID} on GPUs ${JUDGE_GPU_IDS}"
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
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "[judge] ready after ${i}s"
      return 0
    fi
    sleep 2
  done
  echo "[judge] healthcheck timed out — see ${JUDGE_LOG}" >&2
  return 1
}

stop_judge() {
  if [ -f "${JUDGE_PIDFILE}" ]; then
    local pid
    pid=$(cat "${JUDGE_PIDFILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[judge] stopping pid=${pid}"
      kill "${pid}" || true
      wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${JUDGE_PIDFILE}"
  fi
}
if [ "${USE_JUDGE}" = "1" ]; then
  trap stop_judge EXIT

  # Skip judge launch if the user already has one running (set SKIP_JUDGE=1).
  if [ "${SKIP_JUDGE:-0}" != "1" ]; then
    launch_judge
    wait_for_judge
  else
    echo "[judge] SKIP_JUDGE=1 — assuming judge is already up at ${JUDGE_HOST}:${JUDGE_PORT}"
  fi

  # Reward function reads these env vars (see custom_reward_functions/judge_client.py)
  export JUDGE_URL="http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions"
  export JUDGE_NAME="${JUDGE_SERVED_NAME}"
  export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-128}"
  export JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-0}"
  export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-256}"
  export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
else
  # Judge disabled. Unset JUDGE_NAME so cosine_reward.is_cosine_mode()
  # flips on for any clue rows in the dataset. Guess rows never read
  # these envs, so a guess-only run works regardless of GloVe state.
  unset JUDGE_NAME JUDGE_URL

  # GLOVE_SRC is the raw text file; .npy and _vocab.pkl are derived
  # artifacts produced by scripts/build_glove_lookup.py.  When this
  # script is run on a fresh GPU server, the artifacts won't exist —
  # we build them on the fly if the source text is present.  The
  # build script is idempotent (re-running on fresh artifacts costs
  # ~100ms).
  # Prefer the raw .txt if it's already extracted (faster — one
  # decompression pass less); otherwise fall back to the .zip which
  # the build script can stream from directly without extracting.
  GLOVE_TXT="${REPO_ROOT}/custom_data/glove_vectors/dolma_300_2024_1.2M.100_combined.txt"
  GLOVE_ZIP="${REPO_ROOT}/custom_data/glove_vectors/glove.2024.dolma.300d.zip"
  if [ -z "${GLOVE_SRC:-}" ]; then
    if [ -f "${GLOVE_TXT}" ]; then
      GLOVE_SRC="${GLOVE_TXT}"
    else
      GLOVE_SRC="${GLOVE_ZIP}"
    fi
  fi
  # Canonical artifact stem — same regardless of source (.txt or .zip),
  # so the .npy/.pkl cache is reusable across servers.
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
      echo "[cosine]        expected one of:" >&2
      echo "[cosine]          ${GLOVE_TXT}" >&2
      echo "[cosine]          ${GLOVE_ZIP}" >&2
      echo "[cosine]        Provide the .txt or .zip, or override GLOVE_SRC=<path>." >&2
      echo "[cosine]        (To run with judge instead, set JUDGE_MODEL_ID=<hf id>.)" >&2
      exit 1
    fi
  fi
  # Post-build verification — the build must have produced both artifacts.
  if [ ! -f "${GLOVE_NPY_PATH}" ] || [ ! -f "${GLOVE_VOCAB_PATH}" ]; then
    echo "[cosine] ERROR: build_glove_lookup.py finished but artifacts are still missing:" >&2
    echo "[cosine]          ${GLOVE_NPY_PATH}" >&2
    echo "[cosine]          ${GLOVE_VOCAB_PATH}" >&2
    exit 1
  fi
fi

# Make `custom_reward_functions` importable when VeRL loads the reward file.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# NCCL 2.27+ defaults NCCL_CUMEM_ENABLE=1, which routes buffer allocation through
# the CUDA VMM driver API (cuMemCreate). On vast.ai containers running driver
# 590.48.01 / RTX 5090, the host-side path (ncclCuMemHostEnable) segfaults inside
# cuMemCreate during the first FSDP broadcast, killing one rank at init. Force
# the classic cudaMalloc path instead.
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# -----------------------------------------------------------------------------
# 3. DAPO hyperparameters (mirrors tests/special_e2e/run_dapo.sh)
# -----------------------------------------------------------------------------
adv_estimator=grpo

kl_coef=0.0
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=2048
max_response_length=16384
enable_overlong_buffer=True
overlong_buffer_len=4096
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10

train_traj_micro_bsz_per_gpu=1
n_resp_per_prompt=8

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * N_TRAIN_GPUS))
train_traj_mini_bsz=$((train_traj_micro_bsz * 2))
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt))
train_prompt_bsz=$((train_prompt_mini_bsz * 2))
gen_prompt_bsz=$((train_prompt_bsz * 4))

total_epochs=4

# Save ~4 checkpoints per run (every 25%). DAPO with filter_groups consumes
# `gen_prompt_bsz` from the dataloader per step (not `train_prompt_bsz`) —
# it over-samples, rolls out, filters low-variance groups, then trains on
# a `train_prompt_bsz` subset. If filter_groups resamples (up to
# max_num_gen_batches), actual step count is LOWER than this estimate,
# which only makes save_freq fire more often — fine.
dataset_rows=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${TRAIN_PARQUET}').num_rows)")
total_train_steps=$(( (dataset_rows * total_epochs + gen_prompt_bsz - 1) / gen_prompt_bsz ))
save_freq=$(( total_train_steps / 4 ))
[ "${save_freq}" -lt 1 ] && save_freq=1
echo "[ckpt] dataset_rows=${dataset_rows} total_train_steps=${total_train_steps} save_freq=${save_freq}"

EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-codenames-dapo-$(basename "${TRAINEE_MODEL_ID,,}")}"
EXP_NAME="${EXP_NAME_PREFIX}-$(date +%Y%m%d-%H%M%S)"

# -----------------------------------------------------------------------------
# 4. Launch DAPO training.
#    Key reward-related flags:
#      reward_model.enable=False                 # no discriminative RM worker
#      reward.reward_manager.name=dapo           # DAPO overlong-buffer aware
#      reward.custom_reward_function.path=...    # our async compute_score
#      reward.custom_reward_function.name=compute_score
# -----------------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size=${train_prompt_bsz} \
    +data.gen_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    reward_model.enable=False \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    actor_rollout_ref.model.path="${TRAINEE_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.project_name='Distant-Association' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.logger='[console,wandb]' \
    trainer.n_gpus_per_node=${N_TRAIN_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    "$@"
