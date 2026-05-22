#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Train Qwen3-4B with DAPO on Codenames — local judge CO-LOCATED with training.
#
# train_codenames_dapo.sh (JUDGE_BACKEND=local) PARTITIONS the GPUs: some run
# the trainee, the rest run the judge. Because VeRL's loop is synchronous
# (rollout -> reward -> update run in series) each half sits idle while the
# other works — roughly 50% of the silicon is dark at any instant.
#
# This script instead CO-LOCATES both models on EVERY GPU:
#
#     +-----------------------------------------------------------+
#     |  GPUs 0..N-1   (ALL GPUs)                                 |
#     |                                                           |
#     |   VeRL driver + Ray + Qwen3-4B FSDP actor                 |
#     |   + vLLM rollout engine   (DP = N/ROLLOUT_TP, TP = ..._TP) |
#     |                                                           |
#     |   vLLM judge server (Qwen3-4B, TP = N)  <--- HTTP ------+  |
#     +-----------------------------------------------------------+
#
# The two engines stay memory-resident at the same time (a 4B model + a 4B
# judge is small next to 96 GB cards), but VeRL's synchronous loop means they
# never COMPUTE at the same instant — so they time-share the GPUs for free.
# Net effect: rollout, reward and update each use all N GPUs instead of N/2
# (~1.8x throughput) with no idle silicon and no sleep/wake orchestration.
#
# The judge is a FROZEN Qwen3-4B (the base checkpoint — it is NOT updated by
# training) reached over HTTP, and runs the guess-generation task only. It is
# a THINKING judge: Qwen3-4B emits <think>...</think> by default, so nothing
# in judge_client.py needs to change. Judge sizing (JUDGE_MAX_TOKENS,
# --max-model-len, KV fraction) is taken from measured Qwen3-4B output
# lengths — see custom_reward_functions/tests/local_judge_length_test.md.
#
# Fixed training hyperparameters live in codenames_dapo.yaml (same folder);
# this script handles environment setup and GPU/path-dependent values only.
# -----------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# Debug mode: smoke-test the full pipeline on a tiny slice of the data.
#   - subsamples TRAIN_PROMPT_BSZ rows from the training parquet
#   - forces total_epochs=1, save_freq=1, val_before_train=True
#   - dumps rollout + validation generations to ${REPO_ROOT}/debug_dumps/<run>
#   - disables algorithm.filter_groups (oversampling would loop over the slice)
# A debug run is ALSO the right way to calibrate the memory split below:
# both vLLM engines reserve their KV pools up front, so OOM (or spare room)
# shows up on the first step. Enable with `DEBUG=1` or `debug`/`--debug`.
# -----------------------------------------------------------------------------
DEBUG="${DEBUG:-0}"
if [ "${1:-}" = "debug" ] || [ "${1:-}" = "--debug" ]; then
  DEBUG=1
  shift
fi

# -----------------------------------------------------------------------------
# Fixed hyperparameters (held constant across runs; do not override via env)
# -----------------------------------------------------------------------------
MICRO_BSZ_PER_GPU=8        # micro-batch trajectories per GPU
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

# Auto-load .env (WANDB_API_KEY etc). Gitignored. `set -a` exports every var
# assigned while sourced; restore the prior allexport state afterward.
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a; . "${REPO_ROOT}/.env"; set +a
fi

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-5/codenames_rlvr_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-5/codenames_rlvr_val.parquet}"
REWARD_FN_PATH="${REPO_ROOT}/custom_reward_functions/codenames_reward.py"

# Debug: subsample training parquet to TRAIN_PROMPT_BSZ rows.
if [ "${DEBUG}" = "1" ]; then
  DEBUG_TRAIN_PARQUET="${DEBUG_TRAIN_PARQUET:-/tmp/codenames_debug_train.parquet}"
  echo "[debug] subsampling ${TRAIN_PROMPT_BSZ} rows from ${TRAIN_PARQUET}"
  python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('${TRAIN_PARQUET}').slice(0, ${TRAIN_PROMPT_BSZ})
pq.write_table(t, '${DEBUG_TRAIN_PARQUET}')
print(f'[debug] wrote {t.num_rows} rows -> ${DEBUG_TRAIN_PARQUET}')
"
  TRAIN_PARQUET="${DEBUG_TRAIN_PARQUET}"
fi

# -----------------------------------------------------------------------------
# Models — trainee and judge are the SAME architecture. The judge is a frozen
# copy of the base checkpoint (never updated by training).
# -----------------------------------------------------------------------------
TRAINEE_MODEL_ID="${TRAINEE_MODEL_ID:-Qwen/Qwen3-4B}"
TRAINEE_MODEL_PATH="${TRAINEE_MODEL_PATH:-${TRAINEE_MODEL_ID}}"

JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-Qwen/Qwen3-4B}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-qwen3-judge}"   # --served-model-name
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"
JUDGE_PORT="${JUDGE_PORT:-8000}"

# -----------------------------------------------------------------------------
# 1. GPU layout — CO-LOCATED: trainee and judge both span every GPU.
# -----------------------------------------------------------------------------
TOTAL_GPUS="$(nvidia-smi -L | wc -l)"
[ "${TOTAL_GPUS}" -lt 1 ] && { echo "[gpu] no GPUs detected via nvidia-smi -L" >&2; exit 1; }

N_TRAIN_GPUS="${TOTAL_GPUS}"                        # trainee FSDP + rollout: all GPUs
JUDGE_GPU_IDS="$(seq -s, 0 $((TOTAL_GPUS - 1)))"    # judge server: all GPUs
ROLLOUT_TP="${ROLLOUT_TP:-2}"                       # rollout: TP=2 → DP = N/2
JUDGE_TP="${JUDGE_TP:-${TOTAL_GPUS}}"               # judge tensor-parallel = all GPUs
                                                    # (must be valid for Qwen3-4B:
                                                    #  8 KV heads → TP ∈ {1,2,4,8})
echo "[layout] CO-LOCATED  train+judge on GPUs ${JUDGE_GPU_IDS}  (rollout TP=${ROLLOUT_TP}, judge TP=${JUDGE_TP})"

# -----------------------------------------------------------------------------
# 2. Memory split  (both vLLM engines are resident on every GPU at once)
#
# vLLM's --gpu-memory-utilization is a fraction of TOTAL card memory, and its
# profiler accounts for memory already held by OTHER processes. The judge is
# launched FIRST and reserves JUDGE_MEM_UTIL; VeRL's rollout engine then comes
# up under ROLLOUT_MEM_UTIL. What is left covers the FSDP actor — its resident
# footprint is small because param/optimizer offload is enabled in the yaml.
#
# These two numbers are the only knobs that need empirical calibration:
# run `debug` once and watch `nvidia-smi`. OOM → lower ROLLOUT_MEM_UTIL;
# lots of free memory → raise it.
# -----------------------------------------------------------------------------
JUDGE_MEM_UTIL="${JUDGE_MEM_UTIL:-0.20}"      # judge weights (~2 GiB/GPU) + KV cache
ROLLOUT_MEM_UTIL="${ROLLOUT_MEM_UTIL:-0.55}"  # trainee vLLM rollout engine

# -----------------------------------------------------------------------------
# 3. Judge sizing — from custom_reward_functions/tests/local_judge_length_test.md
#
# A thinking Qwen3-4B judge on Codenames boards of 4-12 words (the full v5
# train + val range) produced: p99 completion ≈ 1967 tokens, max ≈ 2090,
# with 0.6% degenerate loops. JUDGE_MAX_TOKENS=8192 gives ~4x headroom over
# the real max while capping a runaway loop far below the old 32768 default.
# -----------------------------------------------------------------------------
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-8192}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-9216}"   # judge prompt (≤400) + 8192 + margin
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-128}"      # match JUDGE_CONCURRENCY
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-128}" # reward-loop fan-out semaphore
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
echo "[judge] sizing  max_tokens=${JUDGE_MAX_TOKENS}  max_model_len=${JUDGE_MAX_MODEL_LEN}  concurrency=${JUDGE_CONCURRENCY}"
echo "[mem]   judge_util=${JUDGE_MEM_UTIL}  rollout_util=${ROLLOUT_MEM_UTIL}"

# -----------------------------------------------------------------------------
# 4. Local judge server  (Qwen3-4B, bf16, thinking)
# -----------------------------------------------------------------------------
JUDGE_LOG="${JUDGE_LOG:-/tmp/codenames_judge.log}"
JUDGE_PIDFILE="${JUDGE_PIDFILE:-/tmp/codenames_judge.pid}"

launch_judge() {
  # bf16, NO quantization: co-location removes the memory pressure that
  # motivated bitsandbytes, and bf16 decodes faster — which matters for a
  # thinking judge generating ~1k tokens/call. Qwen3-4B emits <think>...
  # </think> by default, so this is a thinking judge with no extra flags.
  echo "[judge] starting vLLM ${JUDGE_MODEL_ID} (bf16, thinking) on GPUs ${JUDGE_GPU_IDS}, TP=${JUDGE_TP}"
  CUDA_VISIBLE_DEVICES="${JUDGE_GPU_IDS}" \
  nohup vllm serve "${JUDGE_MODEL_ID}" \
      --tensor-parallel-size "${JUDGE_TP}" \
      --dtype bfloat16 \
      --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${JUDGE_MEM_UTIL}" \
      --enable-chunked-prefill \
      --max-num-seqs "${JUDGE_MAX_NUM_SEQS}" \
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

# -----------------------------------------------------------------------------
# 4b. Checkpoint → Google Drive sync daemon  (best-effort; never aborts training)
# -----------------------------------------------------------------------------
SYNC_SCRIPT="${SCRIPT_DIR}/sync_checkpoints_to_gdrive.sh"
SYNC_CHECKPOINTS="${SYNC_CHECKPOINTS:-1}"   # set 0 to disable Drive sync
SYNC_STARTED=0

start_checkpoint_sync() {
  if [ "${SYNC_CHECKPOINTS}" != "1" ]; then
    echo "[ckpt-sync] SYNC_CHECKPOINTS=${SYNC_CHECKPOINTS} — Drive sync disabled"
    return 0
  fi
  if [ "${DEBUG}" = "1" ]; then
    echo "[ckpt-sync] debug run — skipping Drive sync"
    return 0
  fi
  if bash "${SYNC_SCRIPT}" start; then
    SYNC_STARTED=1
  else
    echo "[ckpt-sync] WARN: sync daemon failed to start — training continues" >&2
  fi
}

stop_checkpoint_sync() {
  [ "${SYNC_STARTED}" = "1" ] || return 0
  echo "[ckpt-sync] stopping sync daemon (it runs one final upload first)"
  bash "${SYNC_SCRIPT}" stop || true
}

# Single EXIT handler covering every teardown path. Each step is guarded so
# the trap itself can never abort the script or mask training's exit code.
cleanup() {
  stop_judge || true
  stop_checkpoint_sync || true
}
trap cleanup EXIT

# Bring the judge up before training so the reward function can reach it.
if [ "${SKIP_JUDGE:-0}" != "1" ]; then
  launch_judge
  wait_for_judge
else
  echo "[judge] SKIP_JUDGE=1 — assuming judge is already up at ${JUDGE_HOST}:${JUDGE_PORT}"
fi
export JUDGE_URL="http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions"

# Make custom_reward_functions importable when VeRL loads the reward file.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# NCCL 2.27+: disable VMM path that segfaults on some containers/drivers.
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# -----------------------------------------------------------------------------
# 5. Batch sizes  (TRAIN_PROMPT_BSZ, MICRO_BSZ_PER_GPU, N_RESP_PER_PROMPT
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

# Checkpoint frequency: ~8 saves per run. Read epoch count from yaml so the
# two sources of truth stay in sync.
TOTAL_EPOCHS=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_YAML}'))['trainer']['total_epochs'])")
[ "${DEBUG}" = "1" ] && TOTAL_EPOCHS=1
dataset_rows=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${TRAIN_PARQUET}').num_rows)")
total_steps=$(( (dataset_rows * TOTAL_EPOCHS + GEN_PROMPT_BSZ - 1) / GEN_PROMPT_BSZ ))
save_freq=$(( total_steps / 8 )); [ "${save_freq}" -lt 1 ] && save_freq=1
echo "[ckpt] epochs=${TOTAL_EPOCHS}  rows=${dataset_rows}  steps≈${total_steps}  save_freq=${save_freq}"

EXP_NAME="${EXP_NAME_PREFIX:-codenames-dapo-colocated-$(basename "${TRAINEE_MODEL_ID,,}")}-$(date +%Y%m%d-%H%M%S)"
[ "${DEBUG}" = "1" ] && EXP_NAME="debug-${EXP_NAME}"

# -----------------------------------------------------------------------------
# Debug overrides: force a single training step, run validation, dump both.
# filter_groups is disabled because oversampling would re-iterate the slice.
# -----------------------------------------------------------------------------
debug_overrides=()
if [ "${DEBUG}" = "1" ]; then
  DEBUG_DUMP_DIR="${REPO_ROOT}/debug_dumps/${EXP_NAME}"
  mkdir -p "${DEBUG_DUMP_DIR}/rollout" "${DEBUG_DUMP_DIR}/validation"
  echo "[debug] dump dir=${DEBUG_DUMP_DIR}"

  # Val set = last N_TRAIN_GPUS rows of the original val parquet (one per GPU).
  DEBUG_VAL_PARQUET="${DEBUG_VAL_PARQUET:-/tmp/codenames_debug_val.parquet}"
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

# -----------------------------------------------------------------------------
# 6. Judge reward kwargs
#
# These MUST go under reward.custom_reward_function.reward_kwargs (note the
# leading `+`: the keys are not in the yaml). VeRL routes that block into
# compute_score's kwargs; reward.reward_kwargs is the reward MANAGER's and
# would leave compute_score with judge_model=None → silent cosine fallback.
#
# judge_model = the --served-model-name the local vLLM was launched with.
# judge_thinking=true records intent; on the local backend Qwen3-4B's own
# default-thinking is what actually drives it (see judge_client.py).
# -----------------------------------------------------------------------------
judge_overrides=(
  "+reward.custom_reward_function.reward_kwargs.judge_model=${JUDGE_SERVED_NAME}"
  "+reward.custom_reward_function.reward_kwargs.judge_backend=local"
  "+reward.custom_reward_function.reward_kwargs.judge_thinking=true"
)

# -----------------------------------------------------------------------------
# 7. Launch DAPO training
#    Fixed hyperparameters come from codenames_dapo.yaml via --config-name.
#    rollout.gpu_memory_utilization is overridden here (the yaml default of
#    0.6 assumes the trainee owns the whole card; co-location needs room for
#    the resident judge).
# -----------------------------------------------------------------------------
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
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    "${judge_overrides[@]}" \
    "${debug_overrides[@]}" \
    trainer.n_gpus_per_node=${N_TRAIN_GPUS} \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_freq=${save_freq} \
    "$@"
