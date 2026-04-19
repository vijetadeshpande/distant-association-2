#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Train an LLM with DAPO on Codenames, tuned for a single node of 8x A4000.
#
# Hardware assumption:
#   - 8x NVIDIA RTX A4000 (16 GB each, 128 GB aggregate)
#   - trainee = Qwen/Qwen3-8B (bf16)
#   - judge   = Qwen/Qwen3-14B (bitsandbytes 4-bit, frozen)
#
# Memory budget (per 16 GB A4000):
#   Qwen3-8B bf16 weights  = 16.4 GB; Adam fp32 states = 32 GB; grads = 16 GB.
#   Sharded over 6 GPUs that is ~10.7 GB/GPU — already over budget once you
#   add activations and vLLM KV-cache on the same cards.  Crucially, the 16.4
#   GB bf16 footprint is *already* larger than the 15.6 GB usable on one A4000,
#   which means FSDP v1 OOMs at init: its broadcast requires rank 0 to
#   materialize the full model on GPU before sharding.  So we MUST:
#     - actor.strategy = fsdp2  (shards BEFORE broadcast via DTensor; rank 0
#         only moves its ~2.7 GB shard to GPU — fits on A4000)
#     - fsdp_config.offload_policy = True  (FSDP2 CPU-offload; replaces the
#         post-init param_offload/optimizer_offload flags)
#     - rollout tensor_model_parallel_size = 2 (halves the 8B weights per
#         GPU during generation)
#     - shrink rollout gpu_memory_utilization so FSDP + vLLM coexist
#     - shrink prompt/response lengths and per-GPU micro-batch to 1
#     - run the judge on its own 2 GPUs (TP=2) with a tighter max_model_len
#
# Layout (8 GPUs total):
#   GPUs 0..5  -> trainee pool (FSDP + vLLM rollout, HybridEngine)
#   GPUs 6..7  -> judge vLLM server (bitsandbytes 4-bit, TP=2)
# -----------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Paths and identities
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-4/codenames_rlvr_clue_gen_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${REPO_ROOT}/custom_data/training_prompts/version-4/codenames_rlvr_clue_gen_val.parquet}"
REWARD_FN_PATH="${REPO_ROOT}/custom_reward_functions/codenames_reward.py"

TRAINEE_MODEL_ID="${TRAINEE_MODEL_ID:-Qwen/Qwen3-8B}"
TRAINEE_MODEL_PATH="${TRAINEE_MODEL_PATH:-${TRAINEE_MODEL_ID}}"

JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-Qwen/Qwen3-14B}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-qwen3-judge}"

# -----------------------------------------------------------------------------
# 1. GPU split — 6 training / 2 judge
# -----------------------------------------------------------------------------
TOTAL_GPUS="${TOTAL_GPUS:-8}"
N_TRAIN_GPUS="${N_TRAIN_GPUS:-6}"
N_JUDGE_GPUS="${N_JUDGE_GPUS:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"    # Qwen3-8B: 32 Q heads, 8 KV heads -> TP=2 ok
JUDGE_TP="${JUDGE_TP:-2}"        # Qwen3-14B: 40 Q heads, 8 KV heads -> TP=2 ok

TRAIN_GPU_IDS=$(seq -s, 0 $((N_TRAIN_GPUS - 1)))
JUDGE_GPU_IDS=$(seq -s, ${N_TRAIN_GPUS} $((TOTAL_GPUS - 1)))

echo "[layout] train GPUs=${TRAIN_GPU_IDS}   judge GPUs=${JUDGE_GPU_IDS}"

# -----------------------------------------------------------------------------
# 2. Launch judge vLLM server.
#    A4000 is 16 GB — keep max-model-len and max-num-seqs small so KV cache
#    does not starve the 4-bit weights.
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
      --max-model-len 8192 \
      --gpu-memory-utilization 0.85 \
      --enable-chunked-prefill \
      --max-num-seqs 16 \
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
trap stop_judge EXIT

if [ "${SKIP_JUDGE:-0}" != "1" ]; then
  launch_judge
  wait_for_judge
else
  echo "[judge] SKIP_JUDGE=1 — assuming judge is already up at ${JUDGE_HOST}:${JUDGE_PORT}"
fi

export JUDGE_URL="http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions"
export JUDGE_NAME="${JUDGE_SERVED_NAME}"
# Judge only has ~16 seq slots and 2 A4000s; don't flood it with 128 requests.
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-32}"
export JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-0}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-256}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# 3. DAPO hyperparameters — shrunk for 16 GB A4000s.
#    Sequence lengths: 2048 prompt + 2048 response (was 2048 + 4096).
#    Response shrink is the single biggest activation + KV-cache win.
# -----------------------------------------------------------------------------
adv_estimator=grpo

kl_coef=0.0
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=2048
max_response_length=2048
enable_overlong_buffer=True
overlong_buffer_len=128
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10

# Per-GPU micro batch must stay at 1 on 16 GB for an 8B model.
train_traj_micro_bsz_per_gpu=1
# 4 samples per prompt keeps GRPO group statistics meaningful while halving
# rollout KV-cache pressure vs. the 8-sample default.
n_resp_per_prompt=4

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * N_TRAIN_GPUS))   # 6
train_traj_mini_bsz=${train_traj_micro_bsz}                             # 6  (1 micro/mini, no grad-accum)
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt))      # 24
train_prompt_bsz=${train_prompt_mini_bsz}                               # 24 (1 mini per prompt-batch)
gen_prompt_bsz=$((train_prompt_bsz * 2))                                # 48

EXP_NAME="codenames-dapo-$(basename "${TRAINEE_MODEL_ID,,}")-8xA4000"

# -----------------------------------------------------------------------------
# 4. Launch DAPO training.
#
# A4000-specific overrides vs. the H100 script:
#   actor.strategy=fsdp2 (+ ref via ${actor.strategy})
#     -> FSDP v1 OOMs at init on 16 GB A4000 because rank 0 must put the
#        full 8B bf16 model (16.4 GB) on GPU to broadcast.  FSDP2 shards
#        via DTensor before broadcast, so rank 0 only ever holds its ~2.7
#        GB shard on GPU.  This is THE fix for the init OOM.
#   actor.fsdp_config.offload_policy=True
#     -> FSDP2's native CPUOffloadPolicy. Replaces the v1 param_offload /
#        optimizer_offload flags (they get auto-disabled when FSDP2 owns
#        offload — see fsdp_workers.py:620-623).
#   rollout.gpu_memory_utilization=0.55
#     -> leaves headroom for FSDP shards + activations to coexist with the
#        vLLM HybridEngine on the same cards. 0.80 (the H100 value) OOMs.
#   rollout.max_num_batched_tokens / max_num_seqs -> small, to keep the KV
#     cache inside the remaining budget.
#   Batch sizes: halved from the first A4000 draft after the init-OOM fix,
#     to give post-init rollout + backward activations extra headroom
#     (your priority #1: batch size).
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
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.offload_policy=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.project_name='Distant-Association' \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger='[console,wandb]' \
    trainer.n_gpus_per_node=${N_TRAIN_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=2 \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    "$@"
