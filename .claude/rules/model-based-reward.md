# Wiring an LLM judge into VeRL 0.7.1 — a concrete recipe

**The cleanest way to implement a model-based reward in VeRL 0.7.1 is to run the Qwen3-30B-A3B-Instruct-2507 judge as an independent AWQ-4bit vLLM server on dedicated GPUs and call it via HTTP from a `custom_reward_function`.** VeRL 0.7.1's in-tree `RewardModelWorker` is a *discriminative* classifier (SequenceClassification head), not a generative model, and the in-repo "vLLM-based generative reward model" work (PR #2845, then the Reward Loop / `genrm_remote` recipe) was only partially landed and experimental on that tag. Your custom reward function can still do everything you need — parse `s1/s2/s3` from the rollout, build a judge prompt, call the judge, parse `s4`, and return a scalar — as long as you (a) pick an AWQ-4bit checkpoint (not bitsandbytes), (b) override VeRL's reward manager to call the judge concurrently, and (c) carve GPUs out of Ray's pool for the judge. The rest of this report walks through each decision, gives the exact config keys, and supplies drop-in code.

## What VeRL 0.7.1 actually gives you

VeRL separates two concerns that are easy to confuse. A **reward function** is a per-sample callable with the signature `compute_score(data_source, solution_str, ground_truth, extra_info) -> float | dict`. A **RewardManager** is the class that iterates a `DataProto` batch, calls that function, and places scalar scores onto a `(batch, response_length)` token-level tensor (score deposited on the last valid response token). The two are wired up in `verl/trainer/ppo/reward.py` via `get_custom_reward_fn` and `load_reward_manager`, invoked from `verl/trainer/main_ppo.py`, and the reward is computed on the **driver process** after rollout but before advantage estimation — it is *not* automatically parallelized across Ray workers.

The built-in managers live in `verl/workers/reward_manager/` and are registered by name: **`naive`** (default per-sample loop), **`prime`** (ProcessPoolExecutor-based, async-capable, but requires picklable functions), **`batch`** (passes whole lists to the reward fn), and **`dapo`** (adds `overlong_buffer_cfg` length penalty). Selecting one is a one-line config change: `reward_model.reward_manager=naive`. You can also register your own with `@register("my_rm")` against `AbstractRewardManager`, which is the path you will want here.

Two config blocks control the reward pipeline. `custom_reward_function.{path, name, reward_kwargs}` points at a user Python file and a function inside it (default name `compute_score`); `reward_kwargs` here is merged into every per-sample call. Separately, `reward_model.{enable, reward_manager, reward_kwargs, model.path}` controls the discriminative RM worker and the manager class — and critically, **`reward_model.enable=False` still lets the custom_reward_function run**. The two paths are not mutually exclusive in principle but you do not need `reward_model.enable=True` for an LLM-judge setup; turning it on in 0.7.1 would try to load a classifier model, which is not what you want.

Data flows through `non_tensor_batch` on the `DataProto` object. Each sample exposes `data_source` (string tag), `reward_model["ground_truth"]` (from your parquet preprocessing), and a freeform `extra_info` dict that you populate in `make_map_fn` during dataset construction. Inside `compute_score`, `solution_str` is the already-detokenized model response (special tokens stripped, padding trimmed). This is the hook where you will parse `s1/s2/s3`.

## Why bitsandbytes is the wrong 4-bit choice here

Qwen3-30B-A3B is a **Mixture-of-Experts** model (48 layers, 128 experts, top-8 routing, 3.3B active / 30.5B total), and **vLLM does not implement bitsandbytes 4-bit inference for MoE models** (tracked in vllm-project/vllm issue #20480, still open; issue #32012 specifically shows `Qwen3-VL-30B-A3B` failing with bnb). So `--quantization bitsandbytes --load-format bitsandbytes` will fail or silently misbehave on this checkpoint. Official Qwen only publishes BF16 and FP8 weights for the `-Instruct-2507` variant; **no first-party AWQ exists**. Community AWQ-4bit mirrors that are known-good against vLLM ≥0.9 include `cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit`, `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ`, and the GPTQ-Int4 `btbtyler09/Qwen3-30B-A3B-Instruct-2507-gptq-4bit`. Use `--quantization awq_marlin` for the AWQ mirrors — Marlin kernels give the best throughput on Ampere/Hopper. FP8 (`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`) is an excellent 8-bit alternative on H100/L40S if strict 4-bit is not a hard requirement.

A bonus of the `-Instruct-2507` checkpoint: **it is non-thinking by design**. The model card explicitly states it "supports only non-thinking mode and does not generate `<think></think>` blocks... specifying `enable_thinking=False` is no longer required." You do not need `/no_think`, you do not need `enable_thinking=False` in the chat template, and you do not need `--reasoning-parser deepseek_r1` (that flag targets the *-Thinking-2507* sibling). Your `s4` parser can regex directly over the final answer with no reasoning-tag handling.

## Choosing among three integration approaches

**Approach A — external vLLM OpenAI-compatible server (recommended).** Launch `vllm serve …` once, outside Ray, with `CUDA_VISIBLE_DEVICES` pinning it to GPUs VeRL is not using. Your custom reward function makes async HTTP calls to `/v1/chat/completions`. Zero modifications to VeRL internals; the judge process lives in its own Python env so it can even run a newer vLLM version (0.9+) than the one VeRL 0.7.1 pins for its rollout. Latency overhead is ~1 ms on LAN; vLLM's continuous batching drives 20–50 judge calls/sec per H100 for short prompts. Resilient to Ray restarts, trivially scalable to N replicas behind a round-robin proxy.

**Approach B — `vllm.LLM(...)` in-process inside the RewardManager.** Do not do this. The RewardManager runs on the driver (Ray head controller), which has CPU-only resources in VeRL's default placement group; creating a second, independent `LLM()` there will either fail to find a GPU or collide with VeRL's monkey-patched `vllm.distributed.parallel_state` in `verl/third_party/vllm/`. Even if you force the driver onto a GPU node, the 30B judge cannot share GPUs with the 8B actor + rollout HybridEngine.

**Approach C — Ray actor wrapping vLLM.** Viable as a second choice. Spawn a `@ray.remote(num_gpus=2, lifetime="detached") class JudgeActor` *before* constructing `RayPPOTrainer`, and keep its GPUs outside VeRL's resource pool by shrinking `trainer.n_gpus_per_node`. Cleaner lifecycle than Approach A (tied to the Ray cluster), but more code, and Ray-level TP initialization inside the actor occasionally fights VeRL's vllm monkey-patching. Worth it only if you dislike managing a standalone server process.

## The concurrency trap that will halve your training throughput

**VeRL 0.7.1's `NaiveRewardManager` calls `compute_score` in a Python for-loop.** Even if your judge server supports 256-way concurrent decoding, the loop serializes it to concurrency-1. This is reported as open issues #1507 ("concurrent reward-model calls") and #2236 ("parallelizing rewards across 2 vLLM servers") in the VeRL tracker. The fix is a ~30-line custom RewardManager that `asyncio.gather`s calls across the batch. VeRL does detect `async def compute_score` via `inspect.iscoroutinefunction` in `_call_with_kwargs_async`, and the `prime` manager will parallelize via `ProcessPoolExecutor` — but `prime` requires the reward function and all its closures to be picklable, which conflicts with holding an open `aiohttp.ClientSession`. The safest route is to register a custom async manager. The experimental **Reward Loop** (`verl/experimental/reward/` in 0.7+) does exactly this natively via `RewardLoopManager` + `reward_router_address`; if you want to avoid writing your own manager, you can lift that file into your 0.7.1 checkout, but it was labeled experimental on this tag.

## Recommended layout on one 8×H100 node

| GPUs 0–5 (6 GPUs, VeRL's pool) | GPUs 6–7 (2 GPUs, reserved) |
|---|---|
| Qwen3-8B FSDP actor + ref + vLLM rollout (TP=2), HybridEngine sleep/wake | Qwen3-30B-A3B-Instruct-2507 AWQ-4bit judge (TP=2), persistent |

Peak memory on the training GPUs is ~50 GB each (FSDP bf16 + optimizer + rollout); the judge consumes ~12 GB weights + ~20 GB KV cache on its two reserved cards — comfortable on 80 GB H100s with `--max-model-len 16384`. Start Ray with all 8 GPUs visible, set `trainer.n_gpus_per_node=6` so VeRL only reserves six, and launch the judge with `CUDA_VISIBLE_DEVICES=6,7`. For MoE throughput, pass `--enable-expert-parallel` on vLLM ≥0.9 when TP ≥2, and keep TP as a divisor of both attention-head counts (32 Q / 4 KV) and expert count (128) — **TP=2 or TP=4 are safe; avoid TP=8**.

## Launch command for the judge

```bash
CUDA_VISIBLE_DEVICES=6,7 vllm serve cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \
    --quantization awq_marlin --tensor-parallel-size 2 --dtype float16 \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill --max-num-seqs 256 \
    --served-model-name qwen3-judge --host 0.0.0.0 --port 8000
```

## The custom reward file, end to end

Save this as `/workspace/judge_reward.py`. It implements the seven-step workflow you described: parse `s1/s2/s3` out of the rollout, build a judge prompt from `s2` and `s3`, call the judge, parse `s4`, and run `s4` through a rule-based scorer. Because the default manager serializes, the file also registers a custom async manager that fans out calls with `asyncio.gather`.

```python
# /workspace/judge_reward.py
import asyncio, aiohttp, re, os
from typing import Any
import torch
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

JUDGE_URL  = os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1/chat/completions")
JUDGE_NAME = os.environ.get("JUDGE_NAME", "qwen3-judge")
MAX_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "128"))

S1_RE = re.compile(r"<s1>(.*?)</s1>", re.S)
S2_RE = re.compile(r"<s2>(.*?)</s2>", re.S)
S3_RE = re.compile(r"<s3>(.*?)</s3>", re.S)
S4_RE = re.compile(r"<s4>(.*?)</s4>", re.S)   # judge outputs s4 in this format

JUDGE_TEMPLATE = """You are an expert evaluator.

Context (s2):
{s2}

Candidate (s3):
{s3}

Produce your verdict in the exact format <s4>...</s4>. No explanation."""

def _parse_rollout(solution_str: str):
    s1 = (S1_RE.search(solution_str) or [None, ""])[1].strip() if S1_RE.search(solution_str) else ""
    s2 = (S2_RE.search(solution_str).group(1).strip() if S2_RE.search(solution_str) else "")
    s3 = (S3_RE.search(solution_str).group(1).strip() if S3_RE.search(solution_str) else "")
    return s1, s2, s3

def _rule_based_reward(s4: str, ground_truth: str) -> float:
    # Your rule-based scorer on s4. Example: exact match / numeric / regex.
    if not s4:
        return 0.0
    return float(s4.strip().lower() == str(ground_truth).strip().lower())

async def _judge_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                     s2: str, s3: str) -> str:
    if not s2 or not s3:
        return ""
    payload = {
        "model": JUDGE_NAME,
        "messages": [{"role": "user", "content": JUDGE_TEMPLATE.format(s2=s2, s3=s3)}],
        "temperature": 0.0,
        "max_tokens": 512,
        # Qwen3-Instruct-2507 is non-thinking by design — no extra flags needed.
    }
    async with sem:
        async with session.post(JUDGE_URL, json=payload, timeout=600) as r:
            data = await r.json()
    text = data["choices"][0]["message"]["content"]
    m = S4_RE.search(text)
    return m.group(1).strip() if m else text.strip()

async def _score_batch_async(samples):
    """samples: list of (solution_str, ground_truth) — returns list of (score, s4)."""
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        parsed = [_parse_rollout(sol) for sol, _ in samples]     # (s1, s2, s3)
        tasks  = [_judge_one(session, sem, s2, s3) for (_, s2, s3) in parsed]
        s4s    = await asyncio.gather(*tasks)
    return [(_rule_based_reward(s4, gt), s4) for s4, (_, gt) in zip(s4s, samples)]

# -------- per-sample function (used if you keep the default naive manager) ------
async def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    s1, s2, s3 = _parse_rollout(solution_str)
    if not s2 or not s3:
        return {"score": 0.0, "acc": 0.0, "parse_fail": 1.0}
    sem = asyncio.Semaphore(1)
    async with aiohttp.ClientSession() as session:
        s4 = await _judge_one(session, sem, s2, s3)
    acc = _rule_based_reward(s4, ground_truth)
    return {"score": acc, "acc": acc, "s4": s4}

# -------- batch-concurrent manager (recommended) ---------------------------------
@register("llm_judge_async")
class LLMJudgeRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine: int = 0, compute_score=None,
                 reward_fn_key: str = "data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False):
        responses = data.batch["responses"]                 # (B, T_resp)
        attn_mask = data.batch["attention_mask"]
        prompt_len = data.batch["prompts"].shape[1]
        resp_mask  = attn_mask[:, prompt_len:]
        B, T = responses.shape
        reward_tensor = torch.zeros((B, T), dtype=torch.float32)

        samples = []
        for i in range(B):
            valid_len = int(resp_mask[i].sum().item())
            sol = self.tokenizer.decode(responses[i][:valid_len], skip_special_tokens=True)
            gt  = data.non_tensor_batch["reward_model"][i]["ground_truth"]
            samples.append((sol, gt))

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_score_batch_async(samples))
        finally:
            loop.close()

        extra = {"acc": [], "s4": []}
        for i, (score, s4) in enumerate(results):
            last = max(int(resp_mask[i].sum().item()) - 1, 0)
            reward_tensor[i, last] = float(score)
            extra["acc"].append(float(score))
            extra["s4"].append(s4)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
```

Two subtleties worth flagging. First, the `@register("llm_judge_async")` decorator must execute before `load_reward_manager` runs; the most robust way is to add `import judge_reward` at the top of a small wrapper entrypoint, or re-export it inside `verl/workers/reward_manager/__init__.py`. Second, if you prefer *not* to write a custom manager, keep the async `compute_score` above and set `reward_model.reward_manager=naive` — it will work correctly but serialize, giving you a correctness baseline while you measure throughput impact.

## Exact VeRL launch command

```bash
python3 -m verl.trainer.main_ppo \
  trainer.n_gpus_per_node=6 \
  trainer.nnodes=1 \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  reward_model.enable=False \
  reward_model.reward_manager=llm_judge_async \
  custom_reward_function.path=/workspace/judge_reward.py \
  custom_reward_function.name=compute_score \
  data.train_files=/data/train.parquet \
  data.val_files=/data/val.parquet \
  algorithm.adv_estimator=grpo
```

If you go the server-only route (skipping the custom manager), drop the `reward_model.reward_manager=llm_judge_async` line to fall back to `naive`. `custom_reward_function.reward_kwargs={}` is available if you want to thread extra config into every call without editing code.

## Dataset columns your preprocessing must emit

Each parquet row requires `prompt`, `data_source` (a string tag you can route on inside `compute_score`), `reward_model={"ground_truth": ...}` (read as `non_tensor_batch["reward_model"][i]["ground_truth"]`), and optional `extra_info`. If your rollout is expected to emit `s1/s2/s3` tags, confirm your SFT or prompt template trains the policy to produce them — the parser above silently returns zero reward on a parse miss, which means a broken template will look like "reward always 0" at training time. Log `parse_fail` as an extra metric to catch this early.

## Things that will bite you

Four pitfalls dominate the failure modes here. **Serialized rewards** are the biggest — measure batch-to-batch wall-clock with and without the custom manager to confirm the async fan-out is actually firing. **MoE + bnb incompatibility** means a `--quantization bitsandbytes` attempt will either fail at load or produce garbage; stick to AWQ-Marlin. **GPU double-booking** happens silently when Ray's placement group thinks it owns all 8 GPUs but `CUDA_VISIBLE_DEVICES` on the judge overlaps — always pin the judge to GPUs outside `trainer.n_gpus_per_node`. And **vLLM version drift** between VeRL's rollout (pinned to 0.7.3/0.8.x in 0.7.1) and your judge server (0.9+ needed for reliable Qwen3-MoE) is only a problem if you try to share a Python env; keep them in separate conda/venvs and the two vLLM installs never meet.

## Key takeaways

The native "vLLM-as-generative-reward-model" worker that would make this a one-flag config change does not exist in 0.7.1 — it began as PR #2845 (unmerged), matured via PR #3441 (SGLang, Sept 2025) and the `recipe/genrm_remote/` template, and is still experimental as the Reward Loop. That means the right mental model for 0.7.1 is: **treat the judge as external infrastructure, and use `custom_reward_function` + a custom async `RewardManager` as the integration seam.** The 8B-trainee / 30B-A3B-judge GPU math works comfortably on a single H100 node with a 6+2 split, AWQ-4bit gives you real MoE inference speedups while bitsandbytes does not, and the `-Instruct-2507` checkpoint means you get non-thinking behavior without any chat-template fiddling. Watch the concurrency trap — it is the difference between a judge that keeps up with rollout and one that doubles your step time.