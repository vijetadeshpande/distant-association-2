# Local Qwen3-4B Judge — Thinking Output-Length Profile

**Date:** 2026-05-22
**Judge model:** `Qwen/Qwen3-4B` (local vLLM backend, bf16, TP=4)
**Purpose:** Measure the output-length distribution of a *thinking* local
Qwen3-4B judge so the co-location hyperparameters in
`scripts/train_codenames_dapo_local_colocated_judge.sh` can be sized from
data instead of guessed.

## Setup

All **160** `codenames_guess_generation` rows of
`custom_data/training_prompts/version-5/codenames_rlvr_val.parquet` were
sent through a local vLLM server with the **production prompt builder**
(`build_guess_messages`) and **production parsers/reward**
(`parse_guesses`, `task_reward`) — same methodology as
`openrouter_judge_thining_test.md`, but:

- backend = local vLLM (not OpenRouter); judge = `Qwen/Qwen3-4B`
- the **entire** val set (160 examples), one response each
- **thinking ON** — Qwen3-4B's native `<think>…</think>` (its default)
- temperature 0.0 — the production judge setting (`JUDGE_TEMPERATURE`)
- `max_tokens=16384`, concurrency 128

160/160 calls succeeded; total wall time **98.5 s**.

Board sizes in the val set span **4–12 words** (mean 8.1). The
**training** set's clue-task boards span the **identical 4–12 range**
(mean 8.0, n=1840) — so this profile transfers directly to training with
no board-size extrapolation. (Codenames allows up to 36 words, but
neither the train nor val v5 data exceeds 12.)

Script: `custom_reward_functions/tests/local_judge_length_test.py`.

## Results

### Output-length distribution (tokens)

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| `think_tokens` | 891 | 852 | 1376 | 1612 | 1831 | 2044 |
| `answer_tokens` | 33 \* | 33 | 40 | 42 | 46 | 46 \* |
| `completion_tokens` | 935 \* | 895 | 1442 | 1670 | 1967 | 2090 \* |
| `prompt_tokens` | 375 | 374 | 386 | 390 | 395 | 400 |

\* `answer`/`completion` mean & max **exclude one degenerate-loop
outlier** (Finding 3) that hit the 16384 cap; the script's raw means are
inflated by it. The p-percentiles are unaffected (1/160 sits above p99).

### finish_reason

`stop`: **159 / 160** · `length` (truncated @ 16384): **1 / 160 (0.6 %)**

### Length vs board size (mean tokens)

| board | n | think | completion |
|---|---|---|---|
| 4 | 7 | 718 | 751 |
| 5 | 15 | 611 | 645 |
| 6 | 20 | 774 | 811 |
| 7 | 24 | 820 | 857 |
| 8 | 26 | 994 | ~935 † |
| 9 | 15 | 903 | 941 |
| 10 | 28 | 907 | 948 |
| 11 | 13 | 1163 | 1206 |
| 12 | 12 | 1111 | 1154 |

† board-8 mean is inflated by the single truncated outlier; corrected ≈935.

### Quality (sanity check, not the focus)

mean `task_reward` = **0.733** · guess block parsed in **159 / 160**.

## Findings

### 1. Thinking dominates; the answer is tiny
The judge spends ~890 tokens reasoning and only ~33 on the actual
`[CODENAMES-GUESS-START]` block. **p99 of total completion is 1967
tokens**; the longest non-degenerate response was ~2090.

### 2. Length is bounded and stable across the board range
Mean completion rises only gently with board size (645 → ~1200 from 5 to
12 words) and never approaches 2500. Because the **training** boards are
the same 4–12 range, no example needs more than ~2.1 k output tokens
under normal behaviour. By difficulty: `advance` boards cost the most
(~1400 mean) and `expert` the least (~815) — all comfortably bounded.

### 3. One degenerate loop in 160 (0.6 %)
A single board-8 example never emitted `</think>` and ran to the 16384
cap. This is the failure mode a thinking judge must be budgeted for: at
temperature 0 a looping response runs until `max_tokens`. It already
degrades gracefully — `parse_guesses` finds no guess block → `judge_fail`
→ a ~0 reward for that one clue-task sample. The mitigation is a
*tighter* token cap (so a loop wastes less compute), not a looser one.
Trainee-generated clues early in training may be lower-quality than these
val reference clues, so the production loop rate could run somewhat above
0.6 % — the cap bounds the cost either way.

### 4. The parse pipeline already handles a thinking judge
`parse_guesses` was run on the **full** `message.content` (raw
`<think>…</think>[CODENAMES-GUESS-START]…` text) and extracted the guess
block in **159/160** cases — every non-degenerate one. A local thinking
Qwen3-4B judge therefore needs **no change to `judge_client.py` or the
parsers**: Qwen3-4B thinks by default and the existing code consumes its
output correctly.

### 5. Prompt is small and flat
~375 tokens regardless of board size (max 400). The judge's
`--max-model-len` is set by the output budget, not the prompt.

### 6. Thinking makes the reward phase a real GPU-heavy phase
At ~1000 completion tokens/call the judge does ~30× the decode work of a
non-thinking judge (~30 tokens/call). With ~512 clue-task calls/step that
is ~0.5 M judge tokens per training step — a substantial phase, and the
reason co-locating the judge onto **all** GPUs (so the reward phase is
not stuck on 2 cards) is worth doing.

## Derived co-location hyperparameters

For `scripts/train_codenames_dapo_local_colocated_judge.sh` on the 4×96 GB
box. KV-cache math uses Qwen3-4B = 36 layers, 8 KV heads, head-dim 128 →
**0.1406 MiB/token** (bf16).

| Parameter | Value | Basis |
|---|---|---|
| `JUDGE_MAX_TOKENS` | **8192** | ≈4× the measured non-degenerate max (2090); zero truncation risk on boards 4–12, and caps degenerate loops at 8 k instead of 32 k |
| judge `--max-model-len` | **9216** | prompt (≤400) + 8192 + margin |
| judge `--gpu-memory-utilization` | **0.20** | 128-way KV need ≈ 11 GB/GPU (see below); 0.20 → 19 GB/GPU ≈ **1.9× headroom** |
| judge `--tensor-parallel-size` | **4** (all GPUs) | reward phase then uses every GPU |
| judge `--max-num-seqs` | **128** | matches `JUDGE_CONCURRENCY`; the old default of 32 throttled the fan-out |
| judge dtype / quant | **bfloat16, no quant** | 8 GB of weights is trivial on 96 GB; bf16 decodes faster than bitsandbytes |
| `N_TRAIN_GPUS` | **4** (all GPUs) | trainee FSDP + rollout span every GPU |
| `rollout.gpu_memory_utilization` | **0.55** | leaves headroom for the co-resident judge + 4-way-sharded FSDP; calibrate in the debug run |

**Judge KV-cache need:** 128 concurrent calls × (375 prompt + ~1670 p95
completion) ≈ 262 k tokens → ~36 GB KV + 8 GB weights ≈ **44 GB across 4
GPUs = 11 GB/GPU**. `--gpu-memory-utilization 0.20` gives 19 GB/GPU.

## Note on the trainee side

The judge is **frozen** and only ever runs the guess task, so this
profile is stable for the whole run. The **trainee** is not: its rollouts
lengthen as training proceeds (toward `max_response_length=16384`). That
growth fills the rollout KV pool — it is not new memory — so it shows up
as lower rollout concurrency late in training, never an OOM.
`rollout.gpu_memory_utilization` (0.55) is the knob if late-step
throughput sags; the judge's slice is fixed and small.

## Reproduce

1. `vllm serve Qwen/Qwen3-4B --tensor-parallel-size 4 --dtype bfloat16
   --max-model-len 32768 --gpu-memory-utilization 0.85
   --served-model-name qwen3-judge --host 127.0.0.1 --port 8000`
2. `python custom_reward_functions/tests/local_judge_length_test.py`

Raw per-example results: `/tmp/local_judge_length_test_results.json`.
