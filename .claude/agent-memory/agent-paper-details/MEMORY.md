# Paper Memory: *Playing with Words, Improving with Rewards*

## Paper identity

- **Full title:** *Playing with Words, Improving with Rewards: Training Language Models for Creative Association*
- **Authors:** Vijeta Deshpande, Namrata Shivagunde, Sherin Muckatira, Hadrien Glaude, Mikhail Gronas, Claire Stevenson, Roger Beaty, and Anna Rumshisky.
- **Core question:** Can an LLM be explicitly trained for creativity without relying on subjective human creativity judgments?
- **Proposed answer:** Train on a simplified version of **Codenames**, whose clue-and-guess outcomes are objectively scoreable, using **Reinforcement Learning with Verifiable Rewards (RLVR)**.
- **Models:** Qwen3-1.7B, Qwen3-4B, and Qwen3-8B.
- **Evaluation:** 10 creativity tasks and 4 active reasoning benchmarks.
- **Main finding:** Training produces a **scale-dependent precision-diversity crossover**:
  - 1.7B and 4B become more precise and improve substantially on reasoning, but do not consistently become more creative.
  - 8B becomes more diverse and improves broadly on creativity, with small reasoning losses.
- **One-line takeaway:** The same small, verifiable word-association training recipe helps every tested scale, but the capability that improves depends on model size—“each model gains where it can.”

## Abstract

- Creativity is increasingly important as LLMs tackle broad solution spaces, but its subjective evaluation makes direct training difficult.
- Codenames operationalizes both central components of creative cognition:
  - **Divergence:** finding remote links spanning several concepts.
  - **Convergence:** selecting one clue that captures targets while excluding distractors.
- Verifiable game outcomes remove the need for human preference labels.
- Reported headline outcomes:
  - 8B improves modestly but consistently on 8 of 10 creativity benchmarks (20 of 23 reported metrics in the Results section).
  - 1.7B and 4B obtain the strongest reasoning gains.
  - The result is not a uniform creativity gain; it is a scale-dependent trade-off between diversity and precision.

## 1. Introduction

- **Motivation:** Hard scientific, mathematical, and generative problems demand unconventional but effective solutions, so creativity should be trained deliberately rather than treated only as an emergent effect of scale.
- **Cognitive-science basis:**
  - Creativity is treated as search over semantic memory.
  - Creative people tend to have semantic networks with denser connectivity and shorter paths between concepts.
  - This supports **combinatorial creativity**: novel ideas arise by rearranging and connecting existing knowledge, not from creation *ex nihilo*.
  - LLM learning is also association-heavy, motivating training that encourages integration of distant knowledge.
- **Why Codenames:** Creativity judgments are subjective and annotators often disagree, whereas Codenames demands creative association but gives unambiguous success/failure signals.
- **Game intuition:** A spymaster sees target and non-target words and gives one clue linking as many targets as possible without attracting non-target guesses. For example, *chair*, *altar*, and *stage* may be linked by *platform*.
- **Contributions:**
  - Cast creativity training as a verifiable-reward game with no human preference annotation.
  - Train three Qwen3 sizes with RLVR and evaluate broad out-of-domain transfer.
  - Identify a scale-dependent precision-diversity trade-off.
  - Show broad creativity gains at 8B and substantial reasoning gains at smaller scales.

## 2. Related Work

### Improving LLM creativity

- Existing approaches include:
  - Prompting and decoding strategies.
  - Inference-time analogical reasoning across domains.
  - Multi-agent role-play that moves from divergent ideation to convergence.
  - Fine-tuning on the Divergent Association Task.
  - Preference optimization for novelty, diversity, and surprise (e.g. CrPO/MuCE).
  - Creativity-oriented data selection.
- **Claimed gap:** RLVR is scalable because it avoids preference annotation, but had not yet been studied as a direct route to LLM creativity.

### Games and self-improvement with RLVR

- RLHF depends on costly human-curated data; games offer automatic, verifiable feedback.
- Prior self-play/RLVR environments include adversarial word games, logic puzzles, board games, visual arcade games, and zero-sum multi-turn games.
- A recurring prior result is out-of-distribution reasoning improvement after game training.
- This paper extends that line by using a game designed around creative association and by analyzing how transfer changes with model scale.

## 3. Method

### 3.1 Training Data

#### Game environment

- Original Codenames has two teams, neutral words, and an assassin word.
- The training version removes neutral and assassin words to isolate association quality.
- One LLM alternates between two roles:
  - **Clue generation / Spymaster:** sees target set `T` and non-target set `N`; produces one clue and identifies the intended target subset.
  - **Guess generation / Operative:** sees only shuffled board `B = T ∪ N`, the clue, and a guess limit; returns likely targets.
- Natural multi-turn play is decomposed into independent single-turn tasks to reduce long traces, cost, and instability. Sampled board positions approximate different stages of a complete game.

#### MDP formulation

- State: `S = {T, N, B, w_clue, g_max}`.
  - Clue generation uses `{T, N}`.
  - Guess generation uses `{B, w_clue, g_max}`.
- Action: `A = {w_clue, T_selected, G}`.
  - Clue action produces clue `w_clue` and intended subset `T_selected`.
  - Guess action produces guesses `G`.
- Constraint: `T_selected ⊂/⊆ T`, with `1 ≤ |T_selected| = g_max` (the prose/equations use both strict and non-strict subset notation).
- Transition function is treated as inaccessible; the episode horizon is bounded by maximum generation length.

#### Sampling states

1. Manually seed 50 broad topics, then use Claude Opus to expand them to about 1,000 unique, single-word topics.
2. For each topic, Claude Opus generates 10 common and 10 rare associated words, stored as `R[t] = [W^(t,c), W^(t,r)]`.
3. Sample separate target and non-target topics. Independently sample 2–6 words for each set. The target topic itself supplies the precomputed clue for guess-training examples.
- Using coherent but distinct topics makes the task learnable for small models while maintaining target/non-target separation.

#### Four-level curriculum

- Non-targets always come from the non-target topic’s **common** pool.
- Target construction progresses from easy to hard:
  - **Simple:** all common target-topic words.
  - **Moderate:** about half common and half rare target-topic words.
  - **Advanced:** all rare target-topic words.
  - **Expert:** about half rare target-topic words; the rest are common words from the non-target topic, creating overlap/distractors.
- Training data is ordered Simple → Moderate → Advanced → Expert.
- In Simple–Advanced, all targets are selected and `g_max = |T|`. In Expert, only the rare-topic half is selected and `g_max = floor(n_T/2)`.

#### Data statistics

- 500 unique states × 4 difficulty rules × 2 tasks = **4,000 examples**.
- Split: **3,680 training / 320 validation**, balanced across task and difficulty.
- Checked-in files used by the generic launcher:
  - `custom_data/training_prompts/version-5/codenames_rlvr_train.parquet`
  - `custom_data/training_prompts/version-5/codenames_rlvr_val.parquet`
- Balance is exact:

| Split | Per difficulty × clue task | Per difficulty × guess task | Total |
|---|---:|---:|---:|
| Train | 460 | 460 | 3,680 |
| Validation | 40 | 40 | 320 |
| Combined | 500 | 500 | 4,000 |

- The parquet is physically ordered by difficulty: train row offsets are Simple `0`, Moderate `920`, Advance `1840`, Expert `2760`; validation offsets are `0`, `80`, `160`, `240`.
- Within each tier, clue and guess rows are interleaved. Note that VeRL’s inherited data setting is currently `data.shuffle=True`, so physical ordering alone does **not** guarantee curriculum training; explicitly disable shuffling or supply a curriculum sampler if order is required.
- The literal dataset label is **`advance`**, while the paper calls the tier **Advanced**.

#### Observed v5 data properties

The following statistics were read directly from the checked-in train and validation parquets (4,000 rows total):

| Property | Clue rows | Guess rows |
|---|---:|---:|
| Rows | 2,000 | 2,000 |
| Target-set size | 2–6; mean 4.003 | 2–6; mean 3.929 |
| Non-target-set size | 2–6; mean 4.029 | 2–6; mean 4.013 |
| Total board size | 4–12; mean 8.033 | 4–12; mean 7.942 |
| Selected-target count | 1–6; mean 3.490 | 1–6; mean 3.442 |
| `all_words` / `max_guesses` | Null by design | Present; `max_guesses` 1–6, mean 3.442 |

- Each difficulty has exactly 1,000 combined rows. Mean `(target size, non-target size, selected size, board size)`:
  - Simple: `(3.901, 4.084, 3.901, 7.985)`.
  - Moderate: `(3.978, 3.990, 3.978, 7.968)`.
  - Advance: `(3.985, 3.981, 3.985, 7.966)`.
  - Expert: `(4.000, 4.031, 2.000, 8.031)`.
- Semantic/frequency diagnostics progress as intended (combined means; both task types):

| Tier | Target Zipf frequency | Clue→target cosine | Clue→non-target cosine | Similarity margin | Stored difficulty score |
|---|---:|---:|---:|---:|---:|
| Simple | 4.645 | 0.584 | 0.412 | 0.173 | 0.374 |
| Moderate | 3.713 | 0.528 | 0.422 | 0.106 | 0.575 |
| Advance | 2.784 | 0.469 | 0.419 | 0.049 | 0.632 |
| Expert | 3.709 | 0.444 | 0.416 | 0.028 | 0.658 |

  The clue-target margin narrows monotonically. Advance has the lowest target frequency because every target is rare; Expert’s mean rises because half its target set consists of common-topic trick words.
- Expert is the only tier with `trick_words`; all 1,000 expert rows have them, with `trick_words = target_words - selected_target_words` and `|selected_target_words| = floor(|target_words|/2)`.
- Across the combined files there are 966 distinct `topic_1`/clue values, 971 distinct `topic_2` values, 6,688 distinct target words, 3,171 distinct non-target words, and 6,428 distinct selected-target words.
- Integrity checks on the checked-in files found no target/non-target overlap, duplicate words within sets, selected targets outside `target_words`, count-field mismatches, malformed `all_words` unions, or `max_guesses != len(selected_target_words)`.

#### VeRL parquet schema and required curation fields

Every finalized parquet row has four top-level fields:

| Field | Type | Requirement / use |
|---|---|---|
| `prompt` | `list[{role, content}]` | Required model input. Current rows have exactly two messages: the shared system CoT prompt followed by one task-specific user prompt. |
| `data_source` | string | Required VeRL routing tag; converter overwrites it with the absolute path to `custom_reward_functions/codenames_reward.py`. |
| `reward_model` | `{ground_truth: string}` | Required by the standard VeRL row shape. Current rows store the reference clue as ground truth, but `compute_score` does not directly use this argument. |
| `extra_info` | struct/dict | Required task state, reward inputs, and analysis metadata. |

Minimum semantic keys to curate before prompt rendering:

| `extra_info` key | Clue task | Guess task | Meaning / invariant |
|---|:---:|:---:|---|
| `task` | Required | Required | Exact current values: `codenames_clue_generation` or `codenames_guess_generation`. Missing/unknown values silently route to the guess scorer, so validate this field. |
| `target_words` | Required | Required | Nonempty list of 2–6 unique target words. Used directly by reward computation. |
| `non_target_words` | Required | Required | Nonempty list of 2–6 unique opposing words, disjoint from targets. Used directly by reward computation. |
| `clue` | Strongly required | Required | Precomputed/reference clue. The v5 converter drops rows where it is missing/empty/`unknown`; clue scoring needs it when cosine mode is used, and the guess prompt displays it. |
| `selected_target_words` | Required by curation | Required | Nonempty subset of targets; converter drops missing/`unknown` values. It supplies the intended subset and determines the guess limit, although clue-task runtime scoring uses the trainee’s newly generated selection. |
| `all_words` | Null/omit | Required in data; fallback available | Shuffled multiset-equal union of target and non-target words. Guess scorer reconstructs the union if absent, but an explicit shuffled list is preferred and is shown in the prompt. |
| `max_guesses` | Null/omit | Required | Must equal `len(selected_target_words)`. The scorer also accepts legacy alias `num_max_guesses`; missing values become zero and make every nonempty answer fail the bound check. |

Recommended derived/analysis keys already present in v5:

| Key(s) | Meaning |
|---|---|
| `difficulty_rule` | One of `simple`, `moderate`, `advance`, `expert`; needed for balance, curriculum, and slice analysis. |
| `topic_1`, `topic_2` | Target/clue topic and non-target topic respectively. In the current data, `clue == topic_1`. |
| `trick_words` | Expert-only target words sampled from the non-target topic and deliberately excluded from `selected_target_words`; null otherwise. |
| `num_words_in_target`, `num_words_non_target`, `total_words_on_board` | Redundant integrity/count fields; must equal list lengths and their sum. |
| `avg_clue_target_cosine`, `avg_clue_non_target_cosine` | Mean clue similarity to each side. |
| `diff_clue_similarity` | Stored clue-separation margin; observed values equal target mean minus non-target mean. |
| `intra_target_cosine`, `intra_non_target_cosine`, `cross_target_non_target_cosine` | Within-set and cross-set semantic-cohesion diagnostics. |
| `avg_target_zipf_freq`, `avg_non_target_zipf_freq` | Mean lexical-frequency diagnostics; lower target frequency characterizes harder rare-word tiers. |
| `difficulty_score` | Upstream scalar difficulty feature. It is retained for analysis but its construction is not defined in the checked-in converter and it is not consumed by the reward. |
| `reward_function_name` | Logging metadata; current value is `compute_score`. |

Minimal finalized row shape:

```json
{
  "prompt": [
    {"role": "system", "content": "<shared five-stage CoT instruction>"},
    {"role": "user", "content": "<rendered clue or guess prompt>"}
  ],
  "data_source": "/absolute/path/custom_reward_functions/codenames_reward.py",
  "reward_model": {"ground_truth": "reinforcing"},
  "extra_info": {
    "task": "codenames_guess_generation",
    "target_words": ["concrete", "bond"],
    "non_target_words": ["spring", "curve"],
    "clue": "reinforcing",
    "selected_target_words": ["concrete", "bond"],
    "all_words": ["concrete", "curve", "spring", "bond"],
    "max_guesses": 2,
    "difficulty_rule": "simple"
  }
}
```

- Raw v5 conversion is handled by `custom_data_preparation/convert_v5_rlvr_jsonl_to_parquet.py`:
  - Starts from 4,500 mixed JSONL rows.
  - Drops 500 rows whose clue and selected targets are unavailable/`unknown`.
  - Prepends the system prompt, preserves the user prompt and all `extra_info`, sets `data_source`, and adds `reward_function_name`.
  - The checked-in converter emits one 4,000-row parquet; the code that produced the checked-in 3,680/320 stratified split is not present, so future curation should implement and record that split explicitly.

#### Concrete examples by difficulty

These are real validation examples. The same semantic fields can generate a clue row (`all_words/max_guesses = null`) or a guess row (shown below).

| Tier | `topic_1` / clue | Targets | Non-targets (`topic_2`) | Selected targets | Trick words | `max_guesses` |
|---|---|---|---|---|---|---:|
| Simple | `reinforcing` | `[concrete, bond]` | `[spring, curve]` (`flexible`) | `[concrete, bond]` | — | 2 |
| Moderate | `gouda` | `[round, lactose]` | `[earth, brown]` (`terra`) | `[round, lactose]` | — | 2 |
| Advance | `engineering` | `[modulus, tensile]` | `[brown, tree]` (`leaf`) | `[modulus, tensile]` | — | 2 |
| Expert | `rounded` | `[large, cycloid]` | `[pile, volume]` (`mass`) | `[cycloid]` | `[large]` | 1 |

- Composition illustrated above:
  - Simple targets are common associations.
  - Moderate mixes common and rare target-topic associations.
  - Advance uses rare associations (`modulus`, `tensile`).
  - Expert inserts `large`, a common association of the opposing topic `mass`, into the target set but excludes it from the clue’s intended subset; the model must isolate `cycloid`.

### 3.2 Reward Function

#### Clue-generation reward

- The trainable actor generates a clue and selected targets from `{T, N}`.
- A frozen **Gemini Flash 2.5** judge receives the clue, shuffled board, and guess limit, then predicts guesses `G`.
- Reward measures whether the clue lets the frozen judge recover targets without selecting distractors or off-board words:

```text
reward = |T ∩ G| / |T|
       - |N ∩ G| / |N|
       - |G \ (T ∪ N)| / |G|
```

- Range is `[-2, 1]`; `1` requires recovering every target and nothing else.
- **Implementation-critical difference:** the paper equation divides invalid/off-board guesses by `|G|`, but current `custom_reward_functions/task_reward.py` divides them by board size `|T ∪ N|`. Additional experiments using the repository as-is follow the latter.
- For a clue rollout, `g_max` is computed dynamically as the number of targets selected by the trainee, not read from the row. The external judge sees the generated clue, a shuffled target/non-target union, and this limit.
- Judge-response formatting is logged diagnostically but does not hard-gate the trainee’s clue reward because the trainee cannot control judge formatting. A failed/empty judge call returns task score `0` with `judge_fail=1`.

#### Guess-generation reward

- The actor guesses from `(B, w_clue)` using the dataset’s precomputed clue.
- Its guesses are scored by the same deterministic set-overlap reward against retained `T` and `N`.
- `all_words` falls back to `target_words + non_target_words` if absent, but the explicit shuffled field should be supplied so prompt and scorer state agree.
- Comparisons are whitespace-stripped and case-folded; guesses are converted to sets, so duplicate guesses do not earn extra credit.

#### Format reward

- Validity checks run before semantic scoring. Any failure forces reward `-1`.
- Checks cover required tags, valid clue form, clues not appearing on the board or as morphological variants, valid selected targets, nonempty guesses, on-board guesses, and the guess bound.
- Current code has six clue gate keys: tags present, single whitespace-free clue, no hyphen, nonempty selected list, selected list is a target subset, and no morphological variant. Exact normalized board-word equality is handled inside the morphology check.
- Current code has four guess gate keys: tags present, nonempty list, every guess on the board, and count within `max_guesses`.
- Five-stage thinking presence/order is recorded as diagnostics but explicitly excluded from the hard format gate; malformed/missing CoT tags do not currently change the scalar reward.
- The code checks neither dictionary membership/“real English word” nor proper-noun, abbreviation, direct-translation, or general compound-word status beyond whitespace and hyphens.

#### RLVR objective

- Uses DAPO with group-normalized/GRPO advantages and token-level clipped importance ratios.
- For each state, sample a response group; normalize each reward within the group as `Â_i = (R_i - mean(R))/std(R)`; optimize the token-mean clipped surrogate.
- The objective uses asymmetric clipping (`ε_low`, `ε_high`) and no KL term.

## 4. Experimental Setup

### 4.1 Training

- **Models:** Qwen3-1.7B, 4B, and 8B.
- **Implementation:** VeRL; full-parameter `bf16` fine-tuning, with no adapters or quantization.
- **Training judge:** Gemini Flash 2.5, frozen.
- **Paper-run optimization:** DAPO objective with GRPO advantage estimation.
  - Prompt batch: 32; response group `G = 32`; **1,024 trajectories/update**.
  - 2 epochs × 3,680 prompts / 32 prompts per step = **230 parameter updates**.
  - Constant learning rate `1e-6`; no LR warmup or schedule decay.
  - Asymmetric clip bounds `ε_low = 0.20`, `ε_high = 0.28` (“clip higher”).
  - Token-mean policy loss; group advantages normalized by mean and standard deviation.
  - No KL loss, no KL reward penalty, and no separate learned/discriminative reward model.
- **Current generic execution configuration** (`scripts/train_codenames_dapo.sh` + `scripts/codenames_dapo.yaml` + inherited VeRL defaults):

| Area | Effective setting |
|---|---|
| Policy update | FSDP full-parameter training; one PPO epoch per batch; actor minibatch 32 prompts; microbatch 4 trajectories/GPU; gradient checkpointing enabled. |
| Optimizer | AdamW, LR `1e-6`, constant scheduler, warmup ratio `0`, weight decay `0.01`, betas `(0.9, 0.999)`, gradient clipping `1.0`, entropy coefficient `0`. The paper does not report the inherited weight decay/betas/grad clip. |
| Rollout | vLLM async mode, `bf16`, temperature `1.0`, `top_p=1.0`, `top_k=-1`, sampling enabled, `n=32`, TP default `2`, GPU-memory utilization `0.6`. |
| Model/runtime | Remove padding and fused kernels enabled; FSDP parameter and optimizer CPU offload enabled; reference-model parameter offload enabled; rollout weight-update bucket 3,072 MB. |
| Sequence limits | Prompt 2,048; response 16,384; combined potential context 18,432. |
| Trainer | One node; console + W&B logging under project `Distant-Association`; resume disabled; validation before training disabled. |
| Data | Train/validation parquets above; training dataloader shuffle inherited as `True` with no explicit seed; validation shuffle `False`. |
| Checkpoints | Frequency computed as `floor(total_steps/8)` with minimum 1; for 230 steps this is every 28 steps (about 8 saves). |

- **Dynamic sampling:** Oversample and discard groups whose 32 responses all have the same sequence reward because they provide zero GRPO advantage. Generate at most 10 batches to refill one effective batch. `data.gen_batch_size` is 32 prompts.
- **Length control:** Prompt cap 2,048 and response cap 16,384. DAPO soft-overlength shaping has a 4,096-token buffer: penalty begins above 12,288 response tokens, increases linearly, and reaches factor 1.0 at 16,384.
- **Training-time validation defaults:** inherited rollout validation is greedy (`temperature=0`, `top_p=1`, `top_k=-1`, `n=1`, `do_sample=False`), but the generic launcher sets `val_before_train=False` and does not set a positive periodic `test_freq`.
- **Reward execution and logging:**
  - `reward_model.enable=False`; custom async `compute_score` supplies the reward.
  - Clue rows use a frozen generative judge; guess rows use deterministic set scoring and never call the judge.
  - Scalar reward is hard-gated to `-1` on task-format failure; otherwise the raw task scalar passes through (no averaging of format and task components).
  - Logged diagnostics include target recovery, non-target/invalid penalties, format flags, `parse_fail`, `format_fail`, `judge_fail`, cosine/OOV values, generated clue/targets/guesses, task kind, board context, `format_score`, and `task_score`.
- **Generic judge modes:**
  - Default: `google/gemini-2.5-flash` over OpenRouter, thinking disabled, temperature `0`, max 1,024 output tokens, timeout 300 s, concurrency 128, and up to three total attempts on connection/timeout/5xx failures.
  - `JUDGE_THINKING=1`: requests a dynamic reasoning budget and raises default judge max output to 32,768.
  - `JUDGE_BACKEND=local`: calls an OpenAI-compatible local vLLM endpoint; the launcher can reserve separate training/judge GPU pools.
  - Empty/`none` `JUDGE_MODEL_ID`: replaces judge play with GloVe cosine similarity between the generated clue and the reference clue. This is an implementation alternative, not the paper’s reported reward.
- **Launch-time invariants:** `32 prompts × 32 responses = 1,024` trajectories must be divisible by the number of training GPUs. Default rollout TP is 2. The generic launcher currently defaults the trainee to **Qwen3-14B**, so reproducing a paper model requires explicitly setting `TRAINEE_MODEL_ID`/`TRAINEE_MODEL_PATH`.
- **Debug mode:** `bash scripts/train_codenames_dapo.sh debug` slices training to 32 rows, uses one epoch/one update, selects the last `N_TRAIN_GPUS` validation rows, enables pre-training validation and generation dumps, saves every step, disables group filtering, and disables checkpoint sync.
- **Compute/cost:** Vast.ai GPU pool; 4B/8B runs took roughly 40–50 hours and 1.7B roughly 25 hours; total training and evaluation cost was about **US$3,000**.

### 4.2 Evaluation

#### Shared evaluation protocol

- Every base model is compared with its own 230-step fine-tuned checkpoint; no cross-scale comparison is used as the treatment effect.
- All tasks are zero-shot and Qwen3 is evaluated in thinking mode unless explicitly disabled in the NYT Connections diagnostic.
- Custom creativity tasks use temperature `1.0`, with `top_p` and `top_k` unset so Qwen3’s recommended thinking-mode defaults apply.
- Reasoning uses `lighteval` with temperature `0.6`, `top_p=0.95`, `top_k` unset, one sample/item, and a 32,768-token generation budget.
- Main-paper evaluation reports 23 metrics across 10 creativity task families and 4 reasoning benchmarks. Higher is better for every reported metric.
- The nominal active workload is **17,793 creativity generations + 1,577 reasoning generations = 19,370 policy completions per checkpoint**, excluding judge calls and diagnostic reruns. Across 3 base/FT pairs this is 116,220 policy completions.

#### Creativity evaluation catalog

| Task | Size × samples/item | Required output | Reported metric / scorer |
|---|---:|---|---|
| AUT | 15 × 10 = 150 | Exactly 5 uses, each under 5 sentences | Separate 1–5 LLM-judge creativity and coherence scores. Creativity considers originality, diversity, and physical validity. |
| DAT | 1 × 100 = 100 | 10 distinct, single-word, non-proper, nontechnical English nouns | Mean DSI over 100 lists; vocabulary size = unique fraction among all 1,000 generated words. |
| RAT | 271 × 1 | One word linking three cues | Exact-match accuracy against reference answers. |
| CWT | 7 × 10 = 70 | About 5 sentences containing all 3 cue words | 1–5 LLM-judge creativity/coherence, plus DSI and MTLD. Missing required words cap creativity at 2. |
| EQ-Bench CW v3 | 32 × 3 = 96 | Approximately 1,000-word story obeying topic, genre, POV, tense, and required details | Creativity/rubric score on 0–100 scale; appendix describes a 22-criterion aggregate, while the displayed judge prompt summarizes four creative-writing dimensions. |
| MacGyver | 1,683 × 1 | Solvable flag and <100-word stepwise plan using only supplied tools, or short unsolvability justification | Separate 1–5 LLM-judge feasibility and efficiency scores. |
| MUNCH word | 1,492 × 1 | One of A/B/C/D | Accuracy; 25% chance. |
| MUNCH sentence-implicit | 1,492 × 1 | One of A/B/C/D | Accuracy; 25% chance. |
| MUNCH sentence-marked-word | 1,492 × 1 | One of A/B/C/D | Accuracy; 25% chance. |
| MUNCH generation | 2,953 × 1 | One literal replacement word | Recall@1 against the human reference set. |
| New Yorker matching | 2,691 × 1 | One of five captions (A–E) | Accuracy; 20% chance. |
| New Yorker explanation | 651 × 1 | 2–3 sentence joke explanation | 1–5 LLM judge; must identify the incongruity and connect it to the caption. |
| NYT Connections | 652 × 1 | Four disjoint groups of four; every word used exactly once | Group accuracy and exact/perfect match of all four groups. |
| LLM-Grounding | 1,000 × 4 = 4,000 | One integer association rating on the requested psycholinguistic scale | Spearman `ρ` with mean human ratings; results aggregate 18 axes into non-sensorimotor, sensory, and motor categories. The 1,000-item subset is balanced across concepts and human ratings. |

- New Yorker **ranking** (2,616 two-way items; 50% chance) is fully catalogued and prompted but commented out of the main result table.
- LLM-judge configuration declared by the Experimental Setup/table caption: **Claude Opus 4.7**, temperature `0`, `top_p=0.95`, one sample. See Source Cautions for conflicting model annotations inside individual prompt examples.

#### Reasoning evaluation catalog

| Task | Items | Output/scoring | Important interpretation detail |
|---|---:|---|---|
| GSM8K | 1,319 | Free-form solution; extractive-match final answer | Grade-school multi-step arithmetic; near saturation for these models. |
| AIME 2024 | 30 | Reasoning followed by boxed integer; extractive match | Four changed answers move accuracy by 13.3 points; treat deltas cautiously. |
| AIME 2025 | 30 | Same as AIME24 | Same small-sample caveat. |
| GPQA-Diamond | 198 | Four-way graduate science answer; accuracy | 25% chance; larger and more discriminating than AIME. |

- MuSR murder mystery (250; 2-way), object placement (256; 4-way), and team allocation (250; 3-way) are in the appendix catalog/prompts but commented out of the active main evaluation and result table. Their appendix examples use conditional log-likelihood choice ranking rather than free-form generation.

#### Output parsing and judge rubrics

- Custom tasks use explicit wrappers such as `[Answer-Start]...[Answer-End]`; task-specific forms include `[Replacement-Word-*]`, `[Explanation-*]`, `[SOLUTION-*]`, `[FINAL-GROUPING-*]`, and `[Rating]`.
- AUT and CWT judges ignore grammar when scoring creativity; separate coherence passes handle mechanics.
- MacGyver feasibility ignores path length; efficiency is a separate judge pass.
- EQ-Bench’s displayed creativity judge covers originality/imagination, sensory detail/word choice, character/conflict, and narrative arc/pacing, mapping four quality levels onto 1–100.
- New Yorker explanation judging uses the human reference only as an interpretation guide and scores the model’s explanation itself.

#### Repository qualitative evaluation harness (not the main benchmark suite)

- `custom_qual_eval/run_eval.py` evaluates Codenames validation prompts with local vLLM in `bf16`.
- Default parquet sampling is 3 rows per `(task, difficulty_rule)` group: 2 tasks × 4 difficulties × 3 = **24 prompts**, seed 42. A sampled parquet can be materialized once and reused so every checkpoint sees identical prompts; `ALL=1` uses all 320 validation rows.
- Default generation mirrors training: temperature `1.0`, `top_p=1.0`, `top_k=-1`, one sample, maximum 16,384 new tokens, GPU-memory utilization `0.85`; tensor parallelism auto-detects visible GPUs unless specified.
- Output JSONL records index, model, retained row metadata, chat messages, all completions, finish reasons, and sampling settings.
- The checked-in repository contains this qualitative Codenames harness, but no executable runners for the paper’s full creativity/reasoning benchmark suite; the appendix prompts and configuration table are therefore the available source of truth for recreating those evaluations.

## 5. Results

- Except RAT and NYT Connections, evaluation tasks are structurally unlike Codenames and therefore test out-of-domain transfer.

### 5.1 Creativity Evaluation

- **Headline:** Qwen3-8B improves on **20/23 creativity metrics** and is the only scale to increase both DAT diversity measures:
  - DSI: `0.5837 → 0.5886`.
  - Vocabulary size: `0.259 → 0.283`.
- At all three sizes, 8/23 metric pairs improve; at least two sizes improve on 15/23.
- Improvements common across scales include:
  - New Yorker humor explanation and matching.
  - MacGyver feasibility and efficiency.
  - RAT convergent association (though small-model absolute accuracy remains low).
  - MUNCH generation recall and EQ-Bench creativity.
- For 8B, almost everything improves; the three declining metrics are AUT creativity and both NYT Connections measures.

#### NYT Connections regression is attributed to overthinking

- Thinking-on main results for 8B:
  - Group accuracy: `0.6794 → 0.6515`.
  - Perfect match: `0.5077 → 0.4724`.
- The tuned model violated the one-word-one-group constraint 17 times versus 7 for base.
- With thinking disabled:
  - Perfect match is identical: `0.2730` vs `0.2730`.
  - Group accuracy is effectively identical/slightly higher for FT: `0.4172 → 0.4187`.
  - Across 227 difficulty-tagged puzzles, FT degrades in every bucket with thinking on but improves in every bucket with thinking off.
- Therefore the paper treats the regression as a reasoning-length/constraint artifact, not lost associative ability. No convincing trace explanation was found for the small AUT creativity drop.

#### Diversity, entropy, and length by scale

- DAT vocabulary collapses for the smaller models:
  - 1.7B: `0.201 → 0.101`; DSI `0.5519 → 0.5422`.
  - 4B: `0.158 → 0.106`; DSI `0.5381 → 0.5284`.
- FT/base response-length ratios: **2.45×, 1.52×, 1.36×** for 1.7B, 4B, 8B.
- FT/base mean per-token entropy ratios: **1.07×, 0.69×, 1.03×**.
- Interpretation:
  - At 1.7B, a 7% average entropy rise is outweighed by 2.45× longer context, which pushes final generation into a lower-entropy regime.
  - At 4B, entropy itself drops 31% while responses lengthen.
  - At 8B, modest entropy and length growth coexist and divergent diversity is retained/improved.

### 5.2 Reasoning Evaluation

| Model | AIME24 | AIME25 | GPQA-Diamond | GSM8K | Pattern |
|---|---:|---:|---:|---:|---|
| 1.7B | `.4333 → .5667` | `.2667 → .3333` | `.4141 → .4697` | `.8810 → .8923` | Improves on all 4 |
| 4B | `.6667 → .7000` | `.5667 → .6333` | `.5253 → .5455` | `.9431 → .9416` | Improves on 3; tiny GSM8K drop |
| 8B | `.8000 → .6667` | `.7333 → .7333` | `.6111 → .5909` | `.9507 → .9500` | Flat/slightly worse; AIME24 drops 13 points |

- Strongest small-model result: 1.7B gains 5.6 points on GPQA-Diamond and 13.3 points on AIME24.
- The 8B base has less reasoning headroom; AIME sets contain only 30 questions, so a few answers cause large percentage changes.

### 5.3 Synthesis: Associative Learning Meets RLVR

- Proposed mechanism: RLVR’s known diversity-precision trade-off changes direction with scale.
  - Smaller models **sharpen** their distribution, losing lexical/semantic diversity but becoming more accurate on attainable reasoning problems.
  - 8B keeps a broader distribution and gains remote-association creativity, with slight math/science precision costs.
- Hypothesis for the split: smaller models can earn reward by precisely solving newly reachable/easier cases; the larger model already handles those and explores diverse associations on harder cases.
- Supporting 1.7B trace evidence:
  - Reasoning traces lengthen overall by 1.42×.
  - On newly solved problems, the ratio is only 1.13×.
  - On AIME, tuned traces are shorter than base (`<1.0×`), consistent with quicker convergence on easier/solvable problems.
- This mechanism is interpretive rather than causally isolated because there is no RLVR-only or CoT-only ablation.

## 6. Conclusion

- A simplified, verifiable word game is a practical and scalable way to alter LLM capabilities without subjective creativity labels.
- Its benefit is scale-dependent rather than uniform:
  - 1.7B/4B: stronger math and science reasoning.
  - 8B: broad, modest creativity improvement.
- The study’s central contribution is the observed crossover between precision and diversity across scale.

## Limitations

- Simplifying Codenames by removing neutral/assassin words and replacing multi-turn play with single turns omits strategic and constraint-following dynamics; this may help explain NYT Connections violations.
- Training combines a structured chain-of-thought scaffold with generic RLVR, but no ablation isolates the associative-game contribution.
- No representational probes test whether internal semantic geometry or distant-concept links actually changed.
- Training clue rewards use a frozen LLM judge, and several evaluation metrics use an LLM judge, introducing possible judge bias.
- Reasoning sets can be small—especially 30-item AIME—so reported changes may reflect only a few questions.

## Ethics Statement

- More creative/diverse output can reduce precision; these models/data should not be assumed safe for accuracy-critical medical, legal, or scientific uses.
- Training data is synthetic and contains no personal, sensitive, or human-subject data; evaluation benchmarks are public and license-compliant.
- LLM judges may encode cultural/linguistic bias, and English-only tasks limit cross-lingual generalization.
- The work aims to lower the resource barrier through a small dataset, automatic rewards, and gains at 1.7B scale.
- Claude Opus assisted experimental setup, data curation, and manuscript editing, but authors state that all content was written, reviewed, and edited by humans rather than generated entirely by an LLM.

## Appendix Memory

### A. Training prompts

- Shared system prompt enforces five ordered sections: `<thinking>`, `<reasoning>`, `<reflection>`, `<adjustment>`, `<output>`.
- Clue prompt asks for candidate brainstorming, checking every candidate against both sets, discarding clues linked to any non-target, and selecting the clue covering the most targets.
- Clue output contract:

```text
[CODENAMES-CLUE-START]
[Clue]: <one English word>
[Selected-Targets]: [target words]
[CODENAMES-CLUE-END]
```

- Guess prompt asks the model to assess every board word, consider uncommon/lateral meanings, rank by confidence, and return no more than `Max-Guesses`.
- Guess output contract:

```text
[CODENAMES-GUESS-START]
[Guesses]: [first guess, second guess, ...]
[CODENAMES-GUESS-END]
```

### B. Generated-response checks

- **Clues:** required tags/fields; one whitespace-separated English word; no hyphen; not on board; not a morphological board-word variant; at least one selected target; all selected targets belong to `T`.
- **Guesses:** required tags/field; nonempty; every guess on board; number of guesses within `g_max`.
- Failed checks receive `-1` instead of the task reward.

### C. Evaluation dataset catalog

- Records each task’s purpose, size, generation type/configuration, judge settings, and metrics.
- Notable sample counts: AUT 15 items × 10 generations; DAT 1 prompt × 100; CWT 7 × 10; EQ-Bench 32 × 3; LLM-Grounding 1,000 × 4. Most other tasks use one generation per item.
- Appendix also catalogs New Yorker ranking and three MuSR subsets, but these rows are commented out of the main reported result tables.

### D. Evaluation prompts

- Gives exact policy prompts and output formats for every creativity/reasoning dataset.
- Includes separate judge prompts/rubrics for AUT, CWT, EQ-Bench, MacGyver, and New Yorker explanation. The rubric text itself is model-agnostic; model annotations are inconsistent, as noted below.
- Examples show responses from the 8B checkpoint at 230 steps with thinking traces omitted.

## Source cautions / manuscript consistency notes

- The main Method says there are **6 clue checks**, while the appendix presents **7 conceptual checks**. Current code exposes six gate metrics because exact clue/board equality is folded into `clue_no_morph_variant`; functionally, both equality and morphology are checked.
- The paper’s invalid-guess term is `|G \ (T ∪ N)| / |G|`; current code uses `|G \ (T ∪ N)| / |T ∪ N|`. Record which version an experiment uses.
- The evaluation section and dataset-catalog caption specify Claude Opus 4.7 for custom-task judging. Individual prompt annotations instead name Gemini 2.5 Flash for AUT, CWT, MacGyver, and New Yorker explanation, while the EQ-Bench annotation names Claude Opus 4.7. The manuscript does not reconcile this; model/provider should be explicitly logged in new runs.
- The infrastructure paragraph says 1.7B was trained/evaluated on a single A100, while the runtime sentence says its ~25-hour training used 4× RTX 5090; the source does not reconcile these statements.
- The generic launcher now defaults to Qwen3-14B, not one of the three paper models; set the trainee model explicitly for replication.
- The parquet is difficulty-ordered, but the inherited VeRL data loader shuffles training rows by default. The paper’s claimed Simple→Expert curriculum therefore requires an explicit `data.shuffle=False` or curriculum sampler in the current code.
- Three of 4,000 checked-in rows have a reference clue that also appears on their board (`river`, `sea`, `sun`). These are all clue-generation rows. Generated clues with this overlap fail the current morphology/equality gate, so future curation should reject such reference examples too, especially when using cosine-reference reward.
- `.claude/rules/codenames.md` states a possible total board size of 4–36, but two independently sampled 2–6 sets imply 4–12, which is also the observed parquet range.
- The appendix still contains MuSR and New Yorker ranking configurations/prompts, but their main-table rows and MuSR’s main-text description are commented out.
- Large alternative Results drafts inside LaTeX `comment` environments are inactive and are not treated as paper sections here.
