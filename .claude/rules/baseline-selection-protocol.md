# Baseline Selection and Fair-Comparison Protocol

## 1. Purpose and scope

This protocol governs additional training-data baselines for the Codenames
RLVR paper. Its purpose is to determine whether the effects of Codenames
training are attributable to distant semantic association specifically, rather
than to generic RLVR, ordinary reasoning practice, or game-shaped rewards.

The primary comparison model is **Qwen3-8B**. Qwen3-14B is an optional
confirmatory extension. Do not add 1.7B or 4B baseline runs unless the study
scope is explicitly amended.

The protocol assumes the existing Codenames setup:

- 3,680 training prompts and 320 validation prompts;
- global prompt batch size 32;
- 32 responses per prompt;
- DAPO/GRPO advantages with dynamic filtering;
- full-parameter BF16 training;
- learning rate `1e-6`;
- clip-low/high `0.20/0.28`;
- token-mean loss;
- no KL loss or reward;
- response limit 16,384 tokens;
- a 4,096-token soft-overlength buffer; and
- 115 optimizer updates per epoch, or 230 updates for two epochs.

This document specifies preparation and future execution. **Do not start a
GPU process, load a model onto a GPU, or probe currently occupied GPUs while
performing the CPU preparation stages.** Training and rollout calibration are
to be run later on the separate GPU server.

## 2. Experimental questions and estimands

For model scale `m`, training source `b`, epoch endpoint `e`, and evaluation
metric `k`, define the within-model change from the untouched base checkpoint:

```text
Delta(b, m, e, k) = Score(trained_on=b, m, e, k) - Score(base, m, k)
```

The main Codenames-specific contrast is:

```text
Specificity(b, m, e, k) = Delta(Codenames, m, e, k)
                           - Delta(baseline_b, m, e, k)
```

This is a matched within-scale, within-horizon comparison. Follow these rules:

1. Never use an absolute 8B-versus-14B score as a training effect.
2. Never compare a one-epoch baseline with a two-epoch Codenames checkpoint as
   though their training horizons were matched.
3. Never select the best intermediate checkpoint using downstream benchmark
   scores. Use the predeclared epoch endpoint.
4. Treat the training source as the intended intervention. Hold the optimizer,
   model revision, prompt scaffold, output budget, and evaluation pipeline
   fixed wherever the task permits.
5. Describe math gains after math training and GPQA gains after science
   training, if science is ever added, as trained-domain or domain-adjacent
   transfer rather than wholly out-of-domain transfer.

The three principal questions are:

- **Generic RLVR control:** Does deterministic math RLVR reproduce the
  creativity/diversity changes attributed to Codenames?
- **Reasoning-content control:** Does natural-language rule reasoning reproduce
  those changes?
- **Game-structure control:** Does a non-semantic game with deterministic,
  graded rewards reproduce those changes?

An optional fourth question is whether an orthographic word game produces the
same transfer as a semantic association game.

## 3. Selected baselines and priority

### 3.1 Primary matched baselines

Run these three baselines for the complete 8B comparison:

| ID | Baseline | Experimental role | Status |
|---|---|---|---|
| `math` | Cleaned English DAPO-Math | Generic numerical RLVR | Required |
| `logic` | Cleaned six-domain RuleCollection | Natural-language rule reasoning | Required |
| `mastermind` | TextArena-derived single-decision Mastermind states | Non-semantic game RLVR | Required |

These runs must use the static single-turn VeRL/DAPO path used by Codenames.

### 3.2 Optional supplemental baseline

`wordle-state` may be run as a fourth, supplemental baseline. It is useful as
a lexical/orthographic game control. Derive static TextArena Wordle states with
a visible legal history and ask for exactly one next guess. This keeps the
same single-turn VeRL/DAPO path as Codenames and the primary controls. Treat it
as a predeclared supplemental mechanism analysis, not a fourth baseline needed
to call the primary matrix complete. An online multi-turn Wordle run remains a
different optional experiment and must be reported separately.

### 3.3 Excluded baseline

Nemotron-RL-Science-v1 is not part of the main protocol. Its open-ended answers
require an equivalence judge, most rows are physics, and most of the release
expects a Python tool. It introduces reward-model noise, tool-use differences,
and direct proximity to GPQA. Adding it requires a written protocol amendment,
a no-tools-only subset, source-link-grouped splitting, domain balancing, and a
judge calibration study.

### 3.4 Source records

- DAPO-Math processed dataset:
  <https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed>
- DAPO method and dataset construction:
  <https://arxiv.org/abs/2503.14476>
- RuleCollection dataset:
  <https://huggingface.co/datasets/RuleReasoner/RuleCollection-32K>
- RuleReasoner method:
  <https://arxiv.org/abs/2506.08672>
- TextArena package and game definitions:
  <https://github.com/TextArena/TextArena>
- Online Wordle GRPO example:
  <https://github.com/huggingface/trl/blob/main/docs/source/openenv.md#advanced-example-wordle>
- SPIRAL multi-turn self-play comparator:
  <https://arxiv.org/abs/2506.24119>

Record immutable source revisions in every dataset manifest. Revisions used in
the initial audit were:

- DAPO-Math processed: `31dd309567e3da778038cc87d868b6097a3ccf68`;
- RuleCollection: `f14a766d2e8e46390101154c6f4f51a67d7a5d9f`.

Do not silently replace these with a newer upstream revision. If a newer
revision is selected, rerun all admission checks and record the reason.

## 4. Budget ladder and run matrix

### 4.1 Minimum viable study

The minimum complete comparison matrix is:

| Scale | Checkpoint/source | Endpoint | New training required? |
|---|---|---:|---:|
| 8B | untouched base | 0 updates | No |
| 8B | Codenames | 115 updates / epoch 1 | Reuse if exact checkpoint exists |
| 8B | DAPO-Math | 115 updates / epoch 1 | Yes |
| 8B | RuleCollection | 115 updates / epoch 1 | Yes |
| 8B | Mastermind-state | 115 updates / epoch 1 | Yes |

This requires three new one-epoch 8B training runs, totaling 345 accepted
optimizer updates. Dynamic filtering can make the actual rollout workload much
larger, so “345 updates” is not a token or wall-time estimate.

If an exact 115-update Codenames checkpoint does not exist, do one of the
following:

1. produce a matched Codenames epoch-1 checkpoint using the same code and data;
2. move the entire comparison to the two-epoch/230-update endpoint; or
3. label an approximate checkpoint comparison as exploratory and report the
   update mismatch prominently.

Option 3 must not support the main causal claim.

### 4.2 Two-epoch extension

If the budget decision permits two epochs, continue **all three** primary 8B
baseline runs from their exact epoch-1 optimizer/RNG states to 230 updates.
Compare them with the Codenames 230-update checkpoint.

The decision to fund epoch 2 must be based on available compute and training
health, not on downstream creativity or reasoning results. If budget is
uncertain, stop safely at update 115, preserve complete checkpoints, and make
the budget decision before running downstream benchmarks.

Do not continue only the baseline that looks best on downstream evaluations.
If there is insufficient budget to extend all three, either keep epoch 1 as the
primary complete matrix or predeclare a scientifically motivated subset before
examining benchmark results.

### 4.3 Training-seed extension

One training seed is acceptable only as an exploratory baseline comparison.
Item-level evaluation bootstraps do not measure training-run variance.

After completing a matched epoch endpoint for all three primary baselines, the
next robustness priority is a second complete 8B training seed. Do not describe
small differences between single-seed runs as stable model effects.

### 4.4 Optional 14B extension

Begin 14B baseline training only after the 8B matrix is complete and the
following are available at 14B:

- the untouched base checkpoint;
- a Codenames checkpoint at the same epoch endpoint;
- the exact tokenizer/model revision used by that Codenames run; and
- sufficient budget to answer a predeclared confirmatory question.

Use the same dataset row IDs selected for 8B. Run a short rollout-only
calibration at 14B first to detect saturation; do not silently substitute a
harder 14B-only dataset.

The 14B priority order is:

1. DAPO-Math, testing Codenames against generic RLVR at larger scale;
2. Mastermind-state, testing semantic association against generic game RL;
3. RuleCollection, completing the full three-baseline replication.

A base/Codenames/math-only 14B matrix is a valid narrow confirmation of the
generic-RLVR contrast. It is not a full replication of the 8B baseline study.
Prefer a complete second 8B seed over a fragmented 14B baseline collection when
the main objective is statistical confidence at 8B.

The incremental one-epoch 14B costs are 115 accepted updates for math only,
230 for math plus Mastermind, and 345 for all three primary baselines, before
dynamic-filtering overhead.

### 4.5 Required execution order

Execute the study in this order:

1. freeze the comparison question, baseline list, epoch branch, and training
   seed plan in a dated decision record;
2. curate and audit all three static datasets using CPU-only processes;
3. implement and unit-test deterministic rewards using CPU-only processes;
4. copy the frozen artifacts to the separate GPU server and verify hashes;
5. run rollout-only calibration on untouched Qwen3-8B;
6. run isolated one-update smoke tests;
7. train the complete 8B epoch-1 matrix and save update 115;
8. decide the epoch-2 budget without inspecting downstream benchmark scores;
9. stop at epoch 1 or continue the complete chosen matrix to update 230;
10. run the frozen evaluation suite only at the declared primary endpoint;
11. analyze matched within-scale contrasts; and
12. consider a second 8B seed and then the gated 14B extension.

Store the dated decision record with the run artifacts. Any departure from this
order must include its reason and whether it was made before or after viewing
downstream results.

## 5. Common dataset contract

Every static baseline must materialize exactly two immutable parquet files:

```text
<baseline>_train.parquet  # 3,680 rows
<baseline>_val.parquet    #   320 rows
```

Every row must contain at least:

```text
prompt        list[{"role": string, "content": string}]
data_source   string
reward_model  {"ground_truth": ...}
extra_info    dict
```

`extra_info` must include:

```text
row_id             stable unique ID
source_dataset     upstream dataset/environment name
source_revision    immutable upstream revision or code hash
source_row_id      upstream row ID, if one exists
split_group_id     group used to prevent leakage
stratum            domain/difficulty tier
curation_version   local version string
```

Task-specific hidden state may be stored in `extra_info`, but it must never
appear in the serialized prompt.

Each dataset directory must also contain:

- `manifest.json` with source revisions, licenses, counts, seeds, tokenizer
  revision, filters, and SHA-256 hashes;
- `audit.json` with duplicate, leakage, length, label, and stratum statistics;
- `README.md` describing the transformation and reward;
- a deterministic recreation command; and
- the exact curation code at the same Git commit as the training run.

## 6. Common prompt and answer-format controls

Use the same five-section reasoning scaffold used by Codenames as the system
message for all static baselines. Put task instructions and the task-specific
final-answer contract in the user message.

Do not force Codenames-specific clue/guess tags onto another task. Use:

- math: final line `Answer: <integer>`;
- logic: final answer `<answer>LABEL</answer>`;
- Mastermind: final answer `[Guess]: [d1 d2 ... dL]`.

The five reasoning sections may be logged diagnostically, but must not receive
an auxiliary positive reward. This mirrors the current Codenames reward, which
does not hard-gate the thinking-section diagnostics. A malformed task answer
may receive the task’s minimum reward because it cannot be verified.

Do not add SFT demonstrations, reference reasoning, tool access, retrieval, or
domain-specific system messages to one baseline unless all matched conditions
receive the corresponding intervention.

## 7. Dataset admission gates

A dataset may enter GPU calibration only after passing all CPU-side gates.

### 7.1 Provenance and license gate

- Pin an immutable upstream revision.
- Record the dataset and every material upstream component’s license.
- Confirm that local redistribution of the derived 4,000-row subset is allowed.
- Preserve required source attribution.
- If rights are unclear, internal experimentation may proceed only under the
  project’s data-governance decision; do not publish the subset.

The linked processed DAPO-Math card did not state a dataset license during the
initial audit. Treat redistribution permission as unresolved until confirmed.

### 7.2 Schema and label gate

- Row count before sampling must equal the documented filtered-pool count.
- Required columns must be non-null.
- Every ground truth must parse under the exact training scorer.
- Every prompt must render through the exact model tokenizer/chat template.
- Hidden reward fields must be absent from rendered prompt text.
- Invalid or conflicting labels are removed as groups, not resolved by keeping
  whichever row appears first.

### 7.3 Duplicate and split-leakage gate

Create a canonical prompt form by Unicode-normalizing, case-folding, trimming,
and collapsing whitespace. Then require:

- zero canonical duplicates within the final 4,000 rows;
- zero `split_group_id` overlap between train and validation;
- zero exact canonical prompt overlap between train and validation; and
- manual review of near-duplicate pairs flagged by token-shingle Jaccard or
  MinHash similarity of at least 0.85.

When multiple paraphrases or questions arise from one source item, the source
item—not the row—is the split group.

### 7.4 Evaluation-contamination gate

Compare the candidate pool against all protected paper evaluations, including
GSM8K, AIME 2024, AIME 2025, GPQA-Diamond, RAT, NYT Connections, MUNCH,
MacGyver, New Yorker tasks, LLM-Grounding, AUT, DAT, CWT, and EQ-Bench inputs.

At minimum run:

1. exact canonical matching;
2. containment matching after boilerplate removal;
3. token-shingle/MinHash matching;
4. source-ID and URL matching where metadata exists; and
5. manual review of every flagged pair.

Exclude confirmed duplicates and transformations of evaluation items. Save the
flagged-pair report, reviewer decision, and exclusion reason. Do not use
evaluation labels or benchmark model scores to select training examples.

### 7.5 Length gate

Tokenize with the exact pinned Qwen3 tokenizer and chat template. Require:

- 100% of prompts at or below 2,048 tokens after rendering;
- no silent truncation during dataset loading;
- per-stratum prompt-length summaries (`min`, median, p90, p95, p99, `max`);
  and
- manual review of length outliers before exclusion.

Do not raise the prompt limit for only one baseline. If a common limit changes,
rerun the Codenames control or treat the result as unmatched.

### 7.6 Balance gate

The final 4,000 rows and the 3,680/320 split must match the predeclared
stratum quotas exactly. Sampling is deterministic under a recorded seed. Do not
resample until a preferred downstream result appears.

## 8. Baseline-specific curation

### 8.1 DAPO-Math

Use the English configuration. The initial audit found 14,116 rows, 13,985
canonical unique prompts, 131 duplicate rows, and five canonical duplicate
groups with conflicting integer answers. Recompute these figures at the pinned
revision rather than assuming them.

Procedure:

1. Use `source_prompt` as the source chat content; the processed `prompt`
   column is a plain string.
2. Extract only the mathematical problem, remove upstream prompt boilerplate,
   and wrap it in the common project scaffold.
3. Require an integer ground truth and verify it through the local
   `math_dapo` scorer.
4. Remove all canonical duplicates and every member of a conflicting-answer
   group.
5. Perform evaluation decontamination before selecting final rows.
6. Build deterministic structural strata using available source metadata,
   prompt length, and problem features. Use later rollout calibration to verify
   that the mixture is learnable; do not use protected evaluation performance.
7. Select 4,000 unique rows and then make a stratified 3,680/320 split.

For the primary random sample, set `dataset_curation_seed=20260904` and rank
eligible source IDs by `SHA256("20260904" || source_row_id)`. Select by that
order within the predeclared cross-strata of rendered-token-length quartile and
answer bucket (`negative`, `0_or_1`, `2_to_9`, `10_to_99`, `100_plus`). Allocate
the 4,000-row stratum quotas proportionally by largest remainder, subject to a
minimum of 20 rows for every nonempty cross-stratum. Use a second namespaced
hash (`"split" || row_id`) to select 320 validation rows with proportional
largest-remainder quotas. Hash ranking makes the sample independent of upstream
file order. Freeze the eligible-pool hash and quotas before GPU calibration.

Use `data_source="math_dapo"`. Keep the existing deterministic `+1/-1`
correctness scorer unless a documented scorer bug is found.

Because math is close to GSM8K/AIME, report those results as expected
trained-domain transfer. Creativity transfer remains the primary
Codenames-specific contrast.

### 8.2 RuleCollection

Use these six closed-answer domains:

- AR-LSAT;
- FOLIO;
- LogicNLI;
- Logical Deduction;
- ProntoQA; and
- ProofWriter.

Exclude CLUTRR’s free-text answers and the small LogiQA component from the
primary baseline.

The initial audit found that LogicNLI’s 8,000 training rows contain only 1,000
canonical unique prompts and that 412 of 500 packaged LogicNLI test prompts
appear exactly in its training data. Therefore:

- rebuild the split from the cleaned training pool;
- do not use the packaged test files as validation; and
- group exact and near-duplicate/template variants before splitting.

Use these total-row quotas:

| Domain | Total | Validation | Training |
|---|---:|---:|---:|
| AR-LSAT | 667 | 54 | 613 |
| FOLIO | 667 | 54 | 613 |
| LogicNLI | 667 | 53 | 614 |
| Logical Deduction | 667 | 53 | 614 |
| ProntoQA | 666 | 53 | 613 |
| ProofWriter | 666 | 53 | 613 |
| **Total** | **4,000** | **320** | **3,680** |

Stratify answer labels within each domain. Parse the last
`<answer>...</answer>` block, normalize case and whitespace, and compare only
against the domain’s legal answer set. Reject multiple or illegal final labels.

Use `dataset_curation_seed=20260904`. After canonicalization and grouping, rank
groups by `SHA256("20260904" || domain || split_group_id)`. Within each domain,
allocate its total quota across legal labels proportionally by largest
remainder, then take groups in hash order; never take two canonical duplicate
prompts merely to fill a quota. Select validation groups using the independent
namespace `"split" || domain || split_group_id`, meeting the fixed domain
counts above and proportional label counts as closely as whole groups allow.
If group sizes make an exact domain count impossible, deterministically retain
one canonical row per group for the primary dataset and record the discarded
variants; do not move a group across splits or change the seed.

Do not add RuleReasoner’s DADS domain reweighting. If rule order is shuffled,
materialize one deterministic row-specific permutation during curation and
store its seed. Do not create fresh per-epoch augmentations in the primary
comparison.

### 8.3 Mastermind-state

This is a derived static dataset, not a claim that TextArena publishes 4,000
fixed Mastermind prompts.

Use four declared configuration/difficulty strata, 1,000 rows each. A suggested
starting configuration is:

| Tier | Code length | Symbols | Duplicates | Turn limit/source |
|---|---:|---:|---:|---|
| 1 | 3 | 5 | No | custom easy configuration |
| 2 | 4 | 6 | No | TextArena default |
| 3 | 4 | 8 | No | TextArena hard |
| 4 | 6 | 12 | Yes | TextArena extreme |

The final tier assignment must also account for the number of codes consistent
with the visible history. Verify difficulty empirically during rollout
calibration; configuration name alone is not proof of difficulty.

Generate each row with deterministic game logic:

1. sample a hidden secret under the tier rules;
2. simulate only legal, non-winning prior guesses;
3. compute exact black/white feedback;
4. enumerate or otherwise verify that at least one secret is consistent with
   the complete visible history;
5. render the rules and history without the hidden secret; and
6. ask for exactly one next guess.

Set `split_group_id` to include tier/configuration and hidden-secret identity so
that histories for one secret do not cross train/validation. Use 920 training
and 80 validation rows per tier.

Use deterministic progress reward:

```text
progress = (black_pegs + 0.5 * white_pegs) / code_length
score    = 2 * progress - 1
```

Thus a zero-match or invalid guess receives `-1`, and an exact solution
receives `+1`. Return diagnostic fields for black pegs, white pegs, progress,
format failure, and exact solution. Unit-test the feedback and candidate-set
logic exhaustively for all small configurations and with randomized property
tests for larger configurations.

Do not use an LLM to create histories, choose secrets, or score guesses.

## 9. Reward admission and anti-hacking tests

Each reward implementation must pass a saved CPU test suite before rollout.

Required tests:

- correct canonical answer receives the maximum score;
- incorrect canonical answer receives the minimum/defined lower score;
- whitespace and harmless case variations behave as documented;
- multiple final answers cannot obtain credit accidentally;
- answer text outside the final-answer field cannot override the final answer;
- malformed, truncated, empty, and repeated-tag outputs are handled safely;
- NaN, infinity, exceptions, or missing labels never become positive reward;
- reward is deterministic across repeated calls;
- hidden Mastermind state cannot be recovered from prompt serialization; and
- at least 100 independently constructed gold cases per scorer agree with a
  simple reference implementation or manual oracle.

Use deterministic rewards for all three primary baselines. Do not call an
external judge or Python tool during their reward computation.

Reward ranges need not be numerically identical to Codenames because GRPO
normalizes within group, but reward density and ties still matter. Log and
compare:

- minimum, maximum, mean, and standard deviation;
- number of unique reward values;
- format-failure fraction;
- within-group reward standard deviation; and
- fraction of all-correct, all-incorrect, and otherwise zero-variance groups.

Do not add a positive format bonus. It can make a baseline learn formatting
instead of task performance.

## 10. GPU-server calibration and smoke tests

Perform these stages later on the separate GPU server. They are not authorized
for the currently occupied GPUs.

### 10.1 Rollout-only calibration

Use the untouched 8B base checkpoint, the exact training sampling settings,
and group size 32. Draw at least 16 held-out calibration prompts from every
stratum/domain:

- math: at least 16 per declared difficulty stratum;
- logic: at least 16 per domain;
- Mastermind: at least 16 per tier.

Calibration prompts must be disjoint from the final 4,000 rows. Do not perform
optimizer updates.

Admission targets are:

- scorer/infrastructure failure: exactly 0%;
- final-answer parse failure: below 1% where the base model understands the
  contract, otherwise revise the prompt consistently;
- response truncation: at most 2%;
- at least 50% of prompt groups have non-zero within-group reward variance;
- no stratum/domain has more than 80% zero-variance groups; and
- no evidence that a hidden answer is present in the prompt.

For binary rewards, record pass@1 and the count of successes among 32 samples.
For Mastermind, record exact success, mean progress, reward variance, and legal
move rate.

If a baseline fails the variance gate, recurate using task-intrinsic difficulty
metadata and repeat the calibration. Do not select examples based on their
performance on downstream paper benchmarks.

### 10.2 One-update smoke test

For each admitted baseline, run one isolated 32-prompt update with checkpoint
saving and validation enabled. Confirm:

- the accepted training batch contains 1,024 trajectories, while any additional
  attempted trajectories caused by dynamic filtering are counted separately;
- gradients and losses are finite;
- the intended reward function is actually called;
- reward-extra fields appear in logs;
- dynamic filtering and refill counts are recorded;
- the checkpoint can be reloaded; and
- no judge service, tool sandbox, or Codenames reward path is invoked.

Delete no artifacts automatically. Mark smoke-test checkpoints clearly so they
cannot be mistaken for experimental checkpoints.

### 10.3 14B saturation calibration

Before an optional 14B run, repeat a smaller rollout-only check with at least
eight prompts per stratum/domain and group size 32. If fewer than 30% of groups
have reward variance, do not silently change the dataset. Either omit that 14B
baseline or define a separate, explicitly unmatched hard-set experiment.

## 11. Training invariants

### 11.1 Launcher and resolved-config gate

Do not invoke `scripts/train_codenames_dapo.sh` unchanged for a baseline. It
binds the Codenames reward path and derives checkpoint cadence from the
Codenames YAML. Create a baseline launcher/config layer that imports one common
scientific configuration and changes only the declared dataset, reward, run
name, model path, epoch branch, and hardware-dependent fields.

Before each run, generate a machine-readable diff between the resolved
Codenames configuration and the resolved baseline configuration. The only
permitted differences without a protocol amendment are:

- training and validation parquet paths;
- `data_source` values carried by rows;
- custom reward function path/name and task-specific reward kwargs;
- experiment/run names and artifact paths;
- `trainer.total_epochs`, according to the declared one-/two-epoch branch;
- trainee model path when moving from 8B to 14B; and
- the system-dependent parameters explicitly allowed below.

Any other difference must be justified in the decision record before training.

Set checkpoint frequency to a divisor of 115, preferably 23, or add explicit
epoch-boundary saving. Verify from a smoke test that update 115 is saved and is
reloadable. For a two-epoch run, the same mechanism must save update 230.

### 11.2 Frozen scientific settings

The following scientific hyperparameters are frozen across Codenames and all
static baselines at a matched scale/endpoint:

| Parameter | Required value |
|---|---:|
| Global prompts per accepted update | 32 |
| Rollouts per prompt | 32 |
| Trajectories per accepted update | 1,024 |
| Optimizer updates, epoch 1 | 115 |
| Optimizer updates, epoch 2 | 230 cumulative |
| Learning rate | `1e-6`; same scheduler/warmup as resolved Codenames config |
| Advantage estimator | GRPO |
| Clip low/high | `0.20 / 0.28` |
| Loss aggregation | token mean |
| KL loss/reward | disabled |
| Maximum prompt length | 2,048 |
| Maximum response length | 16,384 |
| Overlength buffer | 4,096 |
| Overlength penalty factor | 1.0 |
| Dynamic filtering | enabled on sequence reward |
| Maximum generation batches per refill | 10 |
| Precision | full-parameter BF16 |

### 11.3 Permitted system-dependent changes

System-dependent parameters may change to fit the server or model scale:

- tensor-parallel size;
- FSDP sharding/offload details;
- per-GPU micro-batch size, provided the global batch is unchanged;
- rollout GPU-memory utilization;
- checkpoint bucket sizes; and
- node/GPU count.

Record every such change. Do not change semantic training parameters merely to
make one baseline faster.

For each run, pin and record:

- model and tokenizer repository plus resolved revision SHA;
- code Git commit and dirty diff;
- dataset and reward hashes;
- Python/package lock and CUDA/NCCL/vLLM/VeRL versions;
- all RNG seeds;
- GPU model and count;
- exact Hydra-resolved configuration; and
- W&B run ID and checkpoint locations.

Checkpoint exactly at updates 115 and, if applicable, 230. Epoch 2 must resume
with optimizer, scheduler, sampler, and RNG state—not restart from weights only.

No early stopping and no best-validation-checkpoint selection are permitted for
the main comparison.

## 12. Compute accounting

Equal dataset rows do not imply equal compute. DAPO discards zero-variance
groups, response lengths differ, and environment games contain multiple model
turns.

For every accepted update and whole run, record:

- unique source prompts visited;
- attempted prompt groups;
- accepted prompt groups;
- all-correct, all-incorrect, and other zero-variance groups;
- number of refill batches and max-refill hits;
- policy completions attempted and accepted;
- prompt, generated, accepted, and discarded policy tokens;
- truncated completions;
- optimizer updates;
- GPU-hours and wall time; and
- reward/scoring wall time.

The primary static comparison matches optimizer updates and configured group
size. Also report token and GPU-hour ratios relative to Codenames.

If a baseline differs from Codenames by more than 25% in total generated policy
tokens at the endpoint, report that imbalance as a limitation. A secondary
token-budget-matched checkpoint may be evaluated if it was defined before
downstream results are inspected; it does not replace the update-matched
primary checkpoint.

For static `wordle-state`, use the same update-count match as the other static
baselines and report its generated-policy-token ratio. For a separate online
Wordle experiment, the matching unit is total policy-generated tokens and
episodes, not “4,000 prompts”; record environment/tool-response tokens
separately from policy tokens.

## 13. Runtime safety and abort rules

Stop and diagnose a run without consuming further GPU budget if any of the
following occurs:

- NaN/Inf loss, gradient norm, log probability, or reward;
- reward/scorer infrastructure failure above 0.1% of trajectories;
- wrong reward function or wrong dataset source is detected;
- hidden labels appear in prompts;
- prompt truncation occurs;
- response truncation exceeds 5% for three consecutive accepted updates;
- the ten-batch refill ceiling is reached in more than 20% of accepted updates
  over a 20-update window;
- checkpoint reload fails; or
- data or code hashes change mid-run.

A high task-format failure rate is not automatically an infrastructure abort,
but a rate above 95% for three consecutive updates indicates that the model is
receiving essentially no task signal and requires inspection.

Do not stop or extend training because a downstream benchmark result is
disappointing or favorable.

## 14. Evaluation plan

### 14.1 Models evaluated

For the 8B primary endpoint, evaluate on identical prompts and settings:

```text
8B base
8B Codenames at matched endpoint
8B DAPO-Math at matched endpoint
8B RuleCollection at matched endpoint
8B Mastermind-state at matched endpoint
```

Add static Wordle only as a predeclared supplemental mechanism analysis. Keep
any online Wordle experiment separate from the static matched table.

For 14B, evaluate only the predeclared available matrix. Do not fill missing
14B cells with 8B scores or compare cross-scale absolute values as treatment
effects.

### 14.2 Choosing the evaluation endpoint under the epoch budget

- If training stops after one epoch, update 115 is the primary endpoint and
  receives the full evaluation suite.
- If two epochs are funded before downstream evaluation, update 230 is the
  primary endpoint. Preserve update 115; evaluate it fully only if learning
  curves are a predeclared secondary analysis and budget permits.
- If epoch-2 funding is decided after reaching epoch 1, make the decision from
  compute availability and training-health logs before viewing downstream
  benchmark scores.

This prevents benchmark-driven horizon selection and avoids paying for the
full 19,370-completion suite at every intermediate checkpoint by default.

### 14.3 Evaluation invariants

Use the paper’s frozen evaluation package:

- 23 reported metrics across the 10 creativity task families;
- GSM8K, AIME 2024, AIME 2025, and GPQA-Diamond;
- the same prompts and item order;
- the same generation sampling parameters and token budgets;
- the same number of generations per item;
- the same parsers and metric implementations;
- the same judge model, exact judge revision if available, temperature, and
  retry policy; and
- thinking mode exactly as used in the main paper, including any separately
  labelled NYT Connections diagnostic.

Freeze evaluation prompt files and item IDs once. Reuse base and Codenames
outputs only if checkpoint identity, evaluation code, prompts, sampling
settings, and judge configuration match exactly. Otherwise rerun them.

Use common random numbers where supported: the same per-item generation seeds
for every checkpoint. A generation seed does not make different model samples
identical, but it reduces avoidable pipeline variation.

For LLM-judged tasks:

- hide checkpoint/baseline identity from the judge;
- keep rubric text and reference answers fixed;
- log raw judge request/response, parser status, retries, and failures;
- never retry selectively based on score; and
- adjudicate persistent failures under a rule defined before model comparison.

### 14.4 Primary and secondary outcomes

Report all original raw metrics. Do not invent a favorable post-hoc aggregate.

The co-primary creativity outcomes for the baseline study are:

1. DAT DSI change from base;
2. DAT vocabulary-size change from base; and
3. the Codenames-versus-baseline specificity contrast on those two measures.

These directly test the paper’s scale-dependent diversity claim at 8B.

The broader creativity profile is the complete 23-metric vector. Report:

- change from the common base for every trained checkpoint;
- Codenames-minus-baseline specificity for every metric;
- counts of positive, tied, and negative changes, with the tie tolerance
  declared before results are inspected; and
- benchmark-family summaries without giving large families more weight merely
  because they expose more metrics.

Reasoning benchmarks are secondary outcomes and mechanism controls. Report raw
correct counts as well as accuracy, especially for the 30-item AIME sets.

### 14.5 Uncertainty and multiplicity

Use paired resampling because checkpoints answer the same items:

- item-level paired bootstrap with at least 10,000 replicates for ordinary
  item-based metrics;
- hierarchical bootstrap for tasks with multiple generations per item;
- generation-level bootstrap for DAT, which has one prompt and many sampled
  lists; and
- paired binary intervals/tests or paired bootstrap for accuracy tasks.

Report 95% confidence intervals for `Delta` and `Specificity`. For the 23
individual creativity comparisons, report Benjamini-Hochberg FDR-adjusted
secondary p-values if p-values are used. Do not let multiplicity-adjusted
significance replace effect sizes and intervals.

With only one training seed, label intervals as evaluation-sampling uncertainty
only. They do not represent variation across RL training runs.

## 15. Analysis and interpretation rules

Use the following terminology:

- **Codenames-specific evidence:** Codenames exceeds both generic RLVR controls
  and the non-semantic game control at the same model scale and epoch.
- **Generic RLVR effect:** Codenames and math/logic move similarly relative to
  base.
- **Generic game effect:** Codenames and Mastermind move similarly, especially
  if math/logic do not.
- **Word-game effect:** Codenames and supplemental Wordle move similarly, while
  Mastermind does not.
- **Inconclusive:** confidence intervals are wide, run-to-run variance is
  unavailable, or compute/reward distributions are materially unmatched.

Do not infer that a baseline “fails to improve creativity” merely because its
point estimate is smaller. Use the matched specificity contrast and uncertainty.

Always discuss these possible confounds:

- different reward density and tie structure;
- different response lengths and actual policy-token budgets;
- DAPO refill rates;
- direct domain overlap with reasoning evaluations;
- prompt/output-format difficulty;
- judge noise on Codenames clue examples and creativity evaluations; and
- single-seed RL variance.

## 16. Required artifacts and completion checklist

A baseline is complete only when all boxes below can be checked.

### Data

- [ ] Source revision and license recorded.
- [ ] Transformation reproducible from code and seed.
- [ ] Exactly 3,680 train and 320 validation rows.
- [ ] Stable unique row IDs and split-group IDs.
- [ ] Zero canonical duplicates in the final 4,000 rows.
- [ ] Zero train/validation split-group overlap.
- [ ] Protected-evaluation contamination audit saved.
- [ ] Exact tokenizer length audit saved.
- [ ] Dataset and manifest SHA-256 hashes recorded.

### Reward

- [ ] Gold and adversarial unit tests pass.
- [ ] Reward deterministic and bounded.
- [ ] Ground truths all parse.
- [ ] Hidden state absent from prompts.
- [ ] No judge/tool dependency for primary baselines.
- [ ] Diagnostic reward fields are logged.

### GPU preflight

- [ ] Rollout-only calibration passed on untouched 8B.
- [ ] Group reward-variance targets passed.
- [ ] One-update smoke test passed.
- [ ] Checkpoint save/reload tested.
- [ ] 14B saturation check passed, if applicable.

### Training

- [ ] Exact model/tokenizer revisions match Codenames.
- [ ] Resolved config and environment captured.
- [ ] Endpoint checkpoint exists at update 115 or 230.
- [ ] No early stopping or downstream-driven checkpoint selection.
- [ ] Attempted/accepted groups and policy tokens recorded.
- [ ] Run completed without an unresolved abort condition.

### Evaluation and reporting

- [ ] Endpoint is epoch-matched to Codenames.
- [ ] Evaluation code/config hashes match across checkpoints.
- [ ] Base/Codenames outputs were safely reused or rerun.
- [ ] All raw metrics and matched deltas reported.
- [ ] DAT co-primary specificity contrasts reported.
- [ ] Confidence intervals use the correct resampling unit.
- [ ] Training-seed limitation stated where applicable.
- [ ] Static Wordle, if run, is labelled supplemental; online Wordle is kept
      separate from static matched baselines.

## 17. Pre-run decision-record template

Complete and commit this record before copying artifacts to the GPU server:

```yaml
study_id: baseline-comparison-YYYYMMDD
decision_timestamp_utc: ""
decision_made_before_downstream_results: true

primary_model:
  id: Qwen/Qwen3-8B
  revision: ""
  tokenizer_revision: ""

primary_endpoint:
  epochs: 1                 # 1 or 2
  optimizer_updates: 115    # 115 or 230
  reason: "compute-budget decision made before benchmark evaluation"

primary_baselines:
  - math
  - logic
  - mastermind
supplemental_wordle_state: false
supplemental_wordle_online: false

training_seeds: []
dataset_curation_seed: null
evaluation_generation_seeds: []

codenames_checkpoint:
  path_or_id: ""
  optimizer_updates: null
  model_revision_matches: false
  resolved_config_sha256: ""

datasets:
  math_manifest_sha256: ""
  logic_manifest_sha256: ""
  mastermind_manifest_sha256: ""

code:
  git_commit: ""
  dirty_diff_sha256: ""
  environment_lock_sha256: ""

evaluation:
  prompt_bundle_sha256: ""
  code_sha256: ""
  judge_model_and_revision: ""
  full_suite_only_at_primary_endpoint: true

optional_14b:
  enabled: false
  question: ""
  baselines: []             # priority: math, then mastermind, then logic
  epochs: null
  optimizer_updates: null

approved_config_differences: []
known_limitations: []
approver_or_owner: ""
```

Use run names of the form:

```text
baseline-<source>-qwen3-<scale>-u<updates>-seed<seed>-<datahash8>
```

Store the decision record, resolved configs, logs, checkpoints, evaluation
outputs, and analysis under one immutable study directory. Never overwrite an
earlier run with a later retry; assign the retry a new run ID and record why it
was repeated.
