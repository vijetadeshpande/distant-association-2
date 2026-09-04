# Baseline selection decision — 2026-09-04

## Decision

Use the following Qwen3-8B comparison matrix at the matched, predeclared epoch
endpoint:

1. untouched Qwen3-8B base;
2. Codenames;
3. cleaned English DAPO-Math (generic numerical RLVR);
4. cleaned six-domain RuleCollection (natural-language rule reasoning); and
5. static single-decision Mastermind states (non-semantic game RLVR).

Mastermind is the selected primary game baseline. If budget supports a fourth
training source, add **static single-decision Wordle states** as a supplemental
lexical/orthographic control. Do not substitute Wordle for Mastermind. Othello
4x4 is the reserve non-semantic game if Mastermind fails the full admission
calibration; it is not currently selected.

This decision confirms the primary baseline list already proposed in
`baseline-selection-protocol.md`, while changing the optional Wordle design
from online-only to a static one-decision design.

## Evidence from the Qwen3-8B selection pilot

### Setup

- Model: `Qwen/Qwen3-8B`, local immutable snapshot
  `b968826d9c46dd6066d109eabc6255188de91218`.
- TextArena: version 0.7.3 source at commit
  `6dfb577c01d0337fe03b05adce84c5ae1878eca1` (2026-08-19).
- Inference: BF16 vLLM 0.11.0 on GPU 6 (RTX 6000 Ada), temperature 1.0,
  unrestricted top-p/top-k, 768-token cap.
- Prompting: condensed version of the project's five-section reasoning
  scaffold; Qwen non-thinking mode; one model decision per state.
- Sample: eight states per game and 16 completions per state (128 per game),
  except the reported Sudoku aggregate retains only the five states verified
  to have a unique solution (80 completions).
- Scoring: lenient extraction accepted the model's harmless omission of an
  extra nested pair of brackets. Illegal or absent actions still received the
  minimum reward.
- This is a comparative selection pilot, not the protocol's 32-rollout
  admission calibration. The default-thinking 512-token screen truncated
  98–100% of completions before their final answers, confirming that the full
  calibration must use the frozen long response budget.
- Environment note: the shared `env_verl` was found with NumPy 2.4.0, which is
  incompatible with its installed Numba 0.61.2 and VeRL package requirements.
  NumPy 1.26.4 was used temporarily for inference and the original 2.4.0 was
  restored afterward. Build a clean lock before the admission calibration.

### Results

| Candidate | Legal action | Maximum-reward action | Non-zero-variance state groups | Mean reward | Median generated tokens | Selection |
|---|---:|---:|---:|---:|---:|---|
| Mastermind | 96.9% | 0.8% exact code | 100% | -0.262 | 493 | **Primary game** |
| Wordle | 64.1% | 0.8% exact word | 100% | -0.648 | 442 | Optional static lexical control |
| Othello 4x4 | 37.5% | 18.0% minimax-optimal | 87.5% | -0.641 | 507 | Reserve only |
| Sudoku (unique subset) | 93.8% | 52.5% correct cell | 80.0% | 0.050 | 418 | Exclude |
| TicTacToe | 88.3% | 52.3% minimax-optimal | 100% | 0.047 | 490 | Exclude |
| SimpleNegotiation accept/deny | 100% | 98.4% utility-optimal | 12.5% | 0.969 | 257 | Exclude |
| Chess | 1.6% | 0% Stockfish-top move | 0% | -1.000 | 327 | Exclude |

Mastermind exposed 12 reward values in `[-1, 1]`, rather than a binary signal.
Its four tested configurations all remained below saturation: `(length,
symbols, duplicates) = (3,5,no), (4,6,no), (4,8,no), (6,12,yes)` had mean
rewards `-0.167, -0.164, -0.258, -0.458`, respectively, while legal-move rates
were 93.8–100%.

Raw pilot artifact hashes:

```text
0505d0bd8cdb563f06c66ae41ac07bf5129a99c537fb08e682e3343c0a9e3ab0  qwen3_8b_textarena_pilot_nothink.jsonl
4bbbb8cf9c1ac5cc67ccfc2e368c1d75904d3f10b0a6371ae56cd73f34949b65  qwen3_8b_textarena_pilot_nothink_rescored.jsonl
e79d741a9f4a25b4dfcbbe08898b515cfff000bfa6956a8137aad9f3478d025e  textarena_baseline_pilot.py
```

The raw artifacts were selection scratch files under `/tmp`; the table and
hashes are the durable decision evidence. A production curation/calibration
implementation must be checked into the repository separately.

## Why each game was selected or rejected

### Mastermind — selected

- Naturally supports a one-decision state: show only prior guesses and
  black/white feedback, then request one next guess.
- Deterministic local scoring needs neither an opponent, LLM judge, nor tool.
- Dense progress reward is directly adapted from TextArena:
  `2 * (black + 0.5 * white) / code_length - 1`.
- The pilot has the desired DAPO profile: almost all outputs are scoreable,
  exact solutions are rare, every group varies, and difficulty changes smoothly
  across configurations.
- It is non-linguistic and therefore isolates generic game/search RL from the
  distant-semantic-association mechanism in Codenames.

### Wordle — optional static supplemental baseline

- A visible Wordle history plus one next guess is also honestly single-turn;
  there is no need to use online environment RL solely because precedent does.
- The same black/white-style graded signal maps cleanly to
  `2 * (green + 0.5 * yellow) / word_length - 1`.
- It provides useful lexical/orthographic triangulation: if Codenames and
  Wordle transfer similarly while Mastermind does not, a word-game mechanism
  is more plausible.
- It remains supplemental because it is linguistically closer to the paper's
  intervention and its legal-word dictionary is an extra source of curation
  and reward behavior.

### Othello 4x4 — reserve only

- A static next move is possible and exact minimax is tractable at 4x4.
- It needs a new minimax labeling oracle; immediate disc flips are a hackable,
  myopic reward and default 8x8 game value is not cheaply exact.
- The pilot's 37.5% legal rate and longer traces make dynamic filtering and
  compute matching less attractive than Mastermind.

### Sudoku — excluded

- One-cell prediction can be static and Qwen's difficulty changed with clue
  count, but the reward is binary and the task is essentially another logic
  reasoning control rather than a clean game-structure control.
- Full-grid solving would restore a dense completion reward only by making the
  action and trace much longer than Codenames/Mastermind.

### TicTacToe — excluded

- Static minimax labels work, but the game has only 4,520 reachable
  non-terminal raw states and just 627 states after rotation/reflection
  canonicalization. A 4,000-row set would nearly enumerate the game and repeat
  many symmetry-equivalent states, making memorization and split leakage the
  dominant concern.
- It also has only binary optimal/non-optimal reward under the proposed static
  scorer.

### SimpleNegotiation — excluded

- The only opponent-independent single-decision conversion is accept/deny of a
  final offer, which Qwen saturated at 98.4% and reduces the game to arithmetic.
- Generating an offer or measuring negotiation quality requires another policy
  and a multi-turn outcome, adding linguistic/social and opponent confounds.

### Chess — excluded

- A one-move prompt is possible, but meaningful reward requires a pinned chess
  engine, search budget, and score shaping. TextArena itself only supplies the
  multi-turn game outcome.
- The pilot produced no engine-top moves and essentially no reward variance.
  Showing all legal moves could fix formatting/legality but not the engine and
  domain-prior confounds.

## Frozen DAPO-Math sampling and reward plan

1. Pin English config revision
   `31dd309567e3da778038cc87d868b6097a3ccf68` and record that the upstream card
   currently does not declare a dataset license; resolve redistribution before
   publishing derived rows.
2. Extract the problem from `source_prompt`; canonicalize Unicode/case/space;
   remove all duplicate rows and every row in conflicting-answer groups; then
   perform the complete protected-evaluation contamination audit.
3. Require the ground truth to parse under the exact local `math_dapo` scorer
   and the rendered Qwen prompt to be at most 2,048 tokens.
4. Use seed `20260904` with stable SHA-256 ranking, not order-dependent RNG.
   Sample within rendered-length quartile × integer-answer bucket and allocate
   4,000 quotas proportionally by largest remainder. The buckets are negative,
   0/1, 2–9, 10–99, and 100+.
5. Split 3,680/320 using an independent hash namespace and proportional strata.
   Freeze eligible-pool hash, quotas, row IDs, and parquet hashes before GPU
   calibration; never resample based on rollout or downstream scores.
6. Reward: require exactly one final line `Answer: <integer>` in the last 300
   characters, normalize only the documented harmless formatting, and return
   `+1` for exact normalized equality and `-1` otherwise. Multiple answer lines,
   non-integers, NaN/Inf, malformed output, or exceptions receive `-1`; no
   positive format bonus.

## Frozen RuleCollection sampling and reward plan

1. Pin revision `f14a766d2e8e46390101154c6f4f51a67d7a5d9f` and retain AR-LSAT,
   FOLIO, LogicNLI, Logical Deduction, ProntoQA, and ProofWriter only. Preserve
   the protocol's exact per-domain 4,000/320 quotas.
2. Canonicalize and group source problems/templates before sampling. Rebuild
   validation from cleaned training pools; do not use packaged test splits,
   especially LogicNLI's contaminated test set.
3. Within domain, allocate quotas proportionally across legal answer labels by
   largest remainder. Rank groups with
   `SHA256("20260904" || domain || split_group_id)` and use a separate `split`
   namespace for validation. Retain one canonical prompt per group if needed to
   meet exact row counts without leakage. Freeze all quota and discard records.
4. If rule order is shuffled, materialize exactly one row-specific permutation
   from a stored seed during curation; do not reshuffle by epoch.
5. Reward: require exactly one `<answer>LABEL</answer>` block, normalize case
   and surrounding whitespace only, verify membership in the domain's legal
   label set, and return `+1/-1` for exact equality. Zero or multiple answer
   blocks, illegal labels, malformed tags, NaN/Inf, or exceptions receive
   `-1`; no format bonus and no DADS domain reweighting.

## Required next gate before training

Materialize and audit all three selected static datasets, then run the full
untouched-Qwen3-8B rollout calibration from the protocol: at least 16 prompts
per math stratum/logic domain/Mastermind tier, 32 rollouts each, the exact full
system prompt and 16,384 response limit, with zero scorer failures and at least
50% non-zero-variance groups. Only after admission should isolated one-update
smoke tests or baseline training begin.
