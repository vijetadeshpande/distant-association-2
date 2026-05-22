# OpenRouter Judge — Thinking On / Off / Omitted

**Date:** 2026-05-22
**Judge model:** `google/gemini-2.5-flash` (OpenRouter backend)
**Purpose:** Verify how the judge behaves under the new `JUDGE_THINKING`
toggle, and confirm the new explicit *thinking-off* default does not
regress against the previous code (which omitted the `reasoning` field).

## Setup

For each of 4 validation examples (one per `difficulty_rule`) the judge
was called with the **production prompt builder** (`build_guess_messages`)
and the response scored with the **production parsers / reward**
(`parse_guesses`, `task_reward`). Three request variants were compared:

| Config | `reasoning` field sent | `max_tokens` | Represents |
|---|---|---|---|
| `thinking_off` | `{"enabled": false}` | 1024 | **new default** (`JUDGE_THINKING=0`) |
| `thinking_on`  | `{"max_tokens": -1}` | 32768 | toggle on (`JUDGE_THINKING=1`), dynamic budget |
| `no_argument`  | *(field omitted)* | 1024 | **old code**, pre-change behaviour |

Examples (all drawn from `custom_data/training_prompts/version-5/codenames_rlvr_val.parquet`,
`codenames_guess_generation` rows — the judge *is* a guess generator):

| Difficulty | Row | Clue | Target set | Non-target set | Max-Guesses |
|---|---|---|---|---|---|
| simple   | 0   | `reinforcing` | concrete, bond   | spring, curve | 2 |
| moderate | 80  | `gouda`       | round, lactose   | earth, brown  | 2 |
| advance  | 160 | `engineering` | modulus, tensile | brown, tree   | 2 |
| expert   | 241 | `rounded`     | large, cycloid   | pile, volume  | 1 |

All four boards have 4 words; "difficulty" here reflects clue/target vs.
clue/non-target cosine separation, not board size.

## Raw results (12 calls)

| Difficulty | Config | Guesses | Reward | pos / negNT / negINV | reasoning_tok | completion_tok | latency | finish |
|---|---|---|---|---|---|---|---|---|
| simple   | thinking_off | `[bond, concrete]`   | **+1.000** | 1.0 / 0.0 / 0.0 | 0   | 28  | 0.70s | stop |
| simple   | thinking_on  | `[concrete, bond]`   | **+1.000** | 1.0 / 0.0 / 0.0 | 473 | 501 | 4.41s | stop |
| simple   | no_argument  | `[bond, concrete]`   | **+1.000** | 1.0 / 0.0 / 0.0 | 0   | 28  | 0.66s | stop |
| moderate | thinking_off | `[lactose, brown]`   | **+0.000** | 0.5 / 0.5 / 0.0 | 0   | 28  | 0.62s | stop |
| moderate | thinking_on  | `[round, brown]`     | **+0.000** | 0.5 / 0.5 / 0.0 | 664 | 692 | 6.02s | stop |
| moderate | no_argument  | `[lactose, brown]`   | **+0.000** | 0.5 / 0.5 / 0.0 | 0   | 28  | 0.69s | stop |
| advance  | thinking_off | `[tensile, modulus]` | **+1.000** | 1.0 / 0.0 / 0.0 | 0   | 28  | 0.53s | stop |
| advance  | thinking_on  | `[modulus, tensile]` | **+1.000** | 1.0 / 0.0 / 0.0 | 648 | 676 | 5.64s | stop |
| advance  | no_argument  | `[tensile, modulus]` | **+1.000** | 1.0 / 0.0 / 0.0 | 0   | 28  | 0.65s | stop |
| expert   | thinking_off | `[cycloid]`          | **+0.500** | 0.5 / 0.0 / 0.0 | 0   | 27  | 0.61s | stop |
| expert   | thinking_on  | `[cycloid]`          | **+0.500** | 0.5 / 0.0 / 0.0 | 478 | 505 | 4.34s | stop |
| expert   | no_argument  | `[cycloid]`          | **+0.500** | 0.5 / 0.0 / 0.0 | 0   | 27  | 0.74s | stop |

## Findings

### 1. `thinking_off` ≡ `no_argument` — verified byte-identical

For **all 4 difficulties** the `thinking_off` and `no_argument` responses
were **byte-identical** (`message.content` exact match) and both reported
`reasoning_tokens = 0`. The new explicit default (`reasoning:
{"enabled": false}`) is therefore a **behavioural no-op** versus the old
code that omitted the field — switching the default carries **zero
regression risk**, while making "no thinking" explicit and enforced
rather than relying on OpenRouter's implicit per-model default.

### 2. `thinking_on` genuinely engages reasoning

The toggle produces real chain-of-thought: 473–664 reasoning tokens
(avg ≈ 566), with visible multi-step deliberation, e.g. the *simple*
example:

> *"I'm prioritizing 'concrete' due to the direct association with
> reinforced structures. 'Bond' also stands out as it relates to
> strengthening connections…"*

The reasoning is hidden from `message.content` (it lands in the separate
`reasoning` field), so **`parse_guesses` is unaffected** — the final
content is still just the `[CODENAMES-GUESS-START] … END]` block.

### 3. Thinking did **not** change the reward on this sample

Final `task_reward` was **identical across all three configs** for every
example (simple +1.0, moderate 0.0, advance +1.0, expert +0.5). On 3/4
examples the guessed *set* was also identical (only ordering differed).

The one set-level difference was **moderate** (`gouda`):
- without thinking → `[lactose, brown]`
- with thinking → `[round, brown]`

Both score 0.0 (one target + the `brown` distractor). Notably, the
thinking trace *reasoned its way into* the distractor — it explicitly
noted "'brown' as a common rind colour" of Gouda and kept it. So on this
trap, deliberation rationalised the wrong pick rather than avoiding it;
thinking neither helped nor hurt.

### 4. No truncation; `max_tokens=32768` is comfortable

Every call finished with `finish_reason=stop`. With thinking on,
completion was only 501–692 tokens — far below the 32768 cap. The
dynamic budget stays modest (~470–660 reasoning tokens) on these 4-word
boards. `32768` leaves a large safety margin for bigger boards / harder
clues where the dynamic budget (capped at 24576 for Gemini 2.5 Flash)
could grow.

### 5. Cost & latency of thinking

| Metric | thinking_off / no_argument | thinking_on | Ratio |
|---|---|---|---|
| Avg latency | ≈ 0.65 s | ≈ 5.10 s | **~7.8× slower** |
| Avg completion tokens | ≈ 28 | ≈ 594 | ~21× |
| Avg reasoning tokens | 0 | ≈ 566 | — |
| Extra OpenRouter cost / call | — | ≈ +$0.0014 (566 tok @ \$2.5/M) | — |

Latency is parallelised by `JUDGE_CONCURRENCY=128`, but a ~7–8× per-call
slowdown still adds meaningfully to clue-task reward time per training
step.

## Conclusion

- **Default = no thinking is correct and safe.** The explicit
  `reasoning: {"enabled": false}` default reproduces the old code's
  output exactly (byte-identical, 0 reasoning tokens) on every
  difficulty level — no regression.
- **`JUDGE_THINKING=1` works as intended:** dynamic reasoning budget,
  no output truncation, parsing unaffected.
- **No measured accuracy benefit** from judge thinking on this 4-example
  probe — rewards were identical — while it costs ~7.8× latency and
  extra tokens. Keeping thinking **off by default** is well justified;
  the toggle is available for ad-hoc experiments if a larger evaluation
  later shows a quality gain.

## Reproduce

Examples: `/tmp/judge_test_examples.json` · Raw results:
`/tmp/judge_test_results.json`. Regenerate by selecting one
`codenames_guess_generation` row per `difficulty_rule` from the v5 val
parquet and POSTing to OpenRouter `/v1/chat/completions` with
`build_guess_messages(...)` under the three `reasoning` variants above.
