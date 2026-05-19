# Qualitative analysis: Codenames DAPO training of Qwen3-8B

Comparing three checkpoints on the same 24 prompts:

| tag    | model                                                                | role               |
|--------|----------------------------------------------------------------------|--------------------|
| `base` | `Qwen/Qwen3-8B` (off-the-shelf)                                      | baseline           |
| `s56`  | `codenames-dapo-qwen3-8b … global_step_56/actor` (~0.5 epoch)        | mid-training       |
| `s112` | `codenames-dapo-qwen3-8b … global_step_112/actor` (~1 epoch)         | end of training    |

Source files: [qwen3-8b.jsonl](qwen3-8b.jsonl), [codenames-dapo-qwen3-8b-step56.jsonl](codenames-dapo-qwen3-8b-step56.jsonl), [codenames-dapo-qwen3-8b-step112.jsonl](codenames-dapo-qwen3-8b-step112.jsonl).
Analysis script: [scripts/analyze_qualitative.py](../scripts/analyze_qualitative.py).
Raw metrics: [qual_metrics.json](qual_metrics.json).

The 24 prompts split evenly into 12 clue-generation and 12 guess-generation tasks, with 6 prompts at each of the four difficulty levels (simple, moderate, advance, expert). All decoding used the same seed (`seed=42`, `temperature=1.0`, `top_p=1.0`, `max_tokens=16384`).

---

## 0. Sanity checks

| check                                                       | result    |
|-------------------------------------------------------------|-----------|
| 24 prompts byte-identical across all three runs             | **pass** (0 mismatches) |
| no completion is byte-identical between any pair of models  | **pass** (0 duplicates) |
| every response contains all five CoT tags                   | **pass** (24/24, all models) |
| every clue response has the four `[CODENAMES-CLUE-…]` tags  | **pass** (12/12, all models) |
| every guess response has the three `[CODENAMES-GUESS-…]` tags | **pass** (12/12, all models) |
| five CoT sections appear in canonical order                 | **pass** (24/24, all models) |

The comparison is apples-to-apples; observed differences are model differences, not prompt or sampler artifacts.

---

## 1. Headline numbers

### Clue generation (12 prompts)

| metric                                | base   | s56    | s112   | Δ (s112−base) |
|---------------------------------------|-------:|-------:|-------:|--------------:|
| completion length (chars, mean)       | 14 565 | 16 773 | 16 526 | +1 961 (+13%) |
| `<think>` length (chars, mean)        | 12 755 | 15 230 | 15 010 | +2 255 (+18%) |
| "wait" tokens / response              |   8.1  |  10.3  |  13.3  |        +5.2   |
| "alternatively" tokens / response     |   5.5  |   7.8  |   6.8  |        +1.3   |
| candidate clues considered (quoted)   |  24.0  |  21.2  |  26.0  |        +2.0   |
| selected-target ÷ total-target ratio  |  0.96  |  0.94  |  0.97  |       +0.01   |
| clue is single English word           |  100%  |  100%  |  100%  |          —    |
| clue appears in either set            |    0%  |    0%  |    0%  |          —    |
| clue is morphological variant of a set word | 0%  |    0%  |    0%  |          —    |

### Guess generation (12 prompts)

| metric                                | base   | s56    | s112   | Δ (s112−base) |
|---------------------------------------|-------:|-------:|-------:|--------------:|
| completion length (chars, mean)       |  5 542 |  4 452 |  5 934 |  +392 (+7%)   |
| `<think>` length (chars, mean)        |  4 121 |  3 267 |  4 753 |  +632 (+15%)  |
| guesses returned / response (mean)    |  3.17  |  3.50  |  3.50  |       +0.33   |
| guesses returned > `max_guesses`      |    0   |    0   |    0   |          —    |
| guesses outside `[All-Words]` (invalid) |  0   |    0   |    0   |          —    |
| **recall = correct ÷ targets**        |  0.715 |  0.847 |  0.826 |       +0.11   |
| **precision = correct ÷ guesses**     |  0.889 |  0.972 |  0.951 |       +0.06   |
| mean wrong (non-target) guesses       |  0.25  |  0.08  |  0.17  |       −0.08   |

### Guess recall by difficulty

| difficulty | base  | s56   | s112  |
|------------|------:|------:|------:|
| simple     | 0.889 | 0.889 | 0.806 |
| moderate   | 0.667 | **1.000** | **1.000** |
| advance    | 0.889 | **1.000** | **1.000** |
| expert     | 0.417 | 0.500 | 0.500 |

Both training checkpoints **saturate the moderate and advance buckets**, leave the expert bucket clearly improved but still hard, and slightly regress on the simple bucket because of two over-clever lateral guesses (see §5).

---

## 2. Are responses longer?

**Clue task — yes, consistently.** `<think>` traces grow ~18% from base to s112, with the largest gain on expert prompts (base 20 045 chars → s112 21 777 chars; cf. simple 10 230 → 13 266). The training signal lengthens deliberation specifically where it pays off.

**Guess task — non-monotonic.** s56 actually thinks *less* than the base model (3 267 vs 4 121 chars on average); s112 thinks more (4 753). Per-difficulty, s56 is shortest on every level except expert, while s112 doubles its thinking on simple guesses (3 590 → 7 839 chars), which is also where its only regressions occur (§5). The simplest reading is that s56 is the most efficient checkpoint on guesses, and s112 has started over-thinking easy cases.

---

## 3. Mistakes — kinds, frequency, patterns

**Format / rule-conformance mistakes — none, across all three models.** No clue is multi-word, hyphenated, present in either set, or a morphological variant. No guess list overflows `max_guesses` or contains an invented word. The format-reward dimension of the reward function is already saturated by the base model on this 24-prompt slice, so all observed differences are on task quality, not format.

**Task mistakes (guess task)** — wrong picks per checkpoint:

| record | difficulty | clue | targets | base | s56 | s112 |
|-------:|-----------|------|---------|------|-----|------|
| 17 | expert | MOLD | democratic, direct, pustule, chitin | CHITIN ✓, **PUBLIC ✗** | DIRECT ✓, CHITIN ✓ | CHITIN ✓, PUSTULE ✓ |
| 20 | moderate | HABITAT | live, refugium | REFUGIUM ✓, **CULTURE ✗** | REFUGIUM ✓, LIVE ✓ | REFUGIUM ✓, LIVE ✓ |
| 22 | simple | SIZE | change, measure, standard | MEASURE ✓, STANDARD ✓, **STRAIN ✗** | MEASURE ✓, STANDARD ✓, **DEFORM ✗** | MEASURE ✓, STANDARD ✓, **STRAIN ✗** |
| 23 | simple | BASIC | plain, start, need, normal | NEED ✓, NORMAL ✓, PLAIN ✓, START ✓ | NEED ✓, START ✓, NORMAL ✓, PLAIN ✓ | NEED ✓, START ✓, PLAIN ✓, **SOIL ✗** |

Three patterns emerge.

1. **Base prefers lateral over literal when a literal answer is sitting right there.** On idx 20 it picks CULTURE (habitat→cultural-habitat) and ignores LIVE; on idx 17 it picks PUBLIC (mold-of-the-public) and ignores PUSTULE. The trained models reliably pick the literal word.
2. **Both base and s112 share the same blind spot on idx 22 ("SIZE")**: all three checkpoints miss CHANGE (which is a clear "size = change in size" link). The trained models do not solve everything — they trade lateral for literal but still under-weight verbs-of-degree in this one case.
3. **s112's one new mistake is a lateral-over-literal in the opposite direction.** On idx 23 it explicitly reasons "basic is not a synonym for normal" and prefers SOIL via the "basic ↔ alkaline ↔ pH ↔ soil" chain. That is a chemistry interpretation of *basic*, and the chain is internally coherent — but loses to the direct synonym. This looks like a side-effect of training the model to chase distant associations on harder problems: on this easy prompt the chase fires when it should not.

**Task mistakes (clue task) — none in terms of validity**, but the *coverage claim* (the size of `[Selected-Targets]`) is interestingly different. See §4.

---

## 4. Does the model correct itself? Reward-hacking signals?

### Self-correction inside `<think>`

The "wait / actually / let me reconsider" tally rises with training:

| pattern (mentions per response, mean) | base | s56  | s112 |
|---------------------------------------|-----:|-----:|-----:|
| "wait"                                |  8.1 | 10.3 | 13.3 |
| "but wait"                            |  0.75| 0.92 | 1.88 |
| "actually"                            |  0.0 | 0.04 | 0.17 |
| "let me"                              |  6.3 |  7.4 |  9.3 |
| "alternatively"                       |  5.5 |  7.8 |  6.8 |

The s112 model interrupts itself ("but wait") about 2.5× as often as the base model. A clear example is idx 13 (`POLYMERIZATION`), where the base model muses "ZIEGLER: Hmm, that's a surname. I don't recall any famous Zieglers in this field. Maybe ZIEGLER? Maybe ZIEGLER?" and gives up — while s56/s112 retrieve "Ziegler-Natta catalysts" and lock the connection in. The trained models are noticeably more willing to keep poking at a candidate they cannot place on first try.

### The `<adjustment>` section is essentially vestigial

Across all three checkpoints, the `<adjustment>` section is filled with "No adjustments needed" 22–23 times out of 24. Self-correction lives inside `<think>`; the post-`</think>` CoT block is a formal recap. Training did not change this — none of the three models is making real use of `<adjustment>` to flip an answer.

### Calibration of the clue's coverage claim

For 12 clue prompts, `[Selected-Targets]` ⊆ `[Target-Set]` 100% of the time across all three models — no invalid selections. But the *number* of selected targets shifts:

| idx | difficulty | targets | base    | s56     | s112    |
|----:|-----------|---------|---------|---------|---------|
|  3 | expert  | 4 | ANIMAL (2/4) | FISH (3/4) | COLOR (3/4) |
|  4 | expert  | 6 | ACT (6/6) | POLITICAL (3/6) | COUNTRY (5/6) |

Idx 4 is the most telling: the base model claims that "ACT" connects to all six of `{gerrymander, incumbent, plant, green, growth, suffrage}` — a clear over-claim, because PLANT/GREEN/GROWTH have nothing to do with "act." Both trained models pull back: s56 to three (the unambiguous political terms), s112 to a more aggressive but still defensible five. Idx 3 goes the other way: the base model under-claims with the safe "ANIMAL → {chromatophore, beak}", and both trained models confidently push to three.

This is **calibration**, not hacking. The reward function pays correct guesses divided by `|Target-Set|` and penalises wrong guesses divided by `|Non-Target-Set|`; over-claiming a vague clue would risk wrong picks at guess-time, under-claiming leaves recall on the table. Training tightens both ends.

### Reward-hacking patterns checked — and **not** found

| candidate hack                                              | observed? |
|-------------------------------------------------------------|-----------|
| clue uses hyphen / multi-word to encode info                | no (0/12 in any checkpoint) |
| clue is morphological variant of a set word                 | no (0/12 in any checkpoint) |
| `[Selected-Targets]` contains words that aren't targets     | no (0/12 in any checkpoint) |
| guesses outside `[All-Words]`                               | no (0/12 in any checkpoint) |
| guesses overflow `[Max-Guesses]`                            | no (0/12 in any checkpoint) |

### One soft signal worth watching: trained models *always* fill `Max-Guesses`

| #guesses returned       | base | s56 | s112 |
|-------------------------|-----:|----:|-----:|
| equal to `max_guesses`  |  10  | 12  | 12   |
| under                   |   2  |  0  |  0   |
| over                    |   0  |  0  |  0   |

The two base under-fills are exactly the two cases where the trained models scored extra correct: idx 13 (base stops at 2 without ZIEGLER; trained returns 3 with ZIEGLER ✓) and idx 19 (base stops at 3 with PLANK/PROLATE/FLAT; trained returns 6 and picks up INGOT/BROAD/SURFACE ✓). So "always fill" is being driven by genuinely better lateral retrieval, not by indiscriminate stuffing — *but* on idx 22 (simple SIZE) and idx 23 (simple BASIC) it forces a low-confidence third/fourth pick that costs one wrong guess. The behavior is net-positive on this slice, but it is a learned reward-shaped behavior, and worth checking on a larger eval whether the cost crosses into negative territory.

---

## 5. Does the model consider multiple options before committing?

Yes, and increasingly so on clue prompts. The trained models float more candidate clues inside `<think>`, more frequently signal "alternatively" / "what about X", and more frequently kill candidates by checking them against `[Non-Target-Set]`. Some quotes:

> **s112, idx 0 (DISASTER/ECHO clue task):** "But wait, does 'disaster' have any connection to the non-target words? Let's check the Non-Target-Set: SIGNAL, FADE, VALLEY, SOUND, VOICE. 'Disaster' doesn't seem to connect with any of these. But wait, 'disaster' could be linked to 'sound' if someone says 'disaster sounds bad,' but that's a stretch."

> **s56, idx 2 (MALLEABLE/PH):** "Another option: 'malleable.' That's similar to pliable. Malleable means able to be shaped … 'Malleable' could relate to 'soap' because soap can be malleable, but again, that's a connection. So 'malleable' might link to SOAP, making it invalid. So 'malleable' is out. Back to 'docile' or 'submissive.' Which is better?"

The base model does the same kind of exploration, but typically over fewer rounds of "no, that links to a non-target, try another." Quantitatively, the candidate-clue count inside `<think>` rises from ~24 (base) to ~26 (s112) on clue prompts; the "stretch" hedge stays roughly constant at ~5 per response.

---

## 6. Repetitions in generated responses

Repetition rises slightly with training, and is more pronounced in s112 than s56:

| metric                                  | base   | s56    | s112   |
|-----------------------------------------|-------:|-------:|-------:|
| within-response 4-gram repeat rate (`<think>`) | 8.7%   | 9.8%   | 10.8% |
| within-response 8-gram repeat rate (`<think>`) | 1.1%   | 1.8%   | 2.0%  |
| max single-response 4-gram repeat        | 16.5% | 19.8%  | 23.4% |

The repeated material is functional ("but that's a stretch", "the non-target words are", "in the non-target set", "to any non-target words") — these are reasoning routines that the model has learned to invoke when checking candidates, not verbatim looping of full sentences. Concretely, s112 on idx 13 says "ZIEGLER refers to a specific catalyst, which is important" → "ZIEGLER is a person's name associated with the Ziegler-Natta catalyst, which is used in polymerization, so that's a strong connection" → "ZIEGLER is a catalyst, so that's a key component" — three near-paraphrases of the same fact. This is not pathological repetition yet, but the trend is monotonic across the two checkpoints and is worth tracking past step 112.

Stylistic markers are *not* moving: 100% of responses in all three models open with "Okay, let's tackle this Codenames…", which is an inherited Qwen3 style and unaffected by training.

---

## 7. Rare-word handling — direct evidence

This is the strongest qualitative result on this slice.

### idx 13 — `POLYMERIZATION` (advance, guess task)

Targets: `{oligomer, termination, ziegler}`. ZIEGLER is the rare/distant target — it refers to the chemist Karl Ziegler and the Ziegler-Natta catalysts.

| model | `<think>` excerpt | output |
|-------|-------------------|--------|
| base  | "ZIEGLER: Hmm, that's a surname. I don't recall any famous Zieglers in this field. Maybe ZIEGLER? Maybe ZIEGLER?" | `OLIGOMER, TERMINATION` (recall 2/3) |
| s56   | "ZIEGLER: I recall that Ziegler was a chemist who worked on polymerization, specifically Ziegler-Natta catalysts. So ZIEGLER is a likely candidate." | `OLIGOMER, ZIEGLER, TERMINATION` (recall 3/3) |
| s112  | "What about ZIEGLER? I recall that Ziegler-Natta catalysts are used in polymerization. So ZIEGLER might refer to that catalyst, which is important in polymerization processes." | `OLIGOMER, ZIEGLER, TERMINATION` (recall 3/3) |

The base model owns the same factual knowledge — it just doesn't *commit* to it. After RL, both trained checkpoints surface the Ziegler-Natta association decisively. This is exactly the failure mode described in the project hypothesis ("representational quality degrades when the underlying statistics are sparse … LLMs encode coherent representations but a growing body of work reveals significant gaps") and exactly the kind of correction RLVR is supposed to produce.

### idx 17 — `MOLD` (expert, guess task)

Targets: `{democratic, direct, pustule, chitin}`. The intended bridge for MOLD is the fungal-disease sense (mold causes skin pustules; chitin is a fungal cell-wall component).

- base: picks `CHITIN, PUBLIC`. CHITIN is right; PUBLIC is wrong — base latched onto "mold of the public" / casting-metaphor sense.
- s56: picks `DIRECT, CHITIN`. CHITIN is right; DIRECT is the secondary "direct mold" (manufacturing cast) sense.
- s112: picks `CHITIN, PUSTULE`. Both right. PUSTULE — a rare medical word for a fluid-filled skin sore — is exactly the distant association the targets were built around.

s112's solution is a clean medical-domain reading of MOLD that base does not access.

### idx 4 — `politics/frond` (expert, clue task)

Target-Set: `{gerrymander, incumbent, plant, green, growth, suffrage}`, Non-Target-Set: `{flat, branch}`. The base model proposes "ACT" as covering all six, asserting "ACT connects to all Target words via abstract action definitions". The trained models instead lean on the political subset and pull back coverage — s56 picks POLITICAL (3 targets), s112 picks COUNTRY (5 targets). Both deliberately decline to claim PLANT/GREEN under "ACT" because the linkage is not actually defensible.

### idx 8 — `OBJECTIVE/TOURNAMENT` (moderate, clue task)

Target-Set: `{benchmark, criterion}`. Base picks STANDARD, s56 picks STANDARD, **s112 picks YARDSTICK** — the same concept but reached via a rarer word.

---

## 8. Distant-concept handling — direct evidence

### idx 19 — `OBLONG` (moderate, guess task)

Targets: `{ingot, flat, broad, plank, surface, prolate}` (with non-targets `{warm, enjoy}`). PROLATE is the technical adjective for elongated; the rest are degrees of "oblongness."

- base: `PLANK, PROLATE, FLAT` (recall 3/6). It explicitly rejects INGOT, SURFACE, BROAD as "semantic mismatch" / "lateral".
- s56: `PROLATE, PLANK, BROAD, FLAT, SURFACE, INGOT` (recall 6/6).
- s112: `PROLATE, PLANK, INGOT, BROAD, SURFACE, FLAT` (recall 6/6).

This is the largest improvement on the slice (3/6 → 6/6). The trained models recognize that:
- INGOT is oblong because metal bars are oblong-shaped,
- BROAD is "flat-and-elongated",
- SURFACE belongs to the surface-of-an-oblong association class.

These are the cross-domain hops that the project hypothesis predicts RLVR-on-Codenames should learn.

### idx 3 — `OCTOPUS/HIGHWAY` (expert, clue task)

Target-Set: `{chromatophore, bridge, beak, travel}`. The targets mix octopus anatomy (chromatophore, beak) with travel terms (bridge, travel) — a deliberately distant set. CHROMATOPHORE is the rare word (octopus color-cells). All three models recognize ANIMAL/FISH, but s112's choice of **COLOR** is the most distant-association-correct: it routes through *chromato-* (color) and arguably *beak* (colored bird beak) and *travel* (color of a journey is a stretch) — a noticeably less obvious bridge than base's ANIMAL.

### Trained models pick literal-direct on lateral-bait

The flip side, on idx 20 / idx 17, is that when there's a literal answer (LIVE for HABITAT, PUSTULE for MOLD) the trained models go there rather than the lateral-but-thematic CULTURE/PUBLIC that base prefers. So "improved distant association" is not the same as "always reaches further" — it's "reaches further *when the targets require it* and stays close when they don't."

---

## 9. Summary table per axis

| axis                                                     | base → s112 direction | strength | caveats |
|----------------------------------------------------------|-----------------------|----------|---------|
| 1. Longer traces                                         | yes (clue +18%; guess mixed) | strong on clue | s56 actually shortens on guess; s112 over-thinks simple guesses |
| 2. Mistake rate (task)                                   | down (wrong guesses 0.25 → 0.17, precision 0.89 → 0.95) | strong | one new pattern: s112 lateral-overshoot on simple ("BASIC → SOIL") |
| 2. Mistake rate (format)                                 | unchanged (already 100% conformant) | n/a | format reward is saturated at the base |
| 3. Self-correction inside `<think>`                      | up ("wait" 8 → 13, "but wait" 0.75 → 1.88) | moderate | `<adjustment>` section unused by any checkpoint |
| 4. Multiple options considered                           | up (candidate clues 24 → 26; "alternatively" 5.5 → 6.8) | moderate | base already considers many |
| 5. Reward hacking                                        | none observed on listed checks | strong | "always fill `max_guesses`" is a learned behavior — net-positive here but worth watching |
| 6. Repetition                                            | up (4-gram repeat 8.7% → 10.8%) | mild | not pathological yet, all functional phrases |
| 7. Rare-word retrieval (Ziegler, Pustule, Prolate, Yardstick, Refugium) | clearly up | **strong** | strongest qualitative result of the analysis |
| 8. Distant-association handling                          | up *and* better-calibrated (reaches further when needed, stays literal otherwise) | **strong** | sample is small (24 prompts) |

---

## 10. Caveats and what this analysis cannot say

- **Sample size is 24 prompts.** Every per-difficulty mean averages three records, and the per-axis effects above swing on roughly 1–3 records each. Treat the *patterns* as hypothesis-quality, not point estimates.
- **One sample per prompt per model** (`n=1`, seed-fixed). The repetition trend, "wait"-count trend, and length trend would tighten or loosen substantially with several samples per prompt; they are confounded with the single rollout's noise.
- **No ground-truth reward computed in this analysis.** The recall/precision numbers above measure agreement with the metadata's `target_words` / `non_target_words` partition, not the actual training reward (which routes through a judge model and a guess-prompt). Take them as a proxy.
- **s56 vs s112 ordering is not monotonic.** s56 is the most efficient guesser (shortest `<think>`, highest recall+precision on this slice), while s112 thinks longer, repeats more, and slightly over-shoots laterally on simple cases. If the goal is to ship a checkpoint *today* on a budget similar to this 24-prompt slice, s56 looks competitive with s112.
- **The "always-fill `max_guesses`" behavior should be monitored on a larger eval.** It is net-positive at n=24 but is unmistakably a learned reward-shaped policy.
- **The dominant style ("Okay, let's tackle this Codenames…") is unchanged**, so any claim of "the model thinks differently" needs to be grounded in the per-axis evidence above and not in surface-level voice.

---
