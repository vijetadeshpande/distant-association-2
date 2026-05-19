# Qualitative analysis (scaled-up, 320 prompts): Codenames DAPO training of Qwen3-8B

This is the scaled-up rerun of [qualitative_analysis.md](qualitative_analysis.md). The earlier
report used 24 prompts; this one uses **320 prompts** (160 clue-generation + 160
guess-generation, 80 at each of simple / moderate / advance / expert). Where the larger
sample changes a conclusion from the 24-prompt report, it is flagged explicitly (see §11).

Three checkpoints, same 320 prompts each:

| tag    | model                                                                | role            |
|--------|----------------------------------------------------------------------|-----------------|
| `base` | `Qwen/Qwen3-8B` (off-the-shelf)                                      | baseline        |
| `s56`  | `codenames-dapo-qwen3-8b … global_step_56/actor` (~0.5 epoch)        | mid-training    |
| `s112` | `codenames-dapo-qwen3-8b … global_step_112/actor` (~1 epoch)         | end of training |

Source files: [qwen3-8b-all.jsonl](qwen3-8b-all.jsonl), [codenames-dapo-qwen3-8b-step56-all.jsonl](codenames-dapo-qwen3-8b-step56-all.jsonl), [codenames-dapo-qwen3-8b-step112-all.jsonl](codenames-dapo-qwen3-8b-step112-all.jsonl).
Analysis script: [scripts/analyze_qualitative.py](../scripts/analyze_qualitative.py) (run with `--all`).
Raw metrics: [qual_metrics-all.json](qual_metrics-all.json).

All decoding used the same sampler (`seed=42`, `temperature=1.0`, `top_p=1.0`, `max_tokens=16384`). Every one of the 960 completions terminated on `stop` — **no truncation**, so length differences below are genuine, not capped.

---

## 0. Sanity checks

| check                                                       | result    |
|-------------------------------------------------------------|-----------|
| 320 prompts byte-identical across all three runs            | **pass** (0 mismatches) |
| no completion byte-identical between any pair of models     | **pass** (0 duplicates) |
| every response terminated on `stop` (no length truncation)  | **pass** (960/960) |
| five CoT sections present and in canonical order            | 318/320 base · 318/320 s56 · 318/320 s112 (see §3) |
| clue tags / guess tags present                              | effectively 100% (see §3 for the handful of slips) |

The comparison is apples-to-apples; observed differences are model differences.

---

## 1. Headline numbers

### Clue generation (160 prompts)

| metric                                | base   | s56    | s112   | trend |
|---------------------------------------|-------:|-------:|-------:|-------|
| completion length (chars, mean)       | 13 907 | 15 405 | 17 990 | ↑ monotone (+29%) |
| `<think>` length (chars, mean)        | 12 083 | 13 828 | 16 447 | ↑ monotone (+36%) |
| "wait" tokens / response              |   8.1  |  10.6  |  14.2  | ↑ monotone |
| "alternatively" tokens / response     |   5.2  |   6.7  |   7.7  | ↑ monotone |
| candidate clues considered (quoted)   |  21.6  |  21.7  |  24.9  | ↑ |
| selected-target ÷ total-target ratio  |  0.914 |  0.944 |  0.915 | s56 best |
| clue is single English word           |  100%  |  100%  |  100%  | — |
| clue appears in either set            |    0%  |    0%  |    0%  | — |
| morphological-variant clue (reward-fn check) | 3/160 | 5/160 | 3/160 | see §4 |
| `[Selected-Targets]` all valid        | 158/160| 160/160| 160/160| see §3 |

### Guess generation (160 prompts)

| metric                                | base   | s56    | s112   | trend |
|---------------------------------------|-------:|-------:|-------:|-------|
| completion length (chars, mean)       |  4 700 |  4 821 |  6 001 | ↑ (s112 +28%) |
| `<think>` length (chars, mean)        |  3 308 |  3 661 |  4 766 | ↑ monotone |
| "wait" tokens / response              |   ~8   |   ~11  |   ~14  | ↑ monotone |
| guesses returned / response (mean)    |  3.47  |  3.58  |  3.57  | ↑ |
| guesses outside `[All-Words]` (invalid) |  0    |  1     |  0     | see §4 |
| guesses overflowing `Max-Guesses`     |  0     |  0     |  0     | — |
| **recall = correct ÷ targets**        |  0.802 |  0.829 |  0.834 | ↑ monotone (+0.032) |
| **precision = correct ÷ guesses**     |  0.942 |  0.950 |  0.956 | ↑ monotone (+0.014) |
| mean wrong (non-target) guesses       |  0.212 |  0.169 |  0.138 | ↓ monotone (−0.074) |

### Guess recall by difficulty (n = 40 prompts per cell)

| difficulty | base  | s56   | s112  |
|------------|------:|------:|------:|
| simple     | 0.894 | 0.936 | 0.944 |
| moderate   | 0.892 | 0.925 | 0.933 |
| advance    | 0.943 | 0.973 | 0.973 |
| **expert** | 0.477 | 0.483 | 0.483 |

### Mean wrong guesses by difficulty

| difficulty | base  | s56   | s112  |
|------------|------:|------:|------:|
| simple     | 0.375 | 0.250 | 0.200 |
| moderate   | 0.225 | 0.225 | 0.200 |
| advance    | 0.125 | 0.100 | 0.100 |
| expert     | 0.125 | 0.100 | 0.050 |

**Two headline takeaways.** First, on the larger sample the picture is **monotone**: s112 ≥ s56 ≥ base on recall, precision, wrong-guess rate, and on every difficulty bucket except the expert ceiling. Training helps, and the second half-epoch keeps helping. Second, the **expert bucket is a hard wall** — recall is essentially frozen at ~0.48 across all three checkpoints. Training improves *precision* on expert prompts (wrong guesses 0.125 → 0.050) but does not move *recall*: the model still cannot find the intended words, it just stops guessing wrong ones.

---

## 2. Are responses longer?

**Yes — and on the larger sample the growth is monotone and substantial**, which the 24-prompt slice did not show cleanly.

- **Clue task:** `<think>` grows +14% (base→s56) then +19% (s56→s112), totalling +36%. The single longest completion rises from 27 015 chars (base) to 37 576 chars (s112).
- **Guess task:** s56 is roughly flat vs base (+11% think chars), but s112 jumps +30%.
- **Records over 25 000 chars:** base 5 → s56 13 → **s112 28**. The long tail is fattening with training.

Length scales with difficulty for every checkpoint (expert clue `<think>`: base 15 548 → s112 21 731 chars), so the extra tokens are partly "harder problem, more work." But s112 also spends more even on easy problems (simple guess `<think>`: base 3 658 → s112 5 180), which is over-thinking, not just difficulty-tracking (see §6).

> **Correction vs the 24-prompt report.** The earlier slice reported the guess-task length as "non-monotonic, s56 shorter than base." At n=320 s56 is *slightly longer* than base on guesses, and the s112 blow-up is unambiguous. The monotone-increase reading is the correct one.

---

## 3. Mistakes — kinds, frequency, patterns

### 3a. Format / rule-conformance mistakes — rare but now non-zero

The 24-prompt slice found zero format mistakes. At n=320 a small number surface:

| mistake                                              | base | s56 | s112 |
|------------------------------------------------------|-----:|----:|-----:|
| CoT sections missing / out of order                  |  1   |  1  |  2   |
| `[Selected-Targets]` contains a non-target word      |  2   |  0  |  0   |
| guess outside `[All-Words]`                          |  0   |  1  |  0   |

Each is informative:

- **Out-of-order sections** are mostly an artifact, not a real failure. On base idx 138 the model emitted a *complete draft* of `<reflection>/<adjustment>/<output>` inside `<think>` and then the real CoT after — the final answer block is well-formed. On s56 idx 122 and s112 idx 314 the model dropped the *opening* `<adjustment>` tag (kept the closing one). On s112 idx 94 it dropped the opening `<thinking>` tag. These are single-tag slips, ~1 per 320.
- **The two base `[Selected-Targets]` errors are the most interesting mistake of the whole analysis: they are misspellings of rare board words.** On idx 7 the Target-Set is `{bother, suffer}` and the base model wrote `[BOOTHER, SUFFER]` — it corrupted "bother" to "BOOTHER". On idx 302 the Target-Set contains `candelabra` and the base model wrote `CANDLELABRA`. The trained models make zero such errors.
- **The one s56 invalid guess** (idx 238) is the same failure mode: `[All-Words]` contains `chemosensory`, and s56 guessed `CEMOSENSORY` — a dropped syllable.

**Pattern: rare/long board words get mis-copied into the answer, and this is a base-model failure that training cleans up.** Three corruptions in base (`boother`, `candlelabra`, plus near-misses), zero clean-copy failures in s112. This is a small but consistent fidelity gain on exactly the rare-word axis the project targets.

### 3b. Task mistakes (guess task)

Wrong-guess rate falls monotonically with training in every difficulty bucket (table in §1). The absolute numbers are small — the average guess response has 0.21 → 0.14 wrong words — but the direction is clean and consistent across 160 prompts.

The residual mistakes cluster on **simple prompts** (wrong rate 0.20 even at s112, higher than advance/expert). The reason is structural: the trained models always fill every `Max-Guesses` slot (§4), so on an easy 3-target prompt where the model is only confident about 2 words, the forced third pick is a coin-flip. This is the price of the "always fill" behavior, and it is concentrated where the board is small.

### 3c. Task mistakes (clue task)

No clue is multi-word, hyphenated, or present in either set, in any checkpoint. The one recurring clue-side mistake is the **morphological-variant clue** — covered in §4 because it interacts with the reward.

---

## 4. Self-correction, and reward-hacking signals

### 4a. Self-correction inside `<think>` rises with training

| pattern (mentions / response, mean over 320) | base | s56  | s112 |
|----------------------------------------------|-----:|-----:|-----:|
| "wait"                                       |  8.1 | 10.6 | 14.2 |
| "but wait"                                   |  0.96| 1.18 | 1.73 |
| "let me"                                     |  5.5 |  7.4 | 10.3 |
| "alternatively"                              |  5.2 |  6.7 |  7.7 |
| "hmm"                                        |  1.6 |  2.1 |  2.7 |
| `n_corrections` on guess task (script tally) |  3.0 |  4.2 |  6.7 |

Every self-monitoring marker rises monotonically. The s112 model interrupts itself ("but wait") ~1.8× as often as base and re-opens its reasoning ("let me …") ~1.9× as often. This is the mechanism behind the longer traces in §2.

As in the 24-prompt slice, the **`<adjustment>` section stays vestigial** — filled with "No adjustments needed" in the overwhelming majority of records for all three checkpoints. Self-correction lives inside `<think>`; the post-`</think>` `<adjustment>` block is a formal recap that training did not teach the model to use.

### 4b. Reward-hacking checks — none of the obvious exploits found

The reward (see [task_reward.py](../../custom_reward_functions/task_reward.py)) is
`task = pos_correct − neg_nontarget − neg_invalid`, with `Max-Guesses = len(Selected-Targets)`,
and a hard format gate that returns −1.0 on any format violation.

| candidate exploit                                            | observed? |
|--------------------------------------------------------------|-----------|
| hyphen / multi-word clue to smuggle information              | no (0/160 every checkpoint) |
| clue literally in Target-Set or Non-Target-Set              | no (0/160 every checkpoint) |
| `[Selected-Targets]` padded with non-target words            | no for trained models (0/160); base 2/160, both misspellings not pad |
| guesses overflowing `Max-Guesses`                            | no (0/160 every checkpoint) |
| guesses invented outside `[All-Words]`                       | no for base/s112; s56 1/160 (a misspelling) |
| over-claiming Selected-Targets to inflate `Max-Guesses`      | no clear signal — see below |

**On the inflate-`Max-Guesses` lever specifically:** the spymaster controls `Max-Guesses` through how many targets it lists, and a larger `Max-Guesses` gives the operative more shots at `pos_correct`. If the model were gaming this, `n_selected` would creep toward the full Target-Set size with training. It does not: average `n_selected` is 3.61 (base) → 3.78 (s56) → 3.64 (s112), and the full-set-claim rate is 0.77 → 0.84 → 0.79. s56 claims marginally more, s112 pulls back to near-base. On **expert** clues (the case where over-claiming would pay off most) s112 claims the *fewest* targets of any checkpoint (3.00 of 4.10 targets, vs base 2.98, s56 3.38). There is no monotone drift toward over-claiming — this reads as calibration, not hacking.

### 4c. The morphological-variant clue — a genuine mistake, *not* an exploit, but worth watching

Running the project's own checker ([morphology.py](../../custom_reward_functions/morphology.py), the exact function the reward uses) over all 480 clue records:

| | base | s56 | s112 |
|---|---:|---:|---:|
| morphological-variant clues | 3/160 | **5/160** | 3/160 |

The clear-cut cases are the trained models picking the *root* of a target word as the clue:
`COLOR` for target `COLORED` (s56 idx 247), `WEIGHT` for `WEIGHTY` (s112 idx 109),
`TOTEM` for `TOTEMIC` (s112 idx 271), `PASSIVITY` for `IMPASSIVITY` (s56 idx 150),
`CARD` for `DISCARD` (s56 idx 100). These are exactly the SWIM→SWIMMING violation the
rules forbid.

Two things make this **not** reward hacking. First, the reward function's `clue_no_morph_variant` check catches every one of these (it uses substring containment and errs toward detection) and applies the −1.0 format penalty — so the model is *punished*, not rewarded, for them. Second, the count does not grow monotonically (base 3 → s56 5 → s112 3). So this is a residual *mistake*: when one target word is rare and the only strong clue the model can find is that word's root, it sometimes reaches for the root anyway. It is slightly elevated at s56 and worth tracking on a larger eval, but it is a reward-penalised error, not a reward-gaming strategy.

### 4d. One genuine learned behavior: trained models *always* fill `Max-Guesses`

| #guesses returned vs `Max-Guesses` | base | s56 | s112 |
|------------------------------------|-----:|----:|-----:|
| exactly equal                      | 148  | 160 | 159  |
| under                              |  12  |  0  |  1   |
| over                               |   0  |  0  |  0   |

Base leaves a guess on the table 12 times; the trained models essentially never do. This is a clearly learned, reward-shaped policy — and §7/§8 show it is **net beneficial** here, because the slot base most often skips is the rare/distant target word. It is not an exploit (it cannot produce reward without correct guesses), but it is the behavior most worth monitoring at larger scale: on small/easy boards the forced last pick costs precision (§3b).

---

## 5. Does the model consider multiple options before committing?

Yes, and more so with training. On clue prompts the count of quoted candidate clues inside
`<think>` rises 21.6 → 21.7 → 24.9, and "alternatively" / "what about X" markers rise
monotonically (§4a). The exploration is real reasoning, not filler — candidates are
typically killed by an explicit check against the Non-Target-Set ("connect to any non
target words" is the single most repeated 6-gram in every checkpoint, and rises in
frequency with training: 315 → 375 → 464 corpus-wide occurrences).

A representative s112 trace (idx 236, clue `LIBERTARIANISM`): it raises CONSEQUENTIALIST,
hesitates ("Wait, but CONSEQUENTIALIST might not fit"), re-derives the link through
"consequentialism is a moral theory where actions are judged by their outcomes," and only
then commits. The base model on the same prompt raises CONSEQUENTIALIST twice, concludes
"not strong," and **drops it** — losing a correct guess.

---

## 6. Repetition

Repetition rises monotonically with training, and faster on the guess task:

| metric                                       | base   | s56    | s112   |
|----------------------------------------------|-------:|-------:|-------:|
| 8-gram repeat rate, clue `<think>`           | 0.014  | 0.017  | 0.022  |
| 8-gram repeat rate, guess `<think>`          | 0.003  | 0.006  | 0.010  |
| records over 25 000 chars                    |   5    |  13    |  28    |

The guess-task 8-gram repeat rate **more than triples** base→s112. The repeated material is
not verbatim sentence looping; it is **circular re-deliberation** — the model re-poses the
same sub-question several times. Example, s112 idx 85 (clue `TOOLS`, deciding whether
`FERRULE` qualifies):

> "But wait, maybe FERRULE is a tool? Let me double-check. A ferrule is a small cylindrical
> part … So maybe FERRULE is part of a tool. … Wait, maybe FERRULE is a type of tool? Let
> me think again. A ferrule is a tool used in plumbing … But maybe the clue-giver is
> thinking of FERRULE as a tool. …"

The same "is FERRULE a tool?" question is re-opened four times without new information.
This is not yet pathological (no infinite loops; all 960 generations terminated on `stop`),
but the trend is monotone over both checkpoints and over both the repeat-rate and the
long-tail-length metrics. **It is the clearest cost of training visible in this analysis**
and should be watched past step 112.

Stylistic markers are unchanged: 320/320 responses in every checkpoint open with "Okay,
let's tackle this Codenames…" — inherited Qwen3 style, untouched by training.

---

## 7. Rare-word handling — direct evidence

This is the strongest positive result, and the 320-prompt sample makes it quantitative.

### 7a. Aggregate: recall on long (rare-proxy) target words

Using target-word length ≥ 9 characters as a proxy for rare/technical vocabulary:

| metric                                          | base  | s56   | s112  |
|-------------------------------------------------|------:|------:|------:|
| recall on prompts whose hardest target ≥ 9 chars | 0.793 | 0.825 | 0.826 |
| recall on prompts whose hardest target ≤ 5 chars | 0.882 | 0.908 | 0.897 |
| **hit rate on individual ≥9-char target words** (n=171) | **0.883** | **0.936** | **0.924** |

The gain is concentrated on long/rare words: individual rare-word hit rate climbs ~5
points with training, while short common words were already near-saturated and barely move.

### 7b. The direct asymmetry test

For every guess prompt, check long target words (≥ 8 chars) that **base missed**:

- **11 prompts** where base missed a long/rare target that **both** trained checkpoints recovered.
- **1 prompt** where the reverse happened.

An 11-to-1 asymmetry. The recovered words are exactly the rare/distant vocabulary the
project hypothesis is about:

| idx | clue | rare word base missed, trained recovered |
|----:|------|------------------------------------------|
| 126 | meander       | RIPARIAN (of a riverbank) |
| 142 | spongy        | MALLEABLE |
| 148 | camcorder     | TIMECODE |
| 149 | hertz         | ATTENUATION |
| 158 | charter       | FIDUCIARY |
| 198 | refined       | DISTILLED |
| 200 | gratefulness  | SUPPLICATION |
| 236 | libertarianism| CONSEQUENTIALIST |
| 238 | anosmia       | NEUROLOGICAL |
| 308 | pegasus       | IMMORTAL |
| 309 | empathy       | INTRINSIC |

### 7c. The mechanism: base *knows* the word but won't *commit*

In nearly every case the base model surfaces the right association and then discards it.
Idx 200 (clue `GRATEFULNESS`, target `SUPPLICATION`):

> **base:** "SUPPLICATION: This is a prayer or request… could be a stretch… might be a
> reach." → drops it, returns only 3 of 5 guesses.
> **s56 / s112:** same definition of supplication, but → "involves seeking… maybe in
> gratitude" → keeps it, returns all 5, SUPPLICATION ✓.

Idx 236 (clue `LIBERTARIANISM`, target `CONSEQUENTIALIST`): base raises the word, says
"not strong," returns 5 guesses without it; both trained models return 6 with it. The
knowledge is present in the base weights — training changes the **decision threshold**, so
the model commits to a plausible-but-uncertain rare word instead of abandoning it. This
dovetails with the "always fill `Max-Guesses`" behavior of §4d: base under-fills 12 times,
and the skipped slot is repeatedly the rare word.

### 7d. Reduced rare-word mis-copying

As noted in §3a, the base model corrupts rare board words when copying them
(`bother`→`BOOTHER`, `candelabra`→`CANDLELABRA`); the trained models do not. Better
handling of rare words shows up both in *retrieval* and in *orthographic fidelity*.

---

## 8. Distant-concept handling — direct evidence

### 8a. Reaches further when the board demands it

Idx 149 (clue `HERTZ` → target `ATTENUATION`): bridging a unit of frequency to the
signal-processing concept of attenuation is a multi-hop, cross-sub-domain link. Base
misses it; both trained models make the hop. Idx 158 (`CHARTER` → `FIDUCIARY`): a
corporate-charter → fiduciary-duty link across the legal domain — recovered only after
training. Idx 309 (`EMPATHY` → `INTRINSIC`, expert): an abstract motivation-theory link
(intrinsic vs extrinsic motivation, empathy as intrinsically driven) — recovered after
training.

### 8b. …but stays literal when a literal answer exists

The §3b finding shows the flip side: the trained models do not simply "reach further
always." Their wrong-guess rate *drops* in every bucket, which means the extra reach is
*selective* — applied when the obvious words run out, not instead of them. "Improved
distant association" here means **better-calibrated** association: commit to the distant
link when the board needs it, keep the literal pick when it does not.

### 8c. The expert ceiling bounds the claim

The expert bucket (recall frozen at ~0.48, §1) is where the *most* distant boards live.
Training makes the model *cleaner* on expert prompts (wrong guesses 0.125 → 0.050) but not
*more able to find* the intended words. So the distant-association improvement is real but
**bounded**: it lifts moderate/advance boards toward saturation and does not crack the
hardest tier. Whatever the expert boards require, one epoch of DAPO does not supply it.

---

## 9. Clue-task quality

Without running the judge we cannot score clue→guess success directly, but two proxies are
available. `selected-target ÷ total-target` ratio (how much of the Target-Set the clue
claims to cover) is highest for s56 (0.944) and roughly equal for base and s112 (~0.915).
On expert clues s56 claims more targets (3.38/4.10) than base (2.98) or s112 (3.00). With
no monotone trend and no over-claiming beyond the Target-Set, the clue-side picture is
"s56 slightly more aggressive, s112 calibrated back to base," consistent with the
no-reward-hacking conclusion of §4.

---

## 10. Summary table per axis

| axis                                          | base → s112 direction | strength | notes |
|-----------------------------------------------|-----------------------|----------|-------|
| 1. Longer traces                              | up, **monotone** (clue +36% think, guess +44% think) | strong | long tail fattening: 5 → 28 records >25k chars |
| 2. Task mistakes                              | down, monotone (wrong 0.21 → 0.14; precision 0.94 → 0.96) | strong | residual errors concentrate on simple boards |
| 2. Format mistakes                            | rare (~1–2 / 320 per checkpoint) | n/a | base also has 2 rare-word misspellings; trained models fix this |
| 3. Self-correction inside `<think>`           | up, monotone ("wait" 8 → 14) | strong | `<adjustment>` section unused by all checkpoints |
| 4. Multiple options considered                | up (candidate clues 21.6 → 24.9) | moderate | exploration is genuine (Non-Target checks) |
| 5. Reward hacking                             | **none of the obvious exploits**; "always fill Max-Guesses" is a learned net-positive behavior | strong | morphological-variant clues are penalised mistakes, not exploits (base 3 / s56 5 / s112 3) |
| 6. Repetition                                 | up, monotone (guess 8-gram repeat 0.003 → 0.010, ×3) | mild–moderate | circular re-deliberation; clearest *cost* of training |
| 7. Rare-word retrieval                        | up; rare-word hit rate 0.88 → 0.94; 11-vs-1 recovery asymmetry | **strong** | base knows the word, training makes it commit |
| 8. Distant-association handling               | up *and* better-calibrated | **strong** | bounded — does not crack the expert tier |

---

## 11. What changed from the 24-prompt report

The scaled-up analysis confirms the direction of every finding in
[qualitative_analysis.md](qualitative_analysis.md) but corrects three things the small
sample got wrong:

1. **Effect size on guess recall was overstated.** The 24-prompt slice reported +0.11
   recall; at n=320 it is **+0.032**. The 24-prompt slice happened to draw prompts where
   the gain looked large. The improvement is real and monotone but modest.
2. **s56-vs-s112 ordering.** The small slice made s56 look like the best guesser ("s56
   competitive with s112"). At n=320 the ordering is cleanly **monotone — s112 ≥ s56 ≥
   base** — on recall, precision, and wrong-guess rate. The second half-epoch helps.
3. **Guess-task length.** The small slice called it "non-monotonic (s56 shorter than
   base)." At n=320 length increases monotonically; the s112 blow-up (+28% chars, 28
   records >25k) is unambiguous.

New findings only visible at scale: the **expert recall ceiling** (~0.48, frozen across
all checkpoints), the **rare-word mis-copy** failure of the base model (`boother`,
`candlelabra`), the **11-vs-1 rare-word recovery asymmetry**, the tripling of guess-task
**repetition**, and the **morphological-variant clue** mistake pattern.

---

## 12. Caveats

- **One sample per prompt** (`n=1`, fixed seed). Length, repetition, and "wait"-count
  trends are still confounded with single-rollout noise; only their monotonicity across
  320 prompts gives confidence. Per-cell difficulty means rest on 40 prompts each.
- **No judge-scored reward in this analysis.** Recall/precision measure agreement with the
  metadata `target_words`/`non_target_words` partition, which is a proxy for the true
  training reward (that routes a clue through a guess-prompt and a judge model). Clue-task
  quality in particular is only proxied (§9).
- **"Rare word" is operationalised as word length ≥ 8–9 characters.** This is a proxy for
  corpus frequency, not a frequency measurement. The recovered examples in §7b are
  qualitatively rare, but the aggregate numbers inherit the proxy's noise.
- **Two checkpoints define the trend.** "Monotone base→s56→s112" is three points. The
  repetition and length costs are rising at step 112; this analysis cannot say whether
  they plateau or worsen afterward.
- **The expert ceiling may be partly a board-construction artifact.** If some expert
  boards have genuinely ambiguous or judge-unrecoverable target sets, a frozen ~0.48
  recall could reflect the boards as much as the model. Worth confirming against the
  judge's own scores before concluding "DAPO cannot improve expert play."

---
