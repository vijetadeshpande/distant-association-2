# Appendix: RuleCollection baseline data construction

## Data source and role in the experiment

We construct the natural-language reasoning control from
[`RuleReasoner/RuleCollection-32K`](https://huggingface.co/datasets/RuleReasoner/RuleCollection-32K),
pinned at revision
`f14a766d2e8e46390101154c6f4f51a67d7a5d9f`. The collection accompanies Liu
et al., [*RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic
Sampling*](https://arxiv.org/abs/2506.08672), published at ICLR 2026. We retain
six in-distribution, closed-answer domains: AR-LSAT, FOLIO, LogicNLI, Logical
Deduction, ProntoQA, and ProofWriter. The release declares an MIT license.

RuleCollection is a reasoning-content control. Its examples require multi-step
deduction over natural-language facts, rules, constraints, or ordered objects,
but do not require distant semantic association. All retained tasks have small,
explicit legal answer sets and can therefore use a deterministic verifier. This
lets us test whether Codenames transfer is reproduced by natural-language
reasoning practice while avoiding an LLM judge. We exclude CLUTRR because its
free-text relational answers introduce a different equivalence problem, and
exclude the small LogiQA component to preserve a clean, balanced six-domain
design. We also omit RuleReasoner's domain-aware dynamic sampling so that the
training algorithm remains matched to the Codenames comparison.

## Why and how we subsample

The six source training pools contain 27,799 rows, nearly seven times the 4,000
Codenames examples. Using all source rows would confound task content with the
number of unique training prompts and optimizer updates. We therefore fix the
same total and split sizes as Codenames: 3,680 training and 320 validation
examples. We assign approximately equal domain totals (666 or 667) so that the
largest synthetic domains cannot dominate the intervention, while preserving
each domain's natural label proportions through stratified sampling.

The exact procedure is:

1. Load only the six upstream `id/<domain>/train.parquet` files at the pinned
   revision. We do not use the packaged test sets: validation is rebuilt from
   the cleaned training pools because the initial audit found extensive exact
   overlap between packaged LogicNLI train and test prompts.
2. Remove the upstream `<think>` formatting instruction, preserve the original
   problem/context/options, and add the same five-section system scaffold used
   for Codenames. The user message requires exactly one
   `<answer>LABEL</answer>` block in its output section.
3. Parse each ground truth as exactly one answer block, normalize label case and
   surrounding whitespace, and require membership in the row/domain legal label
   set. Logical Deduction's row-level legal choices are recovered from its
   displayed options because examples may contain five to seven alternatives.
   All source gold labels passed validation and the final scorer.
4. Canonicalize complete task prompts with Unicode NFKC normalization, case
   folding, and whitespace normalization. For same-label exact duplicates,
   retain the smallest seeded SHA-256-ranked row; remove a complete group if its
   labels conflict. No conflicting group was present.
5. Render all prompts using the Qwen3-8B tokenizer revision
   `b968826d9c46dd6066d109eabc6255188de91218`, including the generation
   prefix, and enforce the shared 2,048-token prompt limit. No eligible source
   row exceeded it.
6. Define the initial split group as the canonical text preceding `Question:`.
   Thus, different questions about the same facts, rules, or scenario cannot
   cross train and validation. We additionally detect token-5-gram pairs with
   Jaccard similarity >=0.85 using deterministic bottom-k candidate generation
   and merge their context groups. This catches near-identical source contexts
   that differ only superficially.
7. Fix domain totals before sampling: 667 each for AR-LSAT, FOLIO, LogicNLI,
   and Logical Deduction, and 666 each for ProntoQA and ProofWriter. Within each
   domain, allocate its total across eligible labels proportionally by the
   largest-remainder method.
8. Allocate the domain's fixed validation count across its selected label
   quotas by largest remainder. Select validation rows by
   `SHA256("20260904", "split", domain, split_group_id, row_id)`, using at
   most one selected validation row from any context group. Exclude every row
   in those context groups from the training candidate pool, then select each
   training label quota by an independent seeded hash rank. This gives random,
   order-independent selection within each domain/label stratum while
   guaranteeing group-disjoint splits. The curation seed is `20260904`.

Equal domain weighting is deliberate: proportional sampling over the combined
pool would allocate roughly 58% of the data to ProntoQA and ProofWriter and only
about 8% to FOLIO and Logical Deduction, turning the control into a comparison
against whichever synthetic dataset happened to be largest. Label
stratification prevents avoidable class shifts, and context-group isolation is
necessary because several domains ask multiple questions about the same base
scenario. Hash-ranked sampling avoids manual cherry-picking and is invariant to
upstream row order.

## Source filtering and domain counts

| Domain | Raw source rows | Canonical unique eligible prompts | Exact duplicates removed | Selected total | Train | Validation |
|---|---:|---:|---:|---:|---:|---:|
| AR-LSAT | 1,636 | 1,623 | 13 | 667 | 613 | 54 |
| FOLIO | 966 | 966 | 0 | 667 | 613 | 54 |
| LogicNLI | 8,000 | 1,000 | 7,000 | 667 | 614 | 53 |
| Logical Deduction | 1,200 | 1,200 | 0 | 667 | 614 | 53 |
| ProntoQA | 8,000 | 7,884 | 116 | 666 | 613 | 53 |
| ProofWriter | 7,997 | 7,997 | 0 | 666 | 613 | 53 |
| **Total** | **27,799** | **20,670** | **7,129** | **4,000** | **3,680** | **320** |

No source row was removed for an illegal/malformed gold answer, a scorer
failure, a conflicting exact-duplicate label, or prompt overlength. The near
duplicate audit found 503 >=0.85 candidate pairs in the eligible pools. Most
already shared an exact source context; 113 ProntoQA pairs connected distinct
but near-identical contexts and were merged for split assignment. In the final
sample, 122 flagged pairs remain within a split and zero cross a split.

## Label counts before and after sampling

| Domain | Label | Eligible pool | Selected total | Train | Validation |
|---|---|---:|---:|---:|---:|
| AR-LSAT | A / B / C / D / E | 322 / 331 / 317 / 318 / 335 | 132 / 136 / 130 / 131 / 138 | 121 / 125 / 120 / 120 / 127 | 11 / 11 / 10 / 11 / 11 |
| FOLIO | False / True / Unknown | 277 / 371 / 318 | 191 / 256 / 220 | 176 / 235 / 202 | 15 / 21 / 18 |
| LogicNLI | contradiction / entailment / neutral / self_contradiction | 271 / 235 / 253 / 241 | 181 / 156 / 169 / 161 | 167 / 144 / 155 / 148 | 14 / 12 / 14 / 13 |
| Logical Deduction | A / B / C / D / E / F / G | 245 / 240 / 238 / 163 / 161 / 81 / 72 | 136 / 133 / 132 / 91 / 90 / 45 / 40 | 125 / 122 / 122 / 84 / 83 / 41 / 37 | 11 / 11 / 10 / 7 / 7 / 4 / 3 |
| ProntoQA | False / True | 3,958 / 3,926 | 334 / 332 | 307 / 306 | 27 / 26 |
| ProofWriter | False / True / Unknown | 2,832 / 2,816 / 2,349 | 236 / 234 / 196 | 217 / 215 / 181 | 19 / 19 / 15 |

Training prompts have median/p95/maximum rendered lengths of 587/720/948
tokens; validation prompts have 590/722/783. The final files contain no
repeated row IDs, canonical duplicate prompts, shared split-group IDs, or
cross-split >=0.85 near-duplicate pairs.

## Reward, artifacts, and reproducibility

`custom_reward_functions/rule_collection_reward.py` requires exactly one
well-formed `<answer>LABEL</answer>` block, normalizes label case and outer
whitespace only, validates the label against the row/domain alternatives, and
returns `+1` exactly when it matches the gold label; every other output receives
`-1`. There is no format bonus. Its admission suite covers malformed, repeated,
empty, illegal, NaN/infinity-like, and text-outside-the-block cases, as well as
more than 100 deterministic gold checks.

The VeRL-ready files are `rule_collection_train.parquet` and
`rule_collection_val.parquet`, each with exactly `prompt`, `data_source`,
`reward_model`, and `extra_info` as top-level columns. `manifest.json` records
source/tokenizer revisions, quotas, reward and artifact hashes; `audit.json`
contains the complete filtering, label, length, duplicate, and leakage results.
Re-running the curator produced byte-identical parquets.

```bash
python custom_data_preparation/curate_rule_collection_baseline.py
pytest -q custom_reward_functions/tests/test_rule_collection_reward.py
```

The protected-evaluation contamination comparison is explicitly pending in
`audit.json`; the files must not enter GPU calibration until that gate is
completed.
