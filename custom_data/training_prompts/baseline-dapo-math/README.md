# Appendix: DAPO-Math baseline data construction

## Data source and role in the experiment

We construct the mathematical RLVR control from the English configuration of
[`open-r1/DAPO-Math-17k-Processed`](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed),
pinned at revision
`31dd309567e3da778038cc87d868b6097a3ccf68`. This release is a processed
version of the data accompanying Yu et al.,
[*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*](https://arxiv.org/abs/2503.14476).
We use the English split because the model, Codenames intervention, and
evaluation suite in the matched experiment are English-language.

DAPO-Math is a generic RLVR control: it supplies challenging problems with
deterministic integer answers and therefore tests whether any observed transfer
from Codenames can be explained by verifiable-reward optimization or sustained
reasoning practice alone. Unlike Codenames, it does not train distant semantic
association or game strategy. The existing local
`verl.utils.reward_score.math_dapo.compute_score` function scores answers with
`+1` for exact normalized correctness and `-1` otherwise, so the baseline does
not require a learned judge or task-specific reward model.

## Why and how we subsample

The English release contains substantially more examples than the 4,000-row
Codenames intervention. Training on the complete release would change both the
number of unique prompts and the optimizer horizon, confounding training source
with training volume. We therefore select exactly 4,000 prompts and use the
same 3,680/320 train/validation sizes as Codenames. A fixed-size subset also
allows the baseline to use the same 115-update one-epoch endpoint.

We use deterministic stratified random sampling rather than either manual
selection or an unstratified random draw. Manual selection could introduce
model- or result-dependent cherry-picking. A uniform draw would be unbiased in
expectation, but could sparsely represent negative and small integer answers or
shift the prompt-length distribution by chance. Prompt length affects context
and compute, while answer magnitude is a simple task-structural feature
available before model evaluation. Stratifying on both therefore improves
coverage without using downstream performance to choose examples.

The exact procedure is:

1. Load only `en/train-00000-of-00001.parquet` at the pinned revision. Recover
   the mathematical problem from `source_prompt`, remove the upstream answer
   boilerplate, and place the problem after the same five-section system
   scaffold used for Codenames.
2. Require the ground truth to match `[+-]?\d+` and verify it by scoring
   `Answer: <ground-truth>` with the exact local DAPO scorer. No source row
   failed either check.
3. Canonicalize problems with Unicode NFKC normalization, case folding,
   whitespace trimming, and whitespace collapsing. For an exact duplicate
   group with one answer, retain the row with the smallest seeded SHA-256 rank.
   Remove every member of a group containing conflicting answers.
4. Detect formatting-level near duplicates using token 5-gram Jaccard
   similarity at a threshold of 0.85, with deterministic bottom-k locality
   sensitive hashing for candidate generation. Conservatively retain one
   seeded representative per same-answer connected component and remove an
   entire component if its answers conflict. This handles mathematically
   identical prompts differing only in notation such as `\frac` versus
   `\tfrac`.
5. Render every prompt with the Qwen3-8B tokenizer pinned at revision
   `b968826d9c46dd6066d109eabc6255188de91218`, including the assistant
   generation prefix. Exclude prompts longer than 2,048 tokens; none exceeded
   the limit.
6. Sort eligible rows by `(rendered token length, SHA256(seed, source_row_id))`
   and assign equal-rank length quartiles. Cross these quartiles with five
   integer-answer buckets: `negative`, `0_or_1`, `2_to_9`, `10_to_99`, and
   `100_plus`.
7. Allocate 4,000 slots proportionally across the 20 cross-strata by the
   largest-remainder method, enforcing at least 20 examples in every nonempty
   stratum. Within a stratum, select the smallest
   `SHA256("20260904", source_row_id)` ranks. Hash ranking makes the result
   independent of upstream file order while retaining random selection within
   each declared stratum.
8. Allocate 320 validation slots proportionally across the selected strata by
   largest remainder and select them using the independent namespace
   `SHA256("20260904", "split", row_id)`. The remaining 3,680 rows form the
   training split. A second namespace determines physical row order. The
   curation seed is fixed at `20260904`.

## Source filtering and final counts

The source release and cleaned pool have the following counts. “Conflicting”
components are removed in full rather than resolving their labels
arbitrarily.

| Stage | Unit | Count |
|---|---|---:|
| Raw English source | rows | 14,116 |
| Exact canonical prompt groups | groups | 13,985 |
| Exact duplicate rows beyond one representative | rows | 131 |
| Same-answer exact duplicates removed | rows | 126 |
| Exact conflicting-answer groups removed | groups / rows | 5 / 10 |
| Pool after exact filtering | rows | 13,980 |
| Near-duplicate candidates at Jaccard >= 0.85 | pairs | 650 |
| Same-answer near-duplicate rows removed | rows | 601 |
| Near-duplicate conflicting components removed | components / rows | 24 / 50 |
| Eligible pool after exact/near filtering and length gate | rows | 13,329 |
| Final training / validation / total | rows | 3,680 / 320 / 4,000 |

Answer-bucket counts before and after sampling are:

| Integer-answer bucket | Raw source | Eligible pool | Selected total | Train | Validation |
|---|---:|---:|---:|---:|---:|
| Negative | 255 | 242 | 86 | 78 | 8 |
| 0 or 1 | 504 | 487 | 143 | 132 | 11 |
| 2--9 | 2,858 | 2,762 | 826 | 760 | 66 |
| 10--99 | 5,165 | 4,895 | 1,465 | 1,348 | 117 |
| 100+ | 5,334 | 4,943 | 1,480 | 1,362 | 118 |
| **Total** | **14,116** | **13,329** | **4,000** | **3,680** | **320** |

The exact selected cross-stratum allocation was:

| Length quartile | Negative | 0 or 1 | 2--9 | 10--99 | 100+ | Total |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 25 | 51 | 220 | 347 | 353 | 996 |
| Q2 | 21 | 34 | 217 | 364 | 359 | 995 |
| Q3 | 20 | 29 | 205 | 380 | 369 | 1,003 |
| Q4 | 20 | 29 | 184 | 374 | 399 | 1,006 |
| **Total** | **86** | **143** | **826** | **1,465** | **1,480** | **4,000** |

The cleaned-pool quartiles contain 3,333, 3,332, 3,332, and 3,332 rows,
respectively; their inclusive token ranges are 435--477, 477--501, 501--533,
and 533--1,911. Final training prompts have median/p95/maximum lengths of
502/616/1,467 tokens; validation prompts have 501/640/1,385. The final splits
contain no repeated row IDs, canonical duplicate prompts, shared split-group
IDs, or detected >=0.85 near-duplicate pairs.

## Artifacts and reproducibility

The VeRL-ready files are `dapo_math_train.parquet` and
`dapo_math_val.parquet`. Each has exactly four top-level columns: `prompt`,
`data_source`, `reward_model`, and `extra_info`; `data_source` is `math_dapo`.
`manifest.json` records immutable source/tokenizer revisions and artifact
SHA-256 hashes, while `audit.json` contains the complete quotas and validation
statistics. Re-running the curator produced byte-identical parquets.

```bash
python custom_data_preparation/curate_dapo_math_baseline.py
```

The processed dataset card does not declare a data license. We therefore treat
the subset as internal-only pending a redistribution/data-governance decision.
The protected-evaluation contamination comparison is also explicitly pending
in `audit.json`; the files must not enter GPU calibration until that gate is
completed.
