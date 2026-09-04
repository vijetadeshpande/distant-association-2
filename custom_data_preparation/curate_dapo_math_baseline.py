#!/usr/bin/env python3
"""Create the pinned, deterministic 4,000-row DAPO-Math RLVR baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from baseline_curation_utils import (  # noqa: E402
    canonical_text,
    counter_dict,
    file_sha256,
    json_dump,
    largest_remainder,
    length_summary,
    near_duplicate_pairs,
    stable_digest,
    validate_records,
)

from custom_data_preparation.system_prompts import SYS_PROMPT  # noqa: E402
from verl.utils.reward_score.math_dapo import compute_score  # noqa: E402

SOURCE_DATASET = "open-r1/DAPO-Math-17k-Processed"
SOURCE_REVISION = "31dd309567e3da778038cc87d868b6097a3ccf68"
SOURCE_FILE = "en/train-00000-of-00001.parquet"
SOURCE_LICENSE = "not declared on the dataset card"
TOKENIZER_ID = "Qwen/Qwen3-8B"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SEED = "20260904"
CURATION_VERSION = "2026-09-04.v1"
TOTAL_ROWS = 4000
VAL_ROWS = 320
MAX_PROMPT_TOKENS = 2048
DEFAULT_OUTPUT_DIR = REPO_ROOT / "custom_data/training_prompts/baseline-dapo-math"
DEFAULT_CACHE_DIR = Path("/home/public/vdeshpan/.cache/huggingface")

UPSTREAM_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
UPSTREAM_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def extract_problem(source_prompt: Any, clean_prompt: str) -> str:
    if len(source_prompt) != 1 or source_prompt[0]["role"] != "user":
        raise ValueError("unexpected source_prompt chat shape")
    content = str(source_prompt[0]["content"])
    if content.startswith(UPSTREAM_PREFIX) and content.endswith(UPSTREAM_SUFFIX):
        extracted = content[len(UPSTREAM_PREFIX) : -len(UPSTREAM_SUFFIX)]
    elif content.count(str(clean_prompt)) == 1:
        extracted = str(clean_prompt)
    else:
        raise ValueError("unable to remove DAPO prompt boilerplate")
    if canonical_text(extracted) != canonical_text(clean_prompt):
        raise ValueError("source_prompt extraction disagrees with clean prompt column")
    return extracted.strip()


def user_prompt(problem: str) -> str:
    return (
        "Solve the following mathematical problem. Show your reasoning using the five-section "
        "format required by the system message. In the <output> section, put the final integer "
        "answer on its own line in exactly this form:\n\n"
        "Answer: <integer>\n\n"
        "Problem:\n"
        f"{problem}"
    )


def answer_bucket(answer: str) -> str:
    value = int(answer)
    if value < 0:
        return "negative"
    if value in (0, 1):
        return "0_or_1"
    if value <= 9:
        return "2_to_9"
    if value <= 99:
        return "10_to_99"
    return "100_plus"


def assign_length_quartiles(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (row["prompt_tokens"], stable_digest(SEED, row["source_row_id"])))
    for rank, row in enumerate(ordered):
        row["length_quartile"] = f"q{min(4, rank * 4 // len(ordered) + 1)}"
        row["stratum"] = f'{row["length_quartile"]}__{row["answer_bucket"]}'


def collapse_near_duplicates(rows: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Conservatively retain one row from every >=0.85 shingle component."""

    pairs = near_duplicate_pairs(
        [(row["row_id"], "pool", row["problem"]) for row in rows], threshold=0.85
    )
    by_id = {row["row_id"]: row for row in rows}
    parent = {row_id: row_id for row_id in by_id}

    def find(row_id: str) -> str:
        while parent[row_id] != row_id:
            parent[row_id] = parent[parent[row_id]]
            row_id = parent[row_id]
        return row_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        union(pair["left_row_id"], pair["right_row_id"])

    components: dict[str, list[dict]] = defaultdict(list)
    for row_id, row in by_id.items():
        components[find(row_id)].append(row)

    retained: list[dict] = []
    conflicting_components = 0
    conflicting_rows = 0
    duplicate_rows = 0
    for members in components.values():
        if len({member["answer"] for member in members}) > 1:
            conflicting_components += 1
            conflicting_rows += len(members)
            continue
        retained.append(
            min(members, key=lambda row: stable_digest(SEED, "near-dedup", row["source_row_id"]))
        )
        duplicate_rows += len(members) - 1

    return retained, {
        "candidate_pairs": len(pairs),
        "components_before_filter": len(components),
        "same_answer_rows_removed": duplicate_rows,
        "conflicting_answer_components_removed": conflicting_components,
        "conflicting_answer_rows_removed": conflicting_rows,
        "pair_examples": pairs[:100],
    }


def build_records(rows: list[dict], split: str) -> list[dict]:
    records = []
    for row in rows:
        records.append(
            {
                "prompt": row["messages"],
                "data_source": "math_dapo",
                "reward_model": {"ground_truth": row["answer"]},
                "extra_info": {
                    "row_id": row["row_id"],
                    "source_dataset": SOURCE_DATASET,
                    "source_revision": SOURCE_REVISION,
                    "source_row_id": row["source_row_id"],
                    "split_group_id": row["split_group_id"],
                    "stratum": row["stratum"],
                    "curation_version": CURATION_VERSION,
                    "split": split,
                    "prompt_tokens": row["prompt_tokens"],
                    "length_quartile": row["length_quartile"],
                    "answer_bucket": row["answer_bucket"],
                    "reward_function_name": "verl.utils.reward_score.math_dapo.compute_score",
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_CACHE_DIR
        / "hub/models--Qwen--Qwen3-8B/snapshots"
        / TOKENIZER_REVISION,
    )
    args = parser.parse_args()

    source_path = Path(
        hf_hub_download(
            repo_id=SOURCE_DATASET,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            filename=SOURCE_FILE,
            cache_dir=args.cache_dir,
        )
    )
    frame = pd.read_parquet(source_path)
    if len(frame) != 14116:
        raise AssertionError(f"pinned English source has {len(frame)} rows, expected 14116")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    invalid_integer = 0
    scorer_failures = 0
    for source_position, source in frame.iterrows():
        answer = str(source["solution"]).strip()
        if not re.fullmatch(r"[+-]?\d+", answer):
            invalid_integer += 1
            continue
        score = compute_score(f"Answer: {answer}", answer)
        if float(score["score"]) != 1.0:
            scorer_failures += 1
            continue
        problem = extract_problem(source["source_prompt"], source["prompt"])
        source_row_id = str(source["extra_info"]["index"])
        canonical_problem = canonical_text(problem)
        grouped[canonical_problem].append(
            {
                "source_position": int(source_position),
                "source_row_id": source_row_id,
                "problem": problem,
                "answer": answer,
            }
        )

    conflict_groups = {
        canonical: members
        for canonical, members in grouped.items()
        if len({member["answer"] for member in members}) > 1
    }
    eligible: list[dict] = []
    same_answer_duplicate_rows = 0
    for canonical_problem, members in grouped.items():
        if canonical_problem in conflict_groups:
            continue
        chosen = min(members, key=lambda row: stable_digest(SEED, "dedup", row["source_row_id"]))
        same_answer_duplicate_rows += len(members) - 1
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_prompt(chosen["problem"])},
        ]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompt_tokens = len(token_ids)
        if prompt_tokens > MAX_PROMPT_TOKENS:
            continue
        split_group_id = f"dapo_prompt:{stable_digest(canonical_problem)}"
        eligible.append(
            {
                **chosen,
                "row_id": f'dapo_math:{chosen["source_row_id"]}',
                "split_group_id": split_group_id,
                "messages": messages,
                "prompt_tokens": prompt_tokens,
                "answer_bucket": answer_bucket(chosen["answer"]),
            }
        )

    eligible, near_dedup_audit = collapse_near_duplicates(eligible)
    assign_length_quartiles(eligible)
    eligible_counts = Counter(row["stratum"] for row in eligible)
    quotas = largest_remainder(eligible_counts, TOTAL_ROWS, minimum=20)
    selected: list[dict] = []
    for stratum, quota in quotas.items():
        candidates = sorted(
            (row for row in eligible if row["stratum"] == stratum),
            key=lambda row: stable_digest(SEED, row["source_row_id"]),
        )
        selected.extend(candidates[:quota])

    selected_counts = Counter(row["stratum"] for row in selected)
    val_quotas = largest_remainder(selected_counts, VAL_ROWS)
    validation: list[dict] = []
    training: list[dict] = []
    for stratum, quota in val_quotas.items():
        candidates = sorted(
            (row for row in selected if row["stratum"] == stratum),
            key=lambda row: stable_digest(SEED, "split", row["row_id"]),
        )
        validation.extend(candidates[:quota])
        training.extend(candidates[quota:])

    training.sort(key=lambda row: stable_digest(SEED, "train-order", row["row_id"]))
    validation.sort(key=lambda row: stable_digest(SEED, "val-order", row["row_id"]))
    train_records = build_records(training, "train")
    val_records = build_records(validation, "validation")
    train_validation = validate_records(train_records, TOTAL_ROWS - VAL_ROWS)
    val_validation = validate_records(val_records, VAL_ROWS)
    if train_validation["canonical_duplicate_prompts"] or val_validation["canonical_duplicate_prompts"]:
        raise AssertionError("canonical duplicates remain within a split")
    train_groups = {record["extra_info"]["split_group_id"] for record in train_records}
    val_groups = {record["extra_info"]["split_group_id"] for record in val_records}
    if train_groups & val_groups:
        raise AssertionError("split-group leakage detected")

    near_pairs = near_duplicate_pairs(
        [
            (row["row_id"], split, row["problem"])
            for split, rows in (("train", training), ("validation", validation))
            for row in rows
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "dapo_math_train.parquet"
    val_path = args.output_dir / "dapo_math_val.parquet"
    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    pd.DataFrame(val_records).to_parquet(val_path, index=False)

    quartile_ranges: dict[str, dict[str, int]] = {}
    for quartile in ("q1", "q2", "q3", "q4"):
        lengths = [row["prompt_tokens"] for row in eligible if row["length_quartile"] == quartile]
        quartile_ranges[quartile] = {"min": min(lengths), "max": max(lengths), "count": len(lengths)}

    eligible_pool_hash = stable_digest(
        *sorted(
            f'{row["source_row_id"]}:{stable_digest(canonical_text(row["problem"]))}:{row["answer"]}'
            for row in eligible
        )
    )
    audit = {
        "baseline": "dapo_math",
        "source_rows": len(frame),
        "source_canonical_groups": len(grouped),
        "source_duplicate_rows": len(frame) - len(grouped),
        "same_answer_duplicate_rows_removed": same_answer_duplicate_rows,
        "conflicting_answer_groups_removed": len(conflict_groups),
        "conflicting_answer_rows_removed": sum(len(value) for value in conflict_groups.values()),
        "invalid_integer_rows": invalid_integer,
        "ground_truth_scorer_failures": scorer_failures,
        "near_duplicate_pool_filter": near_dedup_audit,
        "eligible_rows_after_length_gate": len(eligible),
        "eligible_pool_sha256": eligible_pool_hash,
        "length_quartile_ranges": quartile_ranges,
        "eligible_strata": dict(sorted(eligible_counts.items())),
        "sample_quotas": quotas,
        "validation_quotas": val_quotas,
        "train_strata": counter_dict(row["stratum"] for row in training),
        "validation_strata": counter_dict(row["stratum"] for row in validation),
        "train_answer_buckets": counter_dict(row["answer_bucket"] for row in training),
        "validation_answer_buckets": counter_dict(row["answer_bucket"] for row in validation),
        "train_prompt_lengths": length_summary([row["prompt_tokens"] for row in training]),
        "validation_prompt_lengths": length_summary([row["prompt_tokens"] for row in validation]),
        "train_schema_validation": train_validation,
        "validation_schema_validation": val_validation,
        "cross_split_group_overlap": len(train_groups & val_groups),
        "near_duplicate_threshold": 0.85,
        "near_duplicate_flagged_pairs": near_pairs,
        "near_duplicate_cross_split_pairs": sum(pair["cross_split"] for pair in near_pairs),
        "evaluation_contamination": {
            "status": "pending protected-evaluation corpus availability",
            "note": "Do not begin GPU calibration until the protocol's protected-suite comparison is completed.",
        },
    }
    json_dump(args.output_dir / "audit.json", audit)

    manifest = {
        "schema_version": 1,
        "baseline": "dapo_math",
        "curation_version": CURATION_VERSION,
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "file": SOURCE_FILE,
            "file_sha256": file_sha256(source_path),
            "license": SOURCE_LICENSE,
            "redistribution_status": "unresolved; internal experimentation only pending data-governance review",
        },
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "sampling": {
            "strategy": "deterministic stratified random sampling by SHA-256 rank",
            "seed": int(SEED),
            "total": TOTAL_ROWS,
            "train": TOTAL_ROWS - VAL_ROWS,
            "validation": VAL_ROWS,
            "strata": "rendered-token-length quartile x integer-answer bucket",
        },
        "reward": {
            "data_source": "math_dapo",
            "implementation": "verl.utils.reward_score.math_dapo.compute_score",
            "range": [-1.0, 1.0],
        },
        "files": {
            train_path.name: {"rows": len(train_records), "sha256": file_sha256(train_path)},
            val_path.name: {"rows": len(val_records), "sha256": file_sha256(val_path)},
            "audit.json": {"sha256": file_sha256(args.output_dir / "audit.json")},
        },
        "recreate": "python custom_data_preparation/curate_dapo_math_baseline.py",
    }
    json_dump(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
