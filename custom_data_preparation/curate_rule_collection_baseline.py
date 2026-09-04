#!/usr/bin/env python3
"""Create the pinned, balanced 4,000-row RuleCollection RLVR baseline."""

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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

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
from custom_reward_functions.rule_collection_reward import compute_score, parse_answer  # noqa: E402

SOURCE_DATASET = "RuleReasoner/RuleCollection-32K"
SOURCE_REVISION = "f14a766d2e8e46390101154c6f4f51a67d7a5d9f"
SOURCE_LICENSE = "MIT"
TOKENIZER_ID = "Qwen/Qwen3-8B"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SEED = "20260904"
CURATION_VERSION = "2026-09-04.v1"
MAX_PROMPT_TOKENS = 2048
DEFAULT_OUTPUT_DIR = REPO_ROOT / "custom_data/training_prompts/baseline-rule-collection"
DEFAULT_CACHE_DIR = Path("/home/public/vdeshpan/.cache/huggingface")
REWARD_PATH = REPO_ROOT / "custom_reward_functions/rule_collection_reward.py"

DOMAINS = {
    "ar_lsat": {"data_source": "AR-LSAT", "total": 667, "validation": 54},
    "folio": {"data_source": "Folio", "total": 667, "validation": 54},
    "logic_nli": {"data_source": "Logic NLI", "total": 667, "validation": 53},
    "logical_deduction": {"data_source": "Logical Deduction", "total": 667, "validation": 53},
    "prontoqa": {"data_source": "ProntoQA", "total": 666, "validation": 53},
    "proofwriter": {"data_source": "ProofWriter", "total": 666, "validation": 53},
}

DOMAIN_LABELS = {
    "ar_lsat": ["A", "B", "C", "D", "E"],
    "folio": ["True", "False", "Unknown"],
    "logic_nli": ["contradiction", "self_contradiction", "neutral", "entailment"],
    "logical_deduction": ["A", "B", "C", "D", "E", "F", "G"],
    "prontoqa": ["True", "False", "Unknown"],
    "proofwriter": ["True", "False", "Unknown"],
}

INSTRUCTION_MARKER = " Please answer the question based on"


def source_content(prompt: Any) -> str:
    if len(prompt) != 1 or prompt[0]["role"] != "user":
        raise ValueError("unexpected upstream prompt chat shape")
    return str(prompt[0]["content"])


def remove_upstream_instruction(content: str) -> str:
    marker_position = content.rfind(INSTRUCTION_MARKER)
    if marker_position < 0:
        raise ValueError("RuleCollection instruction marker is missing")
    return content[:marker_position].strip()


def context_for_grouping(core_prompt: str) -> str:
    return core_prompt.split("\nQuestion:", 1)[0].strip()


def row_legal_labels(domain: str, core_prompt: str) -> list[str]:
    if domain != "logical_deduction":
        return DOMAIN_LABELS[domain]
    options = core_prompt.split("Options:", 1)[-1]
    labels = []
    for label in re.findall(r"(?:^|\s)([A-G])\)", options):
        if label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError("unable to extract Logical Deduction options")
    return labels


def render_user_prompt(core_prompt: str, legal_labels: list[str]) -> str:
    choices = "/".join(legal_labels)
    return (
        f"{core_prompt}\n\n"
        "Show your reasoning using the five-section format required by the system message. "
        "In the <output> section, provide exactly one final answer block, with no additional "
        "answer blocks, in this form:\n\n"
        "<answer>LABEL</answer>\n\n"
        f"Replace LABEL with one of [{choices}]."
    )


def connect_near_duplicate_groups(rows: list[dict]) -> dict[str, Any]:
    """Join source-context groups connected by >=0.85 prompt similarity."""

    pairs = near_duplicate_pairs(
        [(row["row_id"], "pool", row["core_prompt"]) for row in rows], threshold=0.85
    )
    by_id = {row["row_id"]: row for row in rows}
    parent = {row["base_group_id"]: row["base_group_id"] for row in rows}

    def find(group_id: str) -> str:
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    cross_context_pairs = 0
    for pair in pairs:
        left_group = by_id[pair["left_row_id"]]["base_group_id"]
        right_group = by_id[pair["right_row_id"]]["base_group_id"]
        if left_group != right_group:
            cross_context_pairs += 1
        union(left_group, right_group)
    for row in rows:
        row["split_group_id"] = f'rule_context:{row["domain"]}:{find(row["base_group_id"])}'
    return {
        "candidate_pairs": len(pairs),
        "cross_context_pairs": cross_context_pairs,
        "pair_examples": pairs[:100],
    }


def select_domain(rows: list[dict], domain: str) -> tuple[list[dict], list[dict], dict[str, Any]]:
    config = DOMAINS[domain]
    available = Counter(row["label"] for row in rows)
    total_quota = largest_remainder(available, config["total"])
    validation_quota = largest_remainder(total_quota, config["validation"])
    training_quota = {label: total_quota[label] - validation_quota.get(label, 0) for label in total_quota}

    # Use distinct validation groups. This makes the subsequent training-group
    # exclusion inexpensive and guarantees whole-context split isolation.
    used_validation_groups: set[str] = set()
    validation: list[dict] = []
    label_order = sorted(
        validation_quota,
        key=lambda label: (
            len({row["split_group_id"] for row in rows if row["label"] == label}),
            label,
        ),
    )
    for label in label_order:
        candidates = sorted(
            (row for row in rows if row["label"] == label),
            key=lambda row: stable_digest(SEED, "split", domain, row["split_group_id"], row["row_id"]),
        )
        chosen = []
        for row in candidates:
            if row["split_group_id"] in used_validation_groups:
                continue
            chosen.append(row)
            used_validation_groups.add(row["split_group_id"])
            if len(chosen) == validation_quota[label]:
                break
        if len(chosen) != validation_quota[label]:
            raise RuntimeError(f"could not satisfy validation quota for {domain}/{label}")
        validation.extend(chosen)

    training: list[dict] = []
    for label, quota in training_quota.items():
        candidates = sorted(
            (
                row
                for row in rows
                if row["label"] == label and row["split_group_id"] not in used_validation_groups
            ),
            key=lambda row: stable_digest(SEED, domain, row["split_group_id"], row["row_id"]),
        )
        if len(candidates) < quota:
            raise RuntimeError(f"not enough training rows for {domain}/{label}: {len(candidates)} < {quota}")
        training.extend(candidates[:quota])

    return training, validation, {
        "available_labels": dict(sorted(available.items())),
        "total_label_quotas": total_quota,
        "training_label_quotas": training_quota,
        "validation_label_quotas": validation_quota,
        "validation_split_groups": len(used_validation_groups),
    }


def to_record(row: dict, split: str) -> dict:
    return {
        "prompt": row["messages"],
        "data_source": row["data_source"],
        "reward_model": {"ground_truth": f'<answer>{row["label_display"]}</answer>'},
        "extra_info": {
            "row_id": row["row_id"],
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_row_id": row["source_row_id"],
            "split_group_id": row["split_group_id"],
            "stratum": row["data_source"],
            "curation_version": CURATION_VERSION,
            "split": split,
            "prompt_tokens": row["prompt_tokens"],
            "label": row["label_display"],
            "legal_labels": row["legal_labels"],
            "reward_function_name": "custom_reward_functions.rule_collection_reward.compute_score",
        },
    }


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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)

    all_training: list[dict] = []
    all_validation: list[dict] = []
    domain_audits: dict[str, Any] = {}
    source_files: dict[str, Any] = {}

    for domain, config in DOMAINS.items():
        source_path = Path(
            hf_hub_download(
                repo_id=SOURCE_DATASET,
                repo_type="dataset",
                revision=SOURCE_REVISION,
                filename=f"id/{domain}/train.parquet",
                cache_dir=args.cache_dir,
            )
        )
        source_files[domain] = {
            "path": f"id/{domain}/train.parquet",
            "sha256": file_sha256(source_path),
        }
        frame = pd.read_parquet(source_path)
        exact_groups: dict[str, list[dict]] = defaultdict(list)
        malformed_ground_truth = 0
        for source_position, source in frame.iterrows():
            full_content = source_content(source["prompt"])
            core_prompt = remove_upstream_instruction(full_content)
            label, label_error = parse_answer(source["reward_model"]["ground_truth"])
            if label_error is not None:
                malformed_ground_truth += 1
                continue
            legal_labels = row_legal_labels(domain, core_prompt)
            display_by_normalized = {value.casefold(): value for value in legal_labels}
            if label not in display_by_normalized:
                malformed_ground_truth += 1
                continue
            source_row_id = f'{domain}:{source["extra_info"]["index"]}'
            exact_groups[canonical_text(core_prompt)].append(
                {
                    "domain": domain,
                    "data_source": config["data_source"],
                    "source_position": int(source_position),
                    "source_row_id": source_row_id,
                    "row_id": f"rule_collection:{source_row_id}",
                    "core_prompt": core_prompt,
                    "label": label,
                    "label_display": display_by_normalized[label],
                    "legal_labels": legal_labels,
                }
            )

        conflicting_groups = {
            canonical: members
            for canonical, members in exact_groups.items()
            if len({member["label"] for member in members}) > 1
        }
        rows: list[dict] = []
        same_label_duplicate_rows = 0
        overlength_rows = 0
        scorer_failures = 0
        for canonical, members in exact_groups.items():
            if canonical in conflicting_groups:
                continue
            chosen = min(
                members,
                key=lambda row: stable_digest(SEED, "dedup", domain, row["source_row_id"]),
            )
            same_label_duplicate_rows += len(members) - 1
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {
                    "role": "user",
                    "content": render_user_prompt(chosen["core_prompt"], chosen["legal_labels"]),
                },
            ]
            prompt_tokens = len(
                tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            )
            if prompt_tokens > MAX_PROMPT_TOKENS:
                overlength_rows += 1
                continue
            gold = f'<answer>{chosen["label_display"]}</answer>'
            score = compute_score(
                config["data_source"],
                gold,
                gold,
                {"legal_labels": chosen["legal_labels"]},
            )
            if score["score"] != 1.0:
                scorer_failures += 1
                continue
            base_group = stable_digest(domain, canonical_text(context_for_grouping(chosen["core_prompt"])))
            rows.append(
                {
                    **chosen,
                    "messages": messages,
                    "prompt_tokens": prompt_tokens,
                    "base_group_id": base_group,
                }
            )

        near_audit = connect_near_duplicate_groups(rows)
        training, validation, sampling_audit = select_domain(rows, domain)
        all_training.extend(training)
        all_validation.extend(validation)
        domain_audits[domain] = {
            "source_rows": len(frame),
            "canonical_unique_prompts": len(exact_groups),
            "exact_duplicate_rows": len(frame) - len(exact_groups),
            "same_label_duplicate_rows_removed": same_label_duplicate_rows,
            "conflicting_label_groups_removed": len(conflicting_groups),
            "conflicting_label_rows_removed": sum(len(value) for value in conflicting_groups.values()),
            "malformed_or_illegal_ground_truth_rows": malformed_ground_truth,
            "overlength_rows_removed": overlength_rows,
            "ground_truth_scorer_failures": scorer_failures,
            "eligible_rows": len(rows),
            "eligible_prompt_lengths": length_summary([row["prompt_tokens"] for row in rows]),
            "near_duplicate_grouping": near_audit,
            "sampling": sampling_audit,
        }

    all_training.sort(key=lambda row: stable_digest(SEED, "train-order", row["row_id"]))
    all_validation.sort(key=lambda row: stable_digest(SEED, "val-order", row["row_id"]))
    train_records = [to_record(row, "train") for row in all_training]
    val_records = [to_record(row, "validation") for row in all_validation]
    train_validation = validate_records(train_records, 3680)
    val_validation = validate_records(val_records, 320)

    train_groups = {record["extra_info"]["split_group_id"] for record in train_records}
    val_groups = {record["extra_info"]["split_group_id"] for record in val_records}
    if train_groups & val_groups:
        raise AssertionError("RuleCollection split-group leakage detected")
    train_prompts = {
        canonical_text(record["prompt"][1]["content"]) for record in train_records
    }
    val_prompts = {canonical_text(record["prompt"][1]["content"]) for record in val_records}
    if train_prompts & val_prompts:
        raise AssertionError("RuleCollection canonical prompt leakage detected")

    final_near_pairs = near_duplicate_pairs(
        [
            (row["row_id"], split, row["core_prompt"])
            for split, rows in (("train", all_training), ("validation", all_validation))
            for row in rows
        ]
    )
    if any(pair["cross_split"] for pair in final_near_pairs):
        raise AssertionError("near-duplicate grouping failed to isolate train and validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "rule_collection_train.parquet"
    val_path = args.output_dir / "rule_collection_val.parquet"
    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    pd.DataFrame(val_records).to_parquet(val_path, index=False)

    audit = {
        "baseline": "rule_collection",
        "domains": domain_audits,
        "train_domains": counter_dict(row["data_source"] for row in all_training),
        "validation_domains": counter_dict(row["data_source"] for row in all_validation),
        "train_domain_labels": counter_dict(
            f'{row["data_source"]}:{row["label"]}' for row in all_training
        ),
        "validation_domain_labels": counter_dict(
            f'{row["data_source"]}:{row["label"]}' for row in all_validation
        ),
        "train_prompt_lengths": length_summary([row["prompt_tokens"] for row in all_training]),
        "validation_prompt_lengths": length_summary([row["prompt_tokens"] for row in all_validation]),
        "train_schema_validation": train_validation,
        "validation_schema_validation": val_validation,
        "cross_split_group_overlap": len(train_groups & val_groups),
        "cross_split_canonical_prompt_overlap": len(train_prompts & val_prompts),
        "near_duplicate_threshold": 0.85,
        "near_duplicate_flagged_pairs": final_near_pairs,
        "near_duplicate_cross_split_pairs": sum(pair["cross_split"] for pair in final_near_pairs),
        "evaluation_contamination": {
            "status": "pending protected-evaluation corpus availability",
            "note": "Do not begin GPU calibration until the protocol's protected-suite comparison is completed.",
        },
    }
    json_dump(args.output_dir / "audit.json", audit)
    manifest = {
        "schema_version": 1,
        "baseline": "rule_collection",
        "curation_version": CURATION_VERSION,
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "files": source_files,
        },
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "sampling": {
            "strategy": "domain-quota and label-stratified SHA-256 sampling with context-group split isolation",
            "seed": int(SEED),
            "total": 4000,
            "train": 3680,
            "validation": 320,
            "domain_quotas": DOMAINS,
        },
        "reward": {
            "path": str(REWARD_PATH.relative_to(REPO_ROOT)),
            "name": "compute_score",
            "range": [-1.0, 1.0],
            "sha256": file_sha256(REWARD_PATH),
        },
        "files": {
            train_path.name: {"rows": len(train_records), "sha256": file_sha256(train_path)},
            val_path.name: {"rows": len(val_records), "sha256": file_sha256(val_path)},
            "audit.json": {"sha256": file_sha256(args.output_dir / "audit.json")},
        },
        "recreate": "python custom_data_preparation/curate_rule_collection_baseline.py",
    }
    json_dump(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
