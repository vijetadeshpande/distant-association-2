"""Shared deterministic helpers for static RLVR baseline curation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def canonical_text(value: str) -> str:
    """Canonical form used for exact deduplication and leakage checks."""

    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def largest_remainder(
    counts: Mapping[str, int], total: int, *, minimum: int = 0
) -> dict[str, int]:
    """Allocate ``total`` proportionally with deterministic largest remainders.

    ``minimum`` is enforced for every non-empty category. Categories and ties
    are ordered lexicographically so input mapping order cannot affect output.
    """

    positive = {str(key): int(value) for key, value in counts.items() if value > 0}
    if total < 0 or total > sum(positive.values()):
        raise ValueError(f"cannot allocate {total} from capacity {sum(positive.values())}")
    if minimum and any(value < minimum for value in positive.values()):
        too_small = {key: value for key, value in positive.items() if value < minimum}
        raise ValueError(f"categories cannot meet minimum={minimum}: {too_small}")
    if minimum * len(positive) > total:
        raise ValueError("minimum allocations exceed requested total")

    population = sum(positive.values())
    exact = {key: total * value / population for key, value in positive.items()}
    quota = {
        key: min(value, max(minimum, math.floor(exact[key])))
        for key, value in positive.items()
    }

    while sum(quota.values()) < total:
        candidates = [key for key in positive if quota[key] < positive[key]]
        if not candidates:
            raise RuntimeError("allocation capacity exhausted")
        key = max(candidates, key=lambda item: (exact[item] - quota[item], -len(item), item))
        quota[key] += 1

    while sum(quota.values()) > total:
        candidates = [key for key in positive if quota[key] > minimum]
        if not candidates:
            raise RuntimeError("cannot reduce allocation without violating minimum")
        key = min(candidates, key=lambda item: (exact[item] - quota[item], item))
        quota[key] -= 1

    return dict(sorted(quota.items()))


def length_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {key: 0 for key in ("count", "min", "median", "p90", "p95", "p99", "max")}
    ordered = sorted(int(value) for value in values)

    def percentile(fraction: float) -> int:
        index = math.ceil(fraction * len(ordered)) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def prompt_content(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def validate_records(records: Sequence[dict], expected_count: int) -> dict[str, Any]:
    if len(records) != expected_count:
        raise AssertionError(f"expected {expected_count} records, found {len(records)}")
    required = {"prompt", "data_source", "reward_model", "extra_info"}
    row_ids: list[str] = []
    canonical_prompts: list[str] = []
    split_groups: list[str] = []
    for index, record in enumerate(records):
        if set(record) != required:
            raise AssertionError(f"row {index} columns are {set(record)}, expected {required}")
        messages = record["prompt"]
        if not isinstance(messages, list) or not messages:
            raise AssertionError(f"row {index} has invalid prompt")
        if any(set(message) != {"role", "content"} for message in messages):
            raise AssertionError(f"row {index} has malformed chat messages")
        if not record["data_source"] or "ground_truth" not in record["reward_model"]:
            raise AssertionError(f"row {index} has incomplete reward routing")
        extra = record["extra_info"]
        for key in (
            "row_id",
            "source_dataset",
            "source_revision",
            "source_row_id",
            "split_group_id",
            "stratum",
            "curation_version",
        ):
            if key not in extra or extra[key] in (None, ""):
                raise AssertionError(f"row {index} missing extra_info[{key!r}]")
        row_ids.append(str(extra["row_id"]))
        split_groups.append(str(extra["split_group_id"]))
        canonical_prompts.append(canonical_text(prompt_content(messages)))

    return {
        "rows": len(records),
        "unique_row_ids": len(set(row_ids)),
        "canonical_unique_prompts": len(set(canonical_prompts)),
        "unique_split_groups": len(set(split_groups)),
        "duplicate_row_ids": len(row_ids) - len(set(row_ids)),
        "canonical_duplicate_prompts": len(canonical_prompts) - len(set(canonical_prompts)),
    }


def token_shingles(text: str, width: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", canonical_text(text))
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def near_duplicate_pairs(
    items: Sequence[tuple[str, str, str]], *, threshold: float = 0.85, signature_size: int = 64
) -> list[dict[str, Any]]:
    """Find high-Jaccard candidates using deterministic bottom-k LSH.

    ``items`` contains ``(row_id, split, text)``. The result is intended as an
    audit flagger, not as a semantic-equivalence oracle.
    """

    shingle_sets: list[set[str]] = []
    signatures: list[tuple[int, ...]] = []
    for _, _, text in items:
        shingles = token_shingles(text)
        shingle_sets.append(shingles)
        hashes = sorted(
            int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")
            for value in shingles
        )[:signature_size]
        signatures.append(tuple(hashes))

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    band_size = 4
    for index, signature in enumerate(signatures):
        for band in range(0, len(signature), band_size):
            chunk = signature[band : band + band_size]
            if len(chunk) == band_size:
                buckets[(band // band_size, chunk)].append(index)

    candidates: set[tuple[int, int]] = set()
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for left_pos in range(len(bucket)):
            for right_pos in range(left_pos + 1, len(bucket)):
                left, right = bucket[left_pos], bucket[right_pos]
                candidates.add((min(left, right), max(left, right)))

    result: list[dict[str, Any]] = []
    for left, right in sorted(candidates):
        union = shingle_sets[left] | shingle_sets[right]
        similarity = len(shingle_sets[left] & shingle_sets[right]) / len(union) if union else 1.0
        if similarity >= threshold:
            left_id, left_split, _ = items[left]
            right_id, right_split, _ = items[right]
            result.append(
                {
                    "left_row_id": left_id,
                    "right_row_id": right_id,
                    "left_split": left_split,
                    "right_split": right_split,
                    "cross_split": left_split != right_split,
                    "token_5gram_jaccard": round(similarity, 6),
                }
            )
    return result


def counter_dict(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))
