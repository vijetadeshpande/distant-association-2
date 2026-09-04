"""Strict deterministic reward for the six-domain RuleCollection baseline."""

from __future__ import annotations

import re
from typing import Any, Iterable

_DOMAIN_LABELS = {
    "ar_lsat": ("a", "b", "c", "d", "e"),
    "folio": ("true", "false", "unknown"),
    "logic_nli": ("contradiction", "self_contradiction", "neutral", "entailment"),
    "logical_deduction": ("a", "b", "c", "d", "e", "f", "g"),
    "prontoqa": ("true", "false", "unknown"),
    "proofwriter": ("true", "false", "unknown"),
}

_ANSWER_PATTERN = re.compile(r"<answer>\s*([^<>]*?)\s*</answer>", re.DOTALL)


def _domain_key(data_source: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(data_source).casefold()).strip("_")


def _normalize_labels(values: Iterable[object]) -> set[str]:
    return {str(value).strip().casefold() for value in values}


def parse_answer(text: object) -> tuple[str | None, str | None]:
    """Return a normalized label and ``None``, or ``(None, error_code)``.

    The tags themselves are intentionally case-sensitive. Exactly one opening
    tag, one closing tag, and one well-formed block are required. Text outside
    the block is allowed because it contains the model's reasoning sections.
    """

    if not isinstance(text, str) or not text:
        return None, "empty_or_non_string"
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None, "answer_tag_count"
    matches = _ANSWER_PATTERN.findall(text)
    if len(matches) != 1:
        return None, "malformed_answer_block"
    label = matches[0].strip().casefold()
    if not label:
        return None, "empty_label"
    return label, None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    """Score one RuleCollection completion with strict exact match.

    Returns ``+1`` only when the completion has exactly one well-formed answer
    block, its normalized label is legal for the row/domain, and it equals the
    normalized gold label. Every failure mode receives ``-1``. There is no
    auxiliary format or reasoning-length reward.
    """

    try:
        domain = _domain_key(data_source)
        domain_labels = set(_DOMAIN_LABELS.get(domain, ()))
        row_labels = (extra_info or {}).get("legal_labels")
        legal_labels = _normalize_labels(row_labels) if row_labels is not None else domain_labels
        if not domain_labels or not legal_labels or not legal_labels.issubset(domain_labels):
            return {
                "score": -1.0,
                "acc": 0.0,
                "format_failure": 0.0,
                "illegal_label": 0.0,
                "ground_truth_failure": 1.0,
            }

        gold, gold_error = parse_answer(ground_truth)
        if gold_error is not None or gold not in legal_labels:
            return {
                "score": -1.0,
                "acc": 0.0,
                "format_failure": 0.0,
                "illegal_label": 0.0,
                "ground_truth_failure": 1.0,
            }

        prediction, prediction_error = parse_answer(solution_str)
        format_failure = float(prediction_error is not None)
        illegal_label = float(prediction_error is None and prediction not in legal_labels)
        correct = prediction_error is None and not illegal_label and prediction == gold
        return {
            "score": 1.0 if correct else -1.0,
            "acc": float(correct),
            "format_failure": format_failure,
            "illegal_label": illegal_label,
            "ground_truth_failure": 0.0,
        }
    except Exception:
        return {
            "score": -1.0,
            "acc": 0.0,
            "format_failure": 1.0,
            "illegal_label": 0.0,
            "ground_truth_failure": 0.0,
        }


__all__ = ["compute_score", "parse_answer"]
