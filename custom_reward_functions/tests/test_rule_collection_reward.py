"""Admission tests for the strict RuleCollection reward."""

from __future__ import annotations

from custom_reward_functions.rule_collection_reward import compute_score, parse_answer


def score(solution: object, gold: str = "<answer>True</answer>", **kwargs) -> dict[str, float]:
    return compute_score("Folio", solution, gold, **kwargs)


def test_canonical_correct_and_incorrect_rewards() -> None:
    assert score("<answer>True</answer>")["score"] == 1.0
    assert score("<answer>False</answer>")["score"] == -1.0


def test_case_and_surrounding_whitespace_are_normalized() -> None:
    assert score("reasoning\n<answer>  tRuE  </answer>")["score"] == 1.0


def test_exactly_one_well_formed_answer_block_is_required() -> None:
    invalid = [
        "True",
        "<answer>True",
        "True</answer>",
        "<ANSWER>True</ANSWER>",
        "<answer></answer>",
        "<answer>False</answer><answer>True</answer>",
        "<answer>True</answer><answer>",
        "</answer><answer>True</answer>",
        "<answer><b>True</b></answer>",
        "",
        None,
    ]
    for value in invalid:
        assert score(value)["score"] == -1.0


def test_text_outside_answer_block_cannot_override_final_answer() -> None:
    result = score("The answer is True.\n<answer>False</answer>")
    assert result["score"] == -1.0
    assert result["format_failure"] == 0.0


def test_illegal_labels_nan_and_infinity_receive_minimum() -> None:
    for label in ("maybe", "nan", "inf", "infinity", "A"):
        result = score(f"<answer>{label}</answer>")
        assert result["score"] == -1.0
        assert result["illegal_label"] == 1.0


def test_unknown_domain_or_bad_ground_truth_fails_closed() -> None:
    assert compute_score("unknown", "<answer>A</answer>", "<answer>A</answer>")["score"] == -1.0
    assert score("<answer>True</answer>", gold="True")["score"] == -1.0
    assert score("<answer>True</answer>", gold="<answer>Maybe</answer>")["score"] == -1.0


def test_row_level_legal_labels_are_enforced() -> None:
    extra = {"legal_labels": ["A", "B", "C", "D", "E"]}
    assert compute_score("Logical Deduction", "<answer>E</answer>", "<answer>E</answer>", extra)["score"] == 1.0
    assert compute_score("Logical Deduction", "<answer>F</answer>", "<answer>F</answer>", extra)["score"] == -1.0


def test_reward_is_deterministic_over_more_than_100_gold_cases() -> None:
    domains = {
        "AR-LSAT": ["A", "B", "C", "D", "E"],
        "Folio": ["True", "False", "Unknown"],
        "Logic NLI": ["contradiction", "self_contradiction", "neutral", "entailment"],
        "Logical Deduction": ["A", "B", "C", "D", "E", "F", "G"],
        "ProntoQA": ["True", "False", "Unknown"],
        "ProofWriter": ["True", "False", "Unknown"],
    }
    checked = 0
    for repetition in range(5):
        for domain, labels in domains.items():
            for label in labels:
                completion = f"reasoning pass {repetition}\n<answer> {label.swapcase()} </answer>"
                gold = f"<answer>{label}</answer>"
                first = compute_score(domain, completion, gold)
                second = compute_score(domain, completion, gold)
                assert first == second
                assert first["score"] == 1.0
                checked += 1
    assert checked >= 100


def test_parser_reports_malformed_values_without_raising() -> None:
    assert parse_answer(None)[0] is None
    assert parse_answer("<answer>x</answer>") == ("x", None)
