from custom_reward_functions.format_reward import (
    clue_format_scores,
    is_clue_format_ok,
    thinking_format_scores,
)
from custom_reward_functions.parsers import parse_clue, parse_thinking
from custom_reward_functions.tests import fixtures_rollouts as F


def test_thinking_pass_is_zero():
    scores = thinking_format_scores(parse_thinking(F.GOOD_ROLLOUT))
    assert scores["thinking_all_present"] == 0.0
    assert scores["thinking_in_order"] == 0.0


def test_thinking_out_of_order_penalty():
    scores = thinking_format_scores(parse_thinking(F.OUT_OF_ORDER_ROLLOUT))
    assert scores["thinking_in_order"] < 0.0


def test_clue_good_passes_all_checks():
    pc = parse_clue(F.GOOD_ROLLOUT)
    scores = clue_format_scores(pc, F.TARGETS, F.NON_TARGETS)
    assert is_clue_format_ok(scores) is True
    assert all(v == 0.0 for v in scores.values())


def test_clue_hyphen_penalty():
    pc = parse_clue(F.HYPHENATED_CLUE_ROLLOUT)
    scores = clue_format_scores(pc, F.TARGETS, F.NON_TARGETS)
    assert scores["clue_no_hyphen"] < 0.0
    assert is_clue_format_ok(scores) is False


def test_clue_out_of_set_target_penalty():
    pc = parse_clue(F.OUT_OF_SET_TARGET_ROLLOUT)
    scores = clue_format_scores(pc, F.TARGETS, F.NON_TARGETS)
    assert scores["clue_selected_subset"] < 0.0


def test_clue_morph_variant_penalty():
    pc = parse_clue(F.MORPH_VARIANT_ROLLOUT)
    scores = clue_format_scores(pc, F.TARGETS, F.NON_TARGETS)
    assert scores["clue_no_morph_variant"] < 0.0


def test_clue_missing_block_fails():
    pc = parse_clue(F.NO_CLUE_BLOCK_ROLLOUT)
    scores = clue_format_scores(pc, F.TARGETS, F.NON_TARGETS)
    assert is_clue_format_ok(scores) is False
