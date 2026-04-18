from custom_reward_functions.parsers import parse_clue, parse_guesses, parse_thinking
from custom_reward_functions.tests import fixtures_rollouts as F


def test_thinking_all_present_in_order():
    r = parse_thinking(F.GOOD_ROLLOUT)
    assert r.all_present is True
    assert r.in_order is True


def test_thinking_out_of_order():
    r = parse_thinking(F.OUT_OF_ORDER_ROLLOUT)
    assert r.all_present is True
    assert r.in_order is False


def test_thinking_missing_section():
    r = parse_thinking(F.MISSING_SECTION_ROLLOUT)
    assert r.all_present is False
    assert "reflection" in r.missing


def test_clue_good():
    r = parse_clue(F.GOOD_ROLLOUT)
    assert r.tags_present is True
    assert r.clue.upper() == "CONTINENT"
    assert r.clue_is_single_word is True
    assert r.clue_has_hyphen is False
    assert r.selected_nonempty is True
    # Compound target name preserved in token split (space-separated).
    assert any("AMERICA" in t.upper() for t in r.selected_targets)


def test_clue_hyphenated():
    r = parse_clue(F.HYPHENATED_CLUE_ROLLOUT)
    assert r.tags_present is True
    assert r.clue_has_hyphen is True


def test_clue_missing_block():
    r = parse_clue(F.NO_CLUE_BLOCK_ROLLOUT)
    assert r.tags_present is False
    assert r.ok is False


def test_guesses_clean():
    r = parse_guesses(F.JUDGE_OUTPUT_CLEAN)
    assert r.tags_present is True
    assert r.guesses_nonempty is True


def test_guesses_no_thinking_wrapper():
    r = parse_guesses(F.JUDGE_OUTPUT_NO_THINKING)
    assert r.tags_present is True
    assert r.guesses_nonempty is True
