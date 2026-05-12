"""Tests for the guess-task reward path and the top-level dispatcher.

The guess task scores the *trainee's own* guess block against the
known board — no judge call, no cosine fallback. We pin
``JUDGE_NAME`` for safety (the dispatcher routes by task, not by
judge mode, but stubs avoid loading any external resource).
"""
import asyncio

import pytest

from custom_reward_functions import codenames_reward
from custom_reward_functions.format_reward import (
    guess_format_scores,
    is_guess_format_ok,
)
from custom_reward_functions.parsers import parse_guesses
from custom_reward_functions.tests import fixtures_rollouts as F


def _run(coro):
    return asyncio.run(coro)


def _extra_info(max_guesses=2):
    return {
        "task": "codenames_guess_generation",
        "target_words": F.TARGETS,
        "non_target_words": F.NON_TARGETS,
        "all_words": F.ALL_WORDS,
        "clue": "continent",
        "num_max_guesses": max_guesses,
    }


# ---------- guess_format_scores unit --------------------------------------

def test_guess_format_scores_ok():
    pg = parse_guesses(F.GOOD_GUESS_ROLLOUT)
    s = guess_format_scores(pg, F.ALL_WORDS, max_guesses=2)
    assert s["guess_tags_present"] == 0.0
    assert s["guess_nonempty"] == 0.0
    assert s["guess_all_in_board"] == 0.0
    assert s["guess_count_ok"] == 0.0
    assert is_guess_format_ok(s)


def test_guess_format_scores_no_block():
    pg = parse_guesses(F.NO_GUESS_BLOCK_ROLLOUT)
    s = guess_format_scores(pg, F.ALL_WORDS, max_guesses=2)
    assert s["guess_tags_present"] < 0.0
    assert not is_guess_format_ok(s)


def test_guess_format_scores_empty_list():
    pg = parse_guesses(F.EMPTY_GUESS_LIST_ROLLOUT)
    s = guess_format_scores(pg, F.ALL_WORDS, max_guesses=2)
    assert s["guess_nonempty"] < 0.0
    assert not is_guess_format_ok(s)


def test_guess_format_scores_off_board():
    pg = parse_guesses(F.OFF_BOARD_GUESS_ROLLOUT)
    s = guess_format_scores(pg, F.ALL_WORDS, max_guesses=2)
    assert s["guess_all_in_board"] < 0.0
    assert not is_guess_format_ok(s)


def test_guess_format_scores_over_limit():
    pg = parse_guesses(F.OVER_LIMIT_GUESS_ROLLOUT)
    s = guess_format_scores(pg, F.ALL_WORDS, max_guesses=2)
    assert s["guess_count_ok"] < 0.0
    # All other format checks should still pass; this is a count violation only.
    assert s["guess_tags_present"] == 0.0
    assert s["guess_nonempty"] == 0.0
    assert s["guess_all_in_board"] == 0.0
    assert not is_guess_format_ok(s)


def test_guess_format_count_check_disabled_when_max_is_none():
    pg = parse_guesses(F.OVER_LIMIT_GUESS_ROLLOUT)
    s = guess_format_scores(pg, F.ALL_WORDS, max_guesses=None)
    # When the caller doesn't set a limit (clue-path diagnostic use),
    # over-limit is not penalized.
    assert s["guess_count_ok"] == 0.0


# ---------- compute_score routing ----------------------------------------

def test_dispatcher_routes_to_guess(monkeypatch):
    monkeypatch.setenv("JUDGE_NAME", "qwen3-judge")

    async def _no_judge(*a, **kw):
        raise AssertionError("judge_guess must not be called for the guess task")

    monkeypatch.setattr(codenames_reward, "judge_guess", _no_judge)
    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_GUESS_ROLLOUT, ground_truth="",
        extra_info=_extra_info(),
    ))
    assert out["format_fail"] == 0
    assert out["pos_correct"] == 1.0
    assert out["task"] == 1.0


def test_dispatcher_routes_to_clue_by_default(monkeypatch):
    """Missing task field defaults to clue."""
    monkeypatch.setenv("JUDGE_NAME", "qwen3-judge")

    async def _ok_judge(*a, **kw):
        return F.JUDGE_OUTPUT_CLEAN

    monkeypatch.setattr(codenames_reward, "judge_guess", _ok_judge)
    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="",
        extra_info={"target_words": F.TARGETS, "non_target_words": F.NON_TARGETS},
    ))
    # Should have produced a judge_guesses string (clue task).
    assert out["judge_guesses"] == "NORTH AMERICA,AFRICA"


# ---------- guess-task end-to-end ----------------------------------------

def test_guess_happy_path(monkeypatch):
    monkeypatch.setenv("JUDGE_NAME", "qwen3-judge")

    async def _no_judge(*a, **kw):
        raise AssertionError("judge must not be called")

    monkeypatch.setattr(codenames_reward, "judge_guess", _no_judge)
    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_GUESS_ROLLOUT, ground_truth="",
        extra_info=_extra_info(),
    ))
    assert out["format_fail"] == 0
    assert out["judge_fail"] == 0
    assert out["pos_correct"] == 1.0   # both targets correctly identified
    assert out["neg_nontarget"] == 0.0
    assert out["neg_invalid"] == 0.0
    assert out["task"] == 1.0
    assert out["guess_tags_present"] == 0.0
    assert out["guess_count_ok"] == 0.0


def test_guess_short_circuit_on_missing_block(monkeypatch):
    out = _run(codenames_reward.compute_score(
        "custom", F.NO_GUESS_BLOCK_ROLLOUT, ground_truth="",
        extra_info=_extra_info(),
    ))
    assert out["format_fail"] == 1
    assert out["task"] == 0.0
    assert out["pos_correct"] == 0.0


def test_guess_short_circuit_on_empty_list(monkeypatch):
    out = _run(codenames_reward.compute_score(
        "custom", F.EMPTY_GUESS_LIST_ROLLOUT, ground_truth="",
        extra_info=_extra_info(),
    ))
    assert out["format_fail"] == 1
    assert out["task"] == 0.0


def test_guess_short_circuit_on_off_board(monkeypatch):
    out = _run(codenames_reward.compute_score(
        "custom", F.OFF_BOARD_GUESS_ROLLOUT, ground_truth="",
        extra_info=_extra_info(),
    ))
    assert out["format_fail"] == 1
    assert out["task"] == 0.0
    assert out["guess_all_in_board"] < 0.0


def test_guess_short_circuit_on_over_limit(monkeypatch):
    out = _run(codenames_reward.compute_score(
        "custom", F.OVER_LIMIT_GUESS_ROLLOUT, ground_truth="",
        extra_info=_extra_info(max_guesses=2),
    ))
    assert out["format_fail"] == 1
    assert out["task"] == 0.0
    assert out["guess_count_ok"] < 0.0


def test_guess_all_wrong(monkeypatch):
    out = _run(codenames_reward.compute_score(
        "custom", F.ALL_WRONG_GUESS_ROLLOUT, ground_truth="",
        extra_info=_extra_info(),
    ))
    assert out["format_fail"] == 0
    assert out["pos_correct"] == 0.0
    assert out["neg_nontarget"] == 1.0
    assert out["task"] == -1.0


def test_guess_partial_credit(monkeypatch):
    out = _run(codenames_reward.compute_score(
        "custom", F.PARTIAL_GUESS_ROLLOUT, ground_truth="",
        extra_info=_extra_info(),
    ))
    assert out["format_fail"] == 0
    assert out["pos_correct"] == 0.5   # 1 of 2 targets
    assert out["neg_nontarget"] == 0.0


def test_guess_reconstructs_all_words_from_target_sets(monkeypatch):
    """When all_words is absent, falls back to target+non_target union."""
    info = _extra_info()
    info.pop("all_words")
    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_GUESS_ROLLOUT, ground_truth="",
        extra_info=info,
    ))
    assert out["format_fail"] == 0
    assert out["pos_correct"] == 1.0


# ---------- key-set parity across both tasks ------------------------------

def test_return_key_set_matches_across_tasks(monkeypatch):
    """Clue and guess paths must emit identical dict keys so VeRL's
    per-step np.array across a mixed-task batch stays homogeneous."""
    monkeypatch.setenv("JUDGE_NAME", "qwen3-judge")

    async def _ok_judge(*a, **kw):
        return F.JUDGE_OUTPUT_CLEAN

    monkeypatch.setattr(codenames_reward, "judge_guess", _ok_judge)

    clue_out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="",
        extra_info={
            "task": "codenames_clue_generation",
            "target_words": F.TARGETS,
            "non_target_words": F.NON_TARGETS,
        },
    ))
    guess_out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_GUESS_ROLLOUT, ground_truth="",
        extra_info=_extra_info(),
    ))
    assert set(clue_out.keys()) == set(guess_out.keys())
