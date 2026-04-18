"""End-to-end test for compute_score with the judge client mocked."""
import asyncio

import pytest

from custom_reward_functions import codenames_reward
from custom_reward_functions.tests import fixtures_rollouts as F


def _run(coro):
    return asyncio.run(coro)


def _extra_info():
    return {"target_words": F.TARGETS, "non_target_words": F.NON_TARGETS}


def test_short_circuits_on_bad_clue_format(monkeypatch):
    called = {"n": 0}

    async def _fail_judge(*a, **kw):
        called["n"] += 1
        return None

    monkeypatch.setattr(codenames_reward, "judge_guess", _fail_judge)

    out = _run(codenames_reward.compute_score(
        "custom", F.HYPHENATED_CLUE_ROLLOUT, ground_truth="continents",
        extra_info=_extra_info(),
    ))
    assert called["n"] == 0  # judge never called
    assert out["format_fail"] == 1
    assert out["pos_correct"] == 0.0


def test_morph_short_circuits_judge(monkeypatch):
    called = {"n": 0}

    async def _fail_judge(*a, **kw):
        called["n"] += 1
        return None

    monkeypatch.setattr(codenames_reward, "judge_guess", _fail_judge)

    out = _run(codenames_reward.compute_score(
        "custom", F.MORPH_VARIANT_ROLLOUT, ground_truth="continents",
        extra_info=_extra_info(),
    ))
    assert called["n"] == 0
    assert out["format_fail"] == 1
    assert out["clue_no_morph_variant"] < 0.0


def test_happy_path_with_mocked_judge(monkeypatch):
    async def _ok_judge(*a, **kw):
        return F.JUDGE_OUTPUT_CLEAN

    monkeypatch.setattr(codenames_reward, "judge_guess", _ok_judge)

    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="continents",
        extra_info=_extra_info(),
    ))
    assert out["format_fail"] == 0
    assert out["judge_fail"] == 0
    assert out["pos_correct"] == 1.0
    assert out["task"] == 1.0
    # score is mean(all fmt zeros + pos - neg_nontarget - neg_invalid terms)
    assert "score" in out
    assert isinstance(out["score"], float)


def test_judge_failure_returns_zero_task(monkeypatch):
    async def _ret_none(*a, **kw):
        return None

    monkeypatch.setattr(codenames_reward, "judge_guess", _ret_none)

    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="continents",
        extra_info=_extra_info(),
    ))
    assert out["judge_fail"] == 1
    assert out["task"] == 0.0
