from custom_reward_functions.task_reward import task_reward, zero_task_reward
from custom_reward_functions.tests import fixtures_rollouts as F


def test_all_correct():
    r = task_reward(["North America", "Africa"], F.TARGETS, F.NON_TARGETS)
    assert r["pos_correct"] == 1.0
    assert r["neg_nontarget"] == 0.0
    assert r["neg_invalid"] == 0.0
    assert r["task"] == 1.0


def test_all_non_target():
    r = task_reward(["thanatophobia", "pyrophobia"], F.TARGETS, F.NON_TARGETS)
    assert r["pos_correct"] == 0.0
    assert r["neg_nontarget"] == 1.0
    assert r["task"] == -1.0


def test_invalid_guesses():
    r = task_reward(["NOT-ON-BOARD"], F.TARGETS, F.NON_TARGETS)
    assert r["neg_invalid"] > 0.0
    assert r["pos_correct"] == 0.0


def test_casefold_normalization():
    r = task_reward(["NORTH AMERICA"], F.TARGETS, F.NON_TARGETS)
    assert r["pos_correct"] == 0.5  # 1/2 targets


def test_empty_guesses():
    r = task_reward([], F.TARGETS, F.NON_TARGETS)
    assert r == zero_task_reward()
