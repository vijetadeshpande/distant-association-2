"""Tests for the GloVe-cosine reward path.

The real GloVe artifacts live under ``custom_data/glove_vectors/`` and
take ~5 s to load; these tests instead inject a tiny in-memory vocab
and matrix into the ``cosine_reward`` module so they run instantly.
"""
import asyncio

import numpy as np
import pytest

from custom_reward_functions import codenames_reward, cosine_reward
from custom_reward_functions.tests import fixtures_rollouts as F


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _stub_table(monkeypatch):
    """Inject a 4-word L2-normalized matrix into the module cache."""
    vocab = {"continent": 0, "land": 1, "ocean": 2, "fire": 3}
    matrix = np.stack([
        _norm([1.0, 0.0, 0.0]),                 # continent
        _norm([0.9, 0.1, 0.0]),                 # land  (close to continent)
        _norm([-1.0, 0.0, 0.0]),                # ocean (opposite of continent)
        _norm([0.0, 1.0, 0.0]),                 # fire  (orthogonal)
    ]).astype(np.float32)
    monkeypatch.setattr(cosine_reward, "_MATRIX", matrix, raising=False)
    monkeypatch.setattr(cosine_reward, "_VOCAB", vocab, raising=False)


def _clear_table_cache(monkeypatch):
    monkeypatch.setattr(cosine_reward, "_MATRIX", None, raising=False)
    monkeypatch.setattr(cosine_reward, "_VOCAB", None, raising=False)


def _run(coro):
    return asyncio.run(coro)


def _extra_info(clue="continent"):
    return {
        "target_words": F.TARGETS,
        "non_target_words": F.NON_TARGETS,
        "clue": clue,
    }


# ---------- is_cosine_mode -------------------------------------------------

def test_is_cosine_mode_when_env_unset(monkeypatch):
    monkeypatch.delenv("JUDGE_NAME", raising=False)
    assert cosine_reward.is_cosine_mode()


def test_is_cosine_mode_when_env_none(monkeypatch):
    monkeypatch.setenv("JUDGE_NAME", "none")
    assert cosine_reward.is_cosine_mode()


def test_is_cosine_mode_when_env_null(monkeypatch):
    monkeypatch.setenv("JUDGE_NAME", "NULL")
    assert cosine_reward.is_cosine_mode()


def test_judge_mode_when_env_set(monkeypatch):
    monkeypatch.setenv("JUDGE_NAME", "qwen3-judge")
    assert not cosine_reward.is_cosine_mode()


# ---------- cosine_reward unit --------------------------------------------

def test_cosine_identical_words(monkeypatch):
    _stub_table(monkeypatch)
    r = cosine_reward.cosine_reward("continent", "continent")
    assert r["oov"] == 0
    assert r["cosine_sim"] == pytest.approx(1.0, abs=1e-6)
    assert r["pos_correct"] == r["cosine_sim"]
    assert r["task"] == r["cosine_sim"]
    assert r["neg_nontarget"] == 0.0
    assert r["neg_invalid"] == 0.0


def test_cosine_close_words(monkeypatch):
    _stub_table(monkeypatch)
    r = cosine_reward.cosine_reward("continent", "land")
    # Both close to [1,0,0]; cosine should be high but < 1.
    assert r["oov"] == 0
    assert 0.9 < r["cosine_sim"] < 1.0


def test_cosine_opposite_words(monkeypatch):
    _stub_table(monkeypatch)
    r = cosine_reward.cosine_reward("continent", "ocean")
    assert r["oov"] == 0
    assert r["cosine_sim"] == pytest.approx(-1.0, abs=1e-6)


def test_cosine_orthogonal_words(monkeypatch):
    _stub_table(monkeypatch)
    r = cosine_reward.cosine_reward("continent", "fire")
    assert r["oov"] == 0
    assert r["cosine_sim"] == pytest.approx(0.0, abs=1e-6)


def test_cosine_oov_parsed_clue(monkeypatch):
    _stub_table(monkeypatch)
    r = cosine_reward.cosine_reward("xyzzy", "continent")
    assert r["oov"] == 1
    assert r["cosine_sim"] == 0.0
    assert r["task"] == 0.0


def test_cosine_oov_target_clue(monkeypatch):
    _stub_table(monkeypatch)
    r = cosine_reward.cosine_reward("continent", "xyzzy")
    assert r["oov"] == 1
    assert r["cosine_sim"] == 0.0


def test_cosine_casefolding(monkeypatch):
    _stub_table(monkeypatch)
    r = cosine_reward.cosine_reward("CONTINENT", "Continent")
    assert r["oov"] == 0
    assert r["cosine_sim"] == pytest.approx(1.0, abs=1e-6)


def test_cosine_empty_parsed_clue(monkeypatch):
    _stub_table(monkeypatch)
    r = cosine_reward.cosine_reward("", "continent")
    assert r["oov"] == 1


def test_cosine_table_missing_raises(monkeypatch, tmp_path):
    """Missing GloVe artifacts must be a hard error, not a silent zero
    — otherwise a misconfigured run would train against an all-zero
    reward signal without anyone noticing."""
    _clear_table_cache(monkeypatch)
    monkeypatch.setenv("GLOVE_NPY_PATH", str(tmp_path / "missing.npy"))
    monkeypatch.setenv("GLOVE_VOCAB_PATH", str(tmp_path / "missing.pkl"))
    with pytest.raises(FileNotFoundError):
        cosine_reward.cosine_reward("continent", "continent")


# ---------- compute_score end-to-end in cosine mode -----------------------

def test_compute_score_cosine_happy_path(monkeypatch):
    monkeypatch.delenv("JUDGE_NAME", raising=False)
    _stub_table(monkeypatch)

    async def _no_judge(*a, **kw):
        raise AssertionError("judge_guess should not be called in cosine mode")

    monkeypatch.setattr(codenames_reward, "judge_guess", _no_judge)

    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="",
        extra_info=_extra_info(clue="continent"),
    ))
    assert out["format_fail"] == 0
    assert out["judge_fail"] == 0
    assert out["oov"] == 0
    assert out["cosine_sim"] == pytest.approx(1.0, abs=1e-6)
    assert out["pos_correct"] == out["cosine_sim"]
    assert out["task"] == out["cosine_sim"]
    assert out["judge_guesses"] == ""
    # Judge-side diagnostics still present so the batch key set is uniform.
    assert "guess_tags_present" in out
    assert "guess_nonempty" in out
    assert "guess_all_in_board" in out


def test_compute_score_cosine_short_circuit_on_format_fail(monkeypatch):
    monkeypatch.delenv("JUDGE_NAME", raising=False)
    _stub_table(monkeypatch)

    async def _no_judge(*a, **kw):
        raise AssertionError("judge_guess should not be called in cosine mode")

    monkeypatch.setattr(codenames_reward, "judge_guess", _no_judge)

    out = _run(codenames_reward.compute_score(
        "custom", F.HYPHENATED_CLUE_ROLLOUT, ground_truth="",
        extra_info=_extra_info(clue="continent"),
    ))
    assert out["format_fail"] == 1
    assert out["cosine_sim"] == 0.0
    assert out["oov"] == 0
    assert out["task"] == 0.0


def test_compute_score_cosine_oov_when_target_missing(monkeypatch):
    monkeypatch.delenv("JUDGE_NAME", raising=False)
    _stub_table(monkeypatch)

    async def _no_judge(*a, **kw):
        raise AssertionError("judge_guess should not be called in cosine mode")

    monkeypatch.setattr(codenames_reward, "judge_guess", _no_judge)

    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="",
        extra_info=_extra_info(clue="not_in_vocab_word"),
    ))
    assert out["format_fail"] == 0
    assert out["oov"] == 1
    assert out["cosine_sim"] == 0.0
    assert out["task"] == 0.0


def test_compute_score_cosine_missing_clue_in_extra_info(monkeypatch):
    monkeypatch.delenv("JUDGE_NAME", raising=False)
    _stub_table(monkeypatch)
    out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="",
        extra_info={"target_words": F.TARGETS, "non_target_words": F.NON_TARGETS},
    ))
    assert out["oov"] == 1
    assert out["cosine_sim"] == 0.0


# ---------- key-set homogeneity across modes ------------------------------

def test_return_key_set_matches_between_modes(monkeypatch):
    """Both modes must emit the same dict keys so VeRL's per-step
    np.array across the batch sees a homogeneous schema."""
    # Run once in cosine mode.
    monkeypatch.delenv("JUDGE_NAME", raising=False)
    _stub_table(monkeypatch)
    cosine_out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="",
        extra_info=_extra_info(clue="continent"),
    ))

    # Then in judge mode.
    monkeypatch.setenv("JUDGE_NAME", "qwen3-judge")

    async def _ok_judge(*a, **kw):
        return F.JUDGE_OUTPUT_CLEAN

    monkeypatch.setattr(codenames_reward, "judge_guess", _ok_judge)
    judge_out = _run(codenames_reward.compute_score(
        "custom", F.GOOD_ROLLOUT, ground_truth="",
        extra_info=_extra_info(clue="continent"),
    ))

    assert set(cosine_out.keys()) == set(judge_out.keys())
