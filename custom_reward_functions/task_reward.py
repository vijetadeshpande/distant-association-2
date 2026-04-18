"""Task reward for the Codenames Spymaster role.

Overview
========

The training loop proceeds in four phases per rollout:

    1. Trainee (Spymaster) generates a clue and a Selected-Targets list
       for a board of ``[Target-Set] + [Non-Target-Set]`` words.
    2. The judge LLM (Operative) receives the shuffled union of all
       board words plus the clue, and is asked for up to
       ``Max-Guesses = len(Selected-Targets)`` guesses.
    3. The judge's guesses are parsed out.
    4. This module scores the guesses against the ground-truth board.

Sub-rewards
===========

Each sub-reward is a value in ``[0, 1]``; the aggregate ``task`` value
combines them with fixed signs.  All comparisons are casefolded and
whitespace-stripped so that the judge's surface form (e.g. ``"North
America"``) matches the board's stored form (e.g. ``"north america"``).

pos_correct
    Fraction of Target-Set words the judge correctly picked::

        pos_correct = |Guesses ∩ Target-Set|     / |Target-Set|

    Scales so a full sweep of the Target-Set returns 1.0 regardless of
    how many guesses the Spymaster allowed.

neg_nontarget
    Fraction of Non-Target-Set words the judge picked by mistake.  This
    penalizes clues that bleed into the opposing team's words::

        neg_nontarget = |Guesses ∩ Non-Target-Set| / |Non-Target-Set|

neg_invalid
    Fraction of guesses that are *not* on the board at all.  The judge
    is instructed to guess only from ``All-Words``; a non-zero value
    here indicates judge hallucination or a parser mismatch::

        neg_invalid = |Guesses \\ All-Words| / |All-Words|

Aggregate
=========

    task = pos_correct - neg_nontarget - neg_invalid

Range: ``[-2, 1]``.  The upstream aggregator in ``codenames_reward`` takes
the arithmetic mean of all format sub-rewards and all task sub-rewards,
so this module only needs to return the individual numbers; the weighting
is controlled at the aggregator level (and kept to a simple mean by
user preference).
"""
from __future__ import annotations

from typing import Iterable


def _norm(word: str) -> str:
    return str(word).strip().casefold()


def _norm_set(words: Iterable[str]) -> set:
    return {_norm(w) for w in words if str(w).strip()}


def task_reward(guesses, target_set, non_target_set) -> dict:
    """Compute the three task sub-rewards plus the combined scalar.

    Args:
        guesses:         Parsed list of judge guesses (raw strings).
        target_set:      Current row's [Target-Set].
        non_target_set:  Current row's [Non-Target-Set].

    Returns:
        Dict with keys ``pos_correct``, ``neg_nontarget``, ``neg_invalid``
        and ``task`` (the combined ``pos_correct - neg_nontarget -
        neg_invalid`` scalar).
    """
    g = _norm_set(guesses)
    t = _norm_set(target_set)
    nt = _norm_set(non_target_set)
    all_board = t | nt

    pos_correct = len(g & t) / max(len(t), 1)
    neg_nontarget = len(g & nt) / max(len(nt), 1)
    neg_invalid = len(g - all_board) / max(len(all_board), 1)

    return {
        "pos_correct": float(pos_correct),
        "neg_nontarget": float(neg_nontarget),
        "neg_invalid": float(neg_invalid),
        "task": float(pos_correct - neg_nontarget - neg_invalid),
    }


def zero_task_reward() -> dict:
    """Return a zero-filled task reward dict.

    Used when we short-circuit the judge call (e.g. the clue format
    failed or the clue is a morphological variant of a board word).
    Returning zeros here lets the aggregator still average the task
    terms in without biasing the signal.
    """
    return {
        "pos_correct": 0.0,
        "neg_nontarget": 0.0,
        "neg_invalid": 0.0,
        "task": 0.0,
    }
