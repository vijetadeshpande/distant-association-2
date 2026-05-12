"""Format-reward scorers for Codenames RLVR.

Each sub-check returns 0.0 when the format is followed and a negative
value (default ``-1.0``) when it is violated.  The aggregator in
``codenames_reward.py`` averages these values together with the task
sub-rewards, so penalties show up as direct subtractions from the final
scalar.

Exported checks
---------------
Thinking format (CoT structure)
  - ``thinking_all_present``: every one of the five sections appears.
  - ``thinking_in_order``:    sections appear in the canonical order.

Clue format
  - ``clue_tags_present``: ``[CODENAMES-CLUE-START]``, ``[Clue]``,
    ``[Selected-Targets]`` and ``[CODENAMES-CLUE-END]`` are all found.
  - ``clue_single_word``:  the clue token is a single whitespace-free word.
  - ``clue_no_hyphen``:    the clue contains no hyphen character.
  - ``clue_selected_nonempty``: Selected-Targets parsed to >= 1 word.
  - ``clue_selected_subset``: every Selected-Target is in Target-Set.
  - ``clue_no_morph_variant``: clue is not a morphological variant of
    any board word (either Target or Non-Target).  On failure we
    short-circuit in the caller and skip the judge call.
"""
from __future__ import annotations

from typing import Iterable

from .morphology import clue_violates_morphology
from .parsers import CluePhase, GuessPhase, ThinkingParse

PENALTY = -1.0


def _pen(passed: bool) -> float:
    return 0.0 if passed else PENALTY


def _normalize_set(words: Iterable[str]) -> set:
    return {w.strip().casefold() for w in words if str(w).strip()}


def thinking_format_scores(parsed: ThinkingParse) -> dict:
    return {
        "thinking_all_present": _pen(parsed.all_present),
        "thinking_in_order": _pen(parsed.in_order),
    }


def clue_format_scores(parsed: CluePhase, target_set, non_target_set) -> dict:
    tags = parsed.tags_present
    single = parsed.clue_is_single_word
    no_hyphen = parsed.tags_present and not parsed.clue_has_hyphen
    nonempty = parsed.selected_nonempty

    targets_norm = _normalize_set(target_set)
    selected_norm = _normalize_set(parsed.selected_targets) if parsed.selected_targets else set()
    subset_ok = bool(selected_norm) and selected_norm.issubset(targets_norm)

    if parsed.clue and single:
        morph_ok = not clue_violates_morphology(parsed.clue, target_set, non_target_set)
    else:
        # Clue missing/malformed — morphological check is vacuously "not violated"
        # here; the other flags already penalize the format issue.
        morph_ok = True

    return {
        "clue_tags_present": _pen(tags),
        "clue_single_word": _pen(single),
        "clue_no_hyphen": _pen(no_hyphen),
        "clue_selected_nonempty": _pen(nonempty),
        "clue_selected_subset": _pen(subset_ok),
        "clue_no_morph_variant": _pen(morph_ok),
    }


def guess_format_scores(parsed: GuessPhase, all_words, max_guesses=None) -> dict:
    """Score the guess block's format.

    Used in two places:

    * **Clue task, judge mode** — diagnostic only.  The trainee can't
      control judge output, so these aren't folded into the reward.
      Called with ``max_guesses=None``, which trivially passes the
      count check (score 0).
    * **Guess task** — the trainee owns the guess block, so every
      sub-score here is a real format-reward term.  Called with
      ``max_guesses`` from the prompt so we can penalize over-limit
      guess lists.
    """
    all_norm = _normalize_set(all_words)
    guess_norm = _normalize_set(parsed.guesses) if parsed.guesses else set()

    if max_guesses is None:
        count_ok = True   # not enforced when caller doesn't supply a limit
    else:
        try:
            limit = int(max_guesses)
        except (TypeError, ValueError):
            limit = 0
        count_ok = len(parsed.guesses) <= limit

    return {
        "guess_tags_present": _pen(parsed.tags_present),
        "guess_nonempty": _pen(parsed.guesses_nonempty),
        "guess_all_in_board": _pen(bool(guess_norm) and guess_norm.issubset(all_norm)),
        "guess_count_ok": _pen(count_ok),
    }


def is_clue_format_ok(scores: dict) -> bool:
    """Every clue-format sub-check passed (score == 0.0)."""
    keys = (
        "clue_tags_present",
        "clue_single_word",
        "clue_no_hyphen",
        "clue_selected_nonempty",
        "clue_selected_subset",
        "clue_no_morph_variant",
    )
    return all(scores.get(k, PENALTY) == 0.0 for k in keys)


def is_guess_format_ok(scores: dict) -> bool:
    """Every guess-format sub-check passed (score == 0.0).

    Used by the guess-task ``_compute_score_guess`` to short-circuit
    the task reward to zero on format failure.  Over-limit guess
    counts (``guess_count_ok``) and off-board guesses
    (``guess_all_in_board``) are treated as hard format fails per
    project decision.
    """
    keys = (
        "guess_tags_present",
        "guess_nonempty",
        "guess_all_in_board",
        "guess_count_ok",
    )
    return all(scores.get(k, PENALTY) == 0.0 for k in keys)
