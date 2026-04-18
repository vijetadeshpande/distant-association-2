"""Morphological-variant check for Codenames clues.

The Clue-Generator rule forbids clues that are morphological variants
(inflections, derivations), abbreviations, or direct translations of any
board word (see ``.claude/rules/codenames.md``).  A full NLP solution
would need a lemmatizer or language model; we instead use a layered
cheap check that catches the common English failure modes (plural,
past/progressive/agentive, simple prefix/suffix overlaps).

The check intentionally errs toward *detecting* overlap — a false
positive costs one format penalty, while a false negative lets an
obvious variant through.
"""
from __future__ import annotations

import re

# A small, deterministic Porter-style stem.  Not exhaustive; covers the
# suffixes that make up the vast majority of English inflections.
_SUFFIXES = (
    "ational",
    "tional",
    "izer",
    "iser",
    "ation",
    "ising",
    "izing",
    "ingly",
    "fully",
    "ively",
    "iness",
    "ness",
    "ment",
    "able",
    "ible",
    "less",
    "ful",
    "ies",
    "ied",
    "ier",
    "ing",
    "est",
    "ous",
    "ive",
    "ity",
    "ate",
    "ers",
    "ed",
    "er",
    "es",
    "ly",
    "en",
    "al",
    "s",
)

_NON_ALPHA_RE = re.compile(r"[^a-z]")


def _normalize(word: str) -> str:
    """Lowercase and strip non-alphabetic characters."""
    return _NON_ALPHA_RE.sub("", word.lower())


def _stem(word: str) -> str:
    """Strip the longest matching suffix; keep at least 3 characters."""
    w = _normalize(word)
    for suf in _SUFFIXES:
        if len(w) - len(suf) >= 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def is_morph_variant(clue: str, board_word: str) -> bool:
    """True if ``clue`` looks like a morphological variant of ``board_word``.

    Layered checks, in order of specificity:

    1. Exact (normalized) equality.
    2. One string contains the other *and* the shorter string is at least
       3 characters.  Catches ``tire`` vs ``tired``, ``swim`` vs ``swimming``.
    3. Shared stem after suffix stripping, as long as both stems are at
       least 3 characters.  Catches ``runner`` vs ``run``, ``happily`` vs
       ``happy``.
    """
    c, b = _normalize(clue), _normalize(board_word)
    if not c or not b:
        return False
    if c == b:
        return True

    shorter, longer = (c, b) if len(c) <= len(b) else (b, c)
    if len(shorter) >= 3 and shorter in longer:
        return True

    sc, sb = _stem(c), _stem(b)
    if len(sc) >= 3 and len(sb) >= 3 and sc == sb:
        return True

    return False


def clue_violates_morphology(clue: str, target_set, non_target_set) -> bool:
    """Return True if the clue is a variant of *any* board word."""
    for word in list(target_set) + list(non_target_set):
        if is_morph_variant(clue, word):
            return True
    return False
