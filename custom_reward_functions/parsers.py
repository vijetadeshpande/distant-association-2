"""Rollout and judge-output parsers for Codenames RLVR.

Three parsers are exposed:

- ``parse_thinking`` extracts the five-section Chain-of-Thought structure
  (``<thinking>`` ... ``<output>``) and reports presence / ordering.
- ``parse_clue`` extracts the Spymaster rollout block
  ``[CODENAMES-CLUE-START] ... [CODENAMES-CLUE-END]`` and pulls the clue
  word and the Selected-Targets list.
- ``parse_guesses`` extracts the Operative judge block
  ``[CODENAMES-GUESS-START] ... [CODENAMES-GUESS-END]`` and pulls the
  guesses list.

Each parser returns a small dataclass with structured flags so the
format-reward scorer can compose per-rule penalties without re-scanning
the string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Five CoT sections, in required order.
THINKING_SECTIONS = ("thinking", "reasoning", "reflection", "adjustment", "output")

_SECTION_OPEN_RE = {name: re.compile(rf"<\s*{name}\s*>", re.IGNORECASE) for name in THINKING_SECTIONS}
_OUTPUT_BLOCK_RE = re.compile(r"<\s*output\s*>(.*?)<\s*/\s*output\s*>", re.IGNORECASE | re.DOTALL)

_CLUE_BLOCK_RE = re.compile(
    r"\[CODENAMES-CLUE-START\](.*?)\[CODENAMES-CLUE-END\]", re.DOTALL
)
_CLUE_LINE_RE = re.compile(r"\[Clue\]\s*:\s*(.+)", re.IGNORECASE)
_SELECTED_LINE_RE = re.compile(
    r"\[Selected-Targets\]\s*:\s*(.+?)(?=\[CODENAMES-CLUE-END\]|$)",
    re.DOTALL | re.IGNORECASE,
)

_GUESS_BLOCK_RE = re.compile(
    r"\[CODENAMES-GUESS-START\](.*?)\[CODENAMES-GUESS-END\]", re.DOTALL
)
_GUESSES_LINE_RE = re.compile(
    r"\[Guesses\]\s*:\s*(.+?)(?=\[CODENAMES-GUESS-END\]|$)", re.DOTALL | re.IGNORECASE
)


@dataclass
class ThinkingParse:
    all_present: bool = False
    in_order: bool = False
    missing: tuple = ()


@dataclass
class CluePhase:
    tags_present: bool = False
    clue: str = ""
    clue_is_single_word: bool = False
    clue_has_hyphen: bool = False
    selected_targets: list = field(default_factory=list)
    selected_nonempty: bool = False

    @property
    def ok(self) -> bool:
        return (
            self.tags_present
            and self.clue_is_single_word
            and not self.clue_has_hyphen
            and self.selected_nonempty
        )


@dataclass
class GuessPhase:
    tags_present: bool = False
    guesses: list = field(default_factory=list)
    guesses_nonempty: bool = False


def parse_thinking(response: str) -> ThinkingParse:
    """Return presence and ordering of the five CoT sections."""
    positions = {}
    for name, pattern in _SECTION_OPEN_RE.items():
        match = pattern.search(response)
        if match is not None:
            positions[name] = match.start()

    missing = tuple(s for s in THINKING_SECTIONS if s not in positions)
    all_present = not missing
    in_order = False
    if all_present:
        last = -1
        in_order = True
        for name in THINKING_SECTIONS:
            if positions[name] <= last:
                in_order = False
                break
            last = positions[name]
    return ThinkingParse(all_present=all_present, in_order=in_order, missing=missing)


def _extract_output_region(response: str) -> str:
    """Return the text inside ``<output>...</output>``; fall back to full response."""
    m = _OUTPUT_BLOCK_RE.search(response)
    return m.group(1) if m else response


def _split_word_list(raw: str) -> list:
    """Split ``[A, B]`` / ``[NORTH AMERICA, AFRICA]`` / ``A B`` into tokens.

    Commas are the authoritative separator when present, so multi-word
    entries like ``NORTH AMERICA`` are preserved.  If no comma appears
    we fall back to whitespace splitting.
    """
    raw = raw.strip().strip("[]").strip()
    if not raw:
        return []
    if "," in raw:
        parts = raw.split(",")
    else:
        parts = re.split(r"\s+", raw)
    return [p.strip().strip("[]").strip() for p in parts if p.strip()]


def parse_clue(response: str) -> CluePhase:
    """Parse the clue-generation block from the rollout's ``<output>`` region."""
    region = _extract_output_region(response)
    block_match = _CLUE_BLOCK_RE.search(region)
    if block_match is None:
        return CluePhase(tags_present=False)

    body = block_match.group(1)
    clue_match = _CLUE_LINE_RE.search(body)
    selected_match = _SELECTED_LINE_RE.search(body)

    tags_present = clue_match is not None and selected_match is not None
    if not tags_present:
        return CluePhase(tags_present=False)

    # Clue is the first line after ``[Clue]:`` up to newline or ``[Selected-Targets]``.
    clue_raw = clue_match.group(1)
    # Stop at the next tag on the same line if present.
    clue_raw = re.split(r"\[Selected-Targets\]", clue_raw, maxsplit=1)[0]
    clue_raw = clue_raw.split("\n", 1)[0].strip()
    # Trim trailing punctuation/markup the model may add.
    clue_token = clue_raw.strip().strip(".,;:\"'()[]").strip()

    has_hyphen = "-" in clue_token
    is_single_word = bool(clue_token) and not re.search(r"\s", clue_token)

    selected_tokens = _split_word_list(selected_match.group(1))

    return CluePhase(
        tags_present=True,
        clue=clue_token,
        clue_is_single_word=is_single_word,
        clue_has_hyphen=has_hyphen,
        selected_targets=selected_tokens,
        selected_nonempty=len(selected_tokens) > 0,
    )


def parse_guesses(judge_text: str) -> GuessPhase:
    """Parse the guess-generation block from the judge response."""
    if not judge_text:
        return GuessPhase(tags_present=False)

    region = _extract_output_region(judge_text)
    block_match = _GUESS_BLOCK_RE.search(region)
    if block_match is None:
        # Some judge modes may skip the CoT wrapper; try the raw text.
        block_match = _GUESS_BLOCK_RE.search(judge_text)
        if block_match is None:
            return GuessPhase(tags_present=False)

    body = block_match.group(1)
    line_match = _GUESSES_LINE_RE.search(body)
    if line_match is None:
        return GuessPhase(tags_present=True)

    tokens = _split_word_list(line_match.group(1))
    return GuessPhase(
        tags_present=True,
        guesses=tokens,
        guesses_nonempty=len(tokens) > 0,
    )
