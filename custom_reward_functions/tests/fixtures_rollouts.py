"""Hand-built rollout strings for parser / reward tests."""

GOOD_ROLLOUT = """\
<thinking>
Identify shared concepts.
</thinking>
<reasoning>
North America and Africa are both continents.
</reasoning>
<reflection>
Continent is clean — no ties to thanatophobia/pyrophobia.
</reflection>
<adjustment>
No adjustments needed.
</adjustment>
<output>
[CODENAMES-CLUE-START]
[Clue]: CONTINENT
[Selected-Targets]: [NORTH AMERICA, AFRICA]
[CODENAMES-CLUE-END]
</output>
"""

# Out of order: <reasoning> before <thinking>.
OUT_OF_ORDER_ROLLOUT = """\
<reasoning>foo</reasoning>
<thinking>bar</thinking>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-CLUE-START]
[Clue]: LAND
[Selected-Targets]: [NORTH AMERICA]
[CODENAMES-CLUE-END]
</output>
"""

# Missing <reflection>.
MISSING_SECTION_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<adjustment>x</adjustment>
<output>
[CODENAMES-CLUE-START]
[Clue]: LAND
[Selected-Targets]: [NORTH AMERICA]
[CODENAMES-CLUE-END]
</output>
"""

HYPHENATED_CLUE_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-CLUE-START]
[Clue]: LAND-MASS
[Selected-Targets]: [NORTH AMERICA, AFRICA]
[CODENAMES-CLUE-END]
</output>
"""

# Selected target not in Target-Set.
OUT_OF_SET_TARGET_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-CLUE-START]
[Clue]: CONTINENT
[Selected-Targets]: [NORTH AMERICA, THANATOPHOBIA]
[CODENAMES-CLUE-END]
</output>
"""

# Clue is a morphological variant of a Target-Set word (AFRICA -> AFRICAN).
MORPH_VARIANT_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-CLUE-START]
[Clue]: AFRICAN
[Selected-Targets]: [AFRICA]
[CODENAMES-CLUE-END]
</output>
"""

# Missing the clue block entirely.
NO_CLUE_BLOCK_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
I cannot find a good clue.
</output>
"""

JUDGE_OUTPUT_CLEAN = """\
<output>
[CODENAMES-GUESS-START]
[Guesses]: [NORTH AMERICA, AFRICA]
[CODENAMES-GUESS-END]
</output>
"""

JUDGE_OUTPUT_NO_THINKING = """\
[CODENAMES-GUESS-START]
[Guesses]: [NORTH AMERICA, AFRICA]
[CODENAMES-GUESS-END]
"""

JUDGE_OUTPUT_MIXED = """\
[CODENAMES-GUESS-START]
[Guesses]: [NORTH AMERICA, PYROPHOBIA]
[CODENAMES-GUESS-END]
"""

TARGETS = ["north america", "africa"]
NON_TARGETS = ["thanatophobia", "pyrophobia"]
ALL_WORDS = ["north america", "thanatophobia", "africa", "pyrophobia"]


# ---------------------------------------------------------------------------
# Trainee-side guess-task rollouts.  Same 5-section CoT shell as the
# clue rollouts; the <output> block carries a [CODENAMES-GUESS-...] block
# (NOT the [CODENAMES-CLUE-...] block).
# ---------------------------------------------------------------------------

GOOD_GUESS_ROLLOUT = """\
<thinking>
The clue is "continent". Look for matching words.
</thinking>
<reasoning>
North America and Africa are both continents.
</reasoning>
<reflection>
Pyrophobia and thanatophobia are fears, not continents — exclude them.
</reflection>
<adjustment>
No adjustments needed.
</adjustment>
<output>
[CODENAMES-GUESS-START]
[Guesses]: [NORTH AMERICA, AFRICA]
[CODENAMES-GUESS-END]
</output>
"""

# Same as GOOD_GUESS_ROLLOUT but only one guess (still <= max_guesses).
# Single-word target (AFRICA) avoids the parser's whitespace-fallback
# split that would otherwise turn "NORTH AMERICA" into two tokens.
PARTIAL_GUESS_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-GUESS-START]
[Guesses]: [AFRICA]
[CODENAMES-GUESS-END]
</output>
"""

# Guess block missing entirely.
NO_GUESS_BLOCK_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
I am not sure which words to guess.
</output>
"""

# Empty [Guesses]: list.
EMPTY_GUESS_LIST_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-GUESS-START]
[Guesses]: []
[CODENAMES-GUESS-END]
</output>
"""

# Includes a word that isn't on the board.
OFF_BOARD_GUESS_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-GUESS-START]
[Guesses]: [NORTH AMERICA, ATLANTIS]
[CODENAMES-GUESS-END]
</output>
"""

# Returns more guesses than max_guesses (used with max_guesses=2 -> 3 guesses).
OVER_LIMIT_GUESS_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-GUESS-START]
[Guesses]: [NORTH AMERICA, AFRICA, PYROPHOBIA]
[CODENAMES-GUESS-END]
</output>
"""

# All guesses land on the non-target set.
ALL_WRONG_GUESS_ROLLOUT = """\
<thinking>x</thinking>
<reasoning>x</reasoning>
<reflection>x</reflection>
<adjustment>x</adjustment>
<output>
[CODENAMES-GUESS-START]
[Guesses]: [PYROPHOBIA, THANATOPHOBIA]
[CODENAMES-GUESS-END]
</output>
"""
