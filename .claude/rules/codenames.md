# Simplified Codenames Game Rules

This project uses a simplified version of Codenames. There is no assassin word,
no neutral words, and no standard 5x5 grid. The board is just two word sets.

## Board Structure
- **[Target-Set]**: own team's words (2 to 6 words)
- **[Non-Target-Set]**: opponent's words (2 to 6 words)
- **Total board size**: 4 to 36 words (sizes of both sets vary independently)
- There are NO other card types (no assassin, no neutral, no bystander)

## Model Roles

### Clue-Generator Role
The model receives the board with both sets visible and labeled.
It must produce:
- **[Clue]**: a single word (no hyphens, no spaces, no compound words)
- **[Selected-Targets]**: one or more words from [Target-Set] that the clue covers
- **N** = number of selected targets (this becomes Max-Num-Guesses)

Clue constraints:
- Must cover at least 1 word from [Target-Set]
- Must have clear semantic similarity (meaning, connotation, association) with the selected targets
- Should NOT share similarity with words in [Non-Target-Set], though some unavoidable overlap is expected — treat this as inherent risk, not a hard failure
- Must be exactly one word, no hyphens allowed

### Guess-Generator Role
The model receives a shuffled mix of all words plus the clue.
It must select up to N words from [All-Words] that are most similar to the clue.

Guess constraints:
- **[All-Words]** = shuffled([Target-Set] + [Non-Target-Set]) — the model does NOT know which words are targets
- Guesses must come ONLY from [All-Words]. No invented words allowed.
- The model may guess fewer than N words but never more than N
- The model should consider all forms of similarity (semantic, connotative, associative) between the clue and each word in [All-Words]

## Prompt Formats

### Clue-Generator Prompt
```
Codenames Clue Generation Task: You are given a [Target-Set] and a [Non-Target-Set] of words.
Generate a single-word clue that connects to as many [Target-Set] words as possible while avoiding any association with [Non-Target-Set] words.
"Connection" is intentionally broad: semantic, conceptual, thematic, associative, phonetic, cultural — any defensible link counts.

Objective:
  - Maximize the number of [Target-Set] words your clue connects to (you may select as few as one).
  - Minimize any plausible connection between your clue and ANY [Non-Target-Set] word, under any common interpretation of the clue.
  - When these goals conflict, weigh the trade-off carefully and thoughtfully.

Clue constraints (violating any of these makes the clue invalid):
  - Must be exactly one real English word (no proper nouns, no hyphenated or compound words, no made-up words).
  - Must NOT appear in either set.
  - Must NOT be a morphological variant (inflection, derivation), abbreviation, or direct translation of any word in either set.
    Example: if SWIM is in a set, SWIMMING, SWIMMER, SWAM are all invalid.

Strategy:
  1. Brainstorm several candidate clues.
  2. For each candidate, mentally check it against every word in BOTH sets.
  3. Discard any candidate that has a plausible link to a [Non-Target-Set] word — even under uncommon meanings.
  4. Among the remaining candidates, choose the one that connects to the most [Target-Set] words.

---

Respond using exactly this format:
[CODENAMES-CLUE-START]
[Clue]: <single-word clue>
[Selected-Targets]: <list of target words your clue relates to, e.g., [WORD1, WORD2, ...]>
[CODENAMES-CLUE-END]

---
Here are the word sets:
[Target-Set]: [FILL-IN-TARGET-WORDS]
[Non-Target-Set]: [FILL-IN-NON-TARGET-WORDS]

```

### Guess-Generator Prompt (constructed on-the-fly during rollout)
```
Codenames Guess Generation Task: You are given a single-word [Clue] and a list of [All-Words] on the board.
Your task is to guess which words the clue-giver intended by listing words from [All-Words] by their connection to the [Clue].
"Connection" is intentionally broad: semantic, conceptual, thematic, associative, phonetic, cultural — any defensible link counts.

Objective:
  - Identify the words in [All-Words] that the clue-giver most plausibly intended with [Clue].
  - Return up to [Max-Guesses] guesses, ordered from most confident to least confident.
  - You may return fewer than [Max-Guesses] if you are not confident in additional guesses. 

Strategy:
  1. For each word in [All-Words], assess how strongly it connects to [Clue].
  2. Consider ALL possible interpretations of [Clue] — the clue-giver may be using an uncommon meaning, thematic link, or lateral association.
  3. Rank candidates by connection strength. Include a word only if you believe the clue-giver plausibly chose [Clue] to point to it.
  4. Be especially cautious with lower-ranked guesses — each additional guess carries increasing risk of selecting an unintended word.

---

Respond using exactly this format:
[CODENAMES-GUESS-START]
[Guesses]: [first guess, second guess, ...]
[CODENAMES-GUESS-END]

---
Here are the word sets:
[All-Words]: [FILL-IN-ALL-WORDS]
[Clue]: [FILL-IN-CLUE]
[Max-Guesses]: [FILL-IN-MAX-GUESSES]
```

Where:
- [All-Words] is the shuffled union of both sets [Target-Set] (labels removed)
- [Clue] is extracted from the clue-generator output
- [Max-Num-Guesses] = number of selected targets from the clue-generator output
- The model does NOT see [Target-Set] or [Non-Target-Set] separately