# Overview
In this file, we will discuss the design of our reward function. Primarily, there will two key components, format reward and task reward. Format reward will be for adhering to specific output formats that we require in this task. And the task reward is to check how well the LLM is playing the Codenames game. 

## Format reward

### Thinking format
The exact system prompt for the training prompts is follows, 
```
You are an AI assistant that uses a structured Chain of Thought (CoT) approach to answer queries accurately and concisely.
Follow these steps in order:
1. **Think** - Identify what the problem is asking and outline your initial approach.
2. **Reason** - Work through the problem step by step, keeping each step focused and atomic.
3. **Reflect** - Check your reasoning for errors, gaps, or improvements.
4. **Adjust** - If your reflection identified an issue, explicitly correct it here. If no issues were found, state that clearly.
5. **Output** - Provide your final answer. This must be fully self-contained and readable without any context from the sections above.

---
Use the following format exactly:
---
<thinking>
[What is the problem asking? What is your initial approach?]
</thinking>

<reasoning>
[Step-by-step reasoning. Each step should be atomic and clearly follow from the previous one. Add as many steps as needed.]
</reasoning>

<reflection>
[Review your reasoning. Are there errors, missing cases, or better approaches? Be critical.]
</reflection>

<adjustment>
[If reflection found an issue: restate the corrected reasoning here.
If no issues were found: explicitly state "No adjustments needed."]
</adjustment>

<output>
[Your final, concise, self-contained answer. If uncertain, state your uncertainty clearly.]
</output>

---
Rules:
- All five sections must always appear in order.
- Only use information given in the problem. Flag missing information rather than assuming.
- Do not state conclusions before completing your reasoning.
- If considering multiple approaches, evaluate each against the original question.
- If uncertain after reflection, say so in <output> rather than guessing.
---
```

#### Rules for reward calculation
Hence, the format reward for the thinking part of the response must be based on following rules,
- All five sections (<thinking>, <reasoning>, <reflection>, <adjustment>, <output>) must always appear
- It's great if they appear in order


### Clue generation format
The exact prompt for the clue generation prompt is as follows, 
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
[Selected-Targets]: <list of target words your clue relates to, e.g., [WORD1, WORD2]>
[CODENAMES-CLUE-END]

---
Here are the word sets:
[Target-Set]: [FILL-IN-TARGET-WORDS]
[Non-Target-Set]: [FILL-IN-NON-TARGET-WORDS]
```

Therefore, the final output of the model must follow this format,
```
[CODENAMES-CLUE-START]
[Clue]: <single-word clue>
[Selected-Targets]: <list of target words your clue relates to, e.g., [WORD1, WORD2, ...]>
[CODENAMES-CLUE-END]
```

#### Rules for reward calculation
Hence, the format reward must be based on following things, 
- Are "[CODENAMES-CLUE-START]", "[CODENAMES-CLUE-END]", "[Clue]:", and "[Selected-Targets]:" tags present in the response string?
- The generated clue (i.e., text following "[Clue]:", but before "[Selected-Targets]:") must be a single word, without any usage of hyphen.
- The selected targets (between "[Selected-Targets]:" and "[CODENAMES-CLUE-END]") must be a list of words separated by comma or space, with atleast one wor present.
- Each word in the selected targets must be present in the [Target-Set]

### Guess generation format
The exact prompt for the guess generation task is as follows, 
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

Therefore, the required format for the guess generation is,
```
[CODENAMES-GUESS-START]
[Guesses]: [first guess, second guess, ...]
[CODENAMES-GUESS-END]
```

#### Rules for reward calculation
Hence, the format reward for the guess generation task must be based on following things,
- The "[CODENAMES-GUESS-START]", "[CODENAMES-GUESS-END]", and "[Guesses]:" tags must be present the final output part of the response
- The guesses (i.e. between [Guesses]: and [CODENAMES-GUESS-END]) must be a list of words separated by comma or space, and must have atleast one word

## Task reward
Keep the task reward simple but make sure that LLMs won't hack it or cheat on the task. 

### Rules for reward calculation
- The task reward must be based on three things [Target-Set], [Non-Target-Set], and [Guesses]
- postive reward = correct guesses = set([Guesses]).intersection(set([Target-Set])) / len([Target-Set])
- negative rewrds
    - incorrect guesses = set([Guesses]).intersection(set([Non-Target-Set])) / len([Non-Target-Set])
    - invalid guesses = (set([Guesses]) - set([All-Words])) / len([All-Words])

## Additional desired properties for the reward
- the final reward must have clear bounds
- keep the format bounds from -1 to 0 
- task related reward should ideally vary between -1 to +1


***Note:*** Take above instructins as a guide and not a finalized design.