"""Async HTTP client for the Codenames judge (Approach A).

The judge is an external vLLM OpenAI-compatible server (e.g. Qwen3-30B-
A3B-Instruct-2507 AWQ-4bit) reserved on dedicated GPUs outside VeRL's
Ray pool.  See ``.claude/rules/model-based-reward.md`` for the server
launch command and GPU layout.

Design choices
--------------
* **No SYS_PROMPT by default.**  We want the judge to generate as little
  as possible per call.  The Qwen3-Instruct-2507 checkpoint is
  non-thinking by design, so the user turn carrying the Guess-Generator
  instruction is sufficient.  Set ``JUDGE_ENABLE_THINKING=1`` to
  restore the five-section CoT wrapper if stronger reasoning is ever
  needed.
* **Single shared ClientSession per reward batch** — see
  ``make_session`` below.  The VeRL 0.7.1 experimental reward-loop
  already fans out ``run_single`` via ``asyncio.gather``, so the only
  concurrency knob we own is ``JUDGE_CONCURRENCY`` (a semaphore over
  the judge calls).
* **Retries** on connection errors only, 2 attempts with exponential
  backoff.  HTTP 4xx is raised immediately; 5xx is retried.
* **Deterministic shuffle** of ``[All-Words]`` via a seeded RNG so the
  same board + clue produces the same prompt (useful for logs and
  repro), while still obeying the "shuffled union" invariant from
  ``.claude/rules/codenames.md``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Iterable

import aiohttp

logger = logging.getLogger(__name__)

JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1/chat/completions")
JUDGE_NAME = os.environ.get("JUDGE_NAME", "qwen3-judge")
JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "128"))
JUDGE_TIMEOUT_S = float(os.environ.get("JUDGE_TIMEOUT_S", "300"))
JUDGE_ENABLE_THINKING = os.environ.get("JUDGE_ENABLE_THINKING", "0") == "1"
JUDGE_MAX_TOKENS = int(os.environ.get(
    "JUDGE_MAX_TOKENS", "2048" if JUDGE_ENABLE_THINKING else "256"
))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))

# Minimal Guess-Generator prompt — mirrors CODENAMES_GUESS_GEN_INSTRUCTION
# in custom_data_preparation/system_prompts.py.  We keep it inlined so
# the reward function has no dependency on that module.
_GUESS_INSTRUCTION = """\
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
[All-Words]: {all_words}
[Clue]: {clue}
[Max-Guesses]: {max_guesses}
"""

# Imported lazily / kept inlined so we don't force every judge call to
# import the dataset-prep module.  Must match SYS_PROMPT in
# custom_data_preparation/system_prompts.py.
_SYS_PROMPT_THINKING = """\
You are an AI assistant that uses a structured Chain of Thought (CoT) approach to answer queries accurately and concisely.
Follow these steps in order:
1. Think   2. Reason   3. Reflect   4. Adjust   5. Output

Use the following format exactly:
<thinking>...</thinking>
<reasoning>...</reasoning>
<reflection>...</reflection>
<adjustment>...</adjustment>
<output>...</output>
"""


def make_session() -> aiohttp.ClientSession:
    """Create an aiohttp session sized for the configured concurrency."""
    connector = aiohttp.TCPConnector(limit=JUDGE_CONCURRENCY * 2)
    timeout = aiohttp.ClientTimeout(total=JUDGE_TIMEOUT_S)
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


def build_guess_messages(all_words: Iterable[str], clue: str, max_guesses: int,
                         seed: int | None = None) -> list[dict]:
    """Build the chat-completions messages list for a single guess call.

    The ``[All-Words]`` list is shuffled with a seed derived from the
    input so repeated calls for the same board produce the same prompt.
    """
    words = list(all_words)
    if seed is None:
        # Deterministic seed from the board + clue so logs are reproducible.
        seed = hash((tuple(sorted(words)), clue)) & 0xFFFFFFFF
    rng = random.Random(seed)
    shuffled = words[:]
    rng.shuffle(shuffled)

    user_content = _GUESS_INSTRUCTION.format(
        all_words=shuffled, clue=clue, max_guesses=int(max_guesses)
    )

    messages = []
    if JUDGE_ENABLE_THINKING:
        messages.append({"role": "system", "content": _SYS_PROMPT_THINKING})
    messages.append({"role": "user", "content": user_content})
    return messages


async def judge_guess(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                      all_words, clue: str, max_guesses: int) -> str | None:
    """Call the judge once; return raw completion text or ``None`` on failure."""
    if not clue or int(max_guesses) < 1:
        return None

    payload = {
        "model": JUDGE_NAME,
        "messages": build_guess_messages(all_words, clue, max_guesses),
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
    }

    last_err: Exception | None = None
    backoff = 1.0
    for attempt in range(3):
        try:
            async with sem:
                async with session.post(JUDGE_URL, json=payload) as r:
                    if 400 <= r.status < 500:
                        text = await r.text()
                        logger.warning("Judge 4xx: %s %s", r.status, text[:256])
                        return None
                    r.raise_for_status()
                    data = await r.json()
            return data["choices"][0]["message"]["content"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt == 2:
                break
            await asyncio.sleep(backoff)
            backoff *= 2
    logger.error("Judge call failed after retries: %r", last_err)
    return None
