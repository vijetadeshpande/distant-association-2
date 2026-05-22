"""Async HTTP client for the Codenames judge (Approach A).

The judge is an external vLLM OpenAI-compatible server (e.g. Qwen3-30B-
A3B-Instruct-2507 AWQ-4bit) reserved on dedicated GPUs outside VeRL's
Ray pool.  See ``.claude/rules/model-based-reward.md`` for the server
launch command and GPU layout.

Design choices
--------------
* **Thinking is off by default.**  The judge should respond directly
  with the guess block (a short pretext is fine, but no CoT scaffold).
  Codenames training relies on the judge being a cheap, fast scorer,
  not a reasoner.  ``judge_guess(..., judge_thinking=True)`` opts into a
  dynamic reasoning budget; reasoning control is wired for the
  OpenRouter backend only — a local vLLM OpenAI server may reject the
  ``reasoning`` field, so that payload is left exactly as before.
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

# ---------------------------------------------------------------------------
# Backend URLs
#   judge_backend="openrouter"  →  _OPENROUTER_URL  (requires OPENROUTER_API_KEY)
#   judge_backend="local"       →  _LOCAL_JUDGE_URL  (local vLLM server on GPUs)
# Model name is passed per-call; no JUDGE_NAME global.
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
_OPENROUTER_URL: str = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
_LOCAL_JUDGE_URL: str = os.environ.get(
    "JUDGE_URL", "http://127.0.0.1:8000/v1/chat/completions"
)

JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "128"))
JUDGE_TIMEOUT_S = float(os.environ.get("JUDGE_TIMEOUT_S", "300"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "1024"))
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

def make_session(backend: str = "openrouter") -> aiohttp.ClientSession:
    """Create an aiohttp session sized for the configured concurrency.

    ``backend="openrouter"`` injects the Authorization + HTTP-Referer
    headers required by the OpenRouter API; raises ``ValueError`` if
    ``OPENROUTER_API_KEY`` is not set.  ``backend="local"`` creates a
    plain session for a local vLLM server.
    """
    headers: dict[str, str] = {}
    if backend == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY env var must be set when judge_backend='openrouter'"
            )
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        headers["HTTP-Referer"] = os.environ.get(
            "OPENROUTER_HTTP_REFERER", "https://github.com/distant-association-2"
        )
    connector = aiohttp.TCPConnector(limit=JUDGE_CONCURRENCY * 2)
    timeout = aiohttp.ClientTimeout(total=JUDGE_TIMEOUT_S)
    return aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers)


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

    # Single user turn, no system prompt. Reasoning (when enabled) is
    # controlled via the `reasoning` field in judge_guess, not here.
    return [{"role": "user", "content": user_content}]


async def judge_guess(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                      all_words, clue: str, max_guesses: int,
                      model: str, backend: str = "openrouter",
                      judge_thinking: bool = False) -> str | None:
    """Call the judge once; return raw completion text or ``None`` on failure.

    ``model`` is the model identifier to send in the payload (e.g.
    ``"openai/gpt-4o"`` for OpenRouter, ``"qwen3-judge"`` for local
    vLLM).  ``backend`` selects the endpoint URL.

    ``judge_thinking`` controls the OpenRouter ``reasoning`` field:
    ``False`` (default) disables reasoning entirely; ``True`` requests a
    dynamic reasoning budget (``thinkingBudget=-1``).  It is a no-op for
    the local backend, whose payload is left exactly as before.
    """
    if not clue or int(max_guesses) < 1:
        return None

    url = _OPENROUTER_URL if backend == "openrouter" else _LOCAL_JUDGE_URL
    payload = {
        "model": model,
        "messages": build_guess_messages(all_words, clue, max_guesses),
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
    }
    # Reasoning control — OpenRouter backend only. A local vLLM OpenAI
    # server may reject an unknown `reasoning` key, so its payload above
    # is left untouched (matching the previous behaviour exactly).
    if backend == "openrouter":
        if judge_thinking:
            # Dynamic budget: thinkingBudget=-1 lets the judge size its
            # own reasoning. JUDGE_MAX_TOKENS must sit comfortably above
            # the max budget (24576 for Gemini 2.5 Flash) so the final
            # answer is never truncated — the launch script raises it to
            # 32768 when JUDGE_THINKING=1.
            payload["reasoning"] = {"max_tokens": -1}
        else:
            # Default: reasoning fully disabled (no thinking).
            payload["reasoning"] = {"enabled": False}

    last_err: Exception | None = None
    backoff = 1.0
    for attempt in range(3):
        try:
            async with sem:
                async with session.post(url, json=payload) as r:
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
