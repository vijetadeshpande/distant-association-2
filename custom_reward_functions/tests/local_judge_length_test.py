#!/usr/bin/env python3
"""Local Qwen3-4B judge — thinking-length measurement on the full val set.

Runs every ``codenames_guess_generation`` row of the v5 validation parquet
(160 rows) through a local vLLM Qwen3-4B server with **thinking ON** and
**temperature 0** (the production judge setting) and measures, per response:

  * thinking-block token length  — native Qwen3 ``<think>...</think>``
  * final-answer token length    — the ``[CODENAMES-GUESS-START]`` block
  * total completion tokens      — from vLLM ``usage``

The resulting output-length distribution sizes the co-location
hyperparameters (``JUDGE_MAX_TOKENS``, judge ``--max-model-len``, judge
KV-cache fraction) for ``scripts/train_codenames_dapo_local_colocated_judge.sh``.

The judge is frozen and only ever runs the guess task, so these stats
stay stable across training — unlike the trainee, whose rollouts grow.

Mirrors the methodology of ``openrouter_judge_thining_test.md`` but on
the local backend, the full val set, and Qwen3-4B.

Reproduce:
  1. vllm serve Qwen/Qwen3-4B --tensor-parallel-size 4 --dtype bfloat16 \
       --max-model-len 32768 --gpu-memory-utilization 0.85 \
       --served-model-name qwen3-judge --host 127.0.0.1 --port 8000
  2. python custom_reward_functions/tests/local_judge_length_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from custom_reward_functions.judge_client import build_guess_messages  # noqa: E402
from custom_reward_functions.parsers import parse_guesses  # noqa: E402
from custom_reward_functions.task_reward import task_reward  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

VAL_PARQUET = REPO_ROOT / "custom_data/training_prompts/version-5/codenames_rlvr_val.parquet"
JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8000/v1/chat/completions")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3-judge")
MAX_TOKENS = int(os.environ.get("TEST_MAX_TOKENS", "16384"))
CONCURRENCY = int(os.environ.get("TEST_CONCURRENCY", "128"))
OUT_JSON = Path(os.environ.get("OUT_JSON", "/tmp/local_judge_length_test_results.json"))

_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


def ntok(text: str) -> int:
    """Token count of ``text`` under the Qwen3-4B tokenizer (no specials)."""
    return len(_tok.encode(text, add_special_tokens=False)) if text else 0


def split_think(content: str) -> tuple[str, str]:
    """Split raw completion into (thinking, answer) on the ``</think>`` tag.

    Robust to either form Qwen3 emits — a leading ``<think>`` in the
    content, or the template having pre-injected it so the content
    starts straight into reasoning.
    """
    if "</think>" in content:
        think, _, answer = content.partition("</think>")
        return think.replace("<think>", "", 1).strip(), answer.strip()
    return "", content.strip()


async def one_call(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                   row: dict) -> dict:
    ei = row["extra_info"]
    all_words = list(ei["all_words"])
    clue = str(ei["clue"])
    max_guesses = int(ei["max_guesses"])
    payload = {
        "model": JUDGE_MODEL,
        "messages": build_guess_messages(all_words, clue, max_guesses),
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    t0 = time.time()
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(JUDGE_URL, json=payload) as r:
                    r.raise_for_status()
                    data = await r.json()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == 2:
                    return {"error": repr(e), "board": ei["total_words_on_board"]}
                await asyncio.sleep(2 * (attempt + 1))
    dt = time.time() - t0

    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    usage = data.get("usage", {})
    think, answer = split_think(content)

    pg = parse_guesses(content)  # production parses the FULL message content
    tr = task_reward(pg.guesses, list(ei["target_words"]), list(ei["non_target_words"]))

    return {
        "board": ei["total_words_on_board"],
        "max_guesses": max_guesses,
        "difficulty": ei["difficulty_rule"],
        "clue": clue,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "think_tokens": ntok(think),
        "answer_tokens": ntok(answer),
        "has_think": bool(think),
        "guesses": list(pg.guesses),
        "tags_present": bool(pg.tags_present),
        "reward": float(tr["task"]),
        "latency_s": round(dt, 2),
    }


def stats(vals: list[float]) -> dict:
    a = np.array(vals, dtype=float)
    return {
        "mean": round(float(a.mean()), 1),
        "p50": round(float(np.percentile(a, 50)), 1),
        "p90": round(float(np.percentile(a, 90)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
        "p99": round(float(np.percentile(a, 99)), 1),
        "max": round(float(a.max()), 1),
        "min": round(float(a.min()), 1),
    }


async def main() -> None:
    rows = [r for r in pq.read_table(VAL_PARQUET).to_pylist()
            if "guess" in r["extra_info"]["task"].lower()]
    print(f"loaded {len(rows)} guess-gen rows; judge={JUDGE_URL} max_tokens={MAX_TOKENS}")

    sem = asyncio.Semaphore(CONCURRENCY)
    conn = aiohttp.TCPConnector(limit=CONCURRENCY * 2)
    timeout = aiohttp.ClientTimeout(total=1200)
    t0 = time.time()
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        results = await asyncio.gather(*(one_call(session, sem, r) for r in rows))
    wall = time.time() - t0

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    print(f"\ncompleted in {wall:.1f}s — {len(ok)} ok, {len(errs)} errors")
    if errs:
        print("errors:", Counter(e["error"] for e in errs))
    if not ok:
        print("no successful calls — is the judge server up?")
        return

    summary = {k: stats([r[k] for r in ok])
               for k in ("think_tokens", "answer_tokens", "completion_tokens",
                         "prompt_tokens", "latency_s")}

    print("\n=== LENGTH STATS (tokens) ===")
    hdr = f"{'metric':<20}{'mean':>9}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>9}"
    print(hdr)
    for k in ("think_tokens", "answer_tokens", "completion_tokens", "prompt_tokens"):
        s = summary[k]
        print(f"{k:<20}{s['mean']:>9}{s['p50']:>9}{s['p90']:>9}"
              f"{s['p95']:>9}{s['p99']:>9}{s['max']:>9}")

    fr = Counter(r["finish_reason"] for r in ok)
    truncated = fr.get("length", 0)
    print(f"\nfinish_reason: {dict(fr)}   (truncated @ {MAX_TOKENS}: {truncated})")
    print(f"has_think: {sum(r['has_think'] for r in ok)}/{len(ok)}")
    print(f"parsed guess block (tags_present): {sum(r['tags_present'] for r in ok)}/{len(ok)}")
    print(f"mean task reward: {np.mean([r['reward'] for r in ok]):.3f}")

    print("\n=== mean think / answer / completion tokens by board size ===")
    by_board: dict[int, list] = defaultdict(list)
    for r in ok:
        by_board[r["board"]].append(r)
    for b in sorted(by_board):
        g = by_board[b]
        print(f"  board={b:>2} (n={len(g):>2}): "
              f"think={np.mean([x['think_tokens'] for x in g]):>7.0f}  "
              f"answer={np.mean([x['answer_tokens'] for x in g]):>6.0f}  "
              f"completion={np.mean([x['completion_tokens'] for x in g]):>7.0f}")

    print("\n=== mean completion tokens by difficulty ===")
    by_diff: dict[str, list] = defaultdict(list)
    for r in ok:
        by_diff[r["difficulty"]].append(r)
    for d in ("simple", "moderate", "advance", "expert"):
        if d in by_diff:
            g = by_diff[d]
            print(f"  {d:<9} (n={len(g):>2}): "
                  f"completion={np.mean([x['completion_tokens'] for x in g]):>7.0f}")

    OUT_JSON.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nraw results -> {OUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
