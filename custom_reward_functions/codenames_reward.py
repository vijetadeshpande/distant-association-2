"""Top-level ``compute_score`` for Codenames RLVR training.

This module is the entry point that VeRL's ``custom_reward_function.path``
config key points at.  The function is declared ``async`` so the
experimental reward-loop in VeRL 0.7.1
(``verl/experimental/reward_loop/reward_loop.py``) fans out judge calls
across the rollout batch via ``asyncio.gather`` — that gives us
Phase-2 throughput without writing a custom RewardManager.

Pipeline per sample
-------------------
1. Parse the trainee rollout:
     - CoT sections (thinking / reasoning / reflection / adjustment /
       output)
     - ``[CODENAMES-CLUE-START] ... END]`` block → clue + Selected-Targets
2. Score the format (0 on pass, negative on violation).
3. If any clue-format check failed, **short-circuit** — skip the judge
   call, return zero task reward alongside the format penalties.
4. Otherwise, call the external vLLM judge with the Guess-Generator
   prompt, parse the ``[Guesses]`` list, and compute the three task
   sub-rewards.
5. Return a dict with:
     - ``score``: arithmetic mean of every sub-reward (format terms +
       task sub-terms).  This is the scalar VeRL writes into
       ``reward_tensor``.
     - Every sub-reward key (for logging via ``reward_extra_info``).
     - ``parse_fail``, ``format_fail``, ``judge_fail`` diagnostics.

Environment variables
---------------------
See ``judge_client.py`` for the judge HTTP config.  No env vars are
read here directly.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from custom_reward_functions.format_reward import (
    clue_format_scores,
    guess_format_scores,
    is_clue_format_ok,
    thinking_format_scores,
)
from custom_reward_functions.judge_client import judge_guess, make_session, JUDGE_CONCURRENCY
from custom_reward_functions.parsers import parse_clue, parse_guesses, parse_thinking
from custom_reward_functions.task_reward import task_reward, zero_task_reward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Side-effect: wrap VeRL's `compute_data_metrics` so every numeric reward
# sub-key lands in wandb as `reward/<key>/{mean,max,min}`.  Done here (not
# in verl/) so the upstream repo stays untouched.  VeRL imports
# `compute_data_metrics` into `ray_trainer.py`'s namespace, so we patch
# that binding — patching `metric_utils` alone would not reach the call
# site.  Import is inside a try so unit tests that stub out verl don't
# fail on this.
def _install_reward_metrics_hook() -> None:
    try:
        import verl.trainer.ppo.ray_trainer as _rt
    except Exception:  # pragma: no cover
        return
    if getattr(_rt, "_codenames_metrics_hooked", False):
        return
    _orig = _rt.compute_data_metrics

    def _wrapped(batch, use_critic=True):
        metrics = _orig(batch, use_critic=use_critic)
        nt = getattr(batch, "non_tensor_batch", None) or {}
        batch_size = 0
        score_arr = None
        for k, v in nt.items():
            if k.startswith("__"):
                continue
            try:
                arr = np.asarray(v, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if arr.size == 0 or arr.ndim != 1:
                continue
            metrics[f"reward/{k}/mean"] = float(np.mean(arr))
            metrics[f"reward/{k}/max"] = float(np.max(arr))
            metrics[f"reward/{k}/min"] = float(np.min(arr))
            batch_size = max(batch_size, arr.size)
            if k == "score":
                score_arr = arr

        # Per-sample text fields as a wandb.Table (N sample rows per step).
        text_keys = [k for k in ("clue", "selected_targets", "judge_guesses") if k in nt]
        if text_keys and batch_size:
            try:
                import wandb  # type: ignore
            except ImportError:
                wandb = None  # type: ignore
            if wandb is not None:
                cols = ["sample_idx", *text_keys]
                if score_arr is not None:
                    cols.append("score")
                tbl = wandb.Table(columns=cols)
                n_rows = min(8, batch_size)
                for i in range(n_rows):
                    row = [i] + [str(nt[k][i]) for k in text_keys]
                    if score_arr is not None:
                        row.append(float(score_arr[i]))
                    tbl.add_data(*row)
                metrics["reward/samples"] = tbl
        return metrics

    _rt.compute_data_metrics = _wrapped
    _rt._codenames_metrics_hooked = True


_install_reward_metrics_hook()

# Module-level semaphore shared across all concurrent compute_score
# coroutines running under the same event loop.  The reward-loop
# orchestrator schedules one coroutine per rollout, all on the driver's
# single event loop, so a module-level Semaphore is the right scope.
_JUDGE_SEM: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _JUDGE_SEM
    if _JUDGE_SEM is None:
        _JUDGE_SEM = asyncio.Semaphore(JUDGE_CONCURRENCY)
    return _JUDGE_SEM


def _as_list(value: Any) -> list:
    """Tolerate ``list``, ``tuple``, ``np.ndarray`` (from parquet)."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _aggregate(fmt_scores: dict, task_scores: dict) -> float:
    """Arithmetic mean over every format sub-reward and every task
    sub-reward except the composite ``task`` key (kept separately for
    logging)."""
    terms: list[float] = []
    terms.extend(float(v) for v in fmt_scores.values())
    for k, v in task_scores.items():
        if k == "task":
            continue
        # Tasks have opposite signs for positive / negative terms.
        if k == "pos_correct":
            terms.append(float(v))
        else:
            terms.append(-float(v))
    if not terms:
        return 0.0
    return sum(terms) / len(terms)


async def compute_score(data_source, solution_str, ground_truth,
                        extra_info=None, **kwargs) -> dict:
    """VeRL-compatible async reward function.

    Returns a dict with ``score`` (scalar used for optimization) plus
    every sub-reward and diagnostic.  All dict values are copied into
    ``reward_extra_info`` by the reward manager, so they show up
    per-step in the training metrics.
    """
    extra_info = dict(extra_info or {})
    target_set = _as_list(extra_info.get("target_words", []))
    non_target_set = _as_list(extra_info.get("non_target_words", []))

    # -- 1. parse rollout --------------------------------------------------
    pt = parse_thinking(solution_str)
    pc = parse_clue(solution_str)

    # -- 2. format scores --------------------------------------------------
    fmt = {}
    fmt.update(thinking_format_scores(pt))
    fmt.update(clue_format_scores(pc, target_set, non_target_set))

    format_ok = is_clue_format_ok(fmt)
    parse_fail = int(not pc.tags_present)

    # Keys from guess_format_scores — must be present in every return path
    # so VeRL's np.array(...) across the batch sees a homogeneous key set.
    _empty_guess_diag = {
        "guess_tags_present": 0.0,
        "guess_nonempty": 0.0,
        "guess_all_in_board": 0.0,
    }

    # -- 3. short-circuit on clue-format failure --------------------------
    if not format_ok:
        task = zero_task_reward()
        score = _aggregate(fmt, task)
        return {
            "score": score,
            **fmt,
            **task,
            **_empty_guess_diag,
            "parse_fail": parse_fail,
            "format_fail": 1,
            "judge_fail": 0,
            "judge_guesses": "",
            "clue": pc.clue,
            "selected_targets": ",".join(pc.selected_targets),
        }

    # -- 4. judge inference ------------------------------------------------
    all_words = list(target_set) + list(non_target_set)
    sem = _get_sem()
    judge_text: str | None = None
    try:
        async with make_session() as session:
            judge_text = await judge_guess(
                session=session,
                sem=sem,
                all_words=all_words,
                clue=pc.clue,
                max_guesses=len(pc.selected_targets),
            )
    except Exception as e:  # network / protocol errors only; log and fall through
        logger.exception("Judge call crashed: %r", e)
        judge_text = None

    if not judge_text:
        task = zero_task_reward()
        score = _aggregate(fmt, task)
        return {
            "score": score,
            **fmt,
            **task,
            **_empty_guess_diag,
            "parse_fail": parse_fail,
            "format_fail": 0,
            "judge_fail": 1,
            "judge_guesses": "",
            "clue": pc.clue,
            "selected_targets": ",".join(pc.selected_targets),
        }

    pg = parse_guesses(judge_text)
    # Judge-side format is logged for diagnostics but does NOT feed into
    # the trainee's reward — the trainee cannot control the judge.
    guess_diag = guess_format_scores(pg, all_words)

    task = task_reward(pg.guesses, target_set, non_target_set)
    score = _aggregate(fmt, task)

    return {
        "score": score,
        **fmt,
        **task,
        **guess_diag,
        "parse_fail": parse_fail,
        "format_fail": 0,
        "judge_fail": 0,
        "judge_guesses": ",".join(pg.guesses),
        "clue": pc.clue,
        "selected_targets": ",".join(pc.selected_targets),
    }
