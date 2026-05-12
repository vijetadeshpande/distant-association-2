"""Top-level ``compute_score`` for Codenames RLVR training.

This module is the entry point that VeRL's ``custom_reward_function.path``
config key points at.  The function is declared ``async`` so the
experimental reward-loop in VeRL 0.7.1
(``verl/experimental/reward_loop/reward_loop.py``) fans out judge calls
across the rollout batch via ``asyncio.gather`` — that gives us
Phase-2 throughput without writing a custom RewardManager.

Two trainee tasks share this entry point
----------------------------------------
* **Clue task** (``extra_info["task"]`` contains ``"clue"``): trainee is
  the Spymaster; format check on the clue block + task reward via
  external judge OR GloVe-cosine. See ``judge_client.py`` and
  ``cosine_reward.py``.
* **Guess task** (``extra_info["task"]`` contains ``"guess"``): trainee
  is the Operative; format check on the guess block + rule-based task
  reward computed directly on the parsed guesses (no judge needed).

The top-level ``compute_score`` is a small dispatcher that routes on
``extra_info["task"]`` and delegates to ``_compute_score_clue`` or
``_compute_score_guess``.  Both helpers emit the *same* return-dict
key set so VeRL's per-step ``np.array(...)`` across the batch stays
homogeneous regardless of the per-row task.

Sub-rewards exposed to wandb
----------------------------
Every float key below lands in wandb as ``reward/<key>/{mean,max,min}``
via a small block added to ``verl/trainer/ppo/metric_utils.py``. Text
fields (``clue``, ``selected_targets``, ``judge_guesses``) are logged
as a per-step ``reward/samples`` wandb.Table. That patch lives in VeRL
because compute_data_metrics runs on the TaskRunner Ray actor, which
does not import this reward file (only the RewardLoopWorker Ray actors
do).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from custom_reward_functions.cosine_reward import (
    cosine_reward,
    is_cosine_mode,
    zero_cosine_reward,
)
from custom_reward_functions.format_reward import (
    clue_format_scores,
    guess_format_scores,
    is_clue_format_ok,
    is_guess_format_ok,
    thinking_format_scores,
)
from custom_reward_functions.judge_client import judge_guess, make_session, JUDGE_CONCURRENCY
from custom_reward_functions.parsers import parse_clue, parse_guesses, parse_thinking
from custom_reward_functions.task_reward import task_reward, zero_task_reward

logger = logging.getLogger(__name__)


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


# Union of every format key produced by either task path. The return
# dict must contain all of these so VeRL sees a homogeneous schema.
_ALL_FORMAT_KEYS: tuple[str, ...] = (
    "thinking_all_present",
    "thinking_in_order",
    "clue_tags_present",
    "clue_single_word",
    "clue_no_hyphen",
    "clue_selected_nonempty",
    "clue_selected_subset",
    "clue_no_morph_variant",
    "guess_tags_present",
    "guess_nonempty",
    "guess_all_in_board",
    "guess_count_ok",
)


def _aggregate(fmt_scores: dict, task_scores: dict) -> float:
    """Arithmetic mean over every format sub-reward and every task
    sub-reward except the composite ``task`` key (kept separately for
    logging).

    Only keys actually present in ``fmt_scores`` are aggregated, so
    a path that scored clue-format keys (e.g. the clue task) ignores
    guess-format keys, and vice versa.
    """
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


def _build_return(
    *,
    fmt: dict,
    task: dict,
    parse_fail: int,
    format_fail: int,
    judge_fail: int,
    judge_guesses: str,
    guess_diag: dict[str, Any] | None,
    cosine_diag: dict[str, Any] | None,
    clue: str,
    selected_targets: list[str],
) -> dict[str, Any]:
    """Build a return dict whose key set is identical in every code path.

    ``fmt`` carries only the format keys the per-task path actually
    scored — the missing keys are padded with 0.0 (passing) here.
    ``guess_diag`` is an OPTIONAL judge-side diagnostic dict used by
    the clue task in judge mode; when present it overrides the
    ``guess_*`` slots so they reflect the judge's output rather than
    the trainee's.  In the guess task, ``guess_diag`` is None — the
    ``guess_*`` slots already hold the trainee's format scores via
    ``fmt``.
    """
    padded_fmt = {k: 0.0 for k in _ALL_FORMAT_KEYS}
    padded_fmt.update(fmt)
    if guess_diag is not None:
        padded_fmt.update(guess_diag)

    cosine = cosine_diag if cosine_diag is not None else {"cosine_sim": 0.0, "oov": 0}

    out: dict[str, Any] = {
        "score": _aggregate(fmt, task),
        **padded_fmt,
        **task,
        **cosine,
        "parse_fail": int(parse_fail),
        "format_fail": int(format_fail),
        "judge_fail": int(judge_fail),
        "judge_guesses": judge_guesses,
        "clue": clue,
        "selected_targets": ",".join(selected_targets),
    }
    return out


# ---------------------------------------------------------------------------
# Clue task
# ---------------------------------------------------------------------------

async def _compute_score_clue(solution_str: str, extra_info: dict) -> dict:
    """Spymaster scoring path.

    Pipeline: parse rollout → score clue+thinking format → short-circuit
    on format fail → step-4 task reward via judge HTTP (or GloVe-cosine
    if ``JUDGE_NAME`` is unset).
    """
    target_set = _as_list(extra_info.get("target_words", []))
    non_target_set = _as_list(extra_info.get("non_target_words", []))
    target_clue = str(extra_info.get("clue", "") or "")

    cosine_mode = is_cosine_mode()

    # -- 1. parse rollout --------------------------------------------------
    pt = parse_thinking(solution_str)
    pc = parse_clue(solution_str)

    # -- 2. format scores --------------------------------------------------
    fmt = {}
    fmt.update(thinking_format_scores(pt))
    fmt.update(clue_format_scores(pc, target_set, non_target_set))

    format_ok = is_clue_format_ok(fmt)
    parse_fail = int(not pc.tags_present)

    # -- 3. short-circuit on clue-format failure --------------------------
    if not format_ok:
        if cosine_mode:
            task = zero_cosine_reward()
            cosine_diag = {"cosine_sim": task["cosine_sim"], "oov": task["oov"]}
        else:
            task = zero_task_reward()
            cosine_diag = None
        return _build_return(
            fmt=fmt,
            task={k: v for k, v in task.items() if k not in ("cosine_sim", "oov")},
            parse_fail=parse_fail,
            format_fail=1,
            judge_fail=0,
            judge_guesses="",
            guess_diag=None,
            cosine_diag=cosine_diag,
            clue=pc.clue,
            selected_targets=pc.selected_targets,
        )

    # -- 4. task reward ----------------------------------------------------
    if cosine_mode:
        if not target_clue:
            logger.warning(
                "Cosine mode: extra_info['clue'] is missing/empty for this sample; "
                "returning cosine_sim=0."
            )
        task = cosine_reward(pc.clue, target_clue)
        cosine_diag = {"cosine_sim": task["cosine_sim"], "oov": task["oov"]}
        return _build_return(
            fmt=fmt,
            task={k: v for k, v in task.items() if k not in ("cosine_sim", "oov")},
            parse_fail=parse_fail,
            format_fail=0,
            judge_fail=0,
            judge_guesses="",
            guess_diag=None,
            cosine_diag=cosine_diag,
            clue=pc.clue,
            selected_targets=pc.selected_targets,
        )

    # -- 4 (judge mode). judge inference ----------------------------------
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
        return _build_return(
            fmt=fmt,
            task=task,
            parse_fail=parse_fail,
            format_fail=0,
            judge_fail=1,
            judge_guesses="",
            guess_diag=None,
            cosine_diag=None,
            clue=pc.clue,
            selected_targets=pc.selected_targets,
        )

    pg = parse_guesses(judge_text)
    # Judge-side format is logged for diagnostics but does NOT feed into
    # the trainee's reward — the trainee cannot control the judge.
    guess_diag = guess_format_scores(pg, all_words)

    task = task_reward(pg.guesses, target_set, non_target_set)
    return _build_return(
        fmt=fmt,
        task=task,
        parse_fail=parse_fail,
        format_fail=0,
        judge_fail=0,
        judge_guesses=",".join(pg.guesses),
        guess_diag=guess_diag,
        cosine_diag=None,
        clue=pc.clue,
        selected_targets=pc.selected_targets,
    )


# ---------------------------------------------------------------------------
# Guess task
# ---------------------------------------------------------------------------

async def _compute_score_guess(solution_str: str, extra_info: dict) -> dict:
    """Operative scoring path.

    Pipeline: parse rollout → score guess+thinking format → short-circuit
    on format fail → rule-based ``task_reward`` directly on the trainee's
    guesses.  No judge, no cosine fallback — the rules already verify
    the guesses against the known board.
    """
    target_set = _as_list(extra_info.get("target_words", []))
    non_target_set = _as_list(extra_info.get("non_target_words", []))
    all_words = _as_list(extra_info.get("all_words", []))
    if not all_words:
        # Fallback: reconstruct from target + non_target if the row
        # didn't ship an explicit all_words list.
        all_words = list(target_set) + list(non_target_set)

    reference_clue = str(extra_info.get("clue", "") or "")
    max_guesses_raw = extra_info.get("num_max_guesses", extra_info.get("max_guesses"))

    # -- 1. parse rollout --------------------------------------------------
    pt = parse_thinking(solution_str)
    pg = parse_guesses(solution_str)

    # -- 2. format scores --------------------------------------------------
    fmt = {}
    fmt.update(thinking_format_scores(pt))
    fmt.update(guess_format_scores(pg, all_words, max_guesses=max_guesses_raw))

    format_ok = is_guess_format_ok(fmt)
    parse_fail = int(not pg.tags_present)

    # -- 3. short-circuit on guess-format failure -------------------------
    if not format_ok:
        task = zero_task_reward()
        return _build_return(
            fmt=fmt,
            task=task,
            parse_fail=parse_fail,
            format_fail=1,
            judge_fail=0,
            judge_guesses=",".join(pg.guesses),
            guess_diag=None,
            cosine_diag=None,
            clue=reference_clue,
            selected_targets=[],
        )

    # -- 4. task reward (rule-based; no judge) ----------------------------
    task = task_reward(pg.guesses, target_set, non_target_set)
    return _build_return(
        fmt=fmt,
        task=task,
        parse_fail=parse_fail,
        format_fail=0,
        judge_fail=0,
        judge_guesses=",".join(pg.guesses),
        guess_diag=None,
        cosine_diag=None,
        clue=reference_clue,
        selected_targets=[],
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _route_task(extra_info: dict) -> str:
    """Return ``"guess"`` or ``"clue"`` based on ``extra_info["task"]``.

    Matches on substring so both ``codenames_clue_generation`` and a
    bare ``clue`` route correctly. Defaults to clue if the field is
    missing or unrecognized.
    """
    raw = str(extra_info.get("task", "")).lower()
    if "guess" in raw:
        return "guess"
    return "clue"


async def compute_score(data_source, solution_str, ground_truth,
                        extra_info=None, **kwargs) -> dict:
    """VeRL-compatible async reward function.

    Returns a dict with ``score`` (scalar used for optimization) plus
    every sub-reward and diagnostic.  All dict values are copied into
    ``reward_extra_info`` by the reward manager, so they show up
    per-step in the training metrics.

    Routes to the clue or guess scorer based on ``extra_info["task"]``.
    """
    extra_info = dict(extra_info or {})
    task_kind = _route_task(extra_info)
    if task_kind == "guess":
        return await _compute_score_guess(solution_str, extra_info)
    return await _compute_score_clue(solution_str, extra_info)
