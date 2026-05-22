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


def _as_bool(value: Any) -> bool:
    """Coerce a config value (bool, int, or string) to ``bool``.

    Hydra passes ``judge_thinking=true/false`` as a real boolean, but a
    stray string like ``"false"`` would otherwise be truthy — so handle
    the string case explicitly.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


# Union of every format key produced by any scenario. The return dict
# must contain all of these so VeRL sees a homogeneous schema. The
# ``guess_*`` block is the trainee's (scenario a), while
# ``judge_guess_*`` is the judge's diagnostic block in the clue task
# judge mode (scenario b) — kept distinct so wandb doesn't blend them.
_ALL_FORMAT_KEYS: tuple[str, ...] = (
    # thinking (both paths)
    "thinking_all_present",
    "thinking_in_order",
    # clue format (trainee, clue task)
    "clue_tags_present",
    "clue_single_word",
    "clue_no_hyphen",
    "clue_selected_nonempty",
    "clue_selected_subset",
    "clue_no_morph_variant",
    # guess format (trainee, guess task)
    "guess_tags_present",
    "guess_nonempty",
    "guess_all_in_board",
    "guess_count_ok",
    # guess format (judge, clue task in judge mode — diagnostic only)
    "judge_guess_tags_present",
    "judge_guess_nonempty",
    "judge_guess_all_in_board",
    "judge_guess_count_ok",
)


# Union of every task-reward key produced by any scenario. Padded with
# 0.0 in _build_return so wandb keeps trainee-side (``pos_correct`` etc.)
# and judge-side (``judge_pos_correct`` etc.) histograms separate.
_ALL_TASK_KEYS: tuple[str, ...] = (
    # trainee guess task (scenario a)
    "pos_correct",
    "neg_nontarget",
    "neg_invalid",
    "task",
    # judge guesses scored under the clue task (scenario b)
    "judge_pos_correct",
    "judge_neg_nontarget",
    "judge_neg_invalid",
    "judge_task",
)


# Format keys computed and logged to wandb but excluded from the
# format gate in _aggregate. They still flow into the return dict via
# _build_return → padded_fmt, so wandb keeps logging them per-step.
_GATE_EXCLUDED_FMT_KEYS: frozenset[str] = frozenset({
    "thinking_all_present",
    "thinking_in_order",
})


def _aggregate(fmt_scores: dict, task_scalar: float) -> float:
    """Hard-gate format, then pass through the task scalar.

    If any non-excluded format sub-reward is negative, return its
    minimum (always -1.0 since format scores are drawn from
    ``{-1.0, 0.0}``). Otherwise return ``task_scalar`` untouched —
    cosine_sim in cosine mode, ``pos_correct - neg_nontarget -
    neg_invalid`` in judge/guess mode. No averaging.

    Thinking-format keys are excluded from the gate but still logged.
    """
    fmt_values = [
        float(v) for k, v in fmt_scores.items()
        if k not in _GATE_EXCLUDED_FMT_KEYS
    ]
    if fmt_values and any(v < 0.0 for v in fmt_values):
        return min(fmt_values)
    return float(task_scalar)


def _rebadge_judge_diag(diag: dict[str, Any]) -> dict[str, Any]:
    """Rename guess-format diagnostic keys to the ``judge_`` namespace.

    The clue task in judge mode parses the JUDGE's guess block to
    diagnose how well-formed the judge's response was — these are not
    trainee-controllable and must not collide with the trainee's own
    ``guess_*`` keys (scenario a).
    """
    return {f"judge_{k}": v for k, v in diag.items()}


def _rebadge_judge_task(task: dict[str, Any]) -> dict[str, Any]:
    """Rename the task-reward keys to the ``judge_`` namespace.

    In the clue task judge mode, ``task_reward(...)`` scores the
    JUDGE's guesses against the trainee's board. We keep these in a
    distinct key namespace so wandb shows judge-side and trainee-side
    distributions separately.
    """
    return {
        "judge_pos_correct":   task["pos_correct"],
        "judge_neg_nontarget": task["neg_nontarget"],
        "judge_neg_invalid":   task["neg_invalid"],
        "judge_task":          task["task"],
    }


def _zero_judge_task() -> dict[str, Any]:
    """Zero-filled judge-side task dict (used on judge-fail / short-circuit)."""
    return {
        "judge_pos_correct": 0.0,
        "judge_neg_nontarget": 0.0,
        "judge_neg_invalid": 0.0,
        "judge_task": 0.0,
    }


def _format_score_of(fmt: dict) -> float:
    """Return the gate-relevant format score: ``min(fmt)`` if any failed, else 0.

    Mirrors the gate logic in :func:`_aggregate` so wandb tables can
    surface the format penalty independently of the final ``score``.
    """
    vals = [float(v) for k, v in fmt.items() if k not in _GATE_EXCLUDED_FMT_KEYS]
    return min(vals) if vals and any(v < 0.0 for v in vals) else 0.0


def _build_return(
    *,
    fmt: dict,
    task: dict,
    task_scalar: float,
    parse_fail: int,
    format_fail: int,
    judge_fail: int,
    judge_guesses: str,
    guess_diag: dict[str, Any] | None,
    cosine_diag: dict[str, Any] | None,
    clue: str,
    selected_targets: list[str],
    # ---- Per-row context surfaced for wandb tables (no effect on score) ----
    task_kind: str = "",
    target_words: list[str] | None = None,
    non_target_words: list[str] | None = None,
    reference_clue: str = "",
    max_num_guesses: int = 0,
    trainee_guesses: str = "",
) -> dict[str, Any]:
    """Build a return dict whose key set is identical in every code path.

    ``fmt`` carries only the format keys the per-task path actually
    scored; ``task`` carries only the task-reward keys for the active
    scenario. Both are zero-padded against ``_ALL_FORMAT_KEYS`` and
    ``_ALL_TASK_KEYS`` so wandb sees a stable schema and never blends
    trainee-side (scenario a) with judge-side (scenario b) values.

    ``guess_diag`` is the judge-side guess-format diagnostic in clue
    task judge mode (already rebadged into the ``judge_guess_*``
    namespace by the caller); it's merged into padded_fmt and only
    overrides ``judge_guess_*`` slots, never trainee ``guess_*`` ones.

    The trailing ``task_kind`` … ``trainee_guesses`` fields are pure
    pass-through context for the per-task wandb tables built in
    ``verl/trainer/ppo/metric_utils.py``; they do not affect the
    optimization scalar.
    """
    padded_fmt = {k: 0.0 for k in _ALL_FORMAT_KEYS}
    padded_fmt.update(fmt)
    if guess_diag is not None:
        padded_fmt.update(guess_diag)

    padded_task = {k: 0.0 for k in _ALL_TASK_KEYS}
    padded_task.update(task)

    cosine = cosine_diag if cosine_diag is not None else {"cosine_sim": 0.0, "oov": 0}

    out: dict[str, Any] = {
        "score": _aggregate(fmt, task_scalar),
        **padded_fmt,
        **padded_task,
        **cosine,
        "parse_fail": int(parse_fail),
        "format_fail": int(format_fail),
        "judge_fail": int(judge_fail),
        "judge_guesses": judge_guesses,
        "clue": clue,
        "selected_targets": ",".join(selected_targets),
        # ---- Table context ----
        "task_kind": task_kind,
        "target_words_str": ",".join(target_words or []),
        "non_target_words_str": ",".join(non_target_words or []),
        "reference_clue": reference_clue,
        "max_num_guesses": int(max_num_guesses),
        "trainee_guesses": trainee_guesses,
        "format_score": _format_score_of(fmt),
        "task_score": float(task_scalar),
    }
    return out


# ---------------------------------------------------------------------------
# Clue task
# ---------------------------------------------------------------------------

async def _compute_score_clue(solution_str: str, extra_info: dict,
                              judge_model: str | None = None,
                              judge_backend: str = "openrouter",
                              judge_thinking: bool = False) -> dict:
    """Spymaster scoring path.

    Pipeline: parse rollout → score clue+thinking format → short-circuit
    on format fail → step-4 task reward via judge HTTP (openrouter or
    local vLLM) or GloVe-cosine when ``judge_model`` is ``None``.

    ``judge_model=None``               → cosine-similarity reward
    ``judge_model=<name>``, backend="openrouter" → OpenRouter API
    ``judge_model=<name>``, backend="local"      → local vLLM server
    """
    target_set = _as_list(extra_info.get("target_words", []))
    non_target_set = _as_list(extra_info.get("non_target_words", []))
    target_clue = str(extra_info.get("clue", "") or "")

    cosine_mode = judge_model is None

    # -- 1. parse rollout --------------------------------------------------
    pt = parse_thinking(solution_str)
    pc = parse_clue(solution_str)

    # Context fields surfaced for the per-task wandb table; pure pass-through.
    ctx = dict(
        task_kind="clue",
        target_words=target_set,
        non_target_words=non_target_set,
        reference_clue=target_clue,
        max_num_guesses=len(pc.selected_targets),
        trainee_guesses="",
    )

    # -- 2. format scores --------------------------------------------------
    fmt = {}
    fmt.update(thinking_format_scores(pt))
    fmt.update(clue_format_scores(pc, target_set, non_target_set))

    format_ok = is_clue_format_ok(fmt)
    parse_fail = int(not pc.tags_present)

    # -- 3. short-circuit on clue-format failure --------------------------
    # Judge inference is gated here: if the clue can't be parsed/extracted
    # cleanly (missing tags, multi-word clue, off-board target, morphology
    # variant, etc.) we return before the judge call. The judge is invoked
    # ONLY when the trainee's clue passed every format check above.
    if not format_ok:
        if cosine_mode:
            # Cosine mode: only cosine_sim/oov are meaningful; trainee
            # and judge task slots stay zero-padded by _build_return.
            task: dict = {}
            cosine_diag = {"cosine_sim": 0.0, "oov": 0}
        else:
            # Judge mode: zero the judge-side task slots explicitly so
            # they record a sample of the format-failed batch rather
            # than being indistinguishable from "scenario not active".
            task = _zero_judge_task()
            cosine_diag = None
        return _build_return(
            fmt=fmt,
            task=task,
            task_scalar=0.0,  # ignored by aggregator — fmt gate fires
            parse_fail=parse_fail,
            format_fail=1,
            judge_fail=0,
            judge_guesses="",
            guess_diag=None,
            cosine_diag=cosine_diag,
            clue=pc.clue,
            selected_targets=pc.selected_targets,
            **ctx,
        )

    # -- 4. task reward ----------------------------------------------------
    if cosine_mode:
        if not target_clue:
            logger.warning(
                "Cosine mode: extra_info['clue'] is missing/empty for this sample; "
                "returning cosine_sim=0."
            )
        cos = cosine_reward(pc.clue, target_clue)
        cosine_diag = {"cosine_sim": cos["cosine_sim"], "oov": cos["oov"]}
        # cosine mode publishes only cosine_sim/oov as its task signal;
        # the trainee/judge task slots stay zero-padded.
        return _build_return(
            fmt=fmt,
            task={},
            task_scalar=cos["cosine_sim"],
            parse_fail=parse_fail,
            format_fail=0,
            judge_fail=0,
            judge_guesses="",
            guess_diag=None,
            cosine_diag=cosine_diag,
            clue=pc.clue,
            selected_targets=pc.selected_targets,
            **ctx,
        )

    # -- 4 (judge mode). judge inference ----------------------------------
    all_words = list(target_set) + list(non_target_set)
    sem = _get_sem()
    judge_text: str | None = None
    try:
        async with make_session(backend=judge_backend) as session:
            judge_text = await judge_guess(
                session=session,
                sem=sem,
                all_words=all_words,
                clue=pc.clue,
                max_guesses=len(pc.selected_targets),
                model=judge_model,
                backend=judge_backend,
                judge_thinking=judge_thinking,
            )
    except Exception as e:  # network / protocol errors only; log and fall through
        logger.exception("Judge call crashed: %r", e)
        judge_text = None

    if not judge_text:
        return _build_return(
            fmt=fmt,
            task=_zero_judge_task(),
            task_scalar=0.0,
            parse_fail=parse_fail,
            format_fail=0,
            judge_fail=1,
            judge_guesses="",
            guess_diag=None,
            cosine_diag=None,
            clue=pc.clue,
            selected_targets=pc.selected_targets,
            **ctx,
        )

    pg = parse_guesses(judge_text)
    # Judge-side format & task scores are rebadged into the ``judge_``
    # namespace so wandb keeps them separate from the trainee's own
    # guess-task metrics (scenario a). The trainee can't control the
    # judge, so these diagnostics never feed the format gate either.
    judge_guess_diag = _rebadge_judge_diag(guess_format_scores(pg, all_words))

    raw_task = task_reward(pg.guesses, target_set, non_target_set)
    return _build_return(
        fmt=fmt,
        task=_rebadge_judge_task(raw_task),
        task_scalar=raw_task["task"],
        parse_fail=parse_fail,
        format_fail=0,
        judge_fail=0,
        judge_guesses=",".join(pg.guesses),
        guess_diag=judge_guess_diag,
        cosine_diag=None,
        clue=pc.clue,
        selected_targets=pc.selected_targets,
        **ctx,
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
    try:
        max_guesses_int = int(max_guesses_raw) if max_guesses_raw is not None else 0
    except (TypeError, ValueError):
        max_guesses_int = 0

    # -- 1. parse rollout --------------------------------------------------
    pt = parse_thinking(solution_str)
    pg = parse_guesses(solution_str)

    # Context fields surfaced for the per-task wandb table; pure pass-through.
    ctx = dict(
        task_kind="guess",
        target_words=target_set,
        non_target_words=non_target_set,
        reference_clue=reference_clue,
        max_num_guesses=max_guesses_int,
        trainee_guesses=",".join(pg.guesses),
    )

    # -- 2. format scores --------------------------------------------------
    fmt = {}
    fmt.update(thinking_format_scores(pt))
    fmt.update(guess_format_scores(pg, all_words, max_guesses=max_guesses_raw))

    format_ok = is_guess_format_ok(fmt)
    parse_fail = int(not pg.tags_present)

    # -- 3. short-circuit on guess-format failure -------------------------
    if not format_ok:
        return _build_return(
            fmt=fmt,
            task=zero_task_reward(),
            task_scalar=0.0,  # ignored by aggregator — fmt gate fires
            parse_fail=parse_fail,
            format_fail=1,
            judge_fail=0,
            judge_guesses=",".join(pg.guesses),
            guess_diag=None,
            cosine_diag=None,
            clue=reference_clue,
            selected_targets=[],
            **ctx,
        )

    # -- 4. task reward (rule-based; no judge) ----------------------------
    task = task_reward(pg.guesses, target_set, non_target_set)
    return _build_return(
        fmt=fmt,
        task=task,
        task_scalar=task["task"],
        parse_fail=parse_fail,
        format_fail=0,
        judge_fail=0,
        judge_guesses=",".join(pg.guesses),
        guess_diag=None,
        cosine_diag=None,
        clue=reference_clue,
        selected_targets=[],
        **ctx,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _route_task(extra_info: dict) -> str:
    """Return ``"guess"`` or ``"clue"`` based on ``extra_info["task"]``.

    Judge inference is reserved for the clue task ONLY, so routing is
    deliberately strict: an unknown or missing ``task`` field falls
    through to the guess path (rule-based, no judge call) rather than
    silently billing the judge on something that isn't a clue task.
    """
    raw = str(extra_info.get("task", "")).lower()
    if "clue" in raw:
        return "clue"
    if "guess" in raw:
        return "guess"
    logger.warning(
        "compute_score: unrecognized task=%r; defaulting to guess path "
        "(no judge call) to avoid spurious judge inference.",
        extra_info.get("task"),
    )
    return "guess"


async def compute_score(data_source, solution_str, ground_truth,
                        extra_info=None, judge_model: str | None = None,
                        judge_backend: str = "openrouter",
                        judge_thinking: bool = False, **kwargs) -> dict:
    """VeRL-compatible async reward function.

    Returns a dict with ``score`` (scalar used for optimization) plus
    every sub-reward and diagnostic.  All dict values are copied into
    ``reward_extra_info`` by the reward manager, so they show up
    per-step in the training metrics.

    Pass via ``custom_reward_function.reward_kwargs`` in the VeRL config:

      ``judge_model=null``                     → cosine-similarity reward
      ``judge_model="openai/gpt-4o"``          → OpenRouter (default backend)
      ``judge_model="qwen3-judge"``
      ``judge_backend="local"``                → local vLLM server on GPUs
      ``judge_thinking=False``                 → judge reasoning disabled (default)
      ``judge_thinking=True``                  → dynamic reasoning budget (OpenRouter)

    Routes to the clue or guess scorer based on ``extra_info["task"]``.
    Guess task never uses the judge; judge args are clue-task-only.
    """
    extra_info = dict(extra_info or {})
    task_kind = _route_task(extra_info)
    if task_kind == "guess":
        return await _compute_score_guess(solution_str, extra_info)
    return await _compute_score_clue(solution_str, extra_info,
                                     judge_model=judge_model,
                                     judge_backend=judge_backend,
                                     judge_thinking=_as_bool(judge_thinking))
