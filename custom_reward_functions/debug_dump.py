"""Checkpoint-time debug dump for Codenames RLVR runs.

Called from ``ray_trainer.py`` right after a checkpoint is saved and the
just-saved actor weights have been pushed to the rollout engine. Picks
the *last N* rows of the validation split (where N = the rollout
engine's worker count, i.e. the ``pad_dataproto_to_divisor`` size so
the inference batch is full with no padding waste), runs inference,
parses each response, and — on clue-task rows — calls the judge. The
saved JSON therefore shows exactly what the just-saved checkpoint
produced on a whole batch's worth of real val examples.

Output: ``<default_local_dir>/global_step_{N}/debug_samples.json``
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _select_debug_row_indices(val_dataset, count: int) -> list[int]:
    """Return the indices of the last ``count`` rows of ``val_dataset``.

    ``count`` is normally the rollout engine's size_divisor (a.k.a.
    ``agent.num_workers``) so the inference batch is full with no
    padding overhead. Whatever mix of clue / guess tasks happens to sit
    in the tail of the val parquet is what gets dumped.
    """
    n = len(val_dataset)
    take = max(1, min(int(count), n))
    return list(range(n - take, n))


def _generate_responses(trainer, indices: list[int]):
    """Run rollout on the given val-dataset indices.

    Returns ``(prompt_texts, response_texts, extra_infos)`` aligned with
    ``indices``. Uses the same rollout manager validate() uses; vLLM is
    expected to be holding the just-saved actor weights at this point.
    """
    from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
    from verl.utils.dataset.rl_dataset import collate_fn

    rows = [trainer.val_dataset[i] for i in indices]
    batch_dict = collate_fn(rows)
    batch = DataProto.from_single_dict(batch_dict)

    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
    )

    gen_batch = trainer._get_gen_batch(batch)
    gen_batch.meta_info = {
        "eos_token_id": trainer.tokenizer.eos_token_id,
        "pad_token_id": trainer.tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": trainer.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
        "validate": True,
        "global_steps": trainer.global_steps,
    }

    size_divisor = trainer.config.actor_rollout_ref.rollout.agent.num_workers
    padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
    output_padded = trainer.async_rollout_manager.generate_sequences(padded)
    output = unpad_dataproto(output_padded, pad_size=pad_size)

    batch = batch.union(output)

    response_texts = [
        trainer.tokenizer.decode(r, skip_special_tokens=True)
        for r in batch.batch["responses"]
    ]
    prompt_texts = [
        trainer.tokenizer.decode(p, skip_special_tokens=True)
        for p in batch.batch["prompts"]
    ]
    extra_infos = list(batch.non_tensor_batch["extra_info"])
    return prompt_texts, response_texts, extra_infos


def _call_judge_sync(*, clue, all_words, max_guesses, model, backend) -> str:
    """Synchronous wrapper around :func:`judge_guess` for the debug dump."""
    from custom_reward_functions.judge_client import judge_guess, make_session

    async def run():
        sem = asyncio.Semaphore(1)
        async with make_session(backend=backend) as session:
            return await judge_guess(
                session=session,
                sem=sem,
                all_words=list(all_words),
                clue=clue,
                max_guesses=max(1, int(max_guesses)),
                model=model,
                backend=backend,
            )

    try:
        return asyncio.run(run()) or ""
    except Exception as e:  # noqa: BLE001 — debug-only, never fail the run
        logger.exception("[debug-dump] judge call failed")
        return f"[judge call failed: {e!r}]"


def _build_record(*, prompt_text, response_text, extra_info, step, judge_cfg) -> dict[str, Any]:
    """Parse the response; for clue rows also call the judge."""
    from custom_reward_functions.parsers import (
        parse_clue,
        parse_guesses,
        parse_thinking,
    )

    ei = dict(extra_info or {})
    task = str(ei.get("task", "")).lower()
    target_words = list(ei.get("target_words", []) or [])
    non_target_words = list(ei.get("non_target_words", []) or [])

    pt = parse_thinking(response_text)
    rec: dict[str, Any] = {
        "step": step,
        "task": ei.get("task", ""),
        "input": {
            "target_words": target_words,
            "non_target_words": non_target_words,
            "clue": ei.get("clue", ""),
        },
        "prompt": prompt_text,
        "raw_response": response_text,
        "thinking_parse": {
            "all_present": pt.all_present,
            "in_order": pt.in_order,
            "missing": list(pt.missing),
        },
        "judge_response": "",
    }

    if "clue" in task:
        pc = parse_clue(response_text)
        rec["parsed"] = {
            "tags_present": pc.tags_present,
            "clue": pc.clue,
            "clue_is_single_word": pc.clue_is_single_word,
            "clue_has_hyphen": pc.clue_has_hyphen,
            "selected_targets": list(pc.selected_targets),
        }
        if judge_cfg.get("model") and pc.tags_present and pc.clue:
            rec["judge_response"] = _call_judge_sync(
                clue=pc.clue,
                all_words=target_words + non_target_words,
                max_guesses=len(pc.selected_targets) or 1,
                model=judge_cfg["model"],
                backend=judge_cfg.get("backend", "openrouter"),
            )
    else:
        pg = parse_guesses(response_text)
        rec["parsed"] = {
            "tags_present": pg.tags_present,
            "guesses": list(pg.guesses),
            "guesses_nonempty": pg.guesses_nonempty,
        }

    return rec


def dump_checkpoint_debug_samples(trainer) -> None:
    """Generate + parse + judge the last clue/guess val examples; write JSON.

    Safe to call anytime after a checkpoint save — failures are caught
    and logged so training never crashes from a debug-dump error.
    """
    try:
        # Match the size_divisor used by validate()'s rollout call so the
        # batch is full with no padding waste.
        size_divisor = trainer.config.actor_rollout_ref.rollout.agent.num_workers
        indices = _select_debug_row_indices(trainer.val_dataset, size_divisor)
        if not indices:
            logger.warning("[debug-dump] val_dataset is empty; skipping")
            return

        prompt_texts, response_texts, extra_infos = _generate_responses(
            trainer, indices
        )

        rk = trainer.config.reward.get("reward_kwargs", {}) or {}
        judge_cfg = {
            "model": rk.get("judge_model", None),
            "backend": rk.get("judge_backend", "openrouter"),
        }

        records = [
            _build_record(
                prompt_text=p,
                response_text=r,
                extra_info=ei,
                step=trainer.global_steps,
                judge_cfg=judge_cfg,
            )
            for p, r, ei in zip(prompt_texts, response_texts, extra_infos)
        ]

        out_dir = os.path.join(
            trainer.config.trainer.default_local_dir,
            f"global_step_{trainer.global_steps}",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "debug_samples.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)
        print(f"[debug-dump] wrote {len(records)} samples to {out_path}")
    except Exception:  # noqa: BLE001 — never break training on debug-dump failure
        logger.exception("[debug-dump] failed; continuing")
