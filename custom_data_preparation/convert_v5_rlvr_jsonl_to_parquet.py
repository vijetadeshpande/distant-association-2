"""Convert the version-5 Codenames RLVR JSONL into a single VeRL-ready parquet.

Input
-----
``custom_data/version-5/codenames_rlvr.jsonl`` — 4500 mixed rows whose
``extra_info["task"]`` is either ``codenames_clue_generation`` or
``codenames_guess_generation``. Each row already carries every key
``custom_reward_functions/codenames_reward.py`` reads from
``extra_info`` (``target_words``, ``non_target_words``, ``clue``, plus
``all_words`` / ``max_guesses`` for the guess task).

Output
------
One parquet under ``custom_data/training_prompts/version-5/``::

    codenames_rlvr.parquet

using VeRL's 4-column schema (``prompt``, ``data_source``,
``reward_model``, ``extra_info``).

Filtering
---------
500 source rows ship ``clue == "unknown"`` and
``selected_target_words == "unknown"`` together (no usable reference
clue); those rows are dropped. The remaining 4000 rows are written in
their original source order — no shuffling, no train/val split.

Transformations
---------------
* The v5 user prompt is kept verbatim; the CoT ``SYS_PROMPT`` from
  ``system_prompts.py`` is prepended as a ``system`` message so the
  prompt list has the same two-message shape as v4.
* ``data_source`` is rewritten to the absolute path of
  ``custom_reward_functions/codenames_reward.py``.
* ``reward_function_name`` is added to ``extra_info`` (v4 parity).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from system_prompts import SYS_PROMPT  # noqa: E402

REPO_ROOT = HERE.parent
DEFAULT_INPUT = REPO_ROOT / "custom_data/version-5/codenames_rlvr.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "custom_data/training_prompts/version-5"
DEFAULT_OUTPUT_NAME = "codenames_rlvr.parquet"
REWARD_FN_PATH = REPO_ROOT / "custom_reward_functions/codenames_reward.py"
REWARD_FN_NAME = "compute_score"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_missing_clue(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        return s == "" or s.lower() == "unknown"
    return False


def _is_missing_selected(v) -> bool:
    # Source data uses the sentinel string ``"unknown"`` when no list is
    # available; a valid value is a non-empty list of target words.
    if v is None:
        return True
    if isinstance(v, str):
        return True
    if isinstance(v, list):
        return len(v) == 0
    return True


def _transform(row: dict) -> dict:
    extra_info = dict(row.get("extra_info") or {})
    extra_info.setdefault("reward_function_name", REWARD_FN_NAME)

    user_messages = list(row.get("prompt") or [])
    messages = [{"role": "system", "content": SYS_PROMPT}, *user_messages]

    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth", "")

    return {
        "prompt": messages,
        "data_source": str(REWARD_FN_PATH),
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": extra_info,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = p.parse_args()

    rows = _load_jsonl(args.input)
    print(f"[load] {len(rows)} rows from {args.input}")

    kept: list[dict] = []
    dropped = 0
    task_counts = {"codenames_clue_generation": 0, "codenames_guess_generation": 0}
    for r in rows:
        ei = r.get("extra_info") or {}
        if _is_missing_clue(ei.get("clue")) or _is_missing_selected(
            ei.get("selected_target_words")
        ):
            dropped += 1
            continue
        kept.append(r)
        task = ei.get("task", "")
        if task in task_counts:
            task_counts[task] += 1

    print(f"[filter] kept={len(kept)}  dropped={dropped}")
    print(f"[task counts] {task_counts}")

    records = [_transform(r) for r in kept]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    pd.DataFrame(records).to_parquet(out_path, index=False)

    print(f"[save] -> {out_path}  ({len(records)} rows)")
    print(f"[reward fn] data_source = {REWARD_FN_PATH}")
    print(f"[reward fn] function    = {REWARD_FN_NAME}")


if __name__ == "__main__":
    main()
