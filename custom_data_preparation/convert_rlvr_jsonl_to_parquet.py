"""Convert the version-4 Codenames RLVR JSONL into VeRL-ready parquet files.

Input
-----
``custom_data/version-4/codenames_rlvr_clue_gen.jsonl``

Each line has keys::

    prompt       : list of messages (user only)
    data_source  : str (ignored — we overwrite to point at our reward fn)
    reward_model : {"ground_truth": ""}
    extra_info   : {"task", "target_words", "non_target_words", ...}

Output
------
Two parquet files with the schema VeRL consumes (same columns as
``custom_data/training_prompts/size_2/only_codenames_trial_train.parquet``)::

    prompt       : list[{"role", "content"}]     # SYS_PROMPT prepended
    data_source  : absolute path to codenames_reward.py
    reward_model : {"ground_truth": str}
    extra_info   : dict preserving every original feature

Notes
-----
* The JSONL user prompt adds a spurious ``[Risk]: ...`` line to the
  response-format spec that is not part of our canonical template.  We
  therefore **rebuild the user message from scratch** using
  ``get_messages_single_turn(..., "clue")`` so the prompt is byte-identical
  to ``CODENAMES_CLUE_GEN_INSTRUCTION`` in
  ``custom_data_preparation/system_prompts.py``, and the system prompt
  carrying the five-section CoT is prepended automatically.
* ``data_source`` is set to the absolute path of
  ``custom_reward_functions/codenames_reward.py``.  VeRL uses the
  ``custom_reward_function.path`` / ``custom_reward_function.name`` CLI
  flags to load the function itself — this field is just a routing tag.
* The source file has 1000 rows.  Requesting 1000 train + 100 val
  therefore overlaps the last 100 rows; the script prints a warning in
  that case.  Pass ``--train-n``/``--val-n`` to override.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Make ``system_prompts`` importable when run as a script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from system_prompts import get_messages_single_turn  # noqa: E402

REPO_ROOT = HERE.parent
DEFAULT_INPUT = REPO_ROOT / "custom_data/version-4/codenames_rlvr_clue_gen.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "custom_data/training_prompts/version-4"
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


def _transform(row: dict) -> dict:
    """Rewrite one JSONL row into the VeRL parquet schema.

    The original user prompt from the JSONL is discarded and rebuilt
    from ``target_words`` / ``non_target_words`` via the canonical
    ``get_messages_single_turn`` helper.  This strips the JSONL's extra
    ``[Risk]:`` line and guarantees the response-format spec matches
    what our reward parser expects.
    """
    extra_info = dict(row.get("extra_info") or {})
    messages = get_messages_single_turn(
        inputs={
            "target_words": list(extra_info.get("target_words") or []),
            "non_target_words": list(extra_info.get("non_target_words") or []),
        },
        task_name="clue",
    )

    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth", "")

    # Preserve every original extra_info field; also tuck the reward
    # function name in so downstream logging can surface it per-sample.
    extra_info.setdefault("reward_function_name", REWARD_FN_NAME)

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
    p.add_argument("--train-n", type=int, default=1000)
    p.add_argument("--val-n", type=int, default=100)
    p.add_argument("--train-name", default="codenames_rlvr_clue_gen_train.parquet")
    p.add_argument("--val-name", default="codenames_rlvr_clue_gen_val.parquet")
    args = p.parse_args()

    rows = _load_jsonl(args.input)
    print(f"[load] {len(rows)} rows from {args.input}")

    if args.train_n > len(rows):
        raise SystemExit(
            f"Requested {args.train_n} train rows but file has only {len(rows)}."
        )
    if args.val_n > len(rows):
        raise SystemExit(
            f"Requested {args.val_n} val rows but file has only {len(rows)}."
        )

    train_rows = rows[: args.train_n]
    val_rows = rows[-args.val_n:]

    train_idx = set(range(args.train_n))
    val_idx = set(range(len(rows) - args.val_n, len(rows)))
    overlap = train_idx & val_idx
    if overlap:
        print(
            f"[WARN] train/val overlap by {len(overlap)} rows — source file has "
            f"only {len(rows)} rows, not enough for {args.train_n} + {args.val_n} "
            "non-overlapping. Pass --train-n/--val-n to change the split."
        )

    train_records = [_transform(r) for r in train_rows]
    val_records = [_transform(r) for r in val_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name

    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    pd.DataFrame(val_records).to_parquet(val_path, index=False)

    print(f"[save] train -> {train_path}  ({len(train_records)} rows)")
    print(f"[save] val   -> {val_path}  ({len(val_records)} rows)")
    print(f"[reward fn] data_source = {REWARD_FN_PATH}")
    print(f"[reward fn] function    = {REWARD_FN_NAME}")


if __name__ == "__main__":
    main()
