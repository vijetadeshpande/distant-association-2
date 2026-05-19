"""Qualitative evaluation harness for Codenames RLVR models.

Runs vLLM inference on a small slice of validation prompts and writes the
prompts + completions to a JSONL file for offline inspection.

Sampling matches the training rollout (see scripts/codenames_dapo.yaml):
    temperature=1.0, top_p=1.0, top_k=-1, max_tokens=max_response_length.

Two data modes:
  - parquet  : sample `--per-group` rows per (task x difficulty_rule)
               (default 3 -> 24 rows on the v5 val parquet)
  - hf       : load an HF dataset and randomly sample `--hf-n` rows (default 24)

Pass `--build-sample-to PATH` to materialise the sample into a fixed parquet and
exit without running inference. Subsequent model runs read that file with
`--use-all-rows`, so every checkpoint is evaluated on identical prompts.

Two model spec forms:
  - HF id            (e.g. "Qwen/Qwen3-8B")
  - rclone path      (e.g. "gdrive-distant-association:foo/bar/merged_hf")
    The remote dir is rclone-copied to --cache-dir before vLLM loads it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# Matches the training rollout (verl/trainer/config/rollout/rollout.yaml).
SAMPLING_DEFAULTS = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=16384,
)


def is_rclone_spec(model: str) -> bool:
    """rclone remote specs look like 'remote-name:path/to/dir'."""
    if ":" not in model:
        return False
    head = model.split(":", 1)[0]
    # HF ids never contain ':' in the org segment; rclone remotes don't contain '/'.
    return "/" not in head and head != ""


def materialise_model(model: str, cache_dir: Path) -> str:
    """Return a local path vLLM can load from.

    For HF ids, returns the id unchanged (vLLM will pull from the hub).
    For rclone specs, rclone-copies to cache_dir/<slug> and returns the path.
    """
    if not is_rclone_spec(model):
        return model

    remote, remote_path = model.split(":", 1)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{remote}__{remote_path}").strip("_")
    local = cache_dir / slug
    marker = local / ".rclone_done"
    if marker.exists():
        print(f"[model] cache hit: {local}", flush=True)
        return str(local)

    local.mkdir(parents=True, exist_ok=True)
    print(f"[model] rclone copy {model} -> {local}", flush=True)
    subprocess.run(
        ["rclone", "copy", model, str(local), "--progress", "--transfers", "8"],
        check=True,
    )
    marker.touch()
    return str(local)


def _attach_task_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper '_task' / '_difficulty' cols. Best-effort: missing on non-v5 data."""
    df = df.copy()
    if "extra_info" in df.columns:
        ei = pd.json_normalize(df["extra_info"])
        df["_task"] = ei["task"].values if "task" in ei.columns else None
        df["_difficulty"] = ei["difficulty_rule"].values if "difficulty_rule" in ei.columns else None
    else:
        df["_task"] = None
        df["_difficulty"] = None
    return df


def sample_parquet(path: str, seed: int, per_group: int = 3) -> pd.DataFrame:
    """`per_group` rows per (task, difficulty_rule)."""
    df = _attach_task_difficulty(pq.read_table(path).to_pandas())
    parts = []
    for _keys, g in df.groupby(["_task", "_difficulty"], sort=True):
        parts.append(g.sample(n=min(per_group, len(g)), random_state=seed))
    picked = pd.concat(parts, ignore_index=True)
    print(
        f"[data] parquet: {len(df)} rows -> {len(picked)} sampled "
        f"({per_group}/group, {picked['_task'].nunique()} tasks x "
        f"{picked['_difficulty'].nunique()} difficulties)",
        flush=True,
    )
    return picked


def load_parquet_all(path: str) -> pd.DataFrame:
    """Load every row from a (pre-sampled) parquet, no resampling."""
    df = _attach_task_difficulty(pq.read_table(path).to_pandas())
    print(f"[data] parquet: loaded all {len(df)} rows from {path}", flush=True)
    return df


def sample_hf(name: str, split: str, n: int, seed: int) -> pd.DataFrame:
    """Random sample of n rows from an HF dataset split. n <= 0 -> all rows."""
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    if n <= 0 or n >= len(ds):
        df = ds.to_pandas()
        print(f"[data] hf {name}/{split}: {len(ds)} rows -> all", flush=True)
        return df
    idx = random.Random(seed).sample(range(len(ds)), n)
    df = ds.select(idx).to_pandas()
    print(f"[data] hf {name}/{split}: {len(ds)} rows -> {n} sampled", flush=True)
    return df


def to_chat_messages(row: pd.Series) -> list[dict]:
    """Extract chat messages from a row.

    Supported shapes (in priority order):
      - row['prompt']  : list of {role, content} dicts (v5 val parquet schema)
      - row['messages']: list of {role, content} dicts (common HF chat schema)
      - row['prompt']  : str (wrapped as a single user message)
    """
    for key in ("prompt", "messages"):
        if key in row and row[key] is not None:
            v = row[key]
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return [{"role": m["role"], "content": m["content"]} for m in v]
    if "prompt" in row and isinstance(row["prompt"], str):
        return [{"role": "user", "content": row["prompt"]}]
    raise ValueError(f"No chat-formatted prompt in row (columns={list(row.index)})")


def row_metadata(row: pd.Series) -> dict:
    """Pull the non-prompt fields useful for offline analysis."""
    meta = {"data_source": row.get("data_source")}
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        keep = (
            "task", "difficulty_rule", "difficulty_score", "clue", "max_guesses",
            "target_words", "non_target_words", "selected_target_words",
            "all_words", "num_words_in_target", "num_words_non_target",
            "topic_1", "topic_2",
        )
        meta.update({k: ei.get(k) for k in keep if k in ei})
    gt = row.get("reward_model")
    if isinstance(gt, dict):
        meta["ground_truth"] = gt.get("ground_truth")
    # Convenience keys for parquet sampling (set by sample_parquet).
    for k in ("_task", "_difficulty"):
        if k in row:
            meta.setdefault(k.lstrip("_"), row[k])
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   help="HF id or rclone path (remote:path/...). "
                        "Required unless --build-sample-to is set.")
    p.add_argument("--data", required=True,
                   help="Path to .parquet OR HF dataset name (with --data-source hf)")
    p.add_argument("--data-source", choices=["parquet", "hf"], default="parquet")
    p.add_argument("--use-all-rows", action="store_true",
                   help="parquet mode: take every row instead of per-group sampling. "
                        "Use this when --data already points at a pre-sampled subset.")
    p.add_argument("--per-group", type=int, default=3,
                   help="parquet mode: rows to sample per (task, difficulty_rule). "
                        "Default 3 -> 24 rows on v5 val (8 combos).")
    p.add_argument("--hf-split", default="validation")
    p.add_argument("--hf-n", type=int, default=24,
                   help="Random sample size for HF datasets. Use 0 or -1 for ALL rows.")
    p.add_argument("--build-sample-to",
                   help="Write the sampled subset to this parquet path and exit "
                        "(no model loading, no inference). Use to fix the eval set "
                        "across multiple model runs.")
    p.add_argument("--output", help="Output .jsonl path (required unless --build-sample-to)")
    p.add_argument("--overwrite", action="store_true",
                   help="Rerun even if --output already exists. Default: skip.")
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "distant_association_models"))
    p.add_argument("--n-samples-per-prompt", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=0,
                   help="0 = auto-detect from CUDA_VISIBLE_DEVICES / nvidia-smi")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Defaults to max_prompt_length + max_response_length = 2048 + 16384")
    p.add_argument("--max-tokens", type=int, default=SAMPLING_DEFAULTS["max_tokens"])
    p.add_argument("--temperature", type=float, default=SAMPLING_DEFAULTS["temperature"])
    p.add_argument("--top-p", type=float, default=SAMPLING_DEFAULTS["top_p"])
    p.add_argument("--top-k", type=int, default=SAMPLING_DEFAULTS["top_k"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Skip rerun if output already exists. Happens before data/model load so
    # already-finished runs cost nothing on the second pass.
    if args.output and not args.build_sample_to and not args.overwrite \
            and Path(args.output).exists():
        print(f"[skip] {args.output} already exists; pass --overwrite to rerun", flush=True)
        return 0

    # --- data ---------------------------------------------------------------
    if args.data_source == "parquet":
        df = (
            load_parquet_all(args.data)
            if args.use_all_rows
            else sample_parquet(args.data, args.seed, per_group=args.per_group)
        )
    else:
        df = sample_hf(args.data, args.hf_split, args.hf_n, args.seed)

    # Sample-build mode: dump the picked rows to parquet and exit before model load.
    if args.build_sample_to:
        out = Path(args.build_sample_to)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_out = df.drop(columns=[c for c in ("_task", "_difficulty") if c in df.columns])
        df_out.to_parquet(out, index=False)
        print(f"[sample] wrote {len(df_out)} rows -> {out}", flush=True)
        return 0

    if not args.model:
        p.error("--model is required unless --build-sample-to is set")
    if not args.output:
        p.error("--output is required unless --build-sample-to is set")

    messages_list = [to_chat_messages(row) for _, row in df.iterrows()]
    metas = [row_metadata(row) for _, row in df.iterrows()]

    # --- model --------------------------------------------------------------
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = materialise_model(args.model, cache_dir)

    if args.tensor_parallel_size > 0:
        tp = args.tensor_parallel_size
    else:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            tp = len([x for x in cvd.split(",") if x.strip() != ""])
        else:
            tp = int(subprocess.check_output(["nvidia-smi", "-L"]).decode().strip().count("\n") + 1)
    print(f"[vllm] tensor_parallel_size={tp}", flush=True)

    from vllm import LLM, SamplingParams

    max_model_len = args.max_model_len or (2048 + args.max_tokens)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        seed=args.seed,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        n=args.n_samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    print(f"[vllm] running chat on {len(messages_list)} prompts "
          f"x n={args.n_samples_per_prompt}", flush=True)
    outputs = llm.chat(messages_list, sampling_params=sampling)

    # --- write --------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, out in enumerate(outputs):
            record = {
                "idx": i,
                "model": args.model,
                "metadata": metas[i],
                "messages": messages_list[i],
                "completions": [o.text for o in out.outputs],
                "finish_reasons": [o.finish_reason for o in out.outputs],
                "sampling": {
                    "temperature": args.temperature, "top_p": args.top_p,
                    "top_k": args.top_k, "max_tokens": args.max_tokens,
                    "n": args.n_samples_per_prompt, "seed": args.seed,
                },
            }
            f.write(json.dumps(record, default=str) + "\n")
    print(f"[done] wrote {len(outputs)} records -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
