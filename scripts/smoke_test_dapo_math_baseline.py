#!/usr/bin/env python3
"""Generate and score a few DAPO-Math responses without training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    REPO_ROOT / "custom_data" / "training_prompts" / "baseline-dapo-math" / "dapo_math_val.parquet"
)
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--indices", type=int, nargs="+", default=[12, 73, 1, 2])
    parser.add_argument("--n", type=int, default=2, help="Responses per prompt")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pq.read_table(args.data).to_pylist()
    if not rows:
        raise ValueError(f"No rows found in {args.data}")
    if min(args.indices) < 0 or max(args.indices) >= len(rows):
        raise IndexError(f"indices must be between 0 and {len(rows) - 1}")

    chosen = [rows[index] for index in args.indices]
    unexpected_sources = {row["data_source"] for row in chosen} - {"math_dapo"}
    if unexpected_sources:
        raise ValueError(f"Unsupported data_source values: {sorted(unexpected_sources)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompts = [
        tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in chosen
    ]

    max_prompt_tokens = max(len(tokenizer(prompt)["input_ids"]) for prompt in prompts)
    max_model_len = max_prompt_tokens + args.max_tokens
    llm = LLM(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    for index, row, request_output in zip(args.indices, chosen, outputs, strict=True):
        problem = row["prompt"][-1]["content"].split("Problem:\n", 1)[-1]
        ground_truth = row["reward_model"]["ground_truth"]
        print(f"\n===== ROW {index} | GROUND TRUTH {ground_truth} =====")
        print(f"PROBLEM: {problem}")
        for sample_index, candidate in enumerate(request_output.outputs):
            result = default_compute_score(
                data_source=row["data_source"],
                solution_str=candidate.text,
                ground_truth=ground_truth,
                extra_info=row["extra_info"],
            )
            print(
                f"\n--- SAMPLE {sample_index} | tokens={len(candidate.token_ids)} "
                f"| finish={candidate.finish_reason} | reward={result} ---"
            )
            print(candidate.text)


if __name__ == "__main__":
    main()
