#!/usr/bin/env python3
"""Generate and score Qwen3-8B responses from all RuleCollection domains."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from custom_reward_functions.rule_collection_reward import compute_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    REPO_ROOT
    / "custom_data"
    / "training_prompts"
    / "baseline-rule-collection"
    / "rule_collection_val.parquet"
)
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_INDICES = [0, 2, 3, 4, 7, 12]
SUPPORTED_SOURCES = {
    "AR-LSAT",
    "Folio",
    "Logic NLI",
    "Logical Deduction",
    "ProntoQA",
    "ProofWriter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    parser.add_argument("--n", type=int, default=2, help="Responses per prompt")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum response tokens; defaults to the shared DAPO training limit",
    )
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
    unexpected_sources = {row["data_source"] for row in chosen} - SUPPORTED_SOURCES
    if unexpected_sources:
        raise ValueError(f"Unsupported data_source values: {sorted(unexpected_sources)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    prompt_ids = [
        tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        for row in chosen
    ]
    for index, row, token_ids in zip(args.indices, chosen, prompt_ids, strict=True):
        recorded_length = row["extra_info"]["prompt_tokens"]
        if len(token_ids) != recorded_length:
            raise ValueError(
                f"row {index}: rendered prompt has {len(token_ids)} tokens, "
                f"but metadata records {recorded_length}"
            )

    max_prompt_tokens = max(map(len, prompt_ids))
    llm = LLM(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_prompt_tokens + args.max_tokens,
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
    # Passing token IDs mirrors VeRL's SingleTurnAgentLoop and avoids a second,
    # potentially different prompt-rendering step inside the inference engine.
    prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_ids]
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    for index, row, token_ids, request_output in zip(
        args.indices, chosen, prompt_ids, outputs, strict=True
    ):
        source = row["data_source"]
        task = row["prompt"][-1]["content"]
        ground_truth = row["reward_model"]["ground_truth"]
        print(
            f"\n===== ROW {index} | DOMAIN {source} | PROMPT TOKENS {len(token_ids)} "
            f"| GROUND TRUTH {ground_truth} ====="
        )
        print(f"TASK:\n{task}")
        for sample_index, candidate in enumerate(request_output.outputs):
            result = compute_score(
                data_source=source,
                solution_str=candidate.text,
                ground_truth=ground_truth,
                extra_info=row["extra_info"],
            )
            print(
                f"\n--- SAMPLE {sample_index} | tokens={len(candidate.token_ids)} "
                f"| finish={candidate.finish_reason} | reward={result['score']} "
                f"| acc={result['acc']} | format_failure={result['format_failure']} "
                f"| illegal_label={result['illegal_label']} ---"
            )
            print(candidate.text)


if __name__ == "__main__":
    main()
