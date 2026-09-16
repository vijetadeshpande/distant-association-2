#!/usr/bin/env python3
"""Generate and score the complete Codenames validation rollout set.

Run this script inside the existing/default VERL image. It does not build,
modify, or launch a container. One invocation evaluates one model checkpoint;
run it once for each base or tuned Qwen3 1.7B/4B/8B/14B checkpoint.

The data and rollout setup are intentionally fixed for comparability:

* data: ``custom_data/training_prompts/version-5/codenames_rlvr_val.parquet``
* prompts: all 320 rows (40 per task/difficulty combination)
* samples: 32 responses per prompt, or 10,240 responses per checkpoint
* sampling: temperature=1.0, top_p=1.0, top_k=-1
* prompt limit: 2,048 tokens
* response limit: 16,384 tokens, matching ``train_codenames_dapo.sh``
* inference: vLLM on all visible GPUs by default, bfloat16, no quantization,
  continuous/dynamic batching, chunked prefill, prefix caching, and CUDA graphs
* clue reward: google/gemini-2.5-flash through OpenRouter, thinking disabled
* guess reward: the repository's deterministic rule-based reward

``--model`` accepts a Hugging Face model ID, a local HF-format directory, or an
rclone reference such as ``gdrive:Distant-Association/.../merged_hf``. Remote
checkpoints are downloaded and cached locally before vLLM is imported. Tuned
VERL checkpoints must be merged to HF format before vLLM can load them.

The output is prompt-grouped JSONL: every line contains the prompt metadata,
32 completions, their token counts and finish reasons, and 32 complete reward
result dictionaries. Passing ``--logprobs 20`` additionally stores compact
per-completion sampled-token likelihood and top-k entropy-lower-bound summaries
without writing the much larger token-level logprob dictionaries. Existing
output is resumed by prompt index by default.

Examples::

    python scripts/generate_codenames_validation_rollouts.py \
        --model Qwen/Qwen3-1.7B \
        --output outputs/codenames/qwen3-1.7b-base.jsonl

    python scripts/generate_codenames_validation_rollouts.py \
        --model checkpoints/.../actor/merged_hf \
        --model-label qwen3-14b-codenames-step230 \
        --tensor-parallel-size 2 \
        --output outputs/codenames/qwen3-14b-codenames-step230.jsonl

    python scripts/generate_codenames_validation_rollouts.py \
        --model gdrive:Distant-Association/.../actor/merged_hf \
        --tensor-parallel-size 2 \
        --output outputs/codenames/qwen3-14b-codenames-step230.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Coroutine


REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_PARQUET = (
    REPO_ROOT
    / "custom_data"
    / "training_prompts"
    / "version-5"
    / "codenames_rlvr_val.parquet"
)

EXPECTED_PROMPTS = 320
RESPONSES_PER_PROMPT = 32
MAX_PROMPT_TOKENS = 2048
MAX_RESPONSE_TOKENS = 16384
MAX_MODEL_LEN = MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS

TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1

JUDGE_MODEL = "google/gemini-2.5-flash"
JUDGE_BACKEND = "openrouter"
JUDGE_THINKING = False

EXPECTED_DATA_DISTRIBUTION = {
    ("codenames_guess_generation", "simple"): 40,
    ("codenames_clue_generation", "simple"): 40,
    ("codenames_guess_generation", "moderate"): 40,
    ("codenames_clue_generation", "moderate"): 40,
    ("codenames_guess_generation", "advance"): 40,
    ("codenames_clue_generation", "advance"): 40,
    ("codenames_guess_generation", "expert"): 40,
    ("codenames_clue_generation", "expert"): 40,
}


def log_progress(message: str) -> None:
    """Emit a timestamped progress line that remains visible beside vLLM logs."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[codenames-progress {timestamp}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    """Format a duration compactly without hiding multi-hour runtimes."""
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def progress_summary(done: int, total: int, elapsed: float, unit: str) -> str:
    """Return count, percentage, observed rate, and a simple rolling ETA."""
    percentage = 100.0 if total == 0 else 100.0 * done / total
    rate = done / elapsed if done > 0 and elapsed > 0 else 0.0
    remaining = max(0, total - done)
    eta = remaining / rate if rate > 0 else None
    rate_text = f"{rate:.2f} {unit}/s" if rate >= 0.1 else f"{rate * 60:.2f} {unit}/min"
    eta_text = format_duration(eta) if eta is not None else "estimating"
    return (
        f"{done:,}/{total:,} {unit} ({percentage:.1f}%) | "
        f"elapsed {format_duration(elapsed)} | {rate_text} | ETA {eta_text}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face ID, local merged-HF directory, or rclone remote",
    )
    parser.add_argument(
        "--model-label",
        help="Short label stored in output; inferred from --model when omitted",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face revision for a base model",
    )
    parser.add_argument(
        "--checkpoint-download-dir",
        type=Path,
        help=(
            "Exact destination directory for an rclone checkpoint. By default, "
            "a unique directory is created under RCLONE_CHECKPOINT_DIR."
        ),
    )
    parser.add_argument(
        "--force-checkpoint-download",
        action="store_true",
        help="Run rclone again even if the cached checkpoint has config.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Prompt-grouped JSONL output path",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="vLLM tensor-parallel GPUs; 0 uses every visible GPU (default: 0)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory available to vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--submission-batch-size",
        type=int,
        default=32,
        help=(
            "Prompt groups submitted at a time (default: 32). This only bounds "
            "host memory and checkpointing intervals; vLLM still schedules "
            "sequences with continuous/dynamic batching."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=0,
        help=(
            "Number of top token logprobs requested from vLLM. Zero disables "
            "logprob summaries; use 20 for checkpoint-trajectory analysis."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start over instead of resuming an existing output file",
    )
    return parser.parse_args()


def load_dotenv_if_present(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding a python-dotenv dependency."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def validate_args(args: argparse.Namespace) -> None:
    if args.tensor_parallel_size < 0:
        raise ValueError("--tensor-parallel-size must be 0 (auto) or a positive integer")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.submission_batch_size < 1:
        raise ValueError("--submission-batch-size must be at least 1")
    if args.logprobs < 0:
        raise ValueError("--logprobs must be non-negative")


def available_gpu_count() -> int:
    """Count GPUs visible to this process, respecting CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
        if not devices or devices == ["-1"]:
            return 0
        return len(devices)

    try:
        listing = subprocess.check_output(
            ["nvidia-smi", "-L"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0
    return sum(bool(line.strip()) for line in listing.splitlines())


def resolve_tensor_parallel_size(requested_size: int) -> int:
    """Resolve 0 to all visible GPUs; preserve a positive explicit override."""
    if requested_size > 0:
        return requested_size
    gpu_count = available_gpu_count()
    if gpu_count < 1:
        raise RuntimeError(
            "No visible GPUs found. Set CUDA_VISIBLE_DEVICES appropriately or pass "
            "--tensor-parallel-size explicitly."
        )
    return gpu_count


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_PROMPTS:
        raise ValueError(
            f"Expected exactly {EXPECTED_PROMPTS} rows in {VAL_PARQUET}, found {len(rows)}"
        )

    distribution: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        extra_info = row.get("extra_info")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Validation row {index} has no chat-formatted prompt")
        if not isinstance(extra_info, dict):
            raise ValueError(f"Validation row {index} has no extra_info dictionary")
        task = extra_info.get("task")
        difficulty = extra_info.get("difficulty_rule")
        difficulty_score = extra_info.get("difficulty_score")
        if task is None or difficulty is None or difficulty_score is None:
            raise ValueError(
                f"Validation row {index} is missing task or difficulty indicators"
            )
        distribution[(str(task), str(difficulty))] += 1

    if dict(distribution) != EXPECTED_DATA_DISTRIBUTION:
        raise ValueError(
            "Unexpected validation task/difficulty distribution. "
            f"Expected {EXPECTED_DATA_DISTRIBUTION}; found {dict(distribution)}"
        )


def sampling_metadata(seed: int, logprobs: int = 0) -> dict[str, Any]:
    metadata = {
        "n": RESPONSES_PER_PROMPT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "seed": seed,
    }
    # Preserve byte-for-byte compatibility with existing JSONLs and their
    # resume validation when logprob collection is disabled.
    if logprobs > 0:
        metadata["logprobs"] = logprobs
    return metadata


def read_completed_indices(
    output: Path,
    model: str,
    revision: str | None,
    seed: int,
    logprobs: int = 0,
) -> set[int]:
    """Validate completed JSONL rows before resuming an expensive run."""
    completed: set[int] = set()
    expected_sampling = sampling_metadata(seed, logprobs)
    with output.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot resume: invalid JSON at {output}:{line_number}: {exc}"
                ) from exc

            if record.get("model") != model or record.get("revision") != revision:
                raise ValueError(
                    f"Cannot resume {output}: model/revision differs at line {line_number}"
                )
            if record.get("sampling") != expected_sampling:
                raise ValueError(
                    f"Cannot resume {output}: sampling settings differ at line {line_number}"
                )
            completions = record.get("completions")
            reward_results = record.get("reward_results")
            if not isinstance(completions, list) or len(completions) != RESPONSES_PER_PROMPT:
                raise ValueError(
                    f"Cannot resume {output}: line {line_number} does not contain "
                    f"{RESPONSES_PER_PROMPT} completions"
                )
            if not isinstance(reward_results, list) or len(reward_results) != RESPONSES_PER_PROMPT:
                raise ValueError(
                    f"Cannot resume {output}: line {line_number} does not contain "
                    f"{RESPONSES_PER_PROMPT} reward results"
                )
            index = int(record["idx"])
            if index in completed:
                raise ValueError(f"Cannot resume {output}: duplicate prompt index {index}")
            completed.add(index)
    return completed


def render_prompt_token_ids(tokenizer: Any, rows: list[dict[str, Any]]) -> list[list[int]]:
    prompt_ids: list[list[int]] = []
    for index, row in enumerate(rows):
        token_ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(token_ids) > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"Validation row {index} renders to {len(token_ids)} prompt tokens; "
                f"the training limit is {MAX_PROMPT_TOKENS}"
            )
        prompt_ids.append(token_ids)
    return prompt_ids


async def score_batch(
    rows: list[dict[str, Any]],
    request_outputs: list[Any],
    compute_score: Callable[..., Coroutine[Any, Any, dict[str, Any]]],
    progress_context: str,
    progress_interval_s: float = 30.0,
) -> list[list[dict[str, Any]]]:
    calls: list[Coroutine[Any, Any, dict[str, Any]]] = []
    for row, request_output in zip(rows, request_outputs, strict=True):
        ground_truth = (row.get("reward_model") or {}).get("ground_truth")
        for candidate in request_output.outputs:
            calls.append(
                compute_score(
                    data_source=row.get("data_source"),
                    solution_str=candidate.text,
                    ground_truth=ground_truth,
                    extra_info=row.get("extra_info"),
                    judge_model=JUDGE_MODEL,
                    judge_backend=JUDGE_BACKEND,
                    judge_thinking=JUDGE_THINKING,
                )
            )

    async def indexed_result(
        result_index: int,
        call: Coroutine[Any, Any, dict[str, Any]],
    ) -> tuple[int, dict[str, Any]]:
        return result_index, await call

    total = len(calls)
    started = time.monotonic()
    tasks = {
        asyncio.create_task(indexed_result(result_index, call))
        for result_index, call in enumerate(calls)
    }
    pending = set(tasks)
    ordered_results: list[dict[str, Any] | None] = [None] * total
    completed_count = 0
    next_report_at = started + progress_interval_s

    try:
        while pending:
            now = time.monotonic()
            timeout = max(0.0, next_report_at - now)
            finished, pending = await asyncio.wait(
                pending,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in finished:
                result_index, result = task.result()
                ordered_results[result_index] = result
                completed_count += 1

            now = time.monotonic()
            if completed_count == total or now >= next_report_at:
                log_progress(
                    f"{progress_context}: "
                    + progress_summary(
                        completed_count,
                        total,
                        now - started,
                        "responses",
                    )
                )
                next_report_at = now + progress_interval_s
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    if any(result is None for result in ordered_results):
        raise RuntimeError("Reward scoring finished without a result for every response")
    flat_results = [result for result in ordered_results if result is not None]
    grouped_results: list[list[dict[str, Any]]] = []
    for start in range(0, len(flat_results), RESPONSES_PER_PROMPT):
        grouped_results.append(flat_results[start : start + RESPONSES_PER_PROMPT])
    return grouped_results


def _numeric_logprob(value: Any) -> float | None:
    """Read a vLLM Logprob object without depending on its concrete version."""
    raw = getattr(value, "logprob", value)
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _logprob_slice_summary(
    token_ids: list[int],
    position_logprobs: list[Any],
) -> dict[str, float | int | None]:
    sampled_logprobs: list[float] = []
    sampled_ranks: list[float] = []
    top1_probabilities: list[float] = []
    top1_top2_margins: list[float] = []
    known_masses: list[float] = []
    entropy_lower_bounds: list[float] = []

    for token_id, raw_entries in zip(token_ids, position_logprobs, strict=False):
        if not isinstance(raw_entries, dict) or not raw_entries:
            continue
        entries: list[tuple[Any, float]] = []
        for entry_token_id, entry in raw_entries.items():
            logprob = _numeric_logprob(entry)
            if logprob is not None:
                entries.append((entry_token_id, logprob))
        if not entries:
            continue

        sampled_entry = raw_entries.get(token_id)
        if sampled_entry is None:
            sampled_entry = raw_entries.get(str(token_id))
        sampled_logprob = _numeric_logprob(sampled_entry)
        if sampled_logprob is not None:
            sampled_logprobs.append(sampled_logprob)
            rank = getattr(sampled_entry, "rank", None)
            if isinstance(rank, (int, float)):
                sampled_ranks.append(float(rank))

        probabilities = sorted(
            (min(1.0, math.exp(logprob)) for _, logprob in entries),
            reverse=True,
        )
        known_mass = min(1.0, sum(probabilities))
        residual_mass = max(0.0, 1.0 - known_mass)
        known_masses.append(known_mass)
        top1_probabilities.append(probabilities[0])
        if len(probabilities) > 1:
            top1_top2_margins.append(probabilities[0] - probabilities[1])

        # Entropy of the observed top-k entries plus one bucket containing the
        # entire unobserved tail. Lumping the tail makes this a lower bound on
        # the full-vocabulary entropy, not an entropy estimate.
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        if residual_mass > 0.0:
            entropy -= residual_mass * math.log(residual_mass)
        entropy_lower_bounds.append(entropy)

    def average(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "logprob_token_count": len(sampled_logprobs),
        "sampled_token_logprob_mean": average(sampled_logprobs),
        "sampled_token_nll_mean": (
            -average(sampled_logprobs) if sampled_logprobs else None
        ),
        "sampled_token_rank_mean": average(sampled_ranks),
        "top1_probability_mean": average(top1_probabilities),
        "top1_top2_margin_mean": average(top1_top2_margins),
        "topk_probability_mass_mean": average(known_masses),
        "topk_entropy_lower_bound_mean": average(entropy_lower_bounds),
    }


def summarize_completion_logprobs(candidate: Any) -> dict[str, float | int | None] | None:
    """Summarize vLLM top-k logprobs without storing token-level dictionaries."""
    raw_logprobs = getattr(candidate, "logprobs", None)
    token_ids = list(getattr(candidate, "token_ids", []) or [])
    if not isinstance(raw_logprobs, list) or not raw_logprobs:
        return None
    full = _logprob_slice_summary(token_ids, raw_logprobs)
    first = _logprob_slice_summary(token_ids[:256], raw_logprobs[:256])
    last = _logprob_slice_summary(token_ids[-128:], raw_logprobs[-128:])
    full.update(
        {
            "first_256_sampled_token_nll_mean": first["sampled_token_nll_mean"],
            "first_256_topk_entropy_lower_bound_mean": first[
                "topk_entropy_lower_bound_mean"
            ],
            "last_128_sampled_token_nll_mean": last["sampled_token_nll_mean"],
            "last_128_topk_entropy_lower_bound_mean": last[
                "topk_entropy_lower_bound_mean"
            ],
        }
    )
    return full


def build_output_record(
    index: int,
    row: dict[str, Any],
    request_output: Any,
    reward_results: list[dict[str, Any]],
    args: argparse.Namespace,
    prompt_token_count: int,
    resolved_model: str,
) -> dict[str, Any]:
    if len(request_output.outputs) != RESPONSES_PER_PROMPT:
        raise RuntimeError(
            f"Prompt {index}: vLLM returned {len(request_output.outputs)} responses; "
            f"expected {RESPONSES_PER_PROMPT}"
        )
    if len(reward_results) != RESPONSES_PER_PROMPT:
        raise RuntimeError(
            f"Prompt {index}: collected {len(reward_results)} rewards; "
            f"expected {RESPONSES_PER_PROMPT}"
        )

    ground_truth = (row.get("reward_model") or {}).get("ground_truth")
    metadata = {
        "data_source": row.get("data_source"),
        "ground_truth": ground_truth,
        **(row.get("extra_info") or {}),
    }
    candidates = request_output.outputs
    record = {
        "idx": index,
        "model": args.model,
        "resolved_model": resolved_model,
        "model_label": args.model_label or args.model,
        "revision": args.revision,
        "metadata": metadata,
        "messages": row["prompt"],
        "prompt_token_count": prompt_token_count,
        "completions": [candidate.text for candidate in candidates],
        "completion_token_counts": [len(candidate.token_ids) for candidate in candidates],
        "finish_reasons": [str(candidate.finish_reason) for candidate in candidates],
        "rewards": [float(result["score"]) for result in reward_results],
        "reward_results": reward_results,
        "sampling": sampling_metadata(args.seed, args.logprobs),
        "inference": {
            "engine": "vllm",
            "dtype": "bfloat16",
            "quantization": None,
            "continuous_batching": True,
            "chunked_prefill": True,
            "prefix_caching": True,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "submission_batch_size": args.submission_batch_size,
        },
        "judge": {
            "model": JUDGE_MODEL,
            "backend": JUDGE_BACKEND,
            "thinking": JUDGE_THINKING,
        },
    }
    if args.logprobs > 0:
        record["completion_logprob_summaries"] = [
            summarize_completion_logprobs(candidate) for candidate in candidates
        ]
    return record


def main() -> int:
    program_started = time.monotonic()
    args = parse_args()
    validate_args(args)

    # Match train_codenames_dapo.sh: load repository-local secrets first and
    # keep explicitly exported environment variables authoritative.
    load_dotenv_if_present(REPO_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError(
            "OPENROUTER_API_KEY is required for the Gemini clue-task judge. "
            "Export it or add it to the repository's .env file."
        )
    os.environ.setdefault("JUDGE_CONCURRENCY", "128")
    os.environ.setdefault("JUDGE_TEMPERATURE", "0.0")
    os.environ.setdefault("JUDGE_MAX_TOKENS", "1024")
    os.environ.setdefault("JUDGE_TIMEOUT_S", "300")
    os.environ.setdefault(
        "RCLONE_CHECKPOINT_DIR",
        str(REPO_ROOT / "checkpoints" / "rclone_downloads"),
    )

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.checkpoint_utils import checkpoint_short_name, resolve_checkpoint_path

    args.model_label = args.model_label or checkpoint_short_name(args.model)
    log_progress(
        f"Starting checkpoint {args.model_label} | output={args.output} | "
        f"seed={args.seed} | logprobs={args.logprobs} | "
        f"batch_size={args.submission_batch_size}"
    )

    # Load only the parquet dependency before checkpoint resolution. In
    # particular, vLLM is not imported until any rclone copy has completed.
    import pyarrow.parquet as pq

    # Import only after judge environment variables have been set; the reward
    # client's configuration is read at import time.
    from custom_reward_functions.codenames_reward import compute_score

    rows = pq.read_table(VAL_PARQUET).to_pylist()
    validate_rows(rows)
    log_progress(f"Validated all {len(rows)} prompts from {VAL_PARQUET}")

    if args.output.exists() and not args.overwrite:
        completed = read_completed_indices(
            args.output,
            model=args.model,
            revision=args.revision,
            seed=args.seed,
            logprobs=args.logprobs,
        )
        log_progress(
            f"Resume check: {len(completed)}/{EXPECTED_PROMPTS} prompts already complete "
            f"in {args.output}"
        )
    else:
        completed = set()

    invalid_completed = completed - set(range(EXPECTED_PROMPTS))
    if invalid_completed:
        raise ValueError(f"Output contains invalid prompt indices: {sorted(invalid_completed)}")
    pending_indices = [index for index in range(EXPECTED_PROMPTS) if index not in completed]
    if not pending_indices:
        log_progress(f"All {EXPECTED_PROMPTS} prompts are already complete; nothing to do")
        return 0

    resolve_started = time.monotonic()
    log_progress(f"Resolving model/checkpoint: {args.model}")
    resolved_model = resolve_checkpoint_path(
        args.model,
        download_dir=(
            str(args.checkpoint_download_dir)
            if args.checkpoint_download_dir is not None
            else None
        ),
        force_download=args.force_checkpoint_download,
    )
    log_progress(
        f"Model ready in {format_duration(time.monotonic() - resolve_started)}: "
        f"{resolved_model}"
    )

    args.tensor_parallel_size = resolve_tensor_parallel_size(args.tensor_parallel_size)
    log_progress(
        f"Using selected GPUs with tensor_parallel_size={args.tensor_parallel_size}"
    )

    # GPU/model imports happen only after the checkpoint is available locally.
    # The default VERL image already provides these dependencies.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.revision is not None:
        tokenizer_kwargs["revision"] = args.revision
    tokenizer_started = time.monotonic()
    log_progress("Loading tokenizer and rendering all prompts")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, **tokenizer_kwargs)
    prompt_token_ids = render_prompt_token_ids(tokenizer, rows)
    log_progress(
        f"Rendered prompts: max={max(map(len, prompt_token_ids))} tokens "
        f"(training limit={MAX_PROMPT_TOKENS}) in "
        f"{format_duration(time.monotonic() - tokenizer_started)}"
    )

    llm_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "tokenizer": resolved_model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "bfloat16",
        "quantization": None,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": MAX_MODEL_LEN,
        "enforce_eager": False,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "seed": args.seed,
    }
    if args.revision is not None:
        llm_kwargs["revision"] = args.revision
        llm_kwargs["tokenizer_revision"] = args.revision
    model_load_started = time.monotonic()
    log_progress("Loading model into vLLM; vLLM will print its own loading progress")
    llm = LLM(**llm_kwargs)
    log_progress(
        f"vLLM model load finished in "
        f"{format_duration(time.monotonic() - model_load_started)}"
    )
    sampling = SamplingParams(
        n=RESPONSES_PER_PROMPT,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_RESPONSE_TOKENS,
        seed=args.seed,
        logprobs=args.logprobs or None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "w" if args.overwrite or not args.output.exists() else "a"
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    processing_started = time.monotonic()
    pending_total = len(pending_indices)
    total_batches = math.ceil(pending_total / args.submission_batch_size)
    log_progress(
        f"Beginning rollout work: {pending_total} pending prompts x "
        f"{RESPONSES_PER_PROMPT} responses in {total_batches} batches; "
        f"{len(completed)}/{EXPECTED_PROMPTS} prompts were already complete"
    )
    try:
        with args.output.open(output_mode, encoding="utf-8") as output_handle:
            for batch_start in range(0, len(pending_indices), args.submission_batch_size):
                batch_number = batch_start // args.submission_batch_size + 1
                batch_started = time.monotonic()
                indices = pending_indices[batch_start : batch_start + args.submission_batch_size]
                batch_rows = [rows[index] for index in indices]
                prompts = [
                    TokensPrompt(prompt_token_ids=prompt_token_ids[index]) for index in indices
                ]
                log_progress(
                    f"Batch {batch_number}/{total_batches} generation started | "
                    f"dataset prompt indices {indices[0]}..{indices[-1]} | "
                    f"{len(indices)} prompts x {RESPONSES_PER_PROMPT} responses"
                )
                generation_started = time.monotonic()
                request_outputs = llm.generate(prompts, sampling, use_tqdm=True)
                generation_elapsed = time.monotonic() - generation_started
                for index, request_output in zip(indices, request_outputs, strict=True):
                    if len(request_output.outputs) != RESPONSES_PER_PROMPT:
                        raise RuntimeError(
                            f"Prompt {index}: vLLM returned {len(request_output.outputs)} "
                            f"responses; expected {RESPONSES_PER_PROMPT}"
                        )

                generated_tokens = sum(
                    len(candidate.token_ids)
                    for request_output in request_outputs
                    for candidate in request_output.outputs
                )
                token_rate = generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
                log_progress(
                    f"Batch {batch_number}/{total_batches} generation finished in "
                    f"{format_duration(generation_elapsed)} | "
                    f"{generated_tokens:,} output tokens | {token_rate:,.1f} tokens/s"
                )
                score_count = len(indices) * RESPONSES_PER_PROMPT
                scoring_started = time.monotonic()
                log_progress(
                    f"Batch {batch_number}/{total_batches} reward scoring started | "
                    f"{score_count:,} responses"
                )
                grouped_rewards = event_loop.run_until_complete(
                    score_batch(
                        batch_rows,
                        request_outputs,
                        compute_score,
                        progress_context=f"Batch {batch_number}/{total_batches} scoring",
                    )
                )
                log_progress(
                    f"Batch {batch_number}/{total_batches} reward scoring finished in "
                    f"{format_duration(time.monotonic() - scoring_started)}"
                )

                log_progress(
                    f"Batch {batch_number}/{total_batches} summarizing logprobs and saving "
                    f"{len(indices)} prompt records"
                )
                for index, row, request_output, reward_results in zip(
                    indices,
                    batch_rows,
                    request_outputs,
                    grouped_rewards,
                    strict=True,
                ):
                    record = build_output_record(
                        index=index,
                        row=row,
                        request_output=request_output,
                        reward_results=reward_results,
                        args=args,
                        prompt_token_count=len(prompt_token_ids[index]),
                        resolved_model=resolved_model,
                    )
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())

                complete_now = len(completed) + min(
                    batch_start + len(indices), len(pending_indices)
                )
                run_done = min(batch_start + len(indices), pending_total)
                run_elapsed = time.monotonic() - processing_started
                log_progress(
                    f"Batch {batch_number}/{total_batches} saved in "
                    f"{format_duration(time.monotonic() - batch_started)} | "
                    f"overall dataset {complete_now}/{EXPECTED_PROMPTS} prompts "
                    f"({complete_now * RESPONSES_PER_PROMPT:,} responses) | "
                    f"this run: {progress_summary(run_done, pending_total, run_elapsed, 'prompts')}"
                )
    finally:
        event_loop.close()
        asyncio.set_event_loop(None)

    log_progress(
        f"Complete: {EXPECTED_PROMPTS} prompts x {RESPONSES_PER_PROMPT} responses "
        f"= {EXPECTED_PROMPTS * RESPONSES_PER_PROMPT:,} scored pairs in {args.output} | "
        f"rollout/scoring time {format_duration(time.monotonic() - processing_started)} | "
        f"total wall time {format_duration(time.monotonic() - program_started)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
