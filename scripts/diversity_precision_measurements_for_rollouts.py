#!/usr/bin/env python3
"""Collect diversity, precision, semantic-path, and decoding metrics for rollouts.

The input is the prompt-level JSONL written by
``scripts/generate_codenames_validation_rollouts.py``: each record contains
``metadata`` and an aligned list of ``completions``. By default, exactly 32
completions are required per prompt.

Guess-task rewards and precision are reconstructed locally with the existing
Codenames parser and reward rules. Clue-task evaluation needs the guesses made
by the judge, so each clue record must also contain aligned reward information
in one of these forms::

    "reward_results": [
        {"score": 0.5, "judge_guesses": "WORD_A,WORD_B",
         "parse_fail": 0, "format_fail": 0, "judge_fail": 0},
        ...
    ]

or aligned top-level ``rewards`` and ``judge_guesses`` lists. A judge's full
text can instead be supplied in ``judge_outputs`` and will be parsed locally.
The script never calls a generation model, judge, or API. Semantic candidate
clustering optionally loads a sentence-transformers embedding model (MiniLM by
default); Hugging Face may download it once when it is not already cached. Use
``--disable-semantic-clustering`` for a fully offline, model-free run.

One ``<output-prefix>.csv`` file is written with one row per rollout. Prompt-
level metrics are repeated across the aligned rollout rows so that both sample-
and group-level metrics remain available in a single rectangular table. Raw
prompt/messages and response strings are not written.

The original MTLD-MA, word-length, and compression metrics analyze responses
exactly as stored. Additional metrics strip only the fixed XML/Codenames markup
and calculate compression over fixed 256- and 512-word prefixes, permitting
length-controlled comparisons. Candidate-clue metrics use a conservative,
auditable rule-based extractor over the reasoning trace; semantic clustering
merges extracted candidates with similar sentence-transformer embeddings.
Numeric ``reward_results`` values are exported under their original keys.

If a newly generated JSONL contains aligned ``completion_logprob_summaries``,
their sampled-token likelihood and top-k entropy lower-bound fields are copied
into the CSV. Old JSONLs do not contain logits, so those columns are null.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from diversity import compression_ratio
from lexical_diversity import lex_div as ld


# Make the repository package imports work when this file is run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_reward_functions.format_reward import (  # noqa: E402
    clue_format_scores,
    guess_format_scores,
    is_clue_format_ok,
    is_guess_format_ok,
)
from custom_reward_functions.parsers import parse_clue, parse_guesses  # noqa: E402
from custom_reward_functions.task_reward import task_reward  # noqa: E402


REWARD_MIN = -2.0
REWARD_MAX = 1.0
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.70
DEFAULT_LENGTH_CONTROL_WORDS = (256, 512)
CANDIDATE_EXTRACTION_METHOD = "rule_based_clue_candidates_v1"

WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
SELF_CORRECTION_RE = re.compile(
    r"\b(?:wait|actually|let me reconsider|reconsider|on second thought|"
    r"hold on|hmm,?\s*wait|but wait|let me recheck|let me re-examine|"
    r"i made a mistake|correcting myself|that's wrong|nope|not quite)\b",
    re.IGNORECASE,
)

# The format itself is intentionally retained in the original metrics. These
# patterns support additional controls that remove markup while retaining the
# response content.
TEMPLATE_MARKUP_RE = re.compile(
    r"</?(?:think|thinking|reasoning|reflection|adjustment|output)>|"
    r"\[(?:CODENAMES-(?:CLUE|GUESS)-(?:START|END)|Clue|Guesses|Selected-Targets)\]\s*:?",
    re.IGNORECASE,
)
FINAL_OUTPUT_SECTION_RE = re.compile(
    r"<output\b[^>]*>.*?</output>|"
    r"\[CODENAMES-(?:CLUE|GUESS)-START\].*?\[CODENAMES-(?:CLUE|GUESS)-END\]",
    re.IGNORECASE | re.DOTALL,
)
QUOTE_WORD_RE = re.compile(r"[\"'`‘’“”]([A-Za-z][A-Za-z'’-]{1,39})[\"'`‘’“”]")
CANDIDATE_CUE_RE = re.compile(
    r"\b(?:clue|candidate|option|choice|possibility|alternative|"
    r"what\s+about|how\s+about|maybe|perhaps|consider|try|choose|"
    r"select|pick|use)\b",
    re.IGNORECASE,
)
CANDIDATE_LIST_RE = re.compile(
    r"\b(?:"
    r"(?:possible|potential|candidate|alternative)\s+(?:clues|words|options|candidates)|"
    r"(?:clues|candidates|options|alternatives)"
    r")\s*(?:are|include|could\s+be|might\s+be|:)\s*"
    r"([^.!?\n;]{1,300})",
    re.IGNORECASE,
)
SYNONYM_LIST_RE = re.compile(
    r"\b(?:synonyms|words\s+related\s+to|related\s+terms)"
    r"(?:\s+(?:for|of)\s+[A-Za-z'’-]+)?\s*(?:are|include|could\s+be|might\s+be|:)\s*"
    r"([^.!?\n;]{1,300})",
    re.IGNORECASE,
)
CANDIDATE_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "anything",
    "association",
    "be",
    "candidate",
    "can",
    "choice",
    "clue",
    "connection",
    "could",
    "english",
    "example",
    "good",
    "invalid",
    "is",
    "it",
    "maybe",
    "might",
    "not",
    "one",
    "option",
    "perhaps",
    "possible",
    "should",
    "something",
    "target",
    "targets",
    "the",
    "this",
    "would",
    "valid",
    "word",
    "words",
}

LOGPROB_SUMMARY_FIELDS = (
    "logprob_token_count",
    "sampled_token_logprob_mean",
    "sampled_token_nll_mean",
    "sampled_token_rank_mean",
    "top1_probability_mean",
    "top1_top2_margin_mean",
    "topk_probability_mass_mean",
    "topk_entropy_lower_bound_mean",
    "first_256_sampled_token_nll_mean",
    "first_256_topk_entropy_lower_bound_mean",
    "last_128_sampled_token_nll_mean",
    "last_128_topk_entropy_lower_bound_mean",
)
MISSING = object()


def lexical_words(text: str) -> list[str]:
    """Return case-folded lexical words without removing tags or boilerplate."""
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def mtld_ma(words: list[str]) -> float | None:
    """Return lexical-diversity's wrapped moving-average MTLD score."""
    if len(words) < 2:
        return None
    return round(float(ld.mtld_ma_wrap(words)), 6)


def strip_response_template(text: str) -> str:
    """Remove fixed markup but retain all substantive response text."""
    return re.sub(r"\s+", " ", TEMPLATE_MARKUP_RE.sub(" ", text)).strip()


def reasoning_without_final_output(text: str) -> str:
    """Remove final-answer blocks before extracting brainstormed candidates."""
    return FINAL_OUTPUT_SECTION_RE.sub(" ", text)


def canonical_candidate(value: str, excluded_words: set[str]) -> str | None:
    """Return a conservative, single-token candidate or ``None``."""
    candidate = value.strip().strip("\"'`‘’“”.,:;!?()[]{}").casefold()
    candidate = candidate.replace("’", "'")
    if not candidate or candidate in excluded_words or candidate in CANDIDATE_STOPWORDS:
        return None
    if not re.fullmatch(r"[a-z][a-z'-]{1,39}", candidate):
        return None
    return candidate


def candidates_from_list_fragment(fragment: str, excluded_words: set[str]) -> list[str]:
    """Extract explicit one-word items from a comma/or/and-separated list."""
    candidates: list[str] = []
    for part in re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", fragment, flags=re.IGNORECASE):
        cleaned = re.sub(r"\([^)]*\)", " ", part)
        cleaned = re.sub(
            r"^(?:maybe|perhaps|possibly|the\s+(?:word|clue)\s+)",
            "",
            cleaned.strip(),
            flags=re.IGNORECASE,
        )
        match = re.fullmatch(
            r"[\"'`‘’“”]?([A-Za-z][A-Za-z'’-]{1,39})"
            r"[\"'`‘’“”]?",
            cleaned.strip(),
        )
        if match:
            candidate = canonical_candidate(match.group(1), excluded_words)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def extract_reasoning_clue_candidates(
    response: str,
    board_words: Iterable[str],
) -> list[str]:
    """Extract explicitly proposed clue words from the non-final reasoning.

    This is deliberately high precision rather than exhaustive. It recognizes
    quoted words in candidate-bearing sentences and explicit candidate or
    synonym lists. Each canonical candidate is counted at most once per trace.
    """
    text = reasoning_without_final_output(response)
    excluded = normalize_words(board_words)
    found: list[str] = []

    # Quoted single words are accepted only when their local sentence contains
    # candidate language. This avoids treating every quoted board word as a
    # proposed clue.
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not CANDIDATE_CUE_RE.search(sentence):
            continue
        for match in QUOTE_WORD_RE.finditer(sentence):
            candidate = canonical_candidate(match.group(1), excluded)
            if candidate is not None:
                found.append(candidate)

    for pattern in (CANDIDATE_LIST_RE, SYNONYM_LIST_RE):
        for match in pattern.finditer(text):
            found.extend(candidates_from_list_fragment(match.group(1), excluded))

    # Preserve proposal order while ensuring repeated discussion of one clue
    # does not dominate the group distribution.
    return list(dict.fromkeys(found))


def distribution_metrics(values: Iterable[str]) -> dict[str, float | int | None]:
    """Return entropy/effective-count summaries for categorical values."""
    counts = Counter(value for value in values if value)
    total = sum(counts.values())
    if total == 0:
        return {
            "count": 0,
            "entropy": None,
            "normalized_entropy": None,
            "effective_count": None,
        }
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    return {
        "count": len(counts),
        "entropy": entropy,
        "normalized_entropy": normalized,
        "effective_count": math.exp(entropy),
    }


class SemanticCandidateEmbedder:
    """Lazy sentence-transformer embedder with an in-process word cache."""

    def __init__(self, model_name: str, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise RuntimeError(
                "sentence-transformers is required for semantic clustering; "
                "run in env_eval or pass --disable-semantic-clustering"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: dict[str, Any] = {}

    def encode(self, candidates: Sequence[str]) -> Any:
        import numpy as np

        missing = [candidate for candidate in candidates if candidate not in self.cache]
        if missing:
            vectors = self.model.encode(
                [f"Codenames clue: {candidate}" for candidate in missing],
                batch_size=min(128, len(missing)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for candidate, vector in zip(missing, vectors, strict=True):
                self.cache[candidate] = vector
        if not candidates:
            return np.empty((0, 0), dtype=float)
        return np.stack([self.cache[candidate] for candidate in candidates])


def semantic_candidate_metrics(
    candidate_lists: Sequence[Sequence[str]],
    final_answers: Sequence[str | None],
    rewards: Sequence[float | None],
    embedder: SemanticCandidateEmbedder | None,
    similarity_threshold: float,
) -> dict[str, Any]:
    """Cluster candidate clues and summarize their cross-rollout distribution."""
    mentions = [candidate for candidates in candidate_lists for candidate in candidates]
    exact = distribution_metrics(mentions)
    result: dict[str, Any] = {
        "candidate_mentions_total": len(mentions),
        "candidate_unique_count": exact["count"],
        "candidate_exact_entropy": exact["entropy"],
        "candidate_exact_normalized_entropy": exact["normalized_entropy"],
        "candidate_exact_effective_count": exact["effective_count"],
        "candidate_semantic_cluster_count": None,
        "candidate_semantic_entropy": None,
        "candidate_semantic_normalized_entropy": None,
        "candidate_semantic_effective_count": None,
        "candidate_semantic_pairwise_distance": None,
        "candidate_final_semantic_cluster_count": None,
        "candidate_semantic_pruning_rate": None,
        "candidate_successful_semantic_cluster_rate": None,
    }
    unique = sorted(set(mentions))
    if embedder is None or not unique:
        return result

    import numpy as np

    vectors = embedder.encode(unique)
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)

    if len(unique) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        from sklearn.cluster import AgglomerativeClustering

        # Average linkage avoids the transitive-chain over-merging produced by
        # simple connected components while retaining a single interpretable
        # cosine-similarity threshold.
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - similarity_threshold,
        ).fit_predict(vectors)

    candidate_to_cluster = {
        candidate: int(labels[index]) for index, candidate in enumerate(unique)
    }
    cluster_mentions = [str(candidate_to_cluster[candidate]) for candidate in mentions]
    clustered = distribution_metrics(cluster_mentions)
    if len(unique) > 1:
        upper = similarities[np.triu_indices(len(unique), k=1)]
        pairwise_distance = float(np.mean(1.0 - upper))
    else:
        pairwise_distance = 0.0

    final_clusters = {
        candidate_to_cluster[answer]
        for answer in final_answers
        if answer is not None and answer in candidate_to_cluster
    }
    all_clusters = set(candidate_to_cluster.values())
    successful_clusters: set[int] = set()
    for candidates, reward in zip(candidate_lists, rewards, strict=True):
        if reward is None or reward <= 0:
            continue
        successful_clusters.update(candidate_to_cluster[candidate] for candidate in candidates)

    result.update(
        {
            "candidate_semantic_cluster_count": clustered["count"],
            "candidate_semantic_entropy": clustered["entropy"],
            "candidate_semantic_normalized_entropy": clustered["normalized_entropy"],
            "candidate_semantic_effective_count": clustered["effective_count"],
            "candidate_semantic_pairwise_distance": pairwise_distance,
            "candidate_final_semantic_cluster_count": len(final_clusters),
            "candidate_semantic_pruning_rate": (
                1.0 - len(final_clusters) / len(all_clusters) if all_clusters else None
            ),
            "candidate_successful_semantic_cluster_rate": (
                len(successful_clusters) / len(all_clusters) if all_clusters else None
            ),
        }
    )
    return result


def normalize_words(words: Iterable[Any]) -> set[str]:
    return {str(word).strip().casefold() for word in words if str(word).strip()}


def as_word_list(value: Any) -> list[str]:
    """Accept a JSON list or the comma-separated form used by reward results."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip().strip("[]").strip()
    if not text:
        return []
    parts = text.split(",") if "," in text else text.split()
    return [part.strip().strip("[]").strip() for part in parts if part.strip()]


def mean_or_none(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def fraction_true(values: Iterable[bool | None]) -> float | None:
    present = [bool(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def finite_population_max_variance(n: int) -> float | None:
    """Maximum population variance for n values bounded by [-2, 1]."""
    if n <= 0:
        return None
    lower_half = n // 2
    upper_half = n - lower_half
    reward_range = REWARD_MAX - REWARD_MIN
    return reward_range**2 * lower_half * upper_half / n**2


def group_compression(responses: list[str]) -> dict[str, float]:
    """Return diversity's gzip ratio over every response in one prompt group."""
    return {"compression_ratio": compression_ratio(responses, algorithm="gzip")}


def controlled_compression_metrics(
    responses: Sequence[str],
    word_budgets: Sequence[int],
) -> dict[str, Any]:
    """Return template-stripped and fixed-prefix compression controls."""
    stripped = [strip_response_template(response) for response in responses]
    word_lists = [lexical_words(response) for response in stripped]
    metrics: dict[str, Any] = {
        "template_stripped_compression_ratio": compression_ratio(
            stripped, algorithm="gzip"
        ),
        "template_stripped_response_words_mean": mean_or_none(
            len(words) for words in word_lists
        ),
    }
    for budget in word_budgets:
        eligible = [words for words in word_lists if len(words) >= budget]
        prefix = f"length_controlled_{budget}w"
        metrics[f"{prefix}_n_responses"] = len(eligible)
        metrics[f"{prefix}_complete"] = len(eligible) == len(word_lists)
        # Only a complete group is comparable: using fewer responses would
        # confound compression with the number of sampled trajectories.
        metrics[f"{prefix}_compression_ratio"] = (
            compression_ratio(
                [" ".join(words[:budget]) for words in eligible],
                algorithm="gzip",
            )
            if len(eligible) == len(word_lists) and eligible
            else None
        )
    return metrics


def logprob_metrics_from_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Return aligned generation-time logprob summaries, or null fields."""
    summary = aligned_value(record, "completion_logprob_summaries", index)
    if not isinstance(summary, dict):
        return {field: None for field in LOGPROB_SUMMARY_FIELDS}
    return {
        field: summary.get(field) if isinstance(summary.get(field), (int, float)) else None
        for field in LOGPROB_SUMMARY_FIELDS
    }


def aligned_value(record: dict[str, Any], key: str, index: int) -> Any:
    value = record.get(key, MISSING)
    if isinstance(value, list) and index < len(value):
        return value[index]
    return MISSING


def reward_payload(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Merge supported aligned reward/judge fields for one rollout."""
    payload: dict[str, Any] = {}
    detailed = aligned_value(record, "reward_results", index)
    if isinstance(detailed, dict):
        payload.update(detailed)
    elif detailed is not MISSING and isinstance(detailed, (int, float)):
        payload["score"] = detailed

    reward = aligned_value(record, "rewards", index)
    if isinstance(reward, dict):
        payload.update(reward)
    elif reward is not MISSING:
        payload.setdefault("score", reward)

    for key in ("judge_guesses", "judge_outputs", "judge_fail"):
        value = aligned_value(record, key, index)
        if value is not MISSING:
            payload.setdefault(key, value)
    return payload


def numeric_reward(payload: dict[str, Any]) -> float | None:
    for key in ("score", "reward"):
        value = payload.get(key, MISSING)
        if value is MISSING or value is None:
            continue
        try:
            reward = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(reward):
            return reward
    return None


def raw_reward_values(payload: dict[str, Any]) -> dict[str, int | float]:
    """Keep numeric reward-result values under their original field names."""
    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def judge_guesses_from_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (was judge information supplied, parsed guesses)."""
    if "judge_guesses" in payload:
        return True, as_word_list(payload["judge_guesses"])
    if "judge_outputs" in payload:
        parsed = parse_guesses(str(payload["judge_outputs"] or ""))
        return True, parsed.guesses
    if "judge_output" in payload:
        parsed = parse_guesses(str(payload["judge_output"] or ""))
        return True, parsed.guesses
    return False, []


def infer_task(raw_task: Any) -> str:
    task = str(raw_task or "").casefold()
    if "clue" in task or "spymaster" in task:
        return "clue"
    if "guess" in task or "operative" in task:
        return "guess"
    raise ValueError(f"Cannot identify Codenames task from metadata.task={raw_task!r}")


def infer_model_size(model: Any) -> str | None:
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])", str(model or ""))
    return f"{match.group(1)}B" if match else None


def infer_checkpoint_step(model: Any) -> int | None:
    matches = re.findall(r"(?:global[_-]?step|step)[_-]?(\d+)", str(model or ""), re.IGNORECASE)
    return int(matches[-1]) if matches else None


def infer_training_run(model: Any) -> str | None:
    """Return the checkpoint run directory immediately before global_step."""
    text = str(model or "").rstrip("/")
    match = re.search(r"([^/]+)/global[_-]?step[_-]?\d+(?:/|$)", text, re.IGNORECASE)
    return match.group(1) if match else None


def precision_metrics(
    guesses: list[str],
    target_words: list[str],
    intended_targets: list[str],
) -> tuple[float, float, float | None]:
    guessed = normalize_words(guesses)
    targets = normalize_words(target_words)
    intended = normalize_words(intended_targets)
    correct = guessed & targets

    target_recall = len(correct) / len(targets) if targets else 0.0
    guess_precision = len(correct) / len(guessed) if guessed else 0.0
    intended_recall = len(guessed & intended) / len(intended) if intended else None
    return target_recall, guess_precision, intended_recall


def score_one_rollout(
    response: str,
    metadata: dict[str, Any],
    task: str,
    payload: dict[str, Any],
    allow_incomplete_clue_data: bool,
) -> dict[str, Any]:
    target_words = as_word_list(metadata.get("target_words"))
    non_target_words = as_word_list(metadata.get("non_target_words"))
    all_words = as_word_list(metadata.get("all_words")) or target_words + non_target_words

    words = lexical_words(response)
    stripped_words = lexical_words(strip_response_template(response))
    self_corrections = list(SELF_CORRECTION_RE.finditer(response))
    base: dict[str, Any] = {
        "response_words": len(words),
        "mtld_ma": mtld_ma(words),
        "template_stripped_response_words": len(stripped_words),
        "template_stripped_mtld_ma": mtld_ma(stripped_words),
        "self_correction_count": len(self_corrections),
        "self_correction_present": bool(self_corrections),
        "self_corrections_per_1000_words": (
            1000.0 * len(self_corrections) / len(words) if words else 0.0
        ),
        **raw_reward_values(payload),
    }

    supplied_reward = numeric_reward(payload)
    payload_parse_fail = payload.get("parse_fail", MISSING)
    payload_format_fail = payload.get("format_fail", MISSING)
    payload_judge_fail = payload.get("judge_fail", MISSING)

    if task == "guess":
        parsed = parse_guesses(response)
        max_guesses = metadata.get("num_max_guesses", metadata.get("max_guesses"))
        format_scores = guess_format_scores(parsed, all_words, max_guesses=max_guesses)
        format_valid = is_guess_format_ok(format_scores)
        local_reward = task_reward(parsed.guesses, target_words, non_target_words)["task"]
        reward = supplied_reward if supplied_reward is not None else (-1.0 if not format_valid else local_reward)
        intended_targets = as_word_list(metadata.get("selected_target_words"))
        if format_valid:
            target_recall, guess_precision, intended_recall = precision_metrics(
                parsed.guesses, target_words, intended_targets
            )
            final_answer_key = ",".join(sorted(normalize_words(parsed.guesses)))
        else:
            target_recall = guess_precision = intended_recall = 0.0
            final_answer_key = None

        base.update(
            {
                "reward": reward,
                "target_recall": target_recall,
                "guess_precision": guess_precision,
                "intended_subset_recall": intended_recall,
                "format_valid": format_valid,
                "parse_failed": bool(payload_parse_fail) if payload_parse_fail is not MISSING else not parsed.tags_present,
                "judge_failed": False,
                "final_answer_key": final_answer_key,
                "candidate_extraction_method": None,
                "reasoning_candidate_clues": [],
                "reasoning_candidate_count": 0,
                "candidate_clues": [],
                "candidate_clue_count": 0,
            }
        )
        return base

    parsed = parse_clue(response)
    format_scores = clue_format_scores(parsed, target_words, non_target_words)
    format_valid = is_clue_format_ok(format_scores)
    reward = supplied_reward if supplied_reward is not None else (-1.0 if not format_valid else None)
    judge_failed = bool(payload_judge_fail) if payload_judge_fail is not MISSING else None
    has_judge_data, guesses = judge_guesses_from_payload(payload)

    if not format_valid:
        target_recall = guess_precision = intended_recall = 0.0
        judge_failed = False if judge_failed is None else judge_failed
        final_answer_key = None
    elif judge_failed:
        target_recall = guess_precision = intended_recall = None
        final_answer_key = parsed.clue.strip().casefold() or None
    elif has_judge_data:
        target_recall, guess_precision, intended_recall = precision_metrics(
            guesses, target_words, parsed.selected_targets
        )
        judge_failed = False if judge_failed is None else judge_failed
        final_answer_key = parsed.clue.strip().casefold() or None
    else:
        target_recall = guess_precision = intended_recall = None
        final_answer_key = parsed.clue.strip().casefold() or None

    reasoning_candidates = extract_reasoning_clue_candidates(response, all_words)
    candidate_clues = list(reasoning_candidates)
    final_candidate = canonical_candidate(parsed.clue, normalize_words(all_words))
    if final_candidate is not None and final_candidate not in candidate_clues:
        candidate_clues.append(final_candidate)

    missing: list[str] = []
    if reward is None:
        missing.append("reward")
    if format_valid and not judge_failed and not has_judge_data:
        missing.append("judge guesses")
    if missing and not allow_incomplete_clue_data:
        raise ValueError(
            "A format-valid clue rollout is missing "
            + " and ".join(missing)
            + ". Add aligned reward_results (recommended), or use aligned rewards "
            "and judge_guesses/judge_outputs."
        )

    base.update(
        {
            "reward": reward,
            "target_recall": target_recall,
            "guess_precision": guess_precision,
            "intended_subset_recall": intended_recall,
            "format_valid": format_valid if payload_format_fail is MISSING else not bool(payload_format_fail),
            "parse_failed": bool(payload_parse_fail) if payload_parse_fail is not MISSING else not parsed.tags_present,
            "judge_failed": judge_failed,
            "final_answer_key": final_answer_key,
            "candidate_extraction_method": CANDIDATE_EXTRACTION_METHOD,
            "reasoning_candidate_clues": reasoning_candidates,
            "reasoning_candidate_count": len(reasoning_candidates),
            "candidate_clues": candidate_clues,
            "candidate_clue_count": len(candidate_clues),
        }
    )
    return base


def analyze_record(
    record: dict[str, Any],
    expected_rollouts: int,
    model_size_override: str | None,
    checkpoint_step_override: int | None,
    allow_incomplete_clue_data: bool,
    semantic_embedder: SemanticCandidateEmbedder | None,
    semantic_similarity_threshold: float,
    length_control_words: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = record.get("metadata") or {}
    responses = record.get("completions")
    if not isinstance(responses, list):
        raise ValueError("Record has no list-valued 'completions' field")
    if len(responses) != expected_rollouts:
        raise ValueError(f"Expected {expected_rollouts} completions, found {len(responses)}")

    prompt_id = metadata.get("row_id", record.get("idx"))
    task = infer_task(metadata.get("task"))
    model = record.get("model")
    model_label = record.get("model_label")
    checkpoint_identity = model or model_label
    model_size = model_size_override or infer_model_size(checkpoint_identity)
    checkpoint_step = (
        checkpoint_step_override
        if checkpoint_step_override is not None
        else infer_checkpoint_step(checkpoint_identity)
    )
    finish_reasons = record.get("finish_reasons") or []
    sampling = record.get("sampling") if isinstance(record.get("sampling"), dict) else {}
    sampling_seed = sampling.get("seed")
    sampling_temperature = sampling.get("temperature")
    sampling_top_p = sampling.get("top_p")
    sampling_top_k = sampling.get("top_k")

    identifiers = {
        "prompt_id": prompt_id,
        "model": model,
        "model_label": model_label,
        "model_revision": record.get("revision"),
        "model_size": model_size,
        "training_run": infer_training_run(checkpoint_identity),
        "training_stage": "checkpoint" if checkpoint_step is not None else "base",
        "checkpoint_step": checkpoint_step,
        "task_kind": task,
        "difficulty_rule": metadata.get("difficulty_rule"),
        "difficulty_score": metadata.get("difficulty_score"),
        "sampling_seed": sampling_seed,
        "sampling_temperature": sampling_temperature,
        "sampling_top_p": sampling_top_p,
        "sampling_top_k": sampling_top_k,
        "sampling_max_tokens": sampling.get("max_tokens"),
        "sampling_logprobs_k": sampling.get("logprobs"),
    }

    sample_rows: list[dict[str, Any]] = []
    for index, raw_response in enumerate(responses):
        response = str(raw_response or "")
        try:
            metrics = score_one_rollout(
                response=response,
                metadata=metadata,
                task=task,
                payload=reward_payload(record, index),
                allow_incomplete_clue_data=allow_incomplete_clue_data,
            )
        except ValueError as exc:
            raise ValueError(f"prompt_id={prompt_id!r}, rollout_index={index}: {exc}") from exc

        finish_reason = finish_reasons[index] if index < len(finish_reasons) else None
        sample_rows.append(
            {
                **identifiers,
                "rollout_index": index,
                **metrics,
                **logprob_metrics_from_record(record, index),
                "truncated": str(finish_reason or "").casefold() in {"length", "max_tokens"},
            }
        )

    rewards = [row["reward"] for row in sample_rows]
    present_rewards = [float(value) for value in rewards if value is not None]
    reward_complete = len(present_rewards) == len(sample_rows)
    valid_answer_keys = [
        row["final_answer_key"] for row in sample_rows if row["final_answer_key"] is not None
    ]
    self_correction_total = sum(row["self_correction_count"] for row in sample_rows)
    total_words = sum(row["response_words"] for row in sample_rows)
    final_distribution = distribution_metrics(valid_answer_keys)
    semantic_metrics = semantic_candidate_metrics(
        candidate_lists=[row["candidate_clues"] for row in sample_rows],
        final_answers=[row["final_answer_key"] for row in sample_rows],
        rewards=rewards,
        embedder=semantic_embedder if task == "clue" else None,
        similarity_threshold=semantic_similarity_threshold,
    )

    group_row: dict[str, Any] = {
        **identifiers,
        "n_rollouts": len(sample_rows),
        "rewards": rewards,
        "n_rewards_present": len(present_rewards),
        "reward_complete": reward_complete,
        "mean_reward": statistics.fmean(present_rewards) if present_rewards else None,
        "reward_variance": statistics.pvariance(present_rewards) if reward_complete else None,
        "maximum_possible_reward_variance": finite_population_max_variance(len(sample_rows)),
        "response_words_mean": mean_or_none(row["response_words"] for row in sample_rows),
        "mtld_ma_mean": mean_or_none(row["mtld_ma"] for row in sample_rows),
        "template_stripped_mtld_ma_mean": mean_or_none(
            row["template_stripped_mtld_ma"] for row in sample_rows
        ),
        "self_correction_prevalence": fraction_true(
            row["self_correction_present"] for row in sample_rows
        ),
        "self_corrections_per_1000_words": (
            1000.0 * self_correction_total / total_words if total_words else 0.0
        ),
        "target_recall_mean": mean_or_none(row["target_recall"] for row in sample_rows),
        "guess_precision_mean": mean_or_none(row["guess_precision"] for row in sample_rows),
        "intended_subset_recall_mean": mean_or_none(
            row["intended_subset_recall"] for row in sample_rows
        ),
        "unique_final_answer_rate": (
            len(set(valid_answer_keys)) / len(valid_answer_keys) if valid_answer_keys else None
        ),
        "final_answer_unique_count": final_distribution["count"],
        "final_answer_entropy": final_distribution["entropy"],
        "final_answer_normalized_entropy": final_distribution["normalized_entropy"],
        "final_answer_effective_count": final_distribution["effective_count"],
        "candidate_extraction_method": (
            CANDIDATE_EXTRACTION_METHOD if task == "clue" else None
        ),
        "candidate_embedding_model": (
            semantic_embedder.model_name if task == "clue" and semantic_embedder is not None else None
        ),
        "candidate_semantic_similarity_threshold": (
            semantic_similarity_threshold if task == "clue" and semantic_embedder is not None else None
        ),
        "candidate_response_coverage": fraction_true(
            row["candidate_clue_count"] > 0 for row in sample_rows
        ),
        "reasoning_candidate_response_coverage": fraction_true(
            row["reasoning_candidate_count"] > 0 for row in sample_rows
        ),
        "reasoning_candidate_count_mean": mean_or_none(
            row["reasoning_candidate_count"] for row in sample_rows
        ),
        "candidate_clue_count_mean": mean_or_none(
            row["candidate_clue_count"] for row in sample_rows
        ),
        **semantic_metrics,
        "format_valid_rate": fraction_true(row["format_valid"] for row in sample_rows),
        "parse_failure_rate": fraction_true(row["parse_failed"] for row in sample_rows),
        "judge_failure_rate": fraction_true(row["judge_failed"] for row in sample_rows),
        "truncation_rate": fraction_true(row["truncated"] for row in sample_rows),
        **group_compression([str(response or "") for response in responses]),
        **controlled_compression_metrics(
            [str(response or "") for response in responses], length_control_words
        ),
    }
    for field in LOGPROB_SUMMARY_FIELDS:
        group_row[f"{field}_group_mean"] = mean_or_none(
            row[field] for row in sample_rows
        )
    return sample_rows, group_row


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, record


def combined_csv_row(
    sample_row: dict[str, Any], group_row: dict[str, Any]
) -> dict[str, Any]:
    """Combine sample and prompt metrics, encoding nested values as JSON."""
    combined = dict(sample_row)
    combined.update(
        (key, value) for key, value in group_row.items() if key not in combined
    )
    return {
        key: json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, dict))
        else value
        for key, value in combined.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Prompt-level rollout JSONL")
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Prefix for the .csv output",
    )
    parser.add_argument(
        "--expected-rollouts",
        type=int,
        default=32,
        help="Required completions per prompt (default: 32)",
    )
    parser.add_argument("--model-size", help="Model-size label, e.g. 14B; inferred if omitted")
    parser.add_argument("--checkpoint-step", type=int, help="Checkpoint step; inferred if omitted")
    parser.add_argument(
        "--length-control-words",
        default=",".join(str(value) for value in DEFAULT_LENGTH_CONTROL_WORDS),
        help=(
            "Comma-separated per-response word budgets for fixed-prefix "
            "compression (default: 256,512)"
        ),
    )
    parser.add_argument(
        "--semantic-embedding-model",
        default=DEFAULT_SEMANTIC_MODEL,
        help=(
            "Sentence-transformers model/path for candidate clustering "
            f"(default: {DEFAULT_SEMANTIC_MODEL})"
        ),
    )
    parser.add_argument(
        "--semantic-device",
        default="cpu",
        help="Sentence-transformers device, e.g. cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--semantic-similarity-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
        help=(
            "Cosine threshold for merging candidate clues into one semantic "
            f"cluster (default: {DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--disable-semantic-clustering",
        action="store_true",
        help="Keep exact candidate metrics but skip sentence-transformer clustering",
    )
    parser.add_argument(
        "--allow-incomplete-clue-data",
        action="store_true",
        help="Write null clue metrics instead of failing when stored rewards/judge guesses are absent",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_rollouts <= 0:
        raise ValueError("--expected-rollouts must be positive")
    try:
        length_control_words = tuple(
            int(value.strip())
            for value in args.length_control_words.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--length-control-words must be comma-separated integers") from exc
    if not length_control_words or any(value <= 0 for value in length_control_words):
        raise ValueError("--length-control-words values must be positive")
    if len(set(length_control_words)) != len(length_control_words):
        raise ValueError("--length-control-words contains duplicates")
    if not -1.0 <= args.semantic_similarity_threshold <= 1.0:
        raise ValueError("--semantic-similarity-threshold must be in [-1, 1]")

    csv_output = Path(f"{args.output_prefix}.csv")
    if csv_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {csv_output}. Pass --overwrite to replace it."
        )

    semantic_embedder = None
    if not args.disable_semantic_clustering:
        print(
            f"Loading semantic embedding model {args.semantic_embedding_model!r} "
            f"on {args.semantic_device}",
            flush=True,
        )
        semantic_embedder = SemanticCandidateEmbedder(
            args.semantic_embedding_model,
            args.semantic_device,
        )

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    prompt_count = 0
    rollout_count = 0

    with csv_output.open("w", encoding="utf-8", newline="") as csv_handle:
        writer: csv.DictWriter[str] | None = None
        for line_number, record in read_jsonl(args.input):
            try:
                samples, group = analyze_record(
                    record=record,
                    expected_rollouts=args.expected_rollouts,
                    model_size_override=args.model_size,
                    checkpoint_step_override=args.checkpoint_step,
                    allow_incomplete_clue_data=args.allow_incomplete_clue_data,
                    semantic_embedder=semantic_embedder,
                    semantic_similarity_threshold=args.semantic_similarity_threshold,
                    length_control_words=length_control_words,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Failed to analyze {args.input}:{line_number}: {exc}") from exc

            for sample in samples:
                row = combined_csv_row(sample, group)
                if writer is None:
                    writer = csv.DictWriter(csv_handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            prompt_count += 1
            rollout_count += len(samples)

    print(
        f"Wrote {rollout_count} rollout rows from {prompt_count} prompt groups "
        f"to {csv_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
