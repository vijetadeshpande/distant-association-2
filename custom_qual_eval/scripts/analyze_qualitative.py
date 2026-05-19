"""Compute quantitative + qualitative metrics across the three Qwen3-8B checkpoints.

Outputs a metrics JSON and a structured findings JSON used by the report writer.
"""
import json
import re
import ast
from collections import Counter
from pathlib import Path

OUT_DIR = Path("custom_qual_eval/outputs")
FILES = {
    "base":  OUT_DIR / "qwen3-8b.jsonl",
    "s56":   OUT_DIR / "codenames-dapo-qwen3-8b-step56.jsonl",
    "s112":  OUT_DIR / "codenames-dapo-qwen3-8b-step112.jsonl",
}

SECTIONS = ["thinking", "reasoning", "reflection", "adjustment", "output"]
CLUE_TAGS = ["[CODENAMES-CLUE-START]", "[Clue]:", "[Selected-Targets]:", "[CODENAMES-CLUE-END]"]
GUESS_TAGS = ["[CODENAMES-GUESS-START]", "[Guesses]:", "[CODENAMES-GUESS-END]"]

CORRECTION_RE = re.compile(
    r"\b(wait|actually|let me reconsider|reconsider|on second thought|hold on|"
    r"hmm,? wait|but wait|let me recheck|let me re-examine|I made a mistake|"
    r"correcting myself|that's wrong|nope,?|not quite)\b",
    re.IGNORECASE,
)
ALTERNATIVE_RE = re.compile(
    r"\b(alternatively|another option|another candidate|or maybe|or perhaps|"
    r"another idea|how about|what about|could be|let me try)\b",
    re.IGNORECASE,
)
HEDGE_RE = re.compile(r"\b(maybe|might|perhaps|possibly|stretch|unsure|uncertain)\b", re.IGNORECASE)

# Candidate-mention patterns: words in quotes inside <think>
CAND_QUOTE_RE = re.compile(r'"([A-Za-z]{3,})"')


def parse_str_list(s):
    """Parse a numpy-printed list like \"['a' 'b' 'c']\" -> ['a','b','c']."""
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x).lower() for x in s]
    s = str(s).strip()
    if not s:
        return []
    # Replace numpy-style ' ' separators with commas
    s2 = re.sub(r"'\s+'", "','", s)
    try:
        return [str(x).lower() for x in ast.literal_eval(s2)]
    except Exception:
        # Fallback: extract bracketed tokens
        return [w.lower() for w in re.findall(r"[A-Za-z]+", s2)]


def split_think_and_after(text):
    """Split into pre-<think> garbage, <think>...</think>, and the remainder."""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if not m:
        return "", "", text
    return text[: m.start()], m.group(1), text[m.end():]


def find_section(text, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None


def extract_clue(output_text):
    """Returns (clue or None, selected_targets list or None)."""
    if not output_text:
        return None, None
    m_clue = re.search(r"\[Clue\]:\s*([^\n\r]*)", output_text)
    clue = m_clue.group(1).strip() if m_clue else None
    if clue:
        clue = clue.strip("`*[]() ").lower()
    m_tgt = re.search(r"\[Selected-Targets\]:\s*(.*?)(?:\[CODENAMES-CLUE-END\]|$)", output_text, re.S)
    sel = None
    if m_tgt:
        raw = m_tgt.group(1)
        # Strip brackets and commas
        sel = [w.lower() for w in re.findall(r"[A-Za-z]+", raw)]
    return clue, sel


def extract_guesses(output_text):
    if not output_text:
        return None
    m = re.search(r"\[Guesses\]:\s*(.*?)(?:\[CODENAMES-GUESS-END\]|$)", output_text, re.S)
    if not m:
        return None
    raw = m.group(1)
    return [w.lower() for w in re.findall(r"[A-Za-z]+", raw)]


def sections_in_order(text):
    """Return True iff the five sections appear in canonical order."""
    positions = []
    for name in SECTIONS:
        m = re.search(rf"<{name}>", text)
        if m is None:
            return False
        positions.append(m.start())
    return positions == sorted(positions)


def ngram_repeat_rate(text, n=8):
    """Fraction of n-grams that already appeared earlier in the same text."""
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < n + 1:
        return 0.0
    seen = set()
    repeats = 0
    total = 0
    for i in range(len(toks) - n + 1):
        gram = tuple(toks[i: i + n])
        total += 1
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / max(total, 1)


def candidates_considered(think_text):
    """Estimate how many distinct candidate words/options the think-trace floats."""
    quoted = set(w.lower() for w in CAND_QUOTE_RE.findall(think_text))
    # Also pull capitalized standalone words (>=4 chars)
    return len(quoted), quoted


def evaluate_clue_record(rec, completion):
    md = rec["metadata"]
    tgt = parse_str_list(md["target_words"])
    non = parse_str_list(md["non_target_words"])
    sel_truth = parse_str_list(md["selected_target_words"])

    _, think, post = split_think_and_after(completion)
    output_block = find_section(post, "output") or ""
    clue, sel = extract_clue(output_block)
    # Fall back: search the whole post if not in <output>
    if clue is None:
        clue, sel = extract_clue(post)

    metrics = {
        "task": "clue",
        "completion_chars": len(completion),
        "completion_words": len(completion.split()),
        "think_chars": len(think),
        "think_words": len(think.split()),
        "post_chars": len(post),
        "n_clue_tags_present": sum(t in completion for t in CLUE_TAGS),
        "sections_in_order": sections_in_order(completion),
        "n_corrections": len(CORRECTION_RE.findall(think)),
        "n_alternatives": len(ALTERNATIVE_RE.findall(think)),
        "n_hedges": len(HEDGE_RE.findall(think)),
        "repeat_rate_8gram": ngram_repeat_rate(think, 8),
        "n_targets": len(tgt),
        "n_non_targets": len(non),
        "clue": clue,
        "selected_targets": sel,
    }

    cand_count, cand_set = candidates_considered(think)
    metrics["n_candidate_clues"] = cand_count

    # ---- Validity checks ----
    clue_word = (clue or "").strip()
    metrics["clue_present"] = bool(clue_word)
    metrics["clue_is_single_word"] = bool(re.fullmatch(r"[A-Za-z]+", clue_word)) if clue_word else False
    metrics["clue_has_hyphen"] = "-" in (clue or "")
    metrics["clue_in_target_set"] = clue_word.lower() in tgt if clue_word else False
    metrics["clue_in_non_target_set"] = clue_word.lower() in non if clue_word else False
    metrics["clue_morphological_overlap"] = False
    if clue_word:
        cl = clue_word.lower()
        for w in tgt + non:
            wl = w.lower()
            if wl == cl:
                continue
            # Crude morphological-variant check
            if (cl.startswith(wl) or wl.startswith(cl) or cl.endswith(wl) or wl.endswith(cl)) \
                    and abs(len(cl) - len(wl)) <= 4 and min(len(cl), len(wl)) >= 4:
                metrics["clue_morphological_overlap"] = True
                metrics["clue_morphological_overlap_word"] = w
                break

    # ---- Selected targets validity ----
    sel = sel or []
    metrics["n_selected"] = len(sel)
    metrics["selected_all_in_target"] = all(s in tgt for s in sel) if sel else False
    metrics["selected_invalid_count"] = sum(1 for s in sel if s not in tgt)
    metrics["selected_full_set"] = (set(sel) == set(tgt)) and len(sel) == len(tgt)
    metrics["selected_match_ratio"] = (len(set(sel) & set(tgt)) / len(tgt)) if tgt else 0.0

    return metrics


def evaluate_guess_record(rec, completion):
    md = rec["metadata"]
    tgt = parse_str_list(md["target_words"])
    non = parse_str_list(md["non_target_words"])
    all_words = parse_str_list(md["all_words"]) or list(set(tgt + non))
    max_guesses = int(md["max_guesses"]) if md["max_guesses"] is not None else None

    _, think, post = split_think_and_after(completion)
    output_block = find_section(post, "output") or ""
    guesses = extract_guesses(output_block)
    if guesses is None:
        guesses = extract_guesses(post)
    guesses = guesses or []

    metrics = {
        "task": "guess",
        "completion_chars": len(completion),
        "completion_words": len(completion.split()),
        "think_chars": len(think),
        "think_words": len(think.split()),
        "post_chars": len(post),
        "sections_in_order": sections_in_order(completion),
        "n_corrections": len(CORRECTION_RE.findall(think)),
        "n_alternatives": len(ALTERNATIVE_RE.findall(think)),
        "n_hedges": len(HEDGE_RE.findall(think)),
        "repeat_rate_8gram": ngram_repeat_rate(think, 8),
        "n_targets": len(tgt),
        "n_non_targets": len(non),
        "max_guesses": max_guesses,
        "guesses": guesses,
        "n_guesses": len(guesses),
    }
    # Hacking signal: guessed more than allowed?
    if max_guesses is not None:
        metrics["guess_overflow"] = len(guesses) > max_guesses
    else:
        metrics["guess_overflow"] = False

    # Validity
    metrics["guesses_in_all_words"] = sum(g in all_words for g in guesses)
    metrics["guesses_invalid"] = [g for g in guesses if g not in all_words]
    metrics["n_correct"] = sum(g in tgt for g in guesses)
    metrics["n_wrong"]  = sum(g in non for g in guesses)
    metrics["n_invalid"] = len(metrics["guesses_invalid"])
    metrics["recall"] = (metrics["n_correct"] / len(tgt)) if tgt else 0.0
    metrics["precision"] = (metrics["n_correct"] / len(guesses)) if guesses else 0.0
    return metrics


def main():
    data = {k: [json.loads(l) for l in open(v)] for k, v in FILES.items()}

    # Sanity checks
    sanity = {"prompt_mismatch_idx": [], "duplicate_completion_pairs": []}
    n = len(data["base"])
    for i in range(n):
        p_b = data["base"][i]["messages"]
        if p_b != data["s56"][i]["messages"] or p_b != data["s112"][i]["messages"]:
            sanity["prompt_mismatch_idx"].append(i)
        for a, b in [("base", "s56"), ("base", "s112"), ("s56", "s112")]:
            if data[a][i]["completions"][0] == data[b][i]["completions"][0]:
                sanity["duplicate_completion_pairs"].append({"idx": i, "models": [a, b]})

    # Per-record metrics
    all_metrics = {k: [] for k in FILES}
    for k, recs in data.items():
        for rec in recs:
            comp = rec["completions"][0]
            if rec["metadata"]["task"] == "codenames_clue_generation":
                m = evaluate_clue_record(rec, comp)
            else:
                m = evaluate_guess_record(rec, comp)
            m["idx"] = rec["idx"]
            m["difficulty"] = rec["metadata"]["difficulty"]
            m["topic_1"] = rec["metadata"]["topic_1"]
            m["topic_2"] = rec["metadata"]["topic_2"]
            all_metrics[k].append(m)

    # Aggregate
    def avg(values):
        vs = [v for v in values if v is not None]
        return sum(vs) / len(vs) if vs else 0.0

    def agg(records, keys, predicate=lambda r: True):
        out = {}
        sel = [r for r in records if predicate(r)]
        if not sel:
            return out
        for key in keys:
            vals = [r.get(key) for r in sel]
            if all(isinstance(v, bool) for v in vals if v is not None):
                out[key + "_rate"] = sum(1 for v in vals if v) / len(vals)
            else:
                out[key + "_mean"] = avg(vals)
        out["n"] = len(sel)
        return out

    aggregates = {}
    for k in FILES:
        recs = all_metrics[k]
        clue_keys = [
            "completion_chars", "think_chars", "completion_words", "think_words",
            "n_corrections", "n_alternatives", "n_hedges", "repeat_rate_8gram",
            "n_candidate_clues", "n_selected",
            "clue_is_single_word", "clue_has_hyphen", "clue_in_target_set",
            "clue_in_non_target_set", "clue_morphological_overlap",
            "selected_all_in_target", "selected_full_set", "selected_match_ratio",
            "sections_in_order",
        ]
        guess_keys = [
            "completion_chars", "think_chars", "completion_words", "think_words",
            "n_corrections", "n_alternatives", "n_hedges", "repeat_rate_8gram",
            "n_guesses", "guess_overflow", "n_correct", "n_wrong", "n_invalid",
            "recall", "precision", "sections_in_order",
        ]
        aggregates[k] = {
            "clue":  agg(recs, clue_keys,  predicate=lambda r: r["task"] == "clue"),
            "guess": agg(recs, guess_keys, predicate=lambda r: r["task"] == "guess"),
        }

    # By difficulty (just recall for guesses, selected_match_ratio for clue)
    by_diff = {}
    for k in FILES:
        by_diff[k] = {}
        for diff in ["simple", "moderate", "advance", "expert"]:
            guess_recall = [r["recall"] for r in all_metrics[k]
                            if r["task"] == "guess" and r["difficulty"] == diff]
            wrong = [r["n_wrong"] for r in all_metrics[k]
                     if r["task"] == "guess" and r["difficulty"] == diff]
            sel_ratio = [r["selected_match_ratio"] for r in all_metrics[k]
                         if r["task"] == "clue" and r["difficulty"] == diff]
            by_diff[k][diff] = {
                "guess_recall_mean": avg(guess_recall),
                "guess_wrong_mean":  avg(wrong),
                "clue_selected_match_ratio_mean": avg(sel_ratio),
                "n_guess": len(guess_recall),
                "n_clue":  len(sel_ratio),
            }

    out = {
        "sanity": sanity,
        "per_record": all_metrics,
        "aggregates": aggregates,
        "by_difficulty": by_diff,
    }
    Path("custom_qual_eval/outputs/qual_metrics.json").write_text(json.dumps(out, indent=2))
    print("wrote custom_qual_eval/outputs/qual_metrics.json")


if __name__ == "__main__":
    main()
