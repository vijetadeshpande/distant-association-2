"""GloVe-cosine reward for the Codenames Spymaster role.

This module is the no-judge alternative to ``judge_client.judge_guess``.
When ``JUDGE_NAME`` is unset (or set to ``"none"``/``"null"``), the
top-level ``compute_score`` skips the judge HTTP call and asks this
module for a scalar reward instead.

Per-sample logic
----------------
1. Read the trainee's parsed clue (``pc.clue``) and the reference
   ``clue`` field from ``extra_info``.
2. Look up both in the GloVe table (casefolded).
3. If either is OOV, return ``cosine_sim = 0`` and set ``oov = 1``.
4. Otherwise, ``cosine_sim`` = dot product of the two L2-normalized
   vectors. This is the entire task reward.

Central, mmap-backed lookup table
---------------------------------
The GloVe matrix and vocab dict are loaded lazily on first call and
cached at module scope, so:

  * Within one Python process, every coroutine shares one load — never
    one-per-sample.
  * The ``.npy`` is opened with ``mmap_mode='r'``, so the 1.4 GB of
    embeddings is shared across all Ray worker processes on the same
    node via the OS page cache.

Pre-build the artifacts once with ``scripts/build_glove_lookup.py``.

Environment variables
---------------------
``GLOVE_NPY_PATH``    Path to the ``.npy`` matrix written by the
                      build script. Default::
                          custom_data/glove_vectors/
                          dolma_300_2024_1.2M.100_combined.npy
``GLOVE_VOCAB_PATH``  Path to the ``_vocab.pkl`` dict. Default is the
                      sibling of ``GLOVE_NPY_PATH`` with ``_vocab.pkl``
                      replacing ``.npy``.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_NPY = _REPO_ROOT / "custom_data/glove_vectors/dolma_300_2024_1.2M.100_combined.npy"


def _default_vocab_for(npy_path: Path) -> Path:
    return npy_path.with_name(npy_path.stem + "_vocab.pkl")


_MATRIX: np.ndarray | None = None
_VOCAB: dict[str, int] | None = None
_LOAD_FAILED: bool = False


def _get_table() -> tuple[np.ndarray | None, dict[str, int] | None]:
    """Lazy module-level load. Returns ``(None, None)`` on failure so
    callers can fall back to ``cosine_sim = 0`` rather than crashing
    the whole training loop."""
    global _MATRIX, _VOCAB, _LOAD_FAILED
    if _MATRIX is not None and _VOCAB is not None:
        return _MATRIX, _VOCAB
    if _LOAD_FAILED:
        return None, None

    npy_path = Path(os.environ.get("GLOVE_NPY_PATH", str(_DEFAULT_NPY)))
    vocab_path = Path(os.environ.get("GLOVE_VOCAB_PATH", str(_default_vocab_for(npy_path))))

    try:
        # mmap_mode='r' lets multiple worker processes share the matrix
        # via the OS page cache instead of each loading its own 1.4 GB.
        matrix = np.load(npy_path, mmap_mode="r")
        with vocab_path.open("rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(
            "GloVe lookup artifacts not found (%s). Run scripts/build_glove_lookup.py first. "
            "Cosine reward will return 0 for every sample.",
            e,
        )
        _LOAD_FAILED = True
        return None, None
    except Exception as e:  # pickle / numpy load errors
        logger.exception("Failed to load GloVe lookup table: %r", e)
        _LOAD_FAILED = True
        return None, None

    if not isinstance(vocab, dict):
        logger.error("GloVe vocab pickle at %s is not a dict (got %s).", vocab_path, type(vocab))
        _LOAD_FAILED = True
        return None, None

    _MATRIX, _VOCAB = matrix, vocab
    logger.info("Loaded GloVe table: %d words, dim=%d from %s", matrix.shape[0], matrix.shape[1], npy_path)
    return _MATRIX, _VOCAB


def is_cosine_mode() -> bool:
    """``True`` iff the JUDGE_NAME env var signals "no judge" — empty
    or the literal strings ``none``/``null`` (case-insensitive)."""
    name = os.environ.get("JUDGE_NAME", "").strip()
    return name == "" or name.lower() in {"none", "null"}


def _lookup(word: str, vocab: dict[str, int], matrix: np.ndarray) -> np.ndarray | None:
    if not word:
        return None
    idx = vocab.get(word.strip().casefold())
    if idx is None:
        return None
    return matrix[idx]


def cosine_reward(parsed_clue: str, target_clue: str) -> dict[str, Any]:
    """Compute the cosine-similarity reward dict.

    Returns the same shape as ``task_reward`` (so the aggregator can
    treat both paths uniformly), plus two diagnostics: ``cosine_sim``
    and ``oov``. ``oov`` is 1 if either word is missing from the table
    (or if the table failed to load).
    """
    matrix, vocab = _get_table()
    if matrix is None or vocab is None:
        return _zero_cosine_reward(oov=1)

    v_clue = _lookup(parsed_clue, vocab, matrix)
    v_target = _lookup(target_clue, vocab, matrix)
    if v_clue is None or v_target is None:
        return _zero_cosine_reward(oov=1)

    # Vectors are pre-normalized at build time, so dot == cosine.
    sim = float(np.dot(v_clue, v_target))
    sim = max(-1.0, min(1.0, sim))  # guard against tiny numerical drift

    return {
        "cosine_sim": sim,
        "oov": 0,
        "pos_correct": sim,
        "neg_nontarget": 0.0,
        "neg_invalid": 0.0,
        "task": sim,
    }


def _zero_cosine_reward(*, oov: int) -> dict[str, Any]:
    return {
        "cosine_sim": 0.0,
        "oov": int(oov),
        "pos_correct": 0.0,
        "neg_nontarget": 0.0,
        "neg_invalid": 0.0,
        "task": 0.0,
    }


def zero_cosine_reward() -> dict[str, Any]:
    """Public zero-fill used by the aggregator when short-circuiting
    (e.g. clue-format failure) in cosine mode."""
    return _zero_cosine_reward(oov=0)
