"""One-shot preprocessor for the cosine-similarity reward path.

Parses a GloVe-format text file (``word v1 v2 ... v_d`` per line) and
writes two artifacts next to it:

  - ``<stem>.npy``       — float32 matrix, shape ``(V, D)``, L2-normalized
                            so cosine similarity is just a dot product.
  - ``<stem>_vocab.pkl`` — pickled ``dict[str, int]`` mapping casefolded
                            word -> row index.

At runtime ``cosine_reward.py`` loads the ``.npy`` with ``mmap_mode='r'``
so the heavy bytes are shared across Ray worker processes via the OS
page cache. The vocab dict is loaded once per process from pickle.

Run once before training in cosine mode::

    python scripts/build_glove_lookup.py \\
        custom_data/glove_vectors/dolma_300_2024_1.2M.100_combined.txt

The script is idempotent: it skips rebuild if both outputs are newer
than the source file. Force a rebuild with ``--force``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def _artifact_paths(source: Path) -> tuple[Path, Path]:
    stem = source.with_suffix("")
    return Path(f"{stem}.npy"), Path(f"{stem}_vocab.pkl")


def _is_fresh(source: Path, npy: Path, pkl: Path) -> bool:
    if not (npy.exists() and pkl.exists()):
        return False
    src_mtime = source.stat().st_mtime
    return npy.stat().st_mtime >= src_mtime and pkl.stat().st_mtime >= src_mtime


def _infer_dim(source: Path) -> int:
    with source.open("r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n")
    parts = first.split()
    if len(parts) < 2:
        raise ValueError(f"First line of {source} has no vector values.")
    return len(parts) - 1


def build(source: Path, force: bool = False) -> None:
    npy_path, pkl_path = _artifact_paths(source)
    if not force and _is_fresh(source, npy_path, pkl_path):
        print(f"[skip] artifacts up-to-date: {npy_path.name}, {pkl_path.name}")
        return

    dim = _infer_dim(source)
    print(f"[build] source={source}  dim={dim}")

    # Stream the file once to count lines so we can allocate the matrix
    # up front (avoids repeated np.concatenate). 1.2M lines counts in
    # ~3 s on local SSD; the parse is the real cost.
    t0 = time.time()
    with source.open("r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    print(f"[count] {n_lines} lines in {time.time() - t0:.1f}s")

    matrix = np.empty((n_lines, dim), dtype=np.float32)
    vocab: dict[str, int] = {}

    t0 = time.time()
    bad = 0
    idx = 0
    with source.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip("\n").split(" ")
            if len(parts) != dim + 1:
                bad += 1
                continue
            word = parts[0].strip().casefold()
            if not word or word in vocab:
                # Skip empty tokens and duplicate keys (first occurrence wins).
                bad += 1
                continue
            try:
                vec = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                bad += 1
                continue
            matrix[idx] = vec
            vocab[word] = idx
            idx += 1
    print(f"[parse] kept {idx} / {n_lines}  skipped={bad}  in {time.time() - t0:.1f}s")

    matrix = matrix[:idx]

    # L2-normalize in place; rows with zero norm stay zero so cosine
    # against them is 0 (which is what we want for an OOV-ish sentinel).
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    t0 = time.time()
    np.save(npy_path, matrix, allow_pickle=False)
    with pkl_path.open("wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = npy_path.stat().st_size / (1024 ** 2)
    print(f"[save] {npy_path.name} ({size_mb:.0f} MB)  {pkl_path.name}  in {time.time() - t0:.1f}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("source", type=Path, help="Path to GloVe text file.")
    p.add_argument("--force", action="store_true", help="Rebuild even if artifacts are fresh.")
    args = p.parse_args()

    if not args.source.exists():
        print(f"[error] not found: {args.source}", file=sys.stderr)
        return 1

    build(args.source, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
