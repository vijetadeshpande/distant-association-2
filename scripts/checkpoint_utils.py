"""Resolve model references to vLLM-loadable local paths.

A model reference is one of:

* a Hugging Face repo ID, for example ``Qwen/Qwen3-8B``;
* a local checkpoint folder, for example ``/home/.../actor/merged_hf``;
* an rclone remote, for example ``gdrive:Distant-Association/.../merged_hf``.

This module is intentionally dependency-free so it can be called cheaply from
shell scripts without importing vLLM::

    python -m scripts.checkpoint_utils resolve <ref> [--download-dir D] [--force]
    python -m scripts.checkpoint_utils short-name <ref>

``resolve`` prints the local path vLLM should load, downloading from rclone if
needed. ``short-name`` prints a filesystem-safe identifier for output names.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Optional


_RCLONE_REMOTE_RE = re.compile(r"^[A-Za-z0-9_-]+:")

_GENERIC_CKPT_DIRS = {
    "merged_hf",
    "actor",
    "critic",
    "hf",
    "merged",
    "checkpoint",
    "checkpoints",
    "model",
}


def _log(msg: str) -> None:
    """Log to stderr so stdout remains clean for the resolved path."""
    print(f"[checkpoint_utils] {msg}", file=sys.stderr)


def is_rclone_remote(path: str) -> bool:
    """Return whether path looks like an rclone remote rather than local/HF."""
    return bool(_RCLONE_REMOTE_RE.match(path)) and "://" not in path and not os.path.exists(path)


def download_checkpoint_with_rclone(
    remote_path: str,
    local_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """Download an rclone checkpoint folder and return its local path."""
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone not found on PATH. Install it from https://rclone.org/install/ "
            "and configure a remote with `rclone config`."
        )

    if local_dir is None:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", remote_path).strip("_")
        base = os.environ.get("RCLONE_CHECKPOINT_DIR", "./rclone_checkpoints")
        local_dir = os.path.join(base, sanitized)
    local_dir = os.path.abspath(os.path.expanduser(local_dir))

    is_complete = os.path.isfile(os.path.join(local_dir, "config.json"))
    if is_complete and not force:
        _log(f"Checkpoint already present at {local_dir} — skipping rclone download.")
        return local_dir

    os.makedirs(local_dir, exist_ok=True)
    _log(f"Downloading checkpoint via rclone: {remote_path} -> {local_dir}")
    subprocess.run(
        ["rclone", "copy", remote_path, local_dir, "--progress"],
        check=True,
    )

    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        raise FileNotFoundError(
            f"rclone download finished but {local_dir} has no config.json — "
            f"check that '{remote_path}' points to a valid HF checkpoint folder."
        )
    return local_dir


def resolve_checkpoint_path(
    model_ref: str,
    download_dir: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """Resolve an rclone, local, or Hugging Face reference for vLLM."""
    if is_rclone_remote(model_ref):
        return download_checkpoint_with_rclone(
            model_ref,
            local_dir=download_dir,
            force=force_download,
        )

    expanded = os.path.abspath(os.path.expanduser(model_ref))
    if os.path.isdir(expanded):
        if not os.path.isfile(os.path.join(expanded, "config.json")):
            raise FileNotFoundError(
                f"Local checkpoint '{expanded}' has no config.json — "
                "not a valid Hugging Face checkpoint folder."
            )
        _log(f"Using local checkpoint: {expanded}")
        return expanded

    # A non-local, non-rclone reference is a Hugging Face repo ID. vLLM and
    # Transformers handle its download through their normal caches.
    return model_ref


def checkpoint_short_name(model_ref: str) -> str:
    """Return a short filesystem-safe identifier for a model reference."""
    ref = model_ref.rstrip("/")
    remote = is_rclone_remote(ref)
    if remote:
        ref = ref.split(":", 1)[1].rstrip("/")

    if not remote and not os.path.exists(ref) and ref.count("/") <= 1:
        return ref.split("/")[-1]

    parts = [part for part in ref.split("/") if part]
    meaningful = [part for part in parts if part not in _GENERIC_CKPT_DIRS]
    name = "-".join(meaningful[-2:]) if meaningful else parts[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve a model reference to a local path vLLM can load.",
    )
    resolve_parser.add_argument(
        "ref",
        help="Hugging Face repo ID, local checkpoint directory, or rclone remote",
    )
    resolve_parser.add_argument(
        "--download-dir",
        default=None,
        help="Destination directory for an rclone download",
    )
    resolve_parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the destination already looks complete",
    )

    short_name_parser = subparsers.add_parser(
        "short-name",
        help="Print a short, filesystem-safe name for a model reference.",
    )
    short_name_parser.add_argument("ref")

    args = parser.parse_args()
    if args.cmd == "resolve":
        print(
            resolve_checkpoint_path(
                args.ref,
                download_dir=args.download_dir,
                force_download=args.force,
            )
        )
    elif args.cmd == "short-name":
        print(checkpoint_short_name(args.ref))


if __name__ == "__main__":
    _main()
