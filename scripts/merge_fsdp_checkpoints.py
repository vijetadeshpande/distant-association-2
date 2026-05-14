#!/usr/bin/env python3
"""Merge per-device FSDP checkpoints into unsharded HuggingFace checkpoints.

VeRL saves DAPO/GRPO actor checkpoints sharded per device, e.g.:

    checkpoints/<project>/<run>/global_step_<N>/actor/
        model_world_size_8_rank_{0..7}.pt
        optim_world_size_8_rank_{0..7}.pt
        extra_state_world_size_8_rank_{0..7}.pt
        fsdp_config.json
        huggingface/            <- tokenizer + model config

This script finds every `global_step_*/actor` directory under a run directory
and merges it into a single HuggingFace-format checkpoint using
`verl.model_merger`.

Pass a single parent run directory; the script merges every `global_step_*`
checkpoint found inside it.

Usage:
    python scripts/merge_fsdp_checkpoints.py \
        checkpoints/Distant-Association/codenames-dapo-qwen3-1.7b-20260512-212152

    # custom output location / re-merge existing
    python scripts/merge_fsdp_checkpoints.py <run_dir> --out-name merged_hf --force
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_actor_dirs(run_dir: Path) -> list[Path]:
    def step_num(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except ValueError:
            return -1

    actor_dirs = []
    for step_dir in sorted(run_dir.glob("global_step_*"), key=step_num):
        if step_num(step_dir) < 0:
            continue
        actor_dir = step_dir / "actor"
        if not actor_dir.is_dir():
            print(f"  [skip] {step_dir.name}: no 'actor' subdir")
            continue
        if not list(actor_dir.glob("model_world_size_*_rank_*.pt")):
            print(f"  [skip] {step_dir.name}: no sharded model_*.pt files")
            continue
        actor_dirs.append(actor_dir)
    return actor_dirs


def merge_one(actor_dir: Path, out_name: str, force: bool) -> bool:
    target_dir = actor_dir / out_name
    if target_dir.exists() and not force:
        if list(target_dir.glob("*.safetensors")) or list(target_dir.glob("*.bin")):
            print(f"  [done] {actor_dir.parent.name}: already merged -> {target_dir}")
            return True

    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(target_dir),
    ]
    print(f"  [merge] {actor_dir.parent.name} -> {target_dir}")
    print(f"          {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAIL] {actor_dir.parent.name}: merger exited {result.returncode}")
        return False
    print(f"  [ok]   {actor_dir.parent.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run directory containing global_step_* folders")
    parser.add_argument(
        "--out-name",
        default="merged_hf",
        help="Name of the merged-checkpoint subdir created inside each 'actor' dir (default: merged_hf)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-merge even if a merged checkpoint already exists",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"error: run dir does not exist: {run_dir}")
        return 1

    print(f"Scanning {run_dir}")
    actor_dirs = find_actor_dirs(run_dir)
    if not actor_dirs:
        print("No mergeable checkpoints found.")
        return 1

    print(f"Found {len(actor_dirs)} checkpoint(s) to merge.\n")
    failed = []
    for actor_dir in actor_dirs:
        if not merge_one(actor_dir, args.out_name, args.force):
            failed.append(actor_dir.parent.name)
        print()

    if failed:
        print(f"Done with errors. Failed: {', '.join(failed)}")
        return 1
    print(f"All {len(actor_dirs)} checkpoint(s) merged successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
