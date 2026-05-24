#!/usr/bin/env python3
"""Verify a merged_hf checkpoint matches the per-device FSDP shards it came from.

Per-rank shard format (saved by VeRL's FSDP checkpointer):
    actor/model_world_size_W_rank_R.pt
        -> OrderedDict[str, torch.distributed.tensor.DTensor]
        -> each DTensor has placements=(Shard(dim=0),) on a W-way fsdp mesh.

Reconstruction: stack the W ranks' `_local_tensor` along dim 0, then trim to
the DTensor's global shape (FSDP pads dim 0 up to a multiple of W).

The merged safetensors is stored in the model's native dtype (bf16 for Qwen3),
while the FSDP shards are fp32, so we cast both to bf16 before comparing — that
matches what HF will actually load at inference time.

Usage:
    python scripts/verify_merged_checkpoint.py <actor_dir>

    # tighter tolerance / fp32 comparison
    python scripts/verify_merged_checkpoint.py <actor_dir> --compare-dtype float32 --atol 1e-5
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open


SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def load_fsdp_full_state_dict(actor_dir: Path) -> dict[str, torch.Tensor]:
    shards = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no FSDP shards in {actor_dir}")

    world = None
    by_rank: dict[int, dict] = {}
    for p in shards:
        m = SHARD_RE.search(p.name)
        if not m:
            continue
        w, r = int(m.group(1)), int(m.group(2))
        world = w if world is None else world
        if w != world:
            raise RuntimeError(f"mixed world sizes: {w} vs {world}")
        by_rank[r] = torch.load(str(p), map_location="cpu", weights_only=False)

    if set(by_rank.keys()) != set(range(world)):
        raise RuntimeError(f"missing ranks: have {sorted(by_rank)} expected 0..{world - 1}")

    keys = list(by_rank[0].keys())
    full: dict[str, torch.Tensor] = {}
    for k in keys:
        dt0 = by_rank[0][k]
        global_shape = tuple(dt0.shape)
        locals_ = [by_rank[r][k]._local_tensor for r in range(world)]
        cat = torch.cat(locals_, dim=0)
        full[k] = cat[: global_shape[0]].contiguous()
        if tuple(full[k].shape) != global_shape:
            raise RuntimeError(f"{k}: rebuilt {tuple(full[k].shape)} != global {global_shape}")
    return full


def load_merged_state_dict(merged_dir: Path) -> dict[str, torch.Tensor]:
    files = sorted(merged_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {merged_dir}")
    sd: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                sd[k] = fh.get_tensor(k)
    return sd


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def verify(
    actor_dir: Path,
    merged_dir: Path,
    compare_dtype: str = "bfloat16",
    atol: float = 0.0,
    rtol: float = 0.0,
    verbose: bool = True,
    max_mismatch_report: int = 10,
) -> tuple[bool, dict]:
    """Compare merged_hf against the FSDP shards in actor_dir.

    Returns (ok, info) where info has keys:
        n_params, max_abs_diff, worst_key, mismatches, only_fsdp, only_hf
    """
    cmp_dtype = _DTYPES[compare_dtype]

    if verbose:
        print(f"  [verify] {actor_dir.parent.name}: comparing in {compare_dtype} (atol={atol}, rtol={rtol})")

    fsdp_sd = load_fsdp_full_state_dict(actor_dir)
    hf_sd = load_merged_state_dict(merged_dir)

    only_fsdp = sorted(set(fsdp_sd) - set(hf_sd))
    only_hf = sorted(set(hf_sd) - set(fsdp_sd))
    common = sorted(set(fsdp_sd) & set(hf_sd))

    mismatches: list[tuple[str, float, float]] = []
    max_abs_diff = 0.0
    worst_key = ""
    for k in common:
        a = fsdp_sd[k].to(cmp_dtype)
        b = hf_sd[k].to(cmp_dtype)
        if a.shape != b.shape:
            mismatches.append((k, float("nan"), float("nan")))
            continue
        diff = (a.float() - b.float()).abs()
        m = float(diff.max())
        mean = float(diff.mean())
        if m > max_abs_diff:
            max_abs_diff, worst_key = m, k
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            mismatches.append((k, m, mean))

    ok = not mismatches and not only_fsdp and not only_hf

    if verbose:
        if only_fsdp:
            print(f"  [verify] {len(only_fsdp)} keys only in FSDP shards (first 5): {only_fsdp[:5]}")
        if only_hf:
            print(f"  [verify] {len(only_hf)} keys only in merged_hf (first 5): {only_hf[:5]}")
        status = "OK" if ok else "FAIL"
        print(
            f"  [verify] {status}: {len(common) - len(mismatches)}/{len(common)} params match, "
            f"max_abs_diff={max_abs_diff:.3e} (worst: {worst_key or 'n/a'})"
        )
        for k, m, mean in mismatches[:max_mismatch_report]:
            print(f"  [verify]   {k}: max_abs={m:.3e}  mean_abs={mean:.3e}")
        if len(mismatches) > max_mismatch_report:
            print(f"  [verify]   ... and {len(mismatches) - max_mismatch_report} more")

    return ok, {
        "n_params": len(common),
        "max_abs_diff": max_abs_diff,
        "worst_key": worst_key,
        "mismatches": mismatches,
        "only_fsdp": only_fsdp,
        "only_hf": only_hf,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_dir", type=Path, help="Path to .../global_step_X/actor")
    parser.add_argument("--out-name", default="merged_hf")
    parser.add_argument(
        "--compare-dtype",
        default="bfloat16",
        choices=list(_DTYPES.keys()),
        help="Cast both tensors to this dtype before comparing (default: bfloat16, matches HF load)",
    )
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance (default 0 = exact)")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance (default 0)")
    parser.add_argument("--max-mismatch-report", type=int, default=10)
    args = parser.parse_args()

    actor_dir = args.actor_dir.resolve()
    merged_dir = actor_dir / args.out_name
    if not actor_dir.is_dir():
        print(f"error: {actor_dir} not a directory")
        return 1
    if not merged_dir.is_dir():
        print(f"error: {merged_dir} not a directory")
        return 1

    print(f"actor:  {actor_dir}")
    print(f"merged: {merged_dir}")
    ok, _ = verify(
        actor_dir,
        merged_dir,
        compare_dtype=args.compare_dtype,
        atol=args.atol,
        rtol=args.rtol,
        max_mismatch_report=args.max_mismatch_report,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
