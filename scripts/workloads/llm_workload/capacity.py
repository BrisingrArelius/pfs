#!/usr/bin/env python3
"""
capacity.py — how much memory a checkpoint read needs, and how many layers fit.

`dcp.load` fills tensors you allocate up front, so a read's memory cost is the
size of the state dict being requested, not of the checkpoint. That is the
binding constraint on a node without GPUs: anjuna2 has 62 GB of RAM, and a full
Llama-3.1-70B eval load needs 141 GB resident.

This turns that arithmetic into a command, so the run scripts can size
`--only-layers` from the machine they are actually on rather than from a
hardcoded guess that goes stale when the model or optimizer changes.

    # what does a full 70B eval load cost?
    python3 capacity.py --model llama31_70b --read-set eval

    # how many layers fit in 30 GB?
    python3 capacity.py --model llama31_70b --read-set eval --ram-gb 30

Sharded tensors divide across ranks; replicated ones (the RMSNorm weights) are
held in full by every rank, so total resident memory grows very slightly with
rank count. That term is included rather than waved away, though at 70B it is
under 0.01%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_spec import (  # noqa: E402
    DTYPE_BYTES, OPTIMIZER_KINDS, ModelSpec, ParallelConfig, preset,
)


def resident_bytes(model: ModelSpec, read_set: str, optimizer: str, tp: int) -> dict:
    """Total bytes across all ranks for the state dict a reader allocates."""
    mb = DTYPE_BYTES[model.torch_dtype]
    sharded = replicated = 0
    for spec in model.tensors():
        n = spec.numel
        if spec.shard_axis() is None:
            replicated += n
        else:
            sharded += n

    model_bytes = sharded * mb + replicated * mb * tp
    opt_bytes = 0
    if read_set == "restart":
        for _, dt in OPTIMIZER_KINDS[optimizer]:
            ob = DTYPE_BYTES[dt]
            opt_bytes += sharded * ob + replicated * ob * tp
    total = model_bytes + opt_bytes
    return {
        "model_bytes": model_bytes,
        "optimizer_bytes": opt_bytes,
        "total_bytes": total,
        "per_rank_bytes": total / tp,
        # bytes actually pulled off the filesystem: replicas are read per rank,
        # and past the saved width DCP re-reads whole shards (see METHODS 4.2)
        "read_bytes": total,
    }


def max_layers(model_name: str, read_set: str, optimizer: str, tp: int,
               budget: int) -> int:
    lo, hi = 0, preset(model_name).num_hidden_layers
    best = 0
    for n in range(0, hi + 1):
        m = preset(model_name, n)
        if resident_bytes(m, read_set, optimizer, tp)["total_bytes"] <= budget:
            best = n
        else:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama31_70b")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--read-set", choices=["eval", "restart"], default="eval")
    ap.add_argument("--optimizer", choices=sorted(OPTIMIZER_KINDS), default="adam")
    ap.add_argument("--load-config", default="TP4")
    ap.add_argument("--ram-gb", type=float, default=None,
                    help="solve for the largest --only-layers fitting this budget")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the layer count (for shell substitution)")
    args = ap.parse_args()

    tp = ParallelConfig.parse(args.load_config).tp

    if args.ram_gb is not None:
        n = max_layers(args.model, args.read_set, args.optimizer, tp,
                       int(args.ram_gb * 1e9))
        if args.quiet:
            print(n)
            return
        full = preset(args.model).num_hidden_layers
        m = preset(args.model, n)
        r = resident_bytes(m, args.read_set, args.optimizer, tp)
        print(f"model          {args.model}  ({full} layers at full size)")
        print(f"read set       {args.read_set}"
              + (f" (+{args.optimizer})" if args.read_set == "restart" else ""))
        print(f"budget         {args.ram_gb:.1f} GB over {tp} ranks")
        print(f"fits           {n} of {full} layers "
              f"({100*n/full:.0f}% of the transformer stack)")
        print(f"resident       {r['total_bytes']/1e9:.2f} GB total, "
              f"{r['per_rank_bytes']/1e9:.2f} GB per rank")
        if n == 0:
            print("\nNothing fits. Use --mode plan for the access pattern (0 GB),"
                  "\nor a smaller model (--model llama31_8b).")
        return

    m = preset(args.model, args.layers)
    r = resident_bytes(m, args.read_set, args.optimizer, tp)
    print(f"model          {m.name}  {m.num_hidden_layers} layers, "
          f"{m.num_parameters():,} params")
    print(f"read set       {args.read_set}"
          + (f" (+{args.optimizer})" if args.read_set == "restart" else ""))
    print(f"load config    TP{tp}")
    print(f"weights        {r['model_bytes']/1e9:.2f} GB")
    if r["optimizer_bytes"]:
        print(f"optimizer      {r['optimizer_bytes']/1e9:.2f} GB")
    print(f"resident TOTAL {r['total_bytes']/1e9:.2f} GB "
          f"({r['per_rank_bytes']/1e9:.2f} GB per rank x {tp})")


if __name__ == "__main__":
    main()
