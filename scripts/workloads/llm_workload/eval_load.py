#!/usr/bin/env python3
"""
eval_load.py — read a Llama-3.1-70B DCP checkpoint the way an *evaluation* job
reads it, and record the per-file access pattern.

This is the measured half of the experiment. Everything under
`torch.distributed.checkpoint` is stock PyTorch: the load planner, the
resharding arithmetic, the `.metadata` fan-out and the per-shard reads are
whatever DCP genuinely does. The only thing this script adds is a
`FileSystemReader` subclass that records `(rank, file, offset, length)` for
every storage item fetched, so the trace exists even where Darshan cannot be
loaded.

What makes this an *eval* read rather than a restart or a fine-tune
-------------------------------------------------------------------
Four knobs, each argued in METHODS.md and marked [CITED] or [ASSUMPTION]:

  state subset        model weights only. No optimizer moments, no RNG state,
                      no dataloader position, no `extra_state`. An eval or
                      serving loader builds its state dict from the module
                      tree alone -- HuggingFace `from_pretrained` and vLLM's
                      `load_weights` have nowhere to put optimizer state.
                      [ASSUMPTION, justified by loader structure]

  parallelism         the reader mesh is not the writer mesh, and it is
                      inference-shaped: tensor-parallel only, no pipeline or
                      data parallelism, and a small rank count.
                      [CITED for "reshard on mismatch", ASSUMPTION for width]

  post-load behaviour a forward-pass stand-in, then exit. The job never writes
                      back to the checkpoint directory. [CITED]

  frequency           driven by the job scheduler, not this script. See
                      `schedule_jobs.py` for the ~19,844:1,870 eval:resumption
                      ratio from ByteCheckpoint's production census. [CITED]

Modes
-----
  --mode plan   run DCP's real load planner against the real `.metadata` and
                emit the read plan without allocating any tensors. This is how
                the full 141 GB, 80-layer access pattern is obtained on a
                machine that cannot hold 141 GB. The plan is what DCP will
                read -- `FileSystemReader.read_data` fetches exactly the
                `(offset, length)` of each planned item.

  --mode load   allocate the destination shards and run `dcp.load` for real.
                Needs `model_bytes / tp` of memory per rank. This is the mode
                to run on the cluster under Darshan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.device_mesh import init_device_mesh
try:  # public since torch 2.5
    from torch.distributed.tensor import DTensor, Replicate, Shard
except ImportError:  # pragma: no cover - version shim
    from torch.distributed._tensor import DTensor, Replicate, Shard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_spec import (  # noqa: E402
    OPTIMIZER_KINDS, ModelSpec, ParallelConfig, _tp_chunks, preset,
)


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------


class RecordingFileSystemReader(FileSystemReader):
    """FileSystemReader that logs every storage item it fetches.

    Subclasses rather than patches, so the read path itself is untouched: DCP
    still groups by file, still opens each file once, still reads whole storage
    items and narrows in memory.
    """

    def __init__(self, path, log: list) -> None:
        super().__init__(path)
        self._log = log

    def read_data(self, plan, planner):
        for item in plan.items:
            si = self.storage_data[item.storage_index]
            self._log.append(
                {
                    "fqn": item.storage_index.fqn,
                    "file": si.relative_path,
                    "offset": si.offset,
                    "length": si.length,
                    "type": str(item.type).rsplit(".", 1)[-1],
                }
            )
        return super().read_data(plan, planner)


# ---------------------------------------------------------------------------
# the eval job's state dict
# ---------------------------------------------------------------------------


def build_eval_state_dict(
    model: ModelSpec, tp: int, rank: int, device: str, fqn_filter=None,
    read_set: str = "eval", optimizer: str = "adam",
) -> dict:
    """The state dict the reading job asks the checkpoint to fill.

    Built from the module geometry, not from the checkpoint. This mirrors how a
    serving loader works -- it instantiates the architecture, then asks the
    checkpoint to fill it. Anything the checkpoint holds that the module tree
    does not name is never requested, and so never read. That is the mechanism
    behind the whole eval-skips-the-optimizer result, and it is worth being
    explicit that it is a *consequence* of how loaders are built, not a policy
    anyone configured.

    read_set="eval"     model weights only. What an evaluation or serving job
                        wants: HuggingFace `from_pretrained` and vLLM's
                        `load_weights` populate a `nn.Module` and have nowhere
                        to put optimizer moments.
    read_set="restart"  weights plus Adam moments, i.e. what a training
                        resumption or fine-tune warm start needs. Present as
                        the contrast case: without it, "eval reads 141 of
                        705 GB" has nothing to be compared against.
    """
    mesh = init_device_mesh(device, (tp,), mesh_dim_names=("tp",))
    dt = getattr(torch, model.torch_dtype)

    def add(sd, fqn, axis, shape, dtype):
        chunks = _tp_chunks_for(shape, axis, tp)
        # replicated tensors yield one whole-tensor chunk, not one per rank
        local_sizes = chunks[0 if axis is None else rank % tp]
        local = torch.empty(local_sizes, dtype=dtype, device=device)
        placements = [Replicate() if axis is None else Shard(axis)]
        sd[fqn] = DTensor.from_local(local, mesh, placements, run_check=False)

    sd: dict = {}
    specs = [s for s in model.tensors() if not (fqn_filter and not fqn_filter(s.fqn))]
    for spec in specs:
        add(sd, spec.fqn, spec.shard_axis(), spec.shape, dt)
    if read_set == "restart":
        for spec in specs:
            for suffix, odt in OPTIMIZER_KINDS[optimizer]:
                add(sd, f"optim.state.{spec.fqn}.{suffix}", spec.shard_axis(),
                    spec.shape, getattr(torch, odt))
    return sd


def _tp_chunks_for(shape, axis, tp):
    """Local shard shapes for each rank -- the sizes half of `_tp_chunks`."""
    if axis is None:
        return [tuple(shape)]
    step = shape[axis] // tp
    return [tuple(step if d == axis else shape[d] for d in range(len(shape)))
            for _ in range(tp)]


def forward_pass_stand_in(sd: dict) -> float:
    """Touch the loaded weights the way a first forward pass would.

    Not a real forward pass -- it exists so the load cannot be optimised away
    and so the job's lifetime resembles an eval task's: load, compute, exit.
    Returns a checksum that is printed, so the reads are observably consumed.

    The checksum is taken over the raw bytes, not the float values: a synthetic
    checkpoint filled with random *bits* decodes to bfloat16 containing NaN and
    Inf, which would poison any arithmetic reduction.
    """
    acc = 0
    for _, t in sd.items():
        local = t.to_local() if isinstance(t, DTensor) else t
        if local.numel() == 0 or local.device.type == "meta":
            continue
        acc = (acc + int(local.contiguous().view(torch.uint8).sum().item())) % (1 << 61)
    return acc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--model", default="llama31_70b")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--load-config", default="TP4",
                    help="reader parallelism; inference-shaped, TP-only")
    ap.add_argument("--mode", choices=["plan", "load"], default="plan")
    ap.add_argument("--read-set", choices=["eval", "restart"], default="eval",
                    help="eval = weights only; restart = weights + optimizer moments")
    ap.add_argument("--optimizer", choices=sorted(OPTIMIZER_KINDS), default="adam",
                    help="which optimizer moments --read-set restart asks for")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "meta"])
    ap.add_argument("--only-layers", type=int, default=None,
                    help="load only the first N transformer layers (memory relief)")
    ap.add_argument("--emit-reads", type=Path, default=None,
                    help="CSV of (rank, fqn, file, offset, length) actually fetched")
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args()

    cfg = ParallelConfig.parse(args.load_config)
    if cfg.pp != 1 or cfg.dp != 1:
        raise SystemExit("eval loads are tensor-parallel only; use TP<n>")

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != cfg.tp:
        raise SystemExit(f"--load-config {cfg} needs world_size={cfg.tp}, got {world}")

    model = preset(args.model, args.layers)
    device = "meta" if args.mode == "plan" else args.device

    keep = None
    if args.only_layers is not None:
        n = args.only_layers
        def keep(fqn: str) -> bool:  # noqa: E306
            if ".layers." not in fqn:
                return True
            return int(fqn.split(".layers.")[1].split(".")[0]) < n

    t0 = time.time()
    sd = build_eval_state_dict(model, cfg.tp, rank, device, keep,
                               args.read_set, args.optimizer)
    t_build = time.time() - t0

    log: list = []
    reader = RecordingFileSystemReader(args.ckpt, log)

    t0 = time.time()
    if args.mode == "load":
        dcp.load(sd, storage_reader=reader)
        t_io = time.time() - t0
        checksum = forward_pass_stand_in(sd)
    else:
        # Planner only: read .metadata, build the local load plan, record the
        # storage items it resolves to. No tensor is allocated or read.
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

        md = reader.read_metadata()
        planner = DefaultLoadPlanner()
        planner.set_up_planner(sd, md, is_coordinator=(rank == 0))
        reader.set_up_storage_reader(md, is_coordinator=(rank == 0))
        plan = reader.prepare_local_plan(planner.create_local_plan())
        for item in plan.items:
            si = reader.storage_data[item.storage_index]
            log.append({"fqn": item.storage_index.fqn, "file": si.relative_path,
                        "offset": si.offset, "length": si.length,
                        "type": str(item.type).rsplit(".", 1)[-1]})
        t_io = time.time() - t0
        checksum = -1

    # ---- per-rank summary ----
    per_file = defaultdict(lambda: {"reqs": 0, "bytes": 0})
    for r in log:
        e = per_file[r["file"]]
        e["reqs"] += 1
        e["bytes"] += r["length"]
    total_bytes = sum(r["length"] for r in log)
    md_bytes = (args.ckpt / ".metadata").stat().st_size

    mine = {
        "rank": rank,
        "world_size": world,
        "load_config": str(cfg),
        "mode": args.mode,
        "requests": len(log),
        "bytes": total_bytes,
        "files_opened": len(per_file),
        "metadata_bytes_read": md_bytes,
        "build_s": round(t_build, 3),
        "io_s": round(t_io, 3),
        "checksum": checksum,
        "per_file": {k: v for k, v in sorted(per_file.items())},
    }

    gathered: list = [None] * world
    dist.all_gather_object(gathered, mine)

    if args.emit_reads:
        rows: list = [None] * world
        dist.all_gather_object(rows, [dict(r, rank=rank) for r in log])
        if rank == 0:
            args.emit_reads.parent.mkdir(parents=True, exist_ok=True)
            with open(args.emit_reads, "w", newline="") as f:
                w = csv.DictWriter(f, ["rank", "fqn", "file", "offset", "length", "type"])
                w.writeheader()
                for chunk in rows:
                    w.writerows(chunk)

    if rank == 0:
        agg_files = defaultdict(lambda: {"readers": 0, "reqs": 0, "bytes": 0})
        for g in gathered:
            for fn, v in g["per_file"].items():
                a = agg_files[fn]
                a["readers"] += 1
                a["reqs"] += v["reqs"]
                a["bytes"] += v["bytes"]
        tot_reqs = sum(g["requests"] for g in gathered)
        tot_bytes = sum(g["bytes"] for g in gathered)
        shard_files = sorted(p.name for p in args.ckpt.glob("*.distcp"))
        never = [f for f in shard_files if f not in agg_files]

        print(f"\n=== {args.read_set} load: {model.name} {model.num_hidden_layers}L  "
              f"{cfg} ({world} ranks)  mode={args.mode} ===")
        print(f"checkpoint          {args.ckpt}")
        print(f".metadata           {md_bytes:,} B, read by all {world} ranks "
              f"({md_bytes*world/1e6:.1f} MB of fan-out)")
        print(f"shard files present {len(shard_files)}")
        print(f"shard files opened  {len(agg_files)}   never opened: {len(never)}")
        print(f"total requests      {tot_reqs:,}")
        print(f"total bytes read    {tot_bytes/1e9:.2f} GB "
              f"(avg request {tot_bytes/max(tot_reqs,1)/1e6:.2f} MB)")
        print(f"slowest rank io     {max(g['io_s'] for g in gathered):.2f} s")
        if args.mode == "load":
            print(f"checksum            {sum(g['checksum'] for g in gathered)}")

        out = {
            "model": model.name, "layers": model.num_hidden_layers,
            "load_config": str(cfg), "mode": args.mode, "read_set": args.read_set,
            "metadata_bytes": md_bytes, "metadata_fanout_bytes": md_bytes * world,
            "shard_files_present": len(shard_files),
            "shard_files_opened": len(agg_files),
            "shard_files_never_opened": never,
            "total_requests": tot_reqs, "total_bytes": tot_bytes,
            "per_rank": [{k: v for k, v in g.items() if k != "per_file"} for g in gathered],
            "per_file": {k: dict(v) for k, v in sorted(agg_files.items())},
        }
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(out, indent=2))
            print(f"summary             {args.summary}")

    dist.barrier()
    dist.destroy_process_group()
    # No write-back: an eval job never modifies the checkpoint it read.


if __name__ == "__main__":
    main()
