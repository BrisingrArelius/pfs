#!/usr/bin/env python3
"""
verify_checkpoint.py — prove that a checkpoint written by populate_checkpoint.py
is a real DCP checkpoint, by checking it two independent ways.

Everything in this experiment rests on one claim: that `dcp.load` reading our
synthetic directory does exactly what it would do reading a directory written
by 32 real training ranks. That claim is worth testing rather than asserting,
so this script does not trust either side.

  Check 1 (structure)  Every shard blob named in `.metadata` is a valid zip
                       whose stored CRC-32 matches its payload, and loads with
                       `torch.load`. This is stricter than DCP itself, which
                       never validates the CRC -- so passing it means the files
                       would survive tooling that does.

  Check 2 (semantics)  Assemble each global tensor by hand, reading chunks
                       directly at the byte offsets `.metadata` records and
                       pasting them at their chunk offsets. Then run a real
                       `dcp.load` over the same directory and require the two
                       to be *bitwise* identical.

Check 2 is the one that matters: it says DCP's reshard arithmetic agrees with
our layout arithmetic. If `llama_spec.plan_shards` put a shard at the wrong
chunk offset, DCP would reassemble a different tensor and the comparison fails.

Comparison is bitwise (`view(torch.uint8)`), not numeric. A checkpoint filled
with random *bits* decodes to bfloat16 full of NaN, and NaN != NaN.
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import zipfile
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_blob(ckpt: Path, si) -> torch.Tensor:
    with open(ckpt / si.relative_path, "rb") as f:
        f.seek(si.offset)
        blob = f.read(si.length)
    if len(blob) != si.length:
        raise SystemExit(f"short read at {si.relative_path}+{si.offset}")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--skip-crc", action="store_true",
                    help="skip check 1 (it reads every byte; slow on a real PFS)")
    ap.add_argument("--max-tensors", type=int, default=None,
                    help="limit check 2 to the first N tensors")
    args = ap.parse_args()

    md = pickle.load(open(args.ckpt / ".metadata", "rb"))
    print(f"checkpoint  {args.ckpt}")
    print(f"tensors     {len(md.state_dict_metadata):,}")
    print(f"shards      {len(md.storage_data):,}")
    # `version` only exists on torch >= 2.6; absent on 2.5
    print(f"dcp version {getattr(md, 'version', 'n/a (pre-2.6 torch)')}\n")

    # ---- check 1: structural validity of every blob --------------------
    if not args.skip_crc:
        n = 0
        for key, si in md.storage_data.items():
            blob = load_blob(args.ckpt, si)
            bad = zipfile.ZipFile(io.BytesIO(blob)).testzip()
            if bad is not None:
                raise SystemExit(f"CRC failure in {key.fqn} ({si.relative_path}): {bad}")
            torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            n += 1
        print(f"check 1  OK  {n:,} shard blobs: zip CRC valid and torch.load succeeds")
    else:
        print("check 1  skipped")

    # ---- check 2: hand-assembled global tensors vs dcp.load ------------
    fqns = list(md.state_dict_metadata)
    if args.max_tensors:
        fqns = fqns[: args.max_tensors]

    expected = {}
    for fqn in fqns:
        tsm = md.state_dict_metadata[fqn]
        full = torch.zeros(tuple(tsm.size), dtype=tsm.properties.dtype)
        for idx, chunk in enumerate(tsm.chunks):
            from torch.distributed.checkpoint.metadata import MetadataIndex
            si = md.storage_data[MetadataIndex(fqn, tuple(chunk.offsets), idx)]
            piece = torch.load(io.BytesIO(load_blob(args.ckpt, si)),
                               map_location="cpu", weights_only=True)
            view = full
            for d, (o, s) in enumerate(zip(chunk.offsets, chunk.sizes)):
                view = view.narrow(d, o, s)
            view.copy_(piece)
        expected[fqn] = full

    # A real dcp.load, single process, full (unsharded) destination tensors --
    # the TP1 reader. No process group: DCP's no-dist path is exercised.
    got = {fqn: torch.zeros(tuple(md.state_dict_metadata[fqn].size),
                            dtype=md.state_dict_metadata[fqn].properties.dtype)
           for fqn in fqns}
    dcp.load(got, checkpoint_id=str(args.ckpt))

    mismatch = []
    for fqn in fqns:
        a = expected[fqn].contiguous().view(torch.uint8)
        b = got[fqn].contiguous().view(torch.uint8)
        if not torch.equal(a, b):
            mismatch.append(fqn)
    if mismatch:
        raise SystemExit(f"check 2 FAILED for {len(mismatch)} tensors, e.g. {mismatch[:3]}")
    total = sum(t.numel() for t in expected.values())
    print(f"check 2  OK  {len(fqns):,} tensors / {total:,} elements "
          f"bitwise identical between hand-assembly and dcp.load")
    print("\nThis directory is a real PyTorch DCP checkpoint.")


if __name__ == "__main__":
    main()
