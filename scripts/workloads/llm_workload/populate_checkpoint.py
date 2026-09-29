#!/usr/bin/env python3
"""
populate_checkpoint.py — materialise a format-exact PyTorch DCP checkpoint for
Llama-3.1-70B on a parallel filesystem, without ever holding 70B parameters.

What "format-exact" means
-------------------------
A DCP checkpoint directory is two contracts, both small and both stable:

  `.metadata`         a plain `pickle.dump` of `torch.distributed.checkpoint.
                      metadata.Metadata`, holding (a) per-FQN global shape,
                      dtype and chunk list, and (b) `storage_data`, a map from
                      `MetadataIndex(fqn, chunk_offsets)` to
                      `_StorageInfo(relative_path, offset, length)`.

  `__{rank}_0.distcp` the raw concatenation of one `torch.save(tensor)` blob
                      per shard that rank owns, at the offsets `.metadata`
                      records.

Nothing else is consulted at load time (see `FileSystemReader.read_data`), so a
checkpoint can be streamed shard by shard from a plan and the result is
indistinguishable, to `dcp.load`, from one written by 32 real training ranks.
That is the whole point: the *read* side of this experiment is then unmodified,
genuine DCP code, which is what the Darshan trace has to capture.

The blob layout is measured, not guessed. `torch.save` of a contiguous tensor
emits a fixed prefix, the payload as one contiguous write, then a fixed suffix
holding the zip data descriptor and central directory. `blob_template` captures
the real framing for each distinct shape by running `torch.save` once and
discarding only the payload write; `_crc_slots` then locates the two copies of
the payload's CRC-32 so they can be corrected when the payload is not zeros.
The result passes `unzip -t` and `torch.load`.

Fill modes
----------
  --fill random   real pseudorandom bytes, CRC corrected. Use on the PFS:
                  bandwidth and any compression the storage layer does are then
                  honest.
  --fill zeros    real blocks, all zero. Fastest byte-real mode.
  --fill sparse   seek over each payload so the filesystem leaves a hole. File
                  *sizes*, offsets, CRCs and `.metadata` are byte-identical to
                  --fill zeros; only the allocated blocks differ. This is how a
                  141 GB checkpoint fits on a laptop for correctness work. It
                  is NOT valid for bandwidth numbers -- reads of a hole are
                  served by the kernel and never reach the media.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import pickle
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint.filesystem import _StorageInfo

try:  # torch >= 2.5 exposes the on-disk metadata version; older builds do not
    from torch.distributed.checkpoint.filesystem import CURRENT_DCP_VERSION
except ImportError:  # pragma: no cover - version shim
    CURRENT_DCP_VERSION = "1.0.0"
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_spec import (  # noqa: E402
    OPTIMIZER_KINDS, ModelSpec, ParallelConfig, ShardEntry, add_optimizer_state,
    assign_files, plan_shards, preset,
)

CHUNK = 32 << 20  # bytes per write when filling for real
PAYLOAD_ENTRY = b"archive/data/0"


# ---------------------------------------------------------------------------
# blob framing
# ---------------------------------------------------------------------------


class _PayloadSpy(io.RawIOBase):
    """Capture torch.save's framing bytes while discarding the payload write.

    The payload cannot be identified by length alone: zip data descriptors are
    exactly 16 bytes, so a 16-byte tensor would latch onto the wrong write. It
    is identified structurally instead. Every zip entry torch.save emits is
    written as (local header, name, alignment padding, data, data descriptor),
    so the payload is the second write after the `archive/data/0` name.
    """

    def __init__(self, payload_len: int) -> None:
        self.payload_len = payload_len
        self.pre = bytearray()
        self.post = bytearray()
        self.stage = 0  # 0 before name, 1 name seen, 2 padding seen, 3 done
        self.seen = False
        self.pos = 0

    def write(self, b) -> int:  # type: ignore[override]
        n = len(b)
        if self.stage == 3:
            self.post.extend(b)
        elif self.stage == 2:
            if n != self.payload_len:
                raise RuntimeError(
                    f"expected a {self.payload_len} B payload write, got {n} B"
                )
            self.stage = 3
            self.seen = True
        else:
            self.pre.extend(b)
            if self.stage == 0 and bytes(b) == PAYLOAD_ENTRY:
                self.stage = 1
            elif self.stage == 1:
                self.stage = 2
        self.pos += n
        return n

    def tell(self) -> int:
        return self.pos

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def _crc_slots(post: bytes, payload_len: int) -> tuple[int, ...]:
    """Byte offsets within `post` holding the payload entry's CRC-32.

    Two copies exist: the streaming data descriptor that immediately follows the
    payload, and the central-directory entry for `archive/data/0`. Both are
    located structurally rather than by pattern-matching a value.
    """
    slots = []
    if post[:4] == b"PK\x07\x08":
        slots.append(4)
    i = 0
    while True:
        i = post.find(b"PK\x01\x02", i)
        if i < 0:
            break
        name_len = struct.unpack_from("<H", post, i + 28)[0]
        if post[i + 46: i + 46 + name_len] == PAYLOAD_ENTRY:
            slots.append(i + 16)
        i += 4
    expected = zlib.crc32(b"\0" * payload_len) & 0xFFFFFFFF
    got = [struct.unpack_from("<I", post, s)[0] for s in slots]
    if len(slots) != 2 or any(g != expected for g in got):
        raise RuntimeError(
            f"could not locate the payload CRC in torch.save framing "
            f"(slots={slots}, values={got}, expected={expected}); "
            "the writer needs updating for this torch version"
        )
    return tuple(slots)


_TEMPLATES: dict[tuple, tuple[bytes, int, bytes, tuple[int, ...]]] = {}


def blob_template(shape: tuple[int, ...], dtype: torch.dtype):
    """(prefix, payload_len, suffix, crc_slots) for `torch.save` of this shape.

    The suffix is the zeros-payload version; `_patch_crc` corrects it for other
    content. Built once per distinct shape -- the 70B inventory has ten.
    """
    key = (shape, dtype)
    if key not in _TEMPLATES:
        t = torch.zeros(shape, dtype=dtype)
        payload = t.numel() * t.element_size()
        if payload >= 1 << 32:
            raise SystemExit(f"shard {shape} is {payload} B; zip64 framing not handled")
        spy = _PayloadSpy(payload)
        torch.save(t, spy)
        del t
        if not spy.seen:
            raise RuntimeError(
                f"torch.save did not emit the payload for {shape} as one write; "
                "the streaming writer needs updating for this torch version"
            )
        post = bytes(spy.post)
        _TEMPLATES[key] = (bytes(spy.pre), payload, post, _crc_slots(post, payload))
    return _TEMPLATES[key]


def _patch_crc(post: bytes, slots: tuple[int, ...], crc: int) -> bytes:
    buf = bytearray(post)
    for s in slots:
        struct.pack_into("<I", buf, s, crc & 0xFFFFFFFF)
    return bytes(buf)


def verify_templates() -> None:
    """Assert template-assembled blobs are byte-identical to real torch.save."""
    for shape in [(6, 5), (8,), (64, 32)]:
        pre, plen, post, slots = blob_template(shape, torch.bfloat16)
        # zeros: framing must match verbatim
        z = torch.zeros(shape, dtype=torch.bfloat16)
        real = io.BytesIO(); torch.save(z, real)
        rebuilt = pre + b"\0" * plen + post
        assert rebuilt == real.getvalue(), f"zeros template mismatch at {shape}"
        # non-zero data: framing must match once the CRC is patched, except for
        # the content-derived serialization_id, which nothing validates on load.
        t = torch.arange(z.numel(), dtype=torch.bfloat16).reshape(shape)
        raw = t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        real2 = io.BytesIO(); torch.save(t, real2)
        mine = pre + raw + _patch_crc(post, slots, zlib.crc32(raw))
        a, b = bytearray(mine), bytearray(real2.getvalue())
        assert len(a) == len(b), f"length mismatch at {shape}"
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        # the only permitted divergence is the 40-byte serialization_id field
        assert len(diff) <= 40, f"unexpected divergence at {shape}: {len(diff)} bytes"
        # and the patched blob must still load
        back = torch.load(io.BytesIO(mine), map_location="cpu", weights_only=True)
        assert torch.equal(back, t), f"patched blob did not round-trip at {shape}"


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _only_known_fields(cls, kwargs: dict, construct: bool = False):
    """Drop kwargs this torch version's dataclass does not define.

    DCP's metadata dataclasses gain fields over time -- `Metadata.version`
    exists on 2.6+ but not 2.5 -- and passing an unknown one is a TypeError.
    Filtering keeps a single writer working across versions, and a field that
    does not exist is also a field the matching reader never looks for.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    kept = {k: v for k, v in kwargs.items() if k in known}
    return cls(**kept) if construct else kept


def _planner_path(fqn: str, kind: str) -> tuple:
    """The nested key path DCP's `flatten_state_dict` would have recorded.

    Model tensors are saved at the top level under their HuggingFace names;
    optimizer state is saved under `{"optim": {"state": {<fqn>: {<moment>: t}}}}`,
    which flattens to `optim.state.<fqn>.<moment>` with the fqn kept whole.
    """
    if kind == "model":
        return (fqn,)
    rest = fqn[len("optim.state."):]
    base, suffix = rest.rsplit(".", 1)
    return ("optim", "state", base, suffix)


def _write_one_file(job) -> tuple[str, int, list]:
    """Write a single `.distcp` file. Returns its per-shard storage info.

    Runs in a worker process: shard files are independent, which is also how a
    real save behaves -- one writer, one file, no coordination until
    `.metadata` is written at the end by the coordinator.
    """
    rel, shards, out, fill, seed = job
    rng = np.random.default_rng([seed, zlib.crc32(rel.encode())])
    offset = 0
    rows = []
    with open(Path(out) / rel, "wb", buffering=1 << 20) as fh:
        for fqn, offsets, sizes, chunk_index, dtype_name in shards:
            dtype = getattr(torch, dtype_name)
            pre, plen, post, slots = blob_template(tuple(sizes), dtype)
            fh.write(pre)
            if fill == "sparse":
                fh.seek(plen, os.SEEK_CUR)
                tail = post  # payload is zeros; template CRC already correct
            elif fill == "zeros":
                remaining = plen
                while remaining:
                    n = min(CHUNK, remaining)
                    fh.write(b"\0" * n)
                    remaining -= n
                tail = post
            else:
                crc = 0
                remaining = plen
                while remaining:
                    n = min(CHUNK, remaining)
                    buf = rng.bytes(n)
                    fh.write(buf)
                    crc = zlib.crc32(buf, crc)
                    remaining -= n
                tail = _patch_crc(post, slots, crc)
            fh.write(tail)
            length = len(pre) + plen + len(tail)
            rows.append((fqn, offsets, chunk_index, offset, length))
            offset += length
        if fill == "sparse":
            fh.truncate(offset)
    return rel, offset, rows


def populate(
    model: ModelSpec,
    cfg: ParallelConfig,
    out: Path,
    fill: str,
    optimizer: str = "none",
    layout: str = "per-rank",
    thread_count: int = 1,
    seed: int = 0,
    workers: int = 1,
) -> dict:
    entries = add_optimizer_state(plan_shards(model, cfg), optimizer)
    files = assign_files(entries, layout, thread_count)

    out.mkdir(parents=True, exist_ok=True)
    # Warm the template cache in the parent: ten distinct shapes, not 14k.
    for e in entries:
        blob_template(tuple(e.sizes), getattr(torch, e.dtype))

    jobs = [
        (rel, [(e.fqn, e.offsets, e.sizes, e.chunk_index, e.dtype) for e in items],
         str(out), fill, seed)
        for rel, items in files
    ]

    t0 = time.time()
    total_bytes = sum(e.nbytes() for e in entries)
    results = []
    done_bytes = 0
    # Report at most ~25 lines, but never less often than every file when there
    # are few of them -- the per-rank layout is 8 huge files, and a silent
    # 10-minute write looks exactly like a hang.
    step = max(1, len(jobs) // 25)

    def report(i, extra=0):
        el = time.time() - t0
        rate = (done_bytes + extra) / el / 1e6 if el else 0
        eta = max(0.0, (total_bytes - done_bytes - extra) / (rate * 1e6)) if rate else 0
        print(f"  [{100*i/len(jobs):5.1f}%] {i:,}/{len(jobs):,} files  "
              f"{(done_bytes+extra)/1e9:6.1f}/{total_bytes/1e9:.1f} GB  "
              f"{rate:6.0f} MB/s  {el:5.0f}s elapsed, ~{eta:.0f}s left", flush=True)

    if workers > 1:
        # as_completed, not map: map yields strictly in submission order, so one
        # slow job stalls all reporting. Chunking is sized from the job count --
        # a fixed chunksize of 8 put all 8 per-rank jobs in ONE worker and
        # serialised an 80 GB write onto a single process.
        n_work = min(workers, len(jobs))
        with ProcessPoolExecutor(max_workers=n_work) as ex:
            futs = [ex.submit(_write_one_file, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                results.append(res)
                done_bytes += res[1]
                if i % step == 0 or i == len(jobs):
                    report(i)
    else:
        for i, job in enumerate(jobs, 1):
            res = _write_one_file(job)
            results.append(res)
            done_bytes += res[1]
            if i % step == 0 or i == len(jobs):
                report(i)

    storage_data: dict[MetadataIndex, _StorageInfo] = {}
    file_rows = []
    kinds = {e.fqn: e.kind for e in entries}
    for rel, size, rows in sorted(results):
        for fqn, offsets, chunk_index, off, length in rows:
            storage_data[MetadataIndex(fqn, offsets, chunk_index)] = _StorageInfo(
                relative_path=rel, offset=off, length=length
            )
        file_rows.append({
            "file": rel, "shards": len(rows), "bytes": size,
            "kinds": sorted({kinds[r[0]] for r in rows}),
        })

    # ---- .metadata ----
    sdm: dict[str, TensorStorageMetadata] = {}
    for e in entries:
        tsm = sdm.get(e.fqn)
        if tsm is None:
            tsm = TensorStorageMetadata(
                properties=TensorProperties(dtype=getattr(torch, e.dtype)),
                size=torch.Size(e.global_shape), chunks=[])
            sdm[e.fqn] = tsm
        tsm.chunks.append(
            ChunkStorageMetadata(offsets=torch.Size(e.offsets), sizes=torch.Size(e.sizes))
        )
    for tsm in sdm.values():
        tsm.chunks.sort(key=lambda c: tuple(c.offsets))

    # Build Metadata from whatever fields this torch actually has. `version`
    # arrived after 2.5, and passing it to an older dataclass is a TypeError --
    # so the fields are filtered rather than assumed. Same for StorageMeta.
    md = Metadata(**_only_known_fields(Metadata, {
        "state_dict_metadata": sdm,
        "planner_data": {e.fqn: _planner_path(e.fqn, e.kind) for e in entries},
        "storage_data": storage_data,
        "storage_meta": _only_known_fields(StorageMeta, {
            "checkpoint_id": str(out), "save_id": f"synthetic-{model.name}",
        }, construct=True),
        "version": CURRENT_DCP_VERSION,
    }))
    tmp = out / ".metadata.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(md, f)
    tmp.replace(out / ".metadata")
    md_bytes = (out / ".metadata").stat().st_size

    opt_bytes = sum(e.nbytes() for e in entries if e.kind == "optim")
    return {
        "model": model.name,
        "num_hidden_layers": model.num_hidden_layers,
        "save_config": str(cfg),
        "world_size": cfg.world_size,
        "dtype": model.torch_dtype,
        "optimizer": optimizer,
        "layout": layout,
        "thread_count": thread_count,
        "parameters": model.num_parameters(),
        "tensors": len(sdm),
        "shards": len(entries),
        "shard_files": len(file_rows),
        "metadata_bytes": md_bytes,
        "payload_bytes": total_bytes,
        "optimizer_bytes": opt_bytes,
        "apparent_bytes": sum(r["bytes"] for r in file_rows) + md_bytes,
        "fill": fill,
        "elapsed_s": round(time.time() - t0, 1),
        "files": file_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama31_70b")
    ap.add_argument("--layers", type=int, default=None,
                    help="override num_hidden_layers (for cheap end-to-end runs)")
    ap.add_argument("--save-config", default="TP8/PP4/DP1",
                    help="writer parallelism: the config the training job ran at")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fill", choices=["random", "zeros", "sparse"], default="random")
    ap.add_argument("--optimizer", choices=sorted(OPTIMIZER_KINDS), default="none",
                    help="optimizer state to include alongside the weights")
    ap.add_argument("--layout", choices=["per-rank", "per-rank-threads", "per-shard"],
                    default="per-rank", help="DCP writer file layout (see llama_spec.assign_files)")
    ap.add_argument("--thread-count", type=int, default=1,
                    help="files per rank when --layout per-rank-threads")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    verify_templates()
    model = preset(args.model, args.layers)
    cfg = ParallelConfig.parse(args.save_config)

    print(f"model      {model.name}  {model.num_hidden_layers} layers, "
          f"{model.num_parameters():,} params, {model.torch_dtype}")
    print(f"save cfg   {cfg}  world_size={cfg.world_size}")
    print(f"layout     {args.layout}"
          + (f" x{args.thread_count}" if args.layout == "per-rank-threads" else "")
          + f"  optimizer={args.optimizer}")
    print(f"target     {args.out}  fill={args.fill} workers={args.workers}\n")

    info = populate(model, cfg, args.out, args.fill, args.optimizer, args.layout,
                    args.thread_count, args.seed, args.workers)

    print(f"\nwrote {info['shard_files']} shard files + .metadata "
          f"({info['metadata_bytes']:,} B) in {info['elapsed_s']}s")
    print(f"apparent size {info['apparent_bytes']/1e9:.2f} GB  "
          f"({info['shards']:,} shards over {info['tensors']:,} tensors)")
    if info["optimizer_bytes"]:
        print(f"of which optimizer state {info['optimizer_bytes']/1e9:.2f} GB "
              f"({100*info['optimizer_bytes']/info['payload_bytes']:.1f}%)")

    manifest = args.manifest or (args.out.parent / f"{args.out.name}.manifest.json")
    manifest.write_text(json.dumps(info, indent=2))
    print(f"manifest   {manifest}")


if __name__ == "__main__":
    main()
