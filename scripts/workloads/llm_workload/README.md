# `llm_workload/` — an eval-shaped checkpoint read of Llama-3.1-70B

Generates a real PyTorch DCP checkpoint for Llama-3.1-70B on a parallel
filesystem, then reads it back the way an **evaluation** job would — model
weights only, under a different and inference-shaped parallelism, forward pass,
exit — and records the per-file access pattern.

The reading side is unmodified `torch.distributed.checkpoint`. We built the
checkpoint it reads and the instrumentation around it, not the I/O behaviour.

[`METHODS.md`](METHODS.md) is the companion: it lists every configuration
choice and marks it as taken from a paper / upstream code, or as our
engineering judgement. Read it before quoting any number here.

## Why this exists

[`scripts/reshard_planner/`](../scripts/reshard_planner/README.md) computes the
same workload analytically, and ends by saying that confirming it against a real
`torch.distributed.checkpoint` reshard is the highest-value next step. This is
that step. It changed one of the planner's conclusions — see *Findings*.

## Files

| file | what it is |
|---|---|
| `llama_spec.py` | Llama-3.1-70B geometry, TP/PP sharding rules, optimizer state, DCP file layouts. Pure arithmetic — no torch runtime needed. |
| `populate_checkpoint.py` | Streams a format-exact DCP checkpoint without holding 70B parameters. |
| `verify_checkpoint.py` | Proves the result is a real DCP checkpoint. Run this if you doubt anything else. |
| `eval_load.py` | The measurement. Runs `dcp.load` and records every storage item fetched. |
| `parse_darshan.py` | Turns a run's Darshan logs into a per-file profile: classification, measured over-fetch, and per-byte-range read amplification. |
| `parse_eval_strace.py` | Same thing from an strace, for when Darshan isn't available. |
| `capacity.py` | How much RAM a given read needs, and how many layers fit in a budget. |
| `run/make_target_map.sh` | Maps each shard file to its real BeeGFS target IDs, so stripe attribution means something. |
| `run/` | Plain-shell runners for a node with no batch scheduler (anjuna2), with BeeGFS pool/stripe pinning, cache control and Darshan wiring. |

## Quick start

```bash
# correctness run: real bytes, ~0.5 GB, a few seconds
python3 populate_checkpoint.py --model llama_tiny --save-config TP4/PP2/DP1 \
        --optimizer adam --fill random --out /tmp/tiny
python3 verify_checkpoint.py --ckpt /tmp/tiny
torchrun --nproc_per_node=2 eval_load.py --ckpt /tmp/tiny --model llama_tiny \
        --load-config TP2 --mode load

# full 70B access pattern on a laptop: 706 GB apparent, ~27 MB on disk
python3 populate_checkpoint.py --model llama31_70b --save-config TP8/PP4/DP1 \
        --optimizer adam --layout per-shard --fill sparse --workers 8 --out /tmp/l70
torchrun --nproc_per_node=4 eval_load.py --ckpt /tmp/l70 --model llama31_70b \
        --load-config TP4 --mode plan --read-set eval
```

`--fill sparse` gives correct file sizes, offsets and CRCs while occupying
almost no blocks. **It is valid for access patterns and invalid for timing** —
reads of a hole never reach the media. Use `--fill random` on the PFS.

`--mode plan` runs DCP's real load planner without allocating tensors, so the
full-scale pattern is obtainable without 141 GB of RAM. Its read list was
verified byte-identical to `--mode load`.

## Key flags

| flag | meaning |
|---|---|
| `--save-config` / `--load-config` | `TP<n>/PP<n>/DP<n>`. Reader is TP-only. TP must divide `num_key_value_heads=8`. |
| `--optimizer` | `none` \| `adam` (m+v, 80% of bytes) \| `adam_master` (+fp32 master, 85.7%) |
| `--layout` | `per-rank` (DCP default, 1 file/rank) \| `per-rank-threads` \| `per-shard` (1 file/shard) |
| `--read-set` | `eval` = weights only \| `restart` = weights + optimizer |
| `--fill` | `random` (cluster) \| `zeros` \| `sparse` (laptop, patterns only) |
| `--only-layers N` | load a prefix of the layers, to bound memory in `--mode load` |

## Findings

Llama-3.1-70B, saved `TP8/PP4/DP1`, Adam m+v: 705.56 GB total (141.11 GB
weights + 564.43 GB optimizer), 13,971 shards, `.metadata` 2.76 MB.

**Eval reads 20% of the checkpoint — but the writer's layout decides whether
that is separable.** Same eval read, same bytes, different writer flag:

| writer layout | DCP writer setting | files | opened by eval | never opened |
|---|---|---|---|---|
| `per-rank` | `single_file_per_rank=True, thread_count=1` (default) | 32 | **32** | **0** |
| `per-rank-threads` ×8 | `single_file_per_rank=True, thread_count=8` | 256 | **256** | **0** |
| `per-shard` | `single_file_per_rank=False` | 13,971 | 4,657 | **9,314 (66.7% of files, 564.4 GB = 80.0% of bytes)** |

Only `single_file_per_rank=False` separates hot weights from cold optimizer
state into distinct files. In both other layouts the cold 564 GB is interleaved
*inside* files the eval job must open, so no file-level policy can act on it.
`per-rank-threads` buckets by size, not type, so it does not help — it happened
to at reduced scale and does not at full scale.

**Resharding causes read amplification, not fragmentation.** The planner
predicted ~524,000 requests of 20 KB once `load_TP > save_TP`. Real DCP never
drops below a 24.4 MB average at any width, because `read_data` reads the whole
stored shard and narrows in memory. Instead the bytes duplicate:

| load TP | requests | bytes | amplification |
|---|---|---|---|
| TP1–TP8 | 4,657–5,784 | 141 GB | 1.00× |
| TP16 | 11,568 | 282 GB | 2.00× |
| TP32 | 23,136 | 565 GB | 4.00× |

The planner's *recommendation* (only `load_TP > save_TP` needs cluster time)
survives; its stated reason does not. Amplification argues for replication or
caching, where fragmentation would have argued for low-latency media.

**The syscall trace is filesystem-dependent — and the ext4 result does not
transfer.** On ext4, 129 DCP requests became 1,449 read syscalls, 91.7% of them
≤64 KB carrying 5.68% of the bytes. That is an artefact of ext4's 4 KB
`st_blksize`, which CPython uses to size its buffered reader. The target BeeGFS
mount reports **`st_blksize = 524288`**, verified by a 100-byte request issuing
a `read(..., 524288)` syscall — so the small-read population does not exist
there. `METHODS.md` §4.3 has the rewritten version and retracts the media
argument that was built on it. Read amplification (above) is unaffected and is
20-80× larger anyway.

**`.metadata` fan-out.** 2.76 MB read in full by every rank before any data
read — 11 MB at TP4, 88 MB at TP32, against one file, at job start.

## On the cluster (anjuna2, no scheduler)

```bash
cd run
./00_check_env.sh                      # read this output before anything else
export POOL_ID=6 MODEL=llama31_70b     # 3 = hdd pool, 6 = ssd pool
./01_populate.sh                       # 706 GB, byte-real
MODE=plan ./05_sweep.sh                # amplification sweep, 0 GB RAM, full 80 layers
MODE=load ./02_eval_read.sh            # timed read, RAM-autosized --only-layers
./04_pool_ab.sh                        # hdd vs ssd, same checkpoint both ways
```

Settings live in `run/env.sh`. Three that will silently ruin results if left
alone:

- **`POOL_ID`** — unset means BeeGFS pool 1, which on this cluster is targets
  401-407: all on colva4, HDD and NVMe mixed. Useless for a media comparison.
- **`OSS_HOSTS`** — dropping page cache on the client does **not** drop it on
  colva1-4, which will serve the checkpoint back out of their own RAM. Set this
  or your pool A/B measures memory.
- **RAM** — a full 70B `--mode load` needs 141 GB resident. On a 62 GB node use
  `--mode plan` for access patterns (needs 0 GB, full 80 layers) and let
  `capacity.py` autosize `--only-layers` for timing runs.

## Requirements

PyTorch ≥ 2.6 (developed against 2.9.0) and NumPy. No GPU required — the CPU
gloo backend is enough for everything above. Darshan is optional; without
`darshan-runtime` the Slurm scripts fall back to strace and
`parse_eval_strace.py`.
