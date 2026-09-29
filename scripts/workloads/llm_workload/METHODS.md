# Methods: an evaluation-shaped checkpoint read of Llama-3.1-70B

This describes how the checkpoint-read workload in `llm_workload/` is
constructed, and — the point of the document — which of its settings come from
a paper or from upstream library code, and which are engineering judgements we
made. Every configuration choice appears in one of the two tables in §3.

Nothing here is a simulation of DCP. The reading side is stock
`torch.distributed.checkpoint`: its load planner, its resharding arithmetic and
its file reads are unmodified. What we built is the checkpoint it reads, and
the instrumentation around it.

---

## 1. What the workload is

A training job saves a Llama-3.1-70B checkpoint under one parallelism
configuration. Later, a *separate* job on a *separate* allocation loads it under
a different, inference-shaped configuration, runs a forward-pass stand-in, and
exits without writing anything back. Darshan (or strace) captures the reading
job.

Three scripts, in order:

| stage | script | what it does |
|---|---|---|
| populate | `populate_checkpoint.py` | writes a real DCP checkpoint to the PFS |
| verify | `verify_checkpoint.py` | proves it is a real DCP checkpoint |
| read | `eval_load.py` | runs `dcp.load` and records the access pattern |

`run/` wraps these as plain shell scripts for a node with no batch scheduler,
handling BeeGFS pool and stripe pinning, cold-cache enforcement and Darshan
setup; `run/schedule_jobs.py` reproduces the production eval-to-restart ratio.
`capacity.py` sizes a run against the machine's actual RAM. `parse_darshan.py`
reduces the resulting logs; `parse_eval_strace.py` is the fallback when Darshan
is unavailable.

Darshan is read two ways, because they answer different questions.
`darshan-parser` gives per-file aggregate counters — enough for the file
classification in §4.1. `darshan-dxt-parser` gives every operation with offset
and timing, and is the only source for two things: proving that distinct ranks
read the *same byte ranges* (§4.2, which otherwise rests on DCP's own plan
rather than on an independent observation), and measuring the buffered
over-fetch in §4.3 rather than estimating it. DXT must be enabled at run time
and **silently truncates** when its per-process buffer fills, so the DXT
operation count is checked against the aggregate counters and a shortfall is
reported as truncation rather than as data.

One caveat on stripe attribution: BeeGFS gives each file its own ordered list of
targets, so `offset / chunksize mod numtargets` yields a *slot within that
file's list*, not a target ID, and slot 0 of two files is usually two different
disks. `run/make_target_map.sh` resolves slots to real IDs via
`beegfs-ctl --getentryinfo`; without it the per-target concurrency figures are
not comparable across files, and the tool says so in its output.

### 1.1 Why the checkpoint is written by us rather than by a training run

A genuine 70B save needs 32 GPUs holding 141 GB of parameters. We do not have
that, and renting it would buy nothing: the *write* side is not what this
experiment measures. So `populate_checkpoint.py` streams the checkpoint shard by
shard, never holding more than one shard in memory.

This is only legitimate if the result is byte-for-byte what a real save
produces. DCP's on-disk contract is small enough to meet exactly:

* `.metadata` is a plain `pickle.dump` of `torch.distributed.checkpoint.
  metadata.Metadata` — per-tensor global shape, dtype and chunk list, plus
  `storage_data`, mapping `MetadataIndex(fqn, chunk_offsets)` to
  `_StorageInfo(relative_path, offset, length)`.
* `__{rank}_{n}.distcp` is the raw concatenation of one `torch.save(tensor)`
  blob per shard, at those offsets.

`FileSystemReader.read_data` consults nothing else. The framing around each
payload is captured from a real `torch.save` per distinct shard shape rather
than reconstructed by hand, and the zip CRC-32 is recomputed when the payload
is not zeros, so the files pass `zipfile.testzip()` — a check DCP itself does
not perform.

**`verify_checkpoint.py` is the argument that this worked**, and it is the
thing to re-run if anything below is doubted. Check 1 confirms every shard blob
is a structurally valid zip with a correct CRC that `torch.load` accepts. Check
2 assembles each global tensor by hand from the recorded byte offsets, runs a
real `dcp.load` over the same directory, and requires the two to be *bitwise*
identical. If our sharding arithmetic disagreed with DCP's anywhere, check 2
fails.

Coverage actually run, stated exactly because the two checks cost very
different amounts: check 1 passed over all 4,657 blobs of the full 80-layer
model (sparse fill, so CRCs are the zeros CRCs) and over all 387 blobs of a
reduced-scale model with real random payloads. Check 2 passed for all 117
tensors at reduced scale with random payloads, and for the first 12 tensors
(1.98 billion elements, offsets up to 4.5 GB) at full 70B scale. Check 2 over
every tensor of the full model is a cluster-sized job, not a laptop one.

### 1.2 Fill modes, and the one that is not valid for timing

`--fill random` writes real pseudorandom bytes and is what the cluster runs use.
`--fill sparse` writes the framing but seeks over each payload, so the file has
the correct size, offsets and CRC while occupying almost no blocks: a 705 GB
checkpoint materialises in under a second and costs 27 MB. That is how the
full-scale access-pattern numbers in §4 were obtained on a laptop with 2.5 GB
free.

**Sparse mode is not valid for bandwidth or latency measurement.** Reads of a
hole are served by the kernel and never reach the media. It is for access
*patterns* — which files, which offsets, which lengths, how many times — and
those are identical to the byte-real case by construction. Any timing number
must come from `--fill random` on the PFS.

---

## 2. The four knobs that make a read an *eval* read

The distinction this experiment rests on is eval versus restart. These are the
only things that differ between `02_eval_read.sbatch` and
`03_restart_read.sbatch`; everything else — model, checkpoint, reader width,
node count — is held fixed so the comparison isolates them.

1. **State subset.** Eval asks for model weights only. Restart also asks for
   Adam moments.
2. **Parallelism mismatch.** The reader mesh differs from the writer mesh and is
   inference-shaped: tensor-parallel only, small rank count.
3. **Post-load behaviour.** A forward-pass stand-in, then exit. Never writes
   back.
4. **Frequency.** Eval fires against every checkpoint generation, about ten
   times as often as restart.

The mechanism behind knob 1 is worth stating precisely, because it is the
result that carries the most weight and it is *not* a policy anyone configures.
`eval_load.py:build_eval_state_dict` builds its state dict from the module
geometry, not from the checkpoint's `.metadata`. A serving loader instantiates
an `nn.Module` tree and asks the checkpoint to fill it; anything on disk that
the module tree does not name is never requested, so DCP's planner never emits a
read for it. The optimizer is skipped because there is nowhere to put it, not
because the job opted out.

---

## 3. Provenance of every setting

### 3.1 Taken from a paper or from upstream code

| setting | value | source |
|---|---|---|
| Model geometry | 80 layers, `hidden=8192`, `ffn=28672`, `vocab=128256`, 64 heads, 8 KV heads, bf16 | `meta-llama/Llama-3.1-70B` `config.json`. Yields 70,553,706,496 parameters = 141.107 GB, matching the published model size. |
| Tensor names | `model.layers.N.self_attn.q_proj.weight`, … | `transformers` `LlamaForCausalLM` state dict — the keys `from_pretrained` and vLLM's `llama.py` loader key off. |
| TP sharding rules | q/k/v, gate/up, embedding, LM head column-parallel (dim 0); o_proj, down_proj row-parallel (dim 1); RMSNorm replicated | Megatron-LM's Llama parallelisation, reproduced by vLLM. |
| Checkpoint format | `.metadata` pickle + `__{rank}_{n}.distcp` | `torch.distributed.checkpoint.filesystem`, PyTorch 2.9. |
| Replicated-tensor placement | lowest-loaded rank in the replication group | `_dedup_save_plans.dedup_save_plans`: `min(plan_indices, key=plan_to_size)` walking write items in plan order. Reproduced exactly. |
| File layouts | `per-rank`, `per-rank-threads`, `per-shard` | `FileSystemWriter.write_data` + `_split_by_size_and_type`. The three layouts correspond to `single_file_per_rank`/`thread_count` settings. |
| Whole-shard reads | DCP reads a full stored shard, then narrows in memory | `FileSystemReader.read_data`: `torch.load(...)` then `narrow_tensor_by_index`. There is no byte-range read below shard granularity. |
| Resharding on mismatch | load plan intersects requested chunks against stored chunks | DCP default planner; the mechanism ByteCheckpoint (NSDI'25 §3.2) calls a parallelism-agnostic representation. |
| Eval : restart ratio | 19,844 : 1,870 = 10.61 : 1 over six months | ByteCheckpoint, NSDI'25 (arXiv:2407.20143), production census. Cross-stage (13,080) reported separately. |
| No redundant-load elimination | each rank independently reads what it needs | Stock DCP has no equivalent of ByteCheckpoint §4.1. This is what produces the amplification in §4.2. |
| Filter operations by size | small framing reads dominate op counts, not bytes | tf-Darshan, Cluster'20, warns that DL traces are dominated by tiny operations; §4.3 measures ours. |
| Separate allocations | read job ≠ write job | A read in the allocation that wrote the file is served from page cache. Standard practice, and the reason `01_populate` and `02_eval_read` are separate scripts. |

### 3.2 Our engineering judgements

Each of these is defensible, none is from a paper, and each is a knob so its
effect can be measured rather than argued.

| setting | value | why, and how to test it |
|---|---|---|
| Eval reads weights only | `--read-set eval` | Justified by loader *structure* (§2), not by a measurement. `--read-set restart` is the contrast. The strongest available check is a Darshan trace of a real eval job on your own cluster — open experiment 5 in the context notes. |
| Writer parallelism | `TP8/PP4/DP1`, 32 ranks | A standard 70B training shape; TP8 within a node, PP4 across. TP must divide `num_key_value_heads=8`, which bounds it to {1,2,4,8}. Arbitrary within that; sweep with `--save-config`. |
| Reader parallelism | `TP4`, 4 ranks | Llama-3.1-70B needs ≥141 GB of weights, so vLLM-class serving uses 4×80 GB or 8×80 GB. Picked to be *smaller* than the writer and pipeline-free. The sweep in §4.2 covers TP1–TP32. |
| Optimizer model | Adam `exp_avg` + `exp_avg_sq`, fp32, sharded like the parameter | 8 B/param → optimizer is 80.0% of the checkpoint, matching the ~80% figure in the literature. `adam_master` adds fp32 master weights (12 B/param, 85.7%). Under ZeRO-1 the moments are re-sharded flat across the DP group, which changes their shapes but not their bytes, and not the fact that eval never asks for them. |
| Pipeline-stage split | layers in `pp` equal contiguous blocks; embedding on the first stage, final norm and LM head on the last | Megatron-LM's default. Real 70B runs often rebalance end stages to offset the LM head; that moves a few tensors between files and changes no total. |
| Key naming | model tensors at top level under HF names; optimizer under `optim.state.<fqn>.<moment>` | Corresponds to `dcp.save({**model_sd, "optim": optim_sd})`. Nesting the model under its own key instead would prefix every name; it changes the strings in `.metadata`, nothing else. |
| Forward-pass stand-in | byte-level checksum over loaded shards | Exists so the load cannot be optimised away and the job's lifetime resembles an eval task's. A byte checksum rather than an arithmetic one because random *bits* decode to bfloat16 full of NaN. |
| Eval covers every generation | `--eval-coverage 1.0` | The census gives totals, not coverage. An eval harness wired to a training run fires on each checkpoint; that is the assumption. |
| Cross-stage excluded | `--crossstage-as exclude` | SFT/RL warm starts are neither of the two jobs here. Folding them into restart gives 1.33:1, into eval 17.61:1 — a large swing, so the flag is explicit and results should state which was used. |
| Darshan config | `DARSHAN_ENABLE_NONMPI=1`, `DXT_ENABLE_IO_TRACE=1` | gloo is not MPI, so without the first there is no log at all. Aggregate POSIX counters cannot show per-offset reuse, so DXT is required to see the amplification in §4.2. |

### 3.3 Known limits

* **Sparse fill cannot produce timing numbers.** §1.2.
* **Darshan records submission, not completion**, and has no `liburing`
  integration (context notes §7). DCP's reader is synchronous and buffered, so
  this bites less here than for async engines — but it still means Darshan
  timestamps mark when a read was issued.
* **§4.3 was measured on ext4 and does not transfer.** This was flagged as a
  falsifiable prediction and has now been falsified: the target BeeGFS mount
  reports `st_blksize = 524288`, so the ≤64 KB read population does not exist
  there. The section has been rewritten and the media-placement argument built
  on it retracted. Any filesystem-level claim in this document should be
  re-derived per filesystem; `run/00_check_env.sh` probes the value.
* **strace must be run with `-ff`, not `-f`.** PyTorch's loader is threaded, so
  `-f` splits concurrent syscalls into `<unfinished ...>` / `<... resumed>`
  pairs across interleaved lines. A parser that misses those recovered **7% of
  the bytes DCP actually read** in our first attempt — plausible-looking and
  completely wrong. `parse_eval_strace.py` now takes `--expect-bytes` and
  compares against DCP's own total; the syscall figure must *exceed* it (by the
  zip framing, ~1.05x measured). Below 1.0 means the trace is incomplete.
* **Page cache is not handled by the environment any more.** The Slurm version
  got cold caches free from separate allocations. On a scheduler-less node it
  is explicit, and dropping caches on the client does **not** drop them on the
  OSSs, which have their own RAM. `run/env.sh:drop_caches` covers both and
  warns loudly when `OSS_HOSTS` is unset.
* **One DCP version.** All of the above is PyTorch 2.9.0. The read path carries
  a `# TODO sort by offset and cache the reading` comment upstream; if that is
  implemented, §4.3 changes.
* **We did not measure a real eval job.** The state-subset knob is justified by
  loader structure, not observation.

---

## 4. What it found

All figures: Llama-3.1-70B, saved `TP8/PP4/DP1` (32 ranks), Adam `exp_avg` +
`exp_avg_sq`. Checkpoint totals 705.56 GB — 141.11 GB weights, 564.43 GB
optimizer (80.0%), 13,971 shards over 2,169 tensors, `.metadata` 2.76 MB.

### 4.1 Eval reads a fifth of the checkpoint, but layout decides whether that is visible

Reader `TP4`, identical in both rows except `--read-set`:

| | bytes read | requests | avg request | files opened |
|---|---|---|---|---|
| eval | 141.12 GB (20.0%) | 5,140 | 27.46 MB | 4,657 of 13,971 |
| restart | 705.60 GB (100%) | 15,420 | 45.76 MB | 13,971 of 13,971 |

The same eval read against the same checkpoint written in a different layout:

| writer layout | DCP writer setting | files | opened by eval | never opened |
|---|---|---|---|---|
| `per-rank` | `single_file_per_rank=True, thread_count=1` (default) | 32 | **32** | **0** |
| `per-rank-threads` ×8 | `single_file_per_rank=True, thread_count=8` | 256 | **256** | **0** |
| `per-shard` | `single_file_per_rank=False` | 13,971 | 4,657 | **9,314 (66.7% of files, 564.4 GB = 80.0% of bytes)** |

This is the result with the most consequence for placement. The bytes read are
identical in all three rows; only the writer's flags differ. Under both layouts
that write whole ranks to files, the cold 564 GB of optimizer state is
interleaved *inside* files the eval job must open, so no file-level policy can
separate hot from cold. Only `single_file_per_rank=False`, which writes one
file per tensor shard, makes the two populations distinct files.

`per-rank-threads` is worth calling out because it looks like it should help and
does not. `_split_by_size_and_type` buckets tensors by *size* only, despite the
name, so model and optimizer shards are mixed across the eight files per rank.
At reduced scale it happened to segregate them (16 of 32 files cold), because
fp32 moments sort ahead of bf16 weights; at full scale that coincidence
disappears entirely and it leaves zero cold files. Partial separation from size
bin-packing is not a property to rely on.

**Whether per-file placement is possible at all is decided by a writer flag the
storage system does not control** — which is worth knowing before designing a
policy around it. The planner in `scripts/reshard_planner/` assumed the
DeepSpeed/DLIO layout, which is file-per-shard-like and therefore separable;
that assumption holds there and fails for DCP's default.

### 4.2 Resharding causes read amplification, not fragmentation

Load width swept against the same checkpoint (`--read-set eval`, model-only):

| load TP | requests | bytes read | amplification | avg request |
|---|---|---|---|---|
| TP1 | 4,657 | 141.11 GB | 1.00× | 30.3 MB |
| TP2 | 4,818 | 141.12 GB | 1.00× | 29.3 MB |
| TP4 | 5,140 | 141.12 GB | 1.00× | 27.5 MB |
| TP8 | 5,784 | 141.13 GB | 1.00× | 24.4 MB |
| TP16 | 11,568 | 282.27 GB | **2.00×** | 24.4 MB |
| TP32 | 23,136 | 564.54 GB | **4.00×** | 24.4 MB |

**This contradicts `scripts/reshard_planner/`**, which predicted that resharding
past the saved width (`load_TP > save_TP = 8`) fragments reads into ~524,000
requests of ~20 KB. Real DCP never fragments: the average request does not drop
below 24.4 MB at any width. The planner's model — a rank reads exactly the byte
ranges it needs — is not how DCP behaves. `read_data` reads the *whole* stored
shard and calls `narrow_tensor_by_index` in memory, so a rank needing half a
shard reads all of it.

Past the saved width the cost reappears as duplication: at TP16 each stored
shard is wanted by two loading ranks and each reads it in full. Bytes scale as
`load_TP / save_TP` while request size stays flat.

The two failure modes call for opposite responses. Fragmentation is a latency
and IOPS problem, which argues for low-latency placement. Amplification is a
bandwidth-and-duplication problem, which argues for replication or a read cache
and is indifferent to media latency. **The planner's conclusion — that only
`load_TP > save_TP` configurations are worth cluster time — survives, but its
stated reason does not.** The amplification is also precisely what
ByteCheckpoint's redundant-load elimination (NSDI'25 §4.1) removes and stock
DCP does not implement, so it is a real gap rather than a defect.

### 4.3 The syscall trace is filesystem-dependent, and the ext4 result does not transfer

strace of a real `--mode load`, counting only checkpoint files, **on ext4**:

| view | operations | mean size |
|---|---|---|
| DCP logical requests | 129 | 0.7 MB |
| POSIX read syscalls | 1,449 | 66.2 KB |

91.7% of read syscalls were ≤64 KB and carried 5.68% of the bytes, with 3.5
`lseek` per read. The small reads are `torch.load` parsing each blob's zip
framing through a buffered reader before touching the payload.

**That result is an artefact of ext4 and must not be quoted for BeeGFS.**
CPython sizes a buffered reader from `os.fstat().st_blksize`, which is 4096 on
ext4. Measured on the target BeeGFS mount:

```
st_blksize = 524288
read(3, "...", 524288) = 524288     # issued for a 100-byte request
```

So on BeeGFS the same code issues **512 KB** reads, and the ≤64 KB population
does not exist. The op-count/byte-count divergence above — which we had
presented as an instance of the tf-Darshan (Cluster'20) hazard — **does not
occur on this cluster**. What replaces it is over-fetch: every framing read
pulls 512 KB to obtain ~1.6 KB of zip header.

The shard inventory is exact (it follows from §3.1); the over-fetch column is
an **estimate**, not a measurement, assuming 1-3 buffer fills per blob before
the payload read passes through. It is recorded here so the cluster run has
something to falsify, and `run/02_eval_read.sh` replaces it with measured
values via `--expect-bytes`.

| shard class | count | shard size | est. over-fetch |
|---|---|---|---|
| embedding / LM head | 16 | 262 MB | <1% |
| MLP gate/up/down | 1,920 | 58.7 MB | ~2% |
| attention q/o | 1,280 | 16.8 MB | ~6% |
| attention k/v | 1,280 | 2.1 MB | ~25-50% |
| RMSNorm | 161 | 16 KB | ~3,000% |
| **total** | **4,657** | **141 GB** | **~5%** |

The relative over-fetch on the smallest tensors is dramatic and the absolute
cost is not: even at one full buffer fill each, the 161 RMSNorm shards waste
~82 MB against a 141 GB read. Read amplification (§4.2) is 20-80× larger than
the whole of this column at TP16/TP32 and remains the dominant effect.

The measured check is simple: `parse_eval_strace.py --expect-bytes` reports the
ratio of syscall bytes to DCP's logical bytes. On ext4 it is **1.05×**. If the
estimate above is right, BeeGFS should land near the same figure by a completely
different route — few large over-fetches rather than many small ones — and if it
lands much higher, the buffer-fill count per blob is worse than assumed.

One property worth noting: the 512 KB buffer **exactly equals the BeeGFS chunk
size**, so each framing read maps to one chunk on one target. Payload reads are
larger than the buffer, so CPython passes them straight through and the client
can stripe them. The two read classes therefore behave quite differently
against the stripe, which is measurable and, as far as we know, uncharacterised.

**Consequence for media placement, stated as a retraction.** We previously
argued that the small-operation population might let NVMe beat HDD through the
IOPS channel even under a 2.5 GbE ceiling. With a 512 KB floor on every read
there is no such population, and that argument does not hold. For sequential
reads a 4-target HDD stripe (~800 MB/s) already exceeds the client NIC
(~312 MB/s), so a straight eval read should be network-bound and media-blind.

The remaining channel where media could still separate is **concurrency**, and
§4.2 is what creates it: at TP32 the load is 32 ranks × 4 stripe targets ≈ 128
concurrent streams over 11 HDD targets, roughly 12 sequential streams per
spindle, which is a seek-thrashing regime that NVMe is indifferent to. That is a
hypothesis, not a result, and `run/04_pool_ab.sh` swept across `WIDTHS` is the
test. It predicts HDD and NVMe converge at low load width and diverge at high
load width — the opposite shape from what we would have predicted before this
measurement.

### 4.4 Metadata fan-out

`.metadata` is 2.76 MB and every rank reads all of it before any data read.
That is 11.0 MB at TP4 and would be 88 MB at TP32, against a single file, at
job start — a shape no framework change removes, since the index genuinely is
global. It grows with shard count: 845 KB model-only, 2.76 MB with Adam state,
3.00 MB in `per-shard` layout.

---

## 5. Reproducing

```bash
# correctness, byte-real, ~0.5 GB, seconds
python3 populate_checkpoint.py --model llama_tiny --save-config TP4/PP2/DP1 \
        --optimizer adam --fill random --out /tmp/tiny
python3 verify_checkpoint.py --ckpt /tmp/tiny
torchrun --nproc_per_node=2 eval_load.py --ckpt /tmp/tiny --model llama_tiny \
        --load-config TP2 --mode load

# full 70B access pattern, sparse, ~27 MB on disk
python3 populate_checkpoint.py --model llama31_70b --save-config TP8/PP4/DP1 \
        --optimizer adam --layout per-shard --fill sparse --workers 8 --out /tmp/l70
torchrun --nproc_per_node=4 eval_load.py --ckpt /tmp/l70 --model llama31_70b \
        --load-config TP4 --mode plan --read-set eval

# on the cluster (no batch scheduler)
cd run && ./00_check_env.sh
POOL_ID=6 ./01_populate.sh          # byte-real, 706 GB, pinned to a pool
MODE=plan ./05_sweep.sh             # full 80-layer pattern, 0 GB RAM
MODE=load ./02_eval_read.sh         # timed, --only-layers autosized to RAM
./04_pool_ab.sh                     # hdd vs ssd
python3 run/schedule_jobs.py --scale 0.005 && ./submit_schedule.sh
```

`--mode plan` runs DCP's real planner without allocating tensors, and was
verified to emit a read list byte-identical to `--mode load`. That is what makes
the full-scale numbers above obtainable without 141 GB of RAM.
