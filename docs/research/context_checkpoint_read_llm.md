# Context: LLM Pipeline I/O from a PFS Lens

Working notes for a trace-driven, ML-informed data placement system for
heterogeneous parallel file systems (BeeGFS + Darshan). Scope: where a PFS
actually sits in the LLM pipeline, and where per-file storage pooling might pay.

---

## 1. The deployment split (the single most important framing)

Storage answers differ by *deployment*, not by pipeline stage:

| | HPC / national lab | Hyperscaler / cloud |
|---|---|---|
| Job model | Slurm batch, fixed allocation | Kubernetes pods, elastic |
| Software | modules, conda, shared home on POSIX | container images |
| Storage | one PFS for everything, thin node-local NVMe | object store + fat node-local NVMe (30 TB/node) |
| Checkpoints | **PFS** | Tectonic / HDFS / Ceph |
| Failure model | job dies, resubmit | job survives node loss |

Most LLM storage literature reflects the hyperscaler world; most LLM
*checkpointing systems* papers (DataStates-LLM, AdaCheck, MLP-Offload) target
the HPC world because they're evaluated on Polaris/Frontier.

BITS Goa + BeeGFS is squarely HPC. That is a legitimate, funded, growing
category — not a legacy environment.

**Defensible framing:** hyperscalers solved LLM storage by *specializing
hardware* (enough local NVMe that shared storage leaves the critical path).
HPC centres can't specialize, so they must solve it in *software*.

---

## 2. PFS by stage

| Stage | HPC | Hyperscaler |
|---|---|---|
| Acquisition, clean, dedup | object store / PFS scratch | object store |
| Tokenize + pack | PFS | object store |
| Pretraining data reads | PFS or node-local | object store behind cache (Alluxio/JuiceFS/Quiver) |
| **Checkpoint write** | **PFS** | Tectonic/HDFS/Ceph |
| Restart | PFS | DFS, read-once-and-broadcast |
| **Eval / fine-tune loads** | **PFS** | DFS |
| Export / quantize | transient | object store |
| Deployment | n/a | object store + P2P |
| Inference serving, KV cache | n/a | purpose-built (3FS/USRBIO, Mooncake, LMCache) |

**Three PFS roles that survive everywhere:**
1. **Job launch** — code, conda env, shared libs. Tens of millions of
   stat/open calls per job start. Metadata-bound. The *only* role where POSIX
   is genuinely required. Nobody writes papers about it.
2. **Checkpoint traffic** — dominant by bytes on HPC.
3. **Eval / fine-tuning loads** — dominant by frequency.

Bytes-based and operation-based rankings of PFS load give **different answers**.
One production Lustre study: metadata ops up to 60% of all filesystem operations.

---

## 3. Access patterns (from Gossman et al., SCA/HPC Asia 2026)

**Checkpoint write — file layout is N × M**
- Each rank writes hundreds of files (file-per-shard). 3B model / 4 GPUs =
  **132 files, 42 GB per checkpoint**.
- Size distribution strongly modal: ~80% of files at one size
  (49 MB @3B, 105 MB @7B, 152 MB @13B); 3–8% >1 GB; small-file fraction
  *grows* with model size.
- Each file = nested structures. Tensors are pre-serialized contiguous byte
  streams; everything else needs pickle.
- Frameworks issue a write per data structure regardless of size → no
  coalescing → MDS contention, unaligned variable-size writes.
- Aggregation (file-per-process / single shared file at disjoint offsets) beats
  file-per-shard by ~34% synthetically; write throughput saturates at ~2 GB/rank.
- O_DIRECT: helps writes up to 4.8× (liburing); *hurts* small reads (buffered
  2.3× faster below 1 GB).

**Restore — serial and blocking**
- N ranks parallel, but each rank's M files read **serially** — next file only
  after previous object fully reconstructed on host AND GPU.
- On Polaris, **restore reads were slower than checkpoint writes** (reversed).
- ~half of restore time was **host memory allocation**, not I/O. Preallocation
  nearly doubled throughput.
- DataStates-LLM issues a read per metadata entry → ~3× read op count.

**Engine overhead end-to-end:** iteration 1.8× (DataStates-LLM), 3.2×
(TorchSnapshot), 4.5× (torch.save) slower than an idealized flush.

---

## 4. Resharding (from ByteCheckpoint, NSDI'25)

Transforming a distributed checkpoint to load under a *different* parallelism
config than it was saved with.

**Frequency, 6 months of ByteDance production:**
| Scenario | Instances |
|---|---|
| Pre-training resumption | 1,870 |
| Cross-stage (SFT/RL) | 13,080 |
| **Evaluation** | **19,844** |

Eval is ~10× more frequent than restart. **This is why restart is the wrong
optimization target.**

**Offline resharding:** download → transform → upload. 593–1,870s per event
before the real job can start. Full read + full write per event, and outputs
are parallelism-coupled so can't be reused.

**Load-time (online) resharding:** parallelism-agnostic representation. One
**global metadata file** (BasicMeta / ShardMeta / ByteMeta +
TensorShardToBasicByteMap) read by *every rank*, then selective byte-range
retrieval. Redundant-load elimination distributes DP-group reads and transfers
by all-to-all over idle inter-GPU bandwidth.

**Load is slower than save at scale:** ViT-7B on 1488 GPUs — save 20.13s,
**load 265.73s**. 405B on 8960 GPUs — save 51.06s, load 129.49s.
Resharding with full states (dataloader token buffer up to 20 GB): 401.21s.

**Small-file pathology:** dataloader state split into ~6 files per loader;
sequential upload accounted for **73.16% of total saving time** for a 7B ViT.

---

## 5. Where per-file pooling looks promising (ranked)

**Strong**
- **Eval / fine-tuning resharded loads.** Frequent, blocking, real cross-job
  reuse, genuine per-file heterogeneity (metadata vs optimizer vs parameter
  shards). Fan-out on the global metadata file is a storage-layer problem no
  framework fix dissolves.
- **Offline format conversion** (HF ↔ Megatron ↔ vLLM/safetensors). Full
  read-modify-write, on the critical path, no compute overlap, derived
  artifacts that can be *expired* not just demoted, and ten groups often hold
  duplicate copies of the same base model. **Nobody has characterized this.**
- **Pool isolation** (not pool speed). Separate OST pools so checkpoint bursts,
  conversion jobs, and dataloader reads don't collide. Costs peak bandwidth,
  buys predictability. Most valuable on a multi-tenant academic cluster.
- **Lifecycle demotion by age.** Newest checkpoint is hot (restart + eval);
  older ones are cold capacity but must be retained.

**Weak**
- **Checkpoint writes.** Async engines put the flush off the critical path;
  makespan is a *max* over concurrently-submitted files, so improving a subset
  is invisible. *Caveat:* holds for async engines only (torch.save blocks), and
  if stragglers correlate with size (3–8% >1 GB) the tail is predictable and
  targetable. Untested.
- **Restore.** Structurally amenable (serial blocking → per-rank time is a
  *sum*, so partial optimization shows up), but: infrequent, ~half the cost is
  memory allocation, and the serial blocking is an **implementation artifact**
  the DataStates-LLM authors have announced they will remove (preallocated
  buffers, batched submissions, multi-threaded completions).

**The baseline to beat:** ByteDance already ships two-tier SSD/HDD cool-down in
production — demote on **last modification time**, remap paths via pure metadata
ops. Justified by exactly the eval-read observation above. A classifier must beat
a timestamp threshold, not just beat no-tiering.

---

## 6. Open experiments (cheapest first)

1. **Per-file completion spread within one checkpoint burst.** Does the tail
   correlate with file size? If yes, the write case reopens.
2. **Restore time split: read vs allocation, on BeeGFS.** Only measured on
   Polaris (fast Lustre, 512 GB DRAM/node). A *different* split on BeeGFS is
   itself a finding.
3. **Job-mix census.** How many jobs on the cluster are format conversion vs
   distributed-checkpoint resume vs training? Does not exist in the literature.
4. **Fan-out scaling test.** Eval job reading one checkpoint from N nodes — does
   per-node read time grow linearly with N? If yes → replication, not tiering.
5. **Does eval actually skip optimizer state?** strace/Darshan an eval job.
   Optimizer state is ~80% of checkpoint bytes; if eval never reads it, that's a
   large, easily-defended result on its own.

**Instrumenting these: run the real libraries, don't synthesize.** Experiments
1, 2, 4 and 5 all need a genuine checkpoint workload underneath Darshan. Both
ByteCheckpoint (§4) and the primitive beneath it are open source and runnable:

PyTorch Distributed Checkpoint (`torch.distributed.checkpoint`, aka DCP) is the
lower-level primitive ByteCheckpoint is built on — it ships with PyTorch itself,
so no extra install, and it also natively supports reloading a saved state dict
under a different world size/sharding layout. If ByteCheckpoint feels heavier
than you need, DCP alone gets you the same core mechanic (metadata-driven
resharded load) with less surface area to configure. Good option if you want
something minimal and well-documented rather than production-featured.

These are actual open-source checkpoint libraries that implement this exact
mechanism, which you can run for real and let Darshan capture whatever they
genuinely do. That gets you a real trace instead of an approximation of one.

---

## 7. Tooling caveats

- Darshan's POSIX module **captures mmap but needs extensions for msync** —
  pretraining `.bin` shards are mmap'd, so page-fault reads are a blind spot.
- Darshan records **request submission, not completion**, and has **no liburing
  integration**. Modern checkpoint engines are async and DataStates-LLM uses
  liburing → Darshan will misattribute *when* checkpoint I/O happened.
- tf-Darshan (Cluster'20) warning: a large fraction of "I/O operations" in a DL
  trace are zero-length EOF probes (TF's read-file op loops on pread until 0).
  Filter by size before counting operations.

---

## 8. Key papers

**Characterization**
- Gossman, Maurya, Nicolae, Calhoun — *Understanding LLM Checkpoint/Restore I/O
  Strategies and Patterns*, SCA/HPC Asia 2026 Workshops, arXiv:2512.24511
  **(the closest thing to a direct answer; read first)**
- Chien et al. — *tf-Darshan*, Cluster'20
- Devarajan et al. — *DLIO*, CCGrid'21; MLPerf Storage v2.0 (adds checkpointing)
- Ren et al. — *I/O Characterizing Study of Offloading LLM Models and KV Caches
  to NVMe SSD*, CHEOPS'25
- Chowdhury et al. — *I/O Characterization and Performance Evaluation of BeeGFS
  for Deep Learning*, ICPP'19
- Patel et al. — *Uncovering Access, Reuse, and Sharing Characteristics of
  I/O-Intensive Files*, FAST'20 (methodological template)

**Checkpointing systems**
- Maurya et al. — *DataStates-LLM*, HPDC'24 (arXiv:2406.10707) + TPDS'26
  follow-up (arXiv:2601.16956)
- Wan et al. — *ByteCheckpoint*, NSDI'25 (arXiv:2407.20143)
- AdaCheck, FAST'26 — redundancy elimination, **6–896× smaller checkpoints**
- Lian et al. — *Universal Checkpointing*, USENIX ATC'25 (arXiv:2406.18820)
- CheckFreq (FAST'21), Check-N-Run (NSDI'22), Gemini (SOSP'23), MLP-Offload (SC'25)

**Framing**
- *Efficient Training of LLMs on Distributed Infrastructures: A Survey*,
  arXiv:2407.20018 — §3.3 storage. Note: §3.3.1 assigns checkpoints to object
  stores while citing two *filesystems* (Tectonic, HDFS) and one object store
  (Ceph) as examples. §3.3.2 assigns data loading to PFS, then undercuts it by
  describing PFS/object store as a cold backend behind Alluxio/JuiceFS.
- Lockwood, *LLM training without a parallel file system* (blog, Feb 2025).
  Claim is "not required," **not** "not used." Concedes PFS works and that the
  shared-home/interactive-debugging role exists — argues the workflow should
  change. Strongest argument is operational (no fragile client-server state, no
  evictions, no LDAP), not bandwidth.

---

## 9. Risks to the framing

- **AdaCheck-class redundancy elimination** could shrink checkpoint traffic by
  2–3 orders of magnitude. Needs a sentence in related work either way.
- **Framework fixes erase restore-side contributions.** Preallocated buffers and
  batched submissions are announced future work.
- **Hierarchical checkpointing eliminates the pathological write shape** before
  bytes reach the PFS — but assumes node-local NVMe budget that HPC allocations
  often lack. That gap is the justification for direct-to-PFS as the evaluation
  baseline.
- Isolation, lifecycle, and conversion-job arguments do **not** have this
  fragility. Prefer them as the primary claim.
