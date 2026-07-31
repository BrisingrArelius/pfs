# BeeGFS Research Context

## Project Goal

Basically what we are trying to do in this project is something like this:

- **Per-file analysis** (prediction based on past logs).
- **Heterogeneous (performance/hardware level) pooling** of a storage cluster — e.g., HDD vs. SSD as 2 pools.
- **Put the appropriate file in the appropriate pool for performance gains.** E.g., a file that is read frequently in random order and not written much → SSD; a file with occasional long writes (checkpoint-type workload) → HDD.
- We can't really measure SSD perf gains on the Dash cluster because of the network bus bandwidth bottleneck. So for now we're pivoting to good HDD vs. bad HDD type comparisons.

## Working Rules

- **Every change to the codebase must be logged in `CHANGELOG.md`** (in this same `MD Files and Context/` folder). Add an entry describing what changed and why whenever a script, config, or pipeline behavior is modified.
- **Task 1 (Profiles Setup)** has its own supporting docs in this same folder — refer to them when working on I/O profile classification: [`Task 1 -  Profiles Setup.md`](Task%201%20-%20%20Profiles%20Setup.md) (goal, current profile params, Darshan-counter constraint) and [`LitReview_task1.md`](LitReview_task1.md) (literature review backing the size/pattern/type/frequency thresholds).

## Overall Plan / Summer Plan

Source: handwritten kickoff/planning notes (for Tejas & Advay, Manoj), summarized from `Summer plan dash annotated.pdf`.

### Kickoff notes (Tejas & Advay)

- **To-do:** run benchmarks on the Dash cluster; compare HDD vs. SSD (noting the network bottleneck), and within each, fast vs. slow variants; work out the BeeGFS placement algorithm and design experiments around it; nail down the math — parameter identification and values from literature, plus the finding that the network bottleneck masks local SSD/HDD gains.
- **Research questions:**
  1. Does pooling even matter under low I/O conditions?
  2. Does disk state (fast vs. slow) matter *when* files are placed?
  3. How do different placement algorithms behave, and what results do they produce?
- **To-do:** build an intelligent placement algorithm that can assess disk state in real time.

### Methodology — deriving application I/O patterns from Darshan logs

- Download ALCF logs → parse → classify each file into categories (small/medium/large; read-only vs. write; contiguous vs. random).
- Worked example: an MPI app with 10 processes and 4 files, illustrating the three access-pattern classes:
  - **File 1** — accessed by all 10 ranks → **Single Shared File (SSF)**
  - **File 2/3** — accessed by one rank only → **File-Per-Process (FPP)**
  - **File 4** — accessed by a subset of ranks (0, 4, 6, 9) → **Partial-Shared File**
- POSIX counters vs. MPI-IO counters record byte/op counts per rank per file (POSIX ~84 counters, MPI-IO ~50) — this is the basis for `parse_darshan.py` / `extract_features.py` in the current pipeline.

### Week-by-week schedule (May 4 → early October)

| Week | Goal |
|---|---|
| May 4–8 | Download Darshan logs to NAS; build file-category taxonomy from literature |
| May 11–15 | Parse a single log into CSV, extract features |
| May 18–22 | Classify files in that CSV; extend to all logs from one day |
| May 25–29 | Extend to one month; start analysis (same app → same/different behavior across users/nprocs) |
| June 1–5 | Extend to 1 month; find most common patterns (bar graphs, heatmaps) |
| June 8–12 | Buffer week |
| June 15–19 | Extend analysis to 1 year of logs |
| June 22–26 | Continue 1-year analysis |
| June 29–Jul 3 | Extend to 2nd year of logs |
| Jul 6–10 | Extend to 3rd year of logs |
| Jul 13–17 | Group multi-year results |
| Jul 20–31 | Start writing the paper (6–7 week budget) |
| End of Aug | Paper submission |
| End of Sep / early Oct | Qualifier exam |

This document is the origin of the pipeline described below (Track 1: Log Predictor) — it explains why files are classified via SSF/FPP/Partial-Shared categories using POSIX + MPI-IO counters, and gives the timeline context for the research.

## Striping in Distributed Storage

In storage and distributed systems, **striping** is the technique of segmenting a logically sequential file or volume into chunks so that consecutive segments are written to different physical storage devices. Instead of writing a massive file to a single disk or server, the data is sliced up and spread across multiple resources.

### Core Striping Parameters

- **Stripe Size (Chunk Size):** The size of individual data blocks written to a single device before moving to the next. Typically ranges from 64 KB up to several megabytes.
- **Stripe Width (Count):** The number of parallel storage devices across which data is distributed.

With a stripe width of 4, the system cycles through servers A, B, C, D distributing chunks sequentially (Chunk1→A, Chunk2→B, Chunk3→C, Chunk4→D, Chunk5→A, etc.).

### Local vs. Distributed Striping

| Aspect | Local (RAID 0) | Distributed (Lustre, BeeGFS, Ceph) |
|---|---|---|
| Target | Multiple drives in a single chassis | Multiple storage servers across a network |
| Bottleneck | PCIe bus, SAS controller, local CPU | Network bandwidth (Infiniband/Ethernet) |
| Blast Radius | One drive failure = entire volume lost | Metadata servers track chunks; clients talk directly to data servers in parallel |

### Pros

- **Massive I/O Concurrency:** Striping across 16 storage servers can theoretically yield up to 16x faster reads/writes.
- **Load Balancing:** Prevents hotspots by distributing I/O load across many storage nodes.
- **Capacity Aggregation:** Allows individual files to grow beyond the capacity of any single node.

### Cons

- **Reliability Penalty:** Without parity or replication, striping lowers MTTF. Distributed systems combine striping with Erasure Coding or Replication.
- **Network Overhead:** Small, random I/O performs poorly with large stripe widths; a 4 KB write against a 1 MB stripe incurs latency without bandwidth gain.
- **File Locking & Coherency:** Multiple clients writing to the same striped file simultaneously requires complex distributed locking.

### Choosing Stripe Width

- **High stripe width:** Best for large, sequential I/O (checkpoint files, scientific datasets, video streaming).
- **Low/no stripe width:** Best for small, random I/O (source code repos, config files).

---

## Parity

**Parity** is a mathematical technique that provides data redundancy and fault tolerance without the high capacity cost of full replication.

### How It Works: XOR

Parity relies on bitwise XOR (⊕). XOR returns 0 if the count of 1s is even, and 1 if odd.

**Generating parity (3 data drives + 1 parity drive):**
```
Data1 ⊕ Data2 ⊕ Data3 = Parity
1 ⊕ 0 ⊕ 1 = 0
```

**Recovery (Drive 2 fails):**
```
Data1 ⊕ Data3 ⊕ Parity = Missing Data
1 ⊕ 1 ⊕ 0 = 0  →  Drive 2 was 0
```

### Parity Layouts

- **Dedicated Parity (RAID 4):** One drive holds all parity. Creates a parity write bottleneck on every write.
- **Distributed Parity (RAID 5):** Parity blocks are rotated across all drives, distributing the write workload evenly.

### Parity vs. Replication

| Feature | Replication (3-way mirror) | Parity / Erasure Coding |
|---|---|---|
| Storage Overhead | High (200%; 1 TB needs 3 TB raw) | Low (25–33%) |
| Write Performance | Fast (sequential copies) | Slower (Read-Modify-Write cycle) |
| CPU Overhead | Negligible | Higher (constant math on write/recovery) |

### Erasure Coding

In massive distributed systems (Ceph, cloud backends), parity scales into **Erasure Coding (EC)**. Using Reed-Solomon, data is split into `k` data chunks + `m` coding chunks. A "10+4" EC scheme spreads data across 14 servers and can survive any 4 simultaneous failures—with far less storage overhead than replication.

---

## BeeGFS Default Placement Strategy

BeeGFS uses a **balanced capacity-pool chooser** algorithm paired with **directory-based inheritance**.

### 1. Capacity Pool Chooser

When a client creates a file, `beegfs-mgmtd` groups all active Storage Targets into three pools:

| Pool | Meaning |
|---|---|
| **Normal** | Ample free space — preferred |
| **Low** | Filling up |
| **Emergency** | Nearly full |

The default algorithm heavily prefers Normal pool targets and only falls back to Low/Emergency if the stripe count can't be achieved otherwise.

### 2. Default Stripe Pattern

| Parameter | Default |
|---|---|
| Stripe Pattern | RAID0 |
| Chunk Size | 512 KiB |
| Stripe Width (Target Count) | Typically 4 targets |

### 3. Directory-Based Layout Inheritance

- All file entries in a directory are managed by a single Metadata Server (MDS) to minimize lookup latency.
- New files and subdirectories inherit the parent directory's storage pool layout, chunk size, and stripe count.

---

## Target Assignment Within a Pool (`tuneTargetChooser`)

When all OSTs are in the same capacity pool, BeeGFS uses the `tuneTargetChooser` setting in `/etc/beegfs/beegfs-meta.conf`.

| Strategy | Behavior | Best Use Case |
|---|---|---|
| `randomized` *(default)* | Randomly picks targets per file | HPC production; acts as a statistical load balancer |
| `roundrobin` | Strict sequential rotation through targets | Benchmarking (IOR, mdtest); guaranteed geometric spread |
| `randomrobin` | Shuffles targets randomly, then round-robins through the shuffled list | Balanced utilization without predictable patterns |

**Note:** Even within the same pool, the capacity balancer still soft-deprioritizes targets with significantly less free space than their peers.

```bash
# Check current setting
grep tuneTargetChooser /etc/beegfs/beegfs-meta.conf
```

**Docs:** [BeeGFS Capacity Pools Architecture Overview](https://doc.beegfs.io/latest/architecture/overview.html)

---

## Research Overview

### Core Idea

1. **Per-file analysis** based on past Darshan logs to predict I/O behavior.
2. **Heterogeneous storage pooling** — e.g., HDD pool vs. SSD pool.
3. **Intelligent file placement** — route files to the appropriate pool for performance gains.

> Since SSD vs. HDD performance gains can't be measured directly on the Dash cluster (network bus bandwidth bottleneck), the current focus is on **good HDD vs. bad/degraded HDD** comparisons.

### File Class → BeeGFS Target Mapping

| Darshan Profile | I/O Blueprint | Ideal BeeGFS Target Allocation |
|---|---|---|
| **Class A: Checkpoint** | Large, contiguous sequential writes; rarely read back | High stripe width; capacity-optimized pool (even if higher latency) |
| **Class B: Random Read Heavy** | Small, non-contiguous random chunks; read-only | Stripe width of 1 (or very low); highest-performing, lowest-latency targets |
| **Class C: File Per Process (FPP)** | Parallel independent files written by individual MPI ranks | Low stripe width per file, widely distributed to prevent cross-rank serialization |

### Real-Time Placement Hook

Without modifying BeeGFS core source, intelligent placement can be implemented using **Storage Pools + Directory Patterns**:

```bash
# Create distinct storage pools
beegfs-ctl --addstoragepool --id=10  # fast targets
beegfs-ctl --addstoragepool --id=20  # slow/degraded targets

# Assign layout to a target directory before heavy I/O starts
beegfs-ctl --setpattern --storagepool=10 /mnt/beegfs/user/app_run/latency_sensitive_dir
```

### Why Investigate BeeGFS Placement?

1. **Establish a baseline:** Document default randomized/roundrobin performance before any intelligent intervention.
2. **Quantify misplacement cost:** Force latency-sensitive files onto degraded targets and measure the penalty.
3. **Find trigger thresholds:** Discover at what point HDD degradation causes different file classes to collapse (e.g., random reads may be devastated at +20% latency while sequential writes are unbothered).
4. **Validate the classification matrix:** Confirm that Darshan-derived file classes match physical behavior on the cluster.

### Research Loop

```
[ Analyze Darshan Logs ] ──> Classify I/O patterns (SSF vs. FPP, Read vs. Write)
          │
          ▼
[ Investigate BeeGFS Placement ] ──> Benchmark default "blind" randomized layout
          │
          ▼
   [ Run Experiments ] ──> Force files onto Good vs. Bad HDDs; measure gains/losses
          │
          ▼
 [ Intelligent Algorithm ] ──> Match right file class to right disk state
```

---

## Repository: `pfs`

**GitHub:** [https://github.com/BrisingrArelius/pfs](https://github.com/BrisingrArelius/pfs)

### Pipeline Architecture

```
[ TRACK 1: THE LOG PREDICTOR ]
download_logs.py ──> parse_darshan.py ──> extract_features.py ──> classify_workloads.py
                                                                         │
                                                                   (Predicted Class)
                                                                         ▼
                                                                 [ run_pipeline.py ]
                                                                         ▲
                                                                   (Actual Performance)
                                                                         │
[ TRACK 2: THE PHYSICAL BENCHMARK ]                                      │
posix_synthetic_workloads.c (Compile) ──> run_workloads.py ─────────────┘
```

### Script Breakdown

#### Track 1 — Log Predictor

| Script | Role |
|---|---|
| `download_logs.py` | Bulk downloads raw `.darshan` log binaries from ALCF Polaris or internal mirrors |
| `parse_darshan.py` | Interprets binary logs via `darshan-parser` or `pydarshan`; outputs readable counters (POSIX_READS, POSIX_WRITES, timestamps, etc.) |
| `extract_features.py` | Condenses parsed data into key metrics (bytes, contiguous vs. random ops, app tags); outputs `Appl.csv` |
| `classify_workloads.py` | Reads `Appl.csv`, applies classification logic, maps entries to file class profiles; outputs `classified_matrix.csv` |
| `analyze_patterns.py` | Aggregates classification matrix across extended timelines; identifies structural behavioral trends |
| `plot_results.py` | Generates performance curves, bar graphs, and heatmaps for the research paper |

#### Track 2 — Physical Benchmark

| Script/File | Role |
|---|---|
| `posix_synthetic_workloads.c` | Native C program that simulates specific application I/O profiles using POSIX I/O (open, write, read, lseek) |
| `run_workloads.py` | Python automation wrapper; passes parameters to the compiled C executable and coordinates test runs against specific BeeGFS directories (good HDD vs. bad HDD) |
| `run_pipeline.py` | Master orchestrator; calls parsing scripts to extract a workload signature, maps it to a file class, triggers `run_workloads.py`, and logs resulting performance metrics |

### Order of Execution

**Step 1 — Compile the C Binary**
```bash
gcc -O3 posix_synthetic_workloads.c -o posix_synthetic_workloads
```

**Step 2 — Extract Your Targets (Predictor Track)**
```bash
python download_logs.py
python parse_darshan.py
python extract_features.py
python classify_workloads.py
```

**Step 3 — Run Isolated Benchmarks (optional)**
```bash
python run_workloads.py --target_dir /mnt/beegfs/fast_hdd/ --mode write_sequential
python run_workloads.py --target_dir /mnt/beegfs/slow_hdd/ --mode write_sequential
```

**Step 4 — Execute the Master Pipeline**
```bash
python run_pipeline.py --config config.json
```

**Step 5 — Chart Results**
```bash
python analyze_patterns.py
python plot_results.py
```

---

## Dash Cluster Hardware Inventory

The cluster has 3 Object Storage Servers (OSSs), each with a mix of HDD and SSD Object Storage Targets (OSTs).

| OSS | HDDs (OST IDs) | SSDs (OST IDs) |
|---|---|---|
| **colva1** | 101, 102, 103 | 104, 105, 106, 107 |
| **colva2** | 201, 202, 203, 204 | 205, 206, 207 |
| **colva3** | 301, 302, 303, 304 | 305, 306, 307 |

**Summary:**
- Total OSSs: 3
- Total HDD OSTs: 11 (3 on colva1, 4 on colva2, 4 on colva3)
- Total SSD OSTs: 10 (4 on colva1, 3 on colva2, 3 on colva3)
- Total OSTs: 21

### Suggested Storage Pool Layout

For the heterogeneous placement experiments, two logical pools can be carved out:

```bash
# Pool 10 — SSDs (low-latency, random-read workloads)
beegfs-ctl --addstoragepool --id=10 --targets=104,105,106,107,205,206,207,305,306,307

# Pool 20 — HDDs (high-capacity, sequential/checkpoint workloads)
beegfs-ctl --addstoragepool --id=20 --targets=101,102,103,201,202,203,204,301,302,303,304
```

For the **good HDD vs. bad HDD** experiments, targets within Pool 20 can be further split by OSS (e.g., colva1 HDDs vs. colva3 HDDs) to isolate per-node performance differences.

---

## Cluster Inspection Commands

### Cluster Architecture & Space (BeeGFS Commands)

```bash
# Topology: which OSTs belong to which OSSs, plus reachability state
beegfs-ctl --listtargets --longnodes --state

# Space: capacity, used, free, and capacity pool status per OST
beegfs-df

# Storage pools (HDD vs. SSD logical groups, if configured)
beegfs-ctl --liststoragepools --verbose
# OR
beegfs-ctl --listtargets --storagepools
```

### Hardware Specs (SSH into Storage Server)

```bash
# Find BeeGFS storage paths on the node
grep -E "^storeStorageTargets" /etc/beegfs/beegfs-storage.conf

# HDD vs. SSD: 1 = spinning HDD, 0 = SSD/NVMe
cat /sys/block/sdb/queue/rotational

# Full hardware specs (model, serial, transport, link speed)
sudo smartctl -i /dev/sdb

# Clean summary: device type, vendor, model, size, transport
lsblk -o NAME,ROTA,TYPE,VENDOR,MODEL,SIZE,TRAN
```

### Combined Workflow

1. `beegfs-ctl --listtargets --longnodes` → identify hostnames of target storage servers
2. `beegfs-df` → monitor which targets are filling up or unbalanced
3. SSH into storage nodes → `lsblk -d -o NAME,MODEL,TRAN` to verify hardware models

---

## Checking & Managing Storage Pool Assignments

### Check which pool a directory is assigned to

```bash
# BeeGFS 7.x
sudo beegfs-ctl --getentryinfo /mnt/beegfs/your/folder

# BeeGFS 8.x
beegfs entry info --verbose /mnt/beegfs/your/folder
```

Output includes the stripe pattern and pool, e.g.:
```
Stripe pattern details:
  + Type: RAID0
  + Chunksize: 512K
  + Number of storage targets: desired: 4
  + Storage Pool: 3 (hdd)
```

**Key gotcha:** `--getentryinfo` on a directory shows what pool *new files will land in*, not where existing files inside already live. Run the same command against individual file paths to check those.

### List all pools and their targets

```bash
sudo beegfs-ctl --liststoragepools
```

### Rename a pool

```bash
sudo beegfs-ctl --modifystoragepool --id=<ID> --desc="new_name"
```

### Assign a directory to a pool

```bash
sudo beegfs-ctl --setpattern --storagepoolid=<ID> /mnt/beegfs/your/folder
```

Only affects **new files** created in that directory going forward. Existing files keep their old pool.

---

## Actual Cluster Pool State (as discovered)

### `beegfs-client` troubleshooting

`beegfs-ctl --getentryinfo` requires the BeeGFS client daemon to be running on the node. If it fails with `Inappropriate ioctl for device`, the client is down:

```bash
systemctl status beegfs-client
sudo systemctl start beegfs-client

# Verify mount
mount | grep beegfs
# Expected: beegfs_nodev on /mnt/beegfs type beegfs (rw,...)
```

### Actual pool layout on the Dash cluster

```
Pool ID   Description               Targets
-------   -----------               -------
1         Default                   (empty)
2         hdd_EMPTY_merged_into_3   (empty — defunct)
3         merged_ssd_hdd_21t        101,102,103,201,202,203,204,301,302,303,304  ← ALL HDDs
4         offline_pool              401,402,403,404,405,406,407                  ← offline/dead
5         pfs_test                  (empty)
6         ssd_only_10t              104,105,106,107,205,206,207,305,306,307      ← ALL SSDs
```

Pool 3 contains all HDD targets and pool 6 contains all SSD targets — targets are correctly separated, only the names are messy.

### Rename pools to clean names

```bash
sudo beegfs-ctl --modifystoragepool --id=3 --desc="hdd"
sudo beegfs-ctl --modifystoragepool --id=6 --desc="ssd"
```

### Assign workload directories to correct pools

```bash
sudo beegfs-ctl --setpattern --storagepoolid=3 /mnt/beegfs/advay/hdd
sudo beegfs-ctl --setpattern --storagepoolid=6 /mnt/beegfs/advay/ssd
```

### Verify

```bash
sudo beegfs-ctl --getentryinfo /mnt/beegfs/advay/hdd
# → Storage Pool: 3 (hdd)

sudo beegfs-ctl --getentryinfo /mnt/beegfs/advay/ssd
# → Storage Pool: 6 (ssd)
```

### `run_workloads.py` path → pool mapping (confirmed)

| Variable | Path | Pool |
|---|---|---|
| `STORAGE_POOLS["hdd"]["workload_dir"]` | `/mnt/beegfs/advay/hdd/workloads/tmp` | Pool 3 (hdd) — inherits from parent |
| `STORAGE_POOLS["ssd"]["workload_dir"]` | `/mnt/beegfs/advay/ssd/workloads/tmp` | Pool 6 (ssd) — inherits from parent |
| `DARSHAN_LOG_DIR` | `/mnt/nfs_shared/darshan-logs` | NFS — outside BeeGFS entirely |

---

## OST Write Distribution Heatmap — Findings

### What was confirmed (empirically)

- **Pool isolation works:** HDD workloads write exclusively to OST101–OST304; SSD workloads write exclusively to OST104–OST307. No cross-pool bleed.
- **Stripe width = 4 confirmed:** Every workload row lights up exactly 4 OSTs, consistent with `desired: 4` from `--getentryinfo`.
- **Randomized placement confirmed:** The 4 chosen OSTs shift between runs of the same profile — matching `tuneTargetChooser = randomized`.
- **0.20 vs 0.30 GiB is a rounding artifact:** Expected per-OST share is exactly 0.25 GiB (1 GB ÷ 4). The heatmap shows 0.20 and 0.30 due to floating point rounding in the GiB conversion — the raw byte counts should be exactly 268,435,456 bytes per OST.

### What was claimed but not sourced (retracted)

The claim that `randomized` uses "weighted random that prefers targets from different OSSs" has **no source** in official BeeGFS documentation or any peer-reviewed paper. The official docs only state:

> *"By default, BeeGFS picks the storage targets for a file randomly."* — `doc.beegfs.io` (all versions)

The cross-OSS spread visible in the heatmap is a **statistical consequence** of pure random selection across 11 HDD targets on 3 OSSs — not a documented BeeGFS preference mechanism.

### Inter-OSS diversity — what IS documented

Inter-server diversity does exist in BeeGFS but only as an **opt-in** feature via a different chooser mode. From `beegfs-meta.conf`:

```ini
# Use randominternode to ensure stripe targets come from different physical hosts
tuneTargetChooser = randominternode

# Define which targets share the same physical host (domain)
# sysTargetAttachmentFile = /etc/beegfs/beegfs-meta-targets.conf
# Format: targetID=domainID, one per line
# e.g.: 101=1
#        102=1   ← 101 and 102 are on the same host (colva1)
#        201=2   ← 201 is on a different host (colva2)
```

This is **not** the default `randomized` behaviour.

---

## BeeGFS Source Code

The official public C/C++ source repo (metadata server, storage server, client kernel module) is at:

```
https://github.com/ThinkParQ/beegfs
```

To find the target chooser implementation, go to the repo and search for `TargetChooser` using GitHub's code search (press `T` in the repo or use the search bar → "Search this repository"). The logic lives under `meta/source/` but the exact filename requires code search to confirm — do not trust any filename cited without verification.

**Note:** Prior to BeeGFS 8, all development was private; the public repo only has full history for BeeGFS 8+ components. Older versions have source squashed into single commits.

---

## Roundrobin Target Chooser — Investigation & Findings

### Changing the target chooser

Edit on the **metadata server** (anjuna3):
```bash
# /etc/beegfs/beegfs-meta.conf
tuneTargetChooser = roundrobin

# Restart beegfs-meta for the change to take effect
sudo systemctl restart beegfs-meta
```

### Confirming roundrobin works (touch test)

Create 3 files back to back and check their OST placement:

```bash
touch /mnt/beegfs/advay/hdd/workloads/tmp/test1
touch /mnt/beegfs/advay/hdd/workloads/tmp/test2
touch /mnt/beegfs/advay/hdd/workloads/tmp/test3

sudo beegfs-ctl --getentryinfo --verbose /mnt/beegfs/advay/hdd/workloads/tmp/test1
sudo beegfs-ctl --getentryinfo --verbose /mnt/beegfs/advay/hdd/workloads/tmp/test2
sudo beegfs-ctl --getentryinfo --verbose /mnt/beegfs/advay/hdd/workloads/tmp/test3
```

**Confirmed output on the Dash cluster (pool 3, 11 HDD targets, stripe width 4):**

```
test1: 102, 103, 201, 202  (colva1 × 2, colva2 × 2)
test2: 203, 204, 301, 302  (colva2 × 2, colva3 × 2)
test3: 303, 304, 101, 102  (colva3 × 2, colva1 × 2) ← wraps after 304
```

This is a perfect sequential rotation through all 11 targets:
```
[101 102 103] [201 202 203 204] [301 302 303 304] → wrap → [101 102...]
      ↑ test1 picks 102,103,201,202
                    ↑ test2 picks 203,204,301,302
                                        ↑ test3 picks 303,304,101,102
```

### Key property: OSS pairing artefact

Because the 3 OSSs have 3+4+4 targets (not equal), the sequential rotation always picks 2 targets from one OSS and 2 from the adjacent one — it never splits evenly across 3 OSSs. This is a geometric consequence of stripe width 4 hitting unequal OSS boundaries, not a BeeGFS preference.

### The roundrobin repeat problem

**Symptom:** Two consecutive workload runs (freq_1 and freq_2) produced **identical OST sets** despite roundrobin being active and confirmed working via the touch test.

```
freq_1 [HDD]: OST102, OST103, OST201, OST202
freq_2 [HDD]: OST102, OST103, OST201, OST202  ← identical
```

**Root cause hypothesis:** The roundrobin counter is a **global persistent counter** on the metadata server that advances by 1 per target assigned, across all file creations cluster-wide — not just workload files. Between two workload runs, `run_workloads.py` does:

1. `cleanup_workload_files` — deletes files (no counter effect)
2. `log_file_ost_layout` — calls `beegfs-ctl --getentryinfo` on each workload file (read-only, no counter effect)
3. `log_ost_space` — calls `beegfs-ctl --listtargets --spaceinfo` (read-only, no counter effect)
4. Cache clear (`sync`, `drop_caches`) — no file creation

None of these should advance the counter. However if any **other process on the cluster** creates exactly a multiple of 11 files between the two runs (advancing the counter by a full cycle), the next workload file lands on the same counter position.

**To verify:** Add counter-position logging before each workload file creation by checking the EntryID sequence, or instrument with back-to-back touch files around each IOR invocation to see where the counter sits.

### roundrobin counter behaviour summary

| Property | Behaviour |
|---|---|
| Counter scope | Global — shared across all clients and all files on the cluster |
| Counter persistence | Survives across script runs; resets only on `beegfs-meta` restart |
| Advances on | Every file **creation** (at metadata assignment time) |
| Does NOT advance on | File deletion, reads, `beegfs-ctl` queries |
| Wrap-around | After last target in pool, cycles back to first |

---

## Fix: Roundrobin Repeat Bug — Unique Filenames Per Run

### Root cause

`run_workloads.py` calls a Python IOR wrapper (`posix_synthetic_workload_IOR.py`, internally documented as `ior_workload_wrapper.py`) twice per run — once in setup mode (mode=0) and once in workload mode (mode=1) — and both calls compute the **same deterministic filename**:

```python
def file_path(p, file_index):
    return os.path.join(p["work_dir"], f"workload_{p['profile_name']}_f{file_index}")
```

Since `cleanup_workload_files()` deletes the file after every run and the next run recreates a file with the **identical path**, BeeGFS's metadata server reuses the same EntryID/metadata slot. With `tuneTargetChooser = roundrobin`, this causes the counter position — and therefore the chosen OSTs — to repeat identically across consecutive runs instead of advancing.

### Fix applied — make filenames unique per run via `run_index`

Three files/call-sites required changes:

**1. `run_workloads.py` — `build_workload_cmd()`**
Added `run_index` as a parameter and as a new 12th CLI argument passed to the wrapper:
```python
def build_workload_cmd(name, params, mode, workload_dir, run_index):
    base_args = [
        "python3", WORKLOAD_BIN,
        name, str(params["read_ratio"]), str(pattern_int),
        str(params["stride_size"]), str(params["op_size"]),
        str(params["num_ops"]), str(params["num_files"]),
        str(params["num_phases"]), str(params["fsync_interval"]),
        workload_dir, str(mode),
        str(run_index),          # NEW — 12th arg
    ]
    return [MPIRUN, "-np", "1"] + base_args if mode == 1 else base_args
```
Both call sites inside `run_profile()` (setup_cmd and workload_cmd) updated to pass `run_index=run_index`.

**2. `posix_synthetic_workload_IOR.py` — `parse_args()` and `file_path()`**
```python
def parse_args(argv):
    if len(argv) < 13:   # was 12
        ...
    return {
        ...
        "mode":      int(argv[11]),
        "run_index": int(argv[12]),   # NEW
    }

def file_path(p, file_index):
    return os.path.join(
        p["work_dir"],
        f"workload_{p['profile_name']}_run{p['run_index']}_f{file_index}"
    )
```
Setup mode and workload mode for the *same run* are invoked with the same `run_index`, so they still agree on the filename and find each other's files correctly — only the filename changes *between* runs now.

**3. `run_workloads.py` — `log_file_ost_layout()` glob pattern (also required)**
The OST layout logger globs workload files by name pattern. This had to be updated to match the new filename scheme, or it would silently log nothing:
```python
# Before
pattern = os.path.join(workload_dir, f"workload_{profile_name}_f*")

# After
pattern = os.path.join(workload_dir, f"workload_{profile_name}_run*_f*")
```

### Why a timestamp wasn't used instead

An earlier proposed fix injected `time.time()` directly inside `file_path()` in the wrapper. This was wrong because `file_path()` is called independently by both the setup-mode process and the workload-mode process (two separate Python invocations) — a timestamp would produce two *different* filenames for the same run, breaking the contract that workload mode reads back the file setup mode just wrote. The fix had to plumb `run_index` through as an explicit, shared argument from the common caller (`run_workloads.py`) instead.

### Verifying the fix

After redeploying, repeat the earlier touch-test-style check using actual workload runs:
```bash
python run_workloads.py --only small_contiguous_read_heavy_freq_1_1gb --storage-type hdd --runs 3
sudo beegfs-ctl --getentryinfo --verbose /mnt/beegfs/advay/hdd/workloads/tmp/workload_small_contiguous_read_heavy_freq_1_1gb_run1_f0
sudo beegfs-ctl --getentryinfo --verbose /mnt/beegfs/advay/hdd/workloads/tmp/workload_small_contiguous_read_heavy_freq_1_1gb_run2_f0
sudo beegfs-ctl --getentryinfo --verbose /mnt/beegfs/advay/hdd/workloads/tmp/workload_small_contiguous_read_heavy_freq_1_1gb_run3_f0
```
Expect each run to show a different, sequentially-advancing OST set (consistent with the earlier confirmed roundrobin rotation: 101-103 → 201-204 → 301-304 → wrap).
