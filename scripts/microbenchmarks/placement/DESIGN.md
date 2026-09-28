# End-to-End BeeGFS Placement Benchmark Design

**Status: offline canonical plan and raw-to-plot visualizer implemented; live
runner, reviewed inventory, restoration watchdog, and pilot not implemented.
The historical pool scripts are not this runner.**

## 1. Research question

Measure complete normal BeeGFS I/O while varying storage pool, target chooser,
stripe count, stripe size, workload, and client concurrency:

> With all other factors fixed, how do Default, SSD-only, and HDD-only target
> eligibility affect delivered performance and placement balance?

```text
IOR -> POSIX -> BeeGFS client -> network -> storage service
    -> target XFS filesystem -> block layer -> HDD/NVMe
```

This is an integrated filesystem result. Local FIO, iperf3, and NetBench help
explain it but cannot be subtracted to derive component overhead.

The design implements sections 6 and 9-12 of
[`MicroBenchmarks.md`](../../../docs/specs/MicroBenchmarks.md) and all applicable
[`Global.md`](../../../docs/specs/Global.md) invariants.

## 2. Scope

Included:

- Normal, NetBench-disabled BeeGFS operation.
- D (all candidate targets), S (NVMe only), and H (HDD only).
- `randomized` and `roundrobin` target choosers.
- Desired stripe counts 1, 2, and 4.
- Stripe sizes 256 KiB, 512 KiB, and 1 MiB.
- Sequential/random read/write file-per-process and sequential shared-file I/O.
- Single- and dual-client MPI concurrency configurations.
- Five independent repetitions and fresh measured files per unit.
- Native IOR, Darshan, placement, network, service, target, and device evidence.

Excluded:

- Metadata-only rates, NetBench, and explicit cache-hit comparisons.
- Mixed read/write phases; each unit has one measured operation.
- Shared-file random I/O in the initial matrix.
- Raw devices or direct access to BeeGFS private storage paths.
- Capacity as an active factor until site bands are approved. The runner must
  support it, but a baseline-capacity run is labelled incomplete for that factor.

## 3. Reviewed inventory and storage configurations

Create `placement_inventory.json` from fresh live evidence. It maps every target
ID to OSS, mount, device, media, capacity, and current pool. The expected geometry
is 28 targets across four OSS hosts, with 14 HDD and 14 NVMe targets; these counts
must be confirmed rather than copied from historical scripts.

| Configuration | Eligible targets |
|---|---|
| D | All 28 reviewed candidate targets in Default |
| S | All 14 reviewed NVMe targets in a dedicated SSD pool |
| H | All 14 reviewed HDD targets in a dedicated HDD pool |

D intentionally has a different media mixture and target count. Report that fact;
do not attribute differences solely to a pool label. Stripe counts 1/2/4 are
feasible only if that many eligible Online/Good targets remain.

The runner owns a reversible cluster-state transaction. Before mutation, save
the exact pool IDs, descriptions, membership, target state, metadata chooser
configuration/checksum, and service identity. Never assume pool IDs or run
`configure_pools.sh`/`reset_pools.sh`; those scripts contain historical mappings.

For each state block:

1. Require exclusive maintenance authorization and a global cluster-state lock.
2. Apply the exact reviewed membership for D or the simultaneous S/H partition.
3. Apply the chooser in `/etc/beegfs/beegfs-meta.conf` using a reviewed privileged
   helper, restart/reload only as required by BeeGFS 7.4.4, and verify live state.
4. Wait a configured stabilization interval and recheck all targets Online/Good.
5. Run only cases belonging to that verified block.
6. Restore the captured pre-run configuration on success, failure, signal, or
   reservation stop, and verify it before releasing the lock.

An external lease/watchdog must restore the saved state if the coordinator dies.
Failure to prove restoration blocks all other benchmark domains.

## 4. Workloads and IOR geometry

Use native MPI-enabled IOR directly with POSIX API. One external invocation is
one repetition and one measured operation.

| Workload ID | Operation | Organization | Transfer size |
|---|---|---|---:|
| `seq_read_fpp` | Sequential read | File per process | 1 MiB |
| `seq_write_fpp` | Sequential write | File per process | 1 MiB |
| `rand_read_fpp` | Random read | File per process | 4 KiB |
| `rand_write_fpp` | Random write | File per process | 4 KiB |
| `seq_read_shared` | Sequential read | Shared file | 1 MiB |
| `seq_write_shared` | Sequential write | Shared file | 1 MiB |

Fixed settings:

| Setting | Value |
|---|---:|
| API | POSIX |
| Block size | 16 GiB per rank |
| Segments | 1 |
| Maximum measured transfer phase | 60 seconds (`-D 60`) |
| IOR iterations | 1 |
| Barriers | Enabled (`-g`) |
| Client direct I/O | Enabled (`--posix.odirect`) |
| Write synchronization | `-e`, included before write phase completion |
| Data checking | Pilot-only, outside timing; not full-matrix overhead |

This is a **size-or-time** protocol. Each rank transfers at most 16 GiB and IOR
stonewalls a phase at 60 seconds. A fast case can finish its byte extent early;
a slow case stops at the stonewall. Preserve actual bytes and phase time and label
completion as byte-limit or stonewall. Do not compare bandwidth without reporting
which limit occurred.

Commands are equivalent to:

```text
mpirun <explicit map> ior -a POSIX -t <1m|4k> -b 16g -s 1 -i 1
  -g -D 60 -k --posix.odirect <-w|-r -E> <-F> <-z> <-e for writes>
  -o <owned-file> -O summaryFormat=JSON -O summaryFile=<summary>
```

Pin exact options to the installed IOR help/version and reject unknown summary
schemas. `-F` selects file-per-process; omitting it selects one shared file. `-z`
is present only for random cases.

## 5. Preparation and cache policy

Every unit uses a fresh marker-owned namespace whose parent receives the selected
pool, desired stripe count, and chunk size before file creation. Existing files
never change layout, so no file can be reused across placement configurations.

Preparation is outside measured time:

- Sequential write: no data preparation; measured IOR creates fresh file(s).
- Sequential read: sequentially write the complete 16-GiB-per-rank extent with
  direct I/O and final fsync, preserving the requested organization/layout.
- Random read: same complete sequential preparation, then measured random reads.
- Random write: same complete preparation, then measured random overwrites with
  IOR existing-file semantics so it cannot recreate the layout.

Verify prepared size, ownership, inode/entry identity, actual stripe count,
target IDs, and successful synchronization. Before measured reads, run `sync`,
drop client cache on participating clients and page cache on all participating
storage hosts through reviewed non-interactive privileged helpers, wait for a
fixed settling interval, and capture state. This is a fixed preparation protocol,
not a cache factor. Direct I/O plus backend device telemetry is still required;
cache dropping alone does not prove every byte came from media.

Measured writes use direct I/O and final fsync. Their primary elapsed interval
must include IOR open, transfer, synchronization/close, and cross-rank completion.
Do not report transfer-only time as persistent-write throughput. Device volatile
caches remain a limitation and their settings are recorded.

## 6. Concurrency and full matrix

| Concurrency ID | Clients | Ranks/client | Total ranks |
|---|---|---:|---:|
| `a2_r1` | `anjuna2` | 1 | 1 |
| `a2_r4` | `anjuna2` | 4 | 4 |
| `dual_r1` | `anjuna2`,`anjuna3` | 1 | 2 |
| `dual_r4` | `anjuna2`,`anjuna3` | 4 | 8 |

`anjuna2` is the non-colocated single-client baseline. `anjuna3` participates in
dual cases and is labelled as colocated with management/metadata. MPI host slots,
core binding, and rank maps are explicit; no oversubscription is allowed.

Baseline-capacity count:

```text
3 storage configurations x 2 choosers x 3 stripe counts x 3 stripe sizes
x 6 workloads x 4 concurrency points x 5 repetitions
= 6,480 measured units
```

Two approved capacity states would double this to 12,960 units. The canonical
plan retains infeasible cells with a recorded reason; it never silently changes
stripe count, dataset size, or eligible targets.

Block execution by capacity state, cluster membership state, and chooser to
minimize destructive transitions. Within a block, use a seeded shuffle and
rotation across repetitions. Persist transition order and configuration order.
Operational blocking does not change the direct-comparison rule: compare D/S/H
only with workload, concurrency, stripe geometry, chooser, capacity, cache
protocol, and instrumentation fixed.

## 7. Capacity extension

The implementation contains explicit capacity-state objects but refuses any
non-baseline capacity run until configuration supplies approved per-target:

- Free-byte and free-percent bands.
- Inode reserve and workload-growth headroom.
- BeeGFS Normal/Low/Emergency byte and inode thresholds.
- Filler ownership path, creation method, target layout, and restoration plan.

Filler is provisioned outside timing, verified on every target, allowed to settle,
and never adjusted during a measured unit. A band excursion invalidates the unit.
Cleanup restores exact pre-run capacity. Report baseline-only results as having
no experimental capacity factor.

## 8. Layout and state validation

Before each measured invocation require:

- NetBench `0` on both clients.
- Expected pool membership, chooser, target health, and capacity state.
- Exact BeeGFS mount, cache mode, transport, and MPI/IOR/Darshan versions.
- Fresh owned path with parent pattern matching pool/count/chunk size.
- Actual created file layouts containing only eligible target IDs and exactly the
  requested stripe count, sampled for every shared file and every FPP rank file.
- Sufficient bytes/inodes for peak dataset plus reserve on every eligible target.

Afterward verify native bytes/time, file sizes, layouts, capacity bands, target
states, and cleanup. Desired count is not accepted as actual count without file
evidence.

## 9. Telemetry and Darshan

Capture before and immediately after each unit:

- Per-client CPU, memory, VM/writeback, BeeGFS connections, and interface counters.
- Metadata-service CPU/RSS/I/O and metadata-target filesystem/device counters.
- Every storage-service CPU/RSS/I/O, interface counters, target free bytes/inodes,
  internal capacity class, and backing-device diskstats.
- Actual file-to-target layouts and per-target bytes written/read where available.

Native IOR is authoritative for performance. Full runs also use Darshan after a
pilot proves all MPI ranks are captured, log naming is deterministic, required
POSIX counters exist, and overhead is measured. Save each log by explicit
run/unit/attempt identity; never select the newest shared log. DXT is disabled
unless separately designed because it changes overhead and artifact volume.

## 10. Validation and analysis

Require zero MPI/IOR exit, exact rank map and options, one expected measured phase,
finite positive bytes/rate/time, plausible size-or-stonewall completion, no short
transfer/error, write sync completion, correct layouts, required telemetry, and
complete owned cleanup. Validate native totals against rank/block geometry and
Darshan-issued bytes within documented API differences.

Primary rows contain all factors plus actual targets, eligible target count,
bytes, IOPS, aggregate MiB/s, total/open/transfer/close time, completion reason,
rank skew, device/network deltas, and provenance. Report all five repetitions,
median, mean, standard deviation, coefficient of variation, min, and max.

Report D/S/H separately before direct ratios. Analyze load balance across targets,
server/client saturation, and scaling efficiency. Never pool unlike workloads,
organizations, completion reasons, target counts, or capacity states.

## 11. Runner and resume contract

Implement a single coordinator on `anjuna3`, with its state and process handling
contained in the placement folder. The visualizer reads raw evidence directly.
Avoid reusing the application-workload wrapper.

```text
placement/
  run_placement.py          canonical plan, setup, MPI/IOR execution, resume
  placement_config.json     workload matrix and planning estimates
  placement_inventory.json reviewed targets/devices/hosts/restoration baseline
  visualize_results.py      validate raw IOR/layout/telemetry and plot performance
  tests/                    fake IOR/native-layout visualization fixtures
```

Pure `plan_units()` and command/layout builders currently have no cluster side
effects. Live `apply_block_state()`, `restore_cluster_state()`, and `execute_unit()`
still need reversible, watchdog-backed state transactions, fresh setup, cache
preparation, validation, and owned cleanup. Visualization never changes cluster
or namespace state.

The manifest fingerprints scientific settings, canonical plan, inventory, tool
versions, Darshan configuration, preparation/cache policy, and baseline restoration
record. Each unit and retry has a stable ID. Persist `pending`, `running`,
`completed`, `failed`, and `interrupted` states atomically. Cleanup, cluster-state
restoration, and analysis are independent states.

Resume first restores NetBench-off, pool/chooser/capacity baseline, and owned
process cleanup before ordinary preflight. It revalidates completed evidence,
recreates all setup for an unfinished unit, and never treats cache state as saved.
The restartable unit is setup + cache preparation + one measured IOR invocation;
successful measurement is not rerun solely because namespace cleanup failed.

Use mutable allocation deadlines, pilot-derived admission estimates, hard failure
timeouts, and a restoration/cleanup reserve. A budget stop restores cluster state,
flushes evidence, and exits resumably.

## 12. Artifacts and pilot

Store configuration/inventory snapshots, baseline and transition journals, exact
commands/environment, IOR JSON/stdout/stderr, rank maps, Darshan logs, layouts,
telemetry, validation, cleanup, and restoration evidence. Derived outputs are
fixed-factor performance PNGs and `plot_manifest.json` beneath the run's `plots/`
directory. No intermediate normalized CSV is required to view performance.

Pilot a bounded set covering all six workloads, both organizations, both choosers,
D/S/H, stripe counts 1/4, stripe sizes 256 KiB/1 MiB, and all concurrency points.
Use pairwise coverage rather than the Cartesian product, plus maximum eight-rank
read and write cases. Pilot gates include:

- Correct size-or-60-second semantics and persistent-write timing.
- Random existing-file behavior without recreation.
- Every requested/actual layout and pool eligibility rule.
- Cold-read preparation with backend evidence.
- Darshan rank coverage and measured overhead.
- Pool/chooser restoration under success, interruption, and coordinator death.
- Raw-evidence visualization corruption tests, safe path/process cleanup, resume, wall-time, capacity,
  and inode estimates.

Do not begin the 6,480-unit run until the pilot provides realistic total-time and
storage-growth estimates and an operator approves the state-transition schedule.
