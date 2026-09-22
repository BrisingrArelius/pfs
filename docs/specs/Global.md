## Objective

Evaluate whether BeeGFS storage pooling is useful across different workload types.

Every workload-specific experiment inherits the factors and invariants defined here.

The full factor matrix applies to end-to-end pooling/application comparisons.
Component benchmark domains use their applicable dimensions: for example,
transport benchmarks do not acquire a physical-fullness factor. Preserve the
full comparison matrix; record infeasible cells and their reasons rather than
silently pruning it.

---

## Factors Varied Between Runs

| Factor | Values / Rule |
|---|---|
| Storage configuration | D — Default (experimental pooling-off), S — SSD-only, H — HDD-only |
| Stripe count | 1, 2, 4 |
| Stripe size | 256 KiB, 512 KiB, 1 MiB |
| Target-selection policy | `randomized`, `roundrobin` |
| Capacity state (proposed; pending user confirmation) | High free space, low free space on the same physical devices; capacity meaning and site-defined bands require confirmation below |
| Workload concurrency | Workload-specific scaling values |

### Storage Configurations

- **D — Default (experimental pooling-off):** all candidate targets belong to the BeeGFS Default pool, and files use that pool. This means no media-specific placement restriction, not absence of a pool.
- **S — SSD pool:** pool containing SSD targets only.
- **H — HDD pool:** pool containing HDD targets only.

BeeGFS does not have an unpooled target state. Every storage target belongs to
exactly one storage pool. Moving a target to a named pool removes it from the
Default pool. Targets cannot simultaneously overlap between Default and named
pools, or between named pools.

D is the baseline; S and H measure restriction to a particular storage class.
D requires different pool membership from the media-separated S/H arrangement;
S and H may coexist as disjoint pools.

### Proposed Capacity State and Site Bands — Pending Confirmation

The proposed global capacity factor is physical filesystem fullness: high versus
low free space on the same devices. This interpretation of “capacity” and the
high/low thresholds are explicitly pending user confirmation before execution.
Installed device/filesystem capacity is a separate property, as are BeeGFS's
internal Normal/Low/Emergency classifications. Those classifications
can affect allocation and must be observed separately; neither experimental
band implies a particular internal class.

Before execution, supply explicit site values for every target or documented
target group. These are required inputs, not assumed thresholds.

| Required site setting | Value to supply |
|---|---|
| High-free-space band | TBD: minimum/maximum free bytes and free percent per target |
| Low-free-space band | TBD: minimum/maximum free bytes and free percent per target |
| Inode reserve and write headroom | TBD: minimum free inodes and bytes per target, allowing peak workload growth |
| Internal classification settings | Installed BeeGFS byte/inode thresholds and dynamic classification settings |

Record installed filesystem capacity separately and define the free-percent
denominator and measurement sources. Provision/remove filler outside timing,
verify its target allocation, and let provisioning and background activity settle
before measurement. Restore the starting band before each independent repetition
and reserve sufficient headroom to maintain the declared band throughout each
measured run; do not adjust filler during timing. Record any band excursions and
repeat affected runs after correcting preparation. Within each D, S, or H
configuration, keep devices, eligible target IDs and counts, working-set size,
and layout identical for high/low comparisons. Record per-target free bytes, free percent, free
inodes, and internal capacity class immediately before and after every run;
record byte and inode classes separately where exposed.

---

## Global Invariants

| Invariant | Rule |
|---|---|
| Repetitions | 5 independent executions of every measured configuration, restoring preparation and starting conditions each time |
| Hardware | Same client nodes, storage servers, storage targets and network |
| Pool membership | Matches D, S, or H; fixed during each run and re-established for all five repetitions |
| Capacity state | Once confirmed, maintain the declared band per run with headroom for workload growth; record band excursions and internal class transitions |
| Workload | Identical workload configuration when comparing BeeGFS settings |
| Input data | Identical dataset/files for comparisons of the same workload |
| Concurrency | Held constant when directly comparing storage configurations |
| Stripe configuration | Held constant when directly comparing storage configurations |
| Target-selection policy | Held constant when directly comparing storage configurations |
| File organization | Same directory/file layout for a given workload |
| Cache handling | Same predefined cache policy for all comparable runs |
| Run ordering | Capacity-state blocks; counterbalance feasible D/S/H and chooser comparisons within blocks and high/low block order across repetitions where feasible; record constraints and order |
| Instrumentation | Same monitoring and tracing enabled for all comparable runs |
| BeeGFS configuration | Unchanged unless the parameter is explicitly being varied |

---

## Comparison Rule

A direct pooling comparison changes only the storage configuration.

For a fixed:

`workload × concurrency × stripe count × stripe size × target-selection policy × capacity state`

compare:

`D vs S vs H`

All other parameters remain identical. D/S/H intentionally change eligible media
and potentially target count; report these differences rather than attributing
results solely to pool naming.

A direct capacity comparison changes only high versus low free space on the
same devices and eligible target IDs/counts within a storage configuration, with identical working set, workload, storage configuration,
chooser, striping, concurrency, and cache preparation. Compare choosers with the
other factors fixed. Use feasible stripe counts and provisioning combinations;
document omissions instead of silently shrinking workloads or changing actual
stripe count. Execute five independent repetitions per measured configuration, with settings
fixed per run. Block expensive capacity changes and counterbalance comparison
order within blocks and high/low block order across repetitions where feasible.
Capacity blocks reduce filler churn; document constraints that
prevent full counterbalancing.

---

## Global Measurements

Collect where applicable:

- Execution / phase time
- Read throughput
- Write throughput
- Bytes read/written
- POSIX read/write time
- Metadata time
- I/O operation counts
- Target-level utilization / traffic
- Per-target free bytes, free percent, free inodes, and internal capacity class before/after each run
- Installed target capacity, eligible target IDs, actual placement, capacity band, and provisioning/run order
- Client cache mode, server RAM, data volume per server, cache preparation, and backend device traffic

Workload-specific specifications may add additional metrics but must retain the
global measurements relevant to that workload.

## Instrumentation and Raw Evidence

Use native benchmark output for performance, Darshan for application-issued I/O
where supported, and client/server/network/backend telemetry for path evidence.
Darshan is a recorder, not a benchmark or a separate I/O-path component. Standard
logs are mostly summaries; detailed DXT tracing is a separately declared option.
Neither establishes whether bytes came from client RAM, server RAM or devices.

Validate instrumentation in a pilot: build/preload integration, enabled modules,
MPI ranks and worker/loader processes, log completeness and overhead. Installation
alone does not instrument a workload. Keep settings consistent across direct
comparisons and distinguish preparation from measured phases. Use the coverage
policy in each suite; local FIO and transport tools use their native metrics.

Associate logs with explicit run/attempt/process identifiers, not the newest file
in a shared directory. Retain raw logs, commands, stdout/stderr, exit status,
configuration and telemetry independently of parsed tables. Parser failures must
be recoverable without rerunning successful expensive measurements.

## Durable Progress and Resume Across Allocations

This is a required runner contract for every benchmark domain and application experiment,
not functionality already supplied by the legacy scripts. Experiment-progress
checkpoints are distinct from a workload's checkpoint-write/recovery operations.

- Persist a suite manifest and configuration fingerprint covering scientific
  factors, workload/tool versions, instrumentation and preparation protocol.
  Give each configuration × repetition × measured phase a stable ID; give each
  retry a separate attempt ID. Resume must reject incompatible configurations.
- Track `pending`, `running`, `completed`, `failed` and `interrupted` measurements
  with atomic durable updates. Keep measurement completion separate from parsing
  and analysis completion. Do not mark success before required raw evidence and
  completion checks are recorded durably.
- On resume skip only validated completed measurements. Treat abandoned running
  attempts as interrupted, preserve partial evidence, and retry the measurement
  after restoring preparation. Parsing failures rerun parsing only; invalid or
  missing required measurement evidence requires a new measurement attempt.
- Record allocation/session IDs, hosts, timestamps, block/order and interruptions.
  Revalidate topology, transport, membership, chooser, stripes, capacity and
  NetBench state on reacquisition. Restore files, filler and cache preparation
  for the pending case; do not assume the previous allocation left them intact.
- Warm cache state cannot be saved in a progress file. If phases/epochs depend on
  preceding warmup or workload state, make that dependent sequence the restartable
  measurement unit; do not splice independently restarted phases into one run.
- Use an artifact/checkpoint location that survives relinquishing the allocation.
  Define the maximum restartable measurement duration and account for cleanup and
  progress flushing before the allocation ends. Unexpected termination must leave
  completed evidence intact and unfinished attempts identifiable.

## Runtime and Allocation Planning

Pilot each group's preparation, measurement, synchronization and cleanup costs.
Estimate total time from the full matrix and repetitions, distinguishing measured
phases from command invocations and serial time from genuinely concurrent work.
Record uncertainty and update estimates from completed measurements. Scheduling
across allocations preserves the matrix; it is not a reason to drop factors.

Reserve time for state transitions, validation, evidence collection and retries.
If one restartable measurement exceeds an allocation, obtain sufficient time or
define scientifically justified segmentation in advance. Arbitrarily pausing a
timed workload changes its semantics. Concrete conditional estimates for FIO and
the full IOR matrix are documented with the microbenchmark suite.
