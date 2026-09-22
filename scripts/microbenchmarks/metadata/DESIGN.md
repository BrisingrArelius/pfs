# BeeGFS Metadata Benchmark Design

**Status: design complete; runner and cluster pilot not implemented.**

## 1. Research question and scope

This experiment measures namespace and file-management behavior through the
mounted BeeGFS filesystem. It answers:

> How do BeeGFS file and directory create, stat, open/read/close, and remove
> rates change with client placement, MPI concurrency, and directory layout?

The path is:

```text
mdtest MPI rank
  -> Linux VFS and BeeGFS client
  -> BeeGFS metadata service on anjuna3
  -> metadata target filesystem/storage
  -> storage-service communication only where the operation requires it
```

This is an end-to-end metadata-path measurement, not isolated metadata-daemon
CPU performance. `anjuna3` also hosts the management and metadata services, so
an `anjuna3` client case is a deliberately labelled colocated case. It is not an
independent hardware replica of `anjuna2`.

The design implements the metadata requirements from:

- [`Global.md`](../../../docs/specs/Global.md)
- [`MicroBenchmarks.md`](../../../docs/specs/MicroBenchmarks.md), section 8
- [`CLUSTER_TOPOLOGY.md`](../../../docs/CLUSTER_TOPOLOGY.md)

### Included

- The registered BeeGFS clients `anjuna2` and `anjuna3`.
- Single-client and simultaneous dual-client execution.
- File and directory creation, stat, and removal.
- Zero-byte file open/read/close as reported by mdtest's read phase.
- A shared flat parent and per-rank subdirectories.
- Three MPI ranks-per-client levels and five independent repetitions.
- Native mdtest output, MPI rank placement, and client/metadata/storage telemetry.
- Durable attempt-level progress and resume across allocations.

### Excluded

- Data-bearing small-file writes and reads. They add storage-target data work and
  belong in a separately labelled extension after this namespace baseline.
- Client or server cache dropping. The experiment does not claim cold metadata
  caches, and system-wide cache eviction would disturb unrelated users.
- Metadata mirroring or multiple metadata targets. The observed system has one
  metadata node and target.
- The D/S/H storage-pool, chooser, stripe-size, stripe-count, and capacity matrix.
  Pools constrain data-target placement, not metadata-server placement. The
  effective inherited data layout is frozen and recorded only as provenance.
- Shared-file contention (`mdtest -S`), collective creation, neighbor-rank stat,
  random stat order, rename, and deep directory trees in the initial matrix.
- Persistent-metadata latency. No `sync`, `fsync`, or mdtest `-Y` durability claim
  is made.
- Darshan in the initial full matrix. It may be added only after a pilot measures
  coverage and overhead, and then must be enabled consistently for all directly
  comparable cases.

## 2. Fixed workload

Use the installed MPI-enabled mdtest binary. Freeze its absolute path, version,
build identity where available, and help output (`-h` for the reviewed version)
in the run inventory. Do not silently substitute a different mdtest or MPI
implementation on resume.

Each rank creates **100,000 zero-byte files and 100,000 directories** as mdtest's
standard file and directory phases require. Each invocation uses one mdtest
iteration; the runner supplies the five independent repetitions. Running all
phases in one invocation preserves the intended lifecycle and barriers:

```text
directory create -> directory stat -> directory remove
file create -> file stat -> file open/read/close -> file remove
```

The exact order must be taken from and validated against the installed mdtest
version's native output rather than assumed from this schematic. A runner must
reject a version whose phases or option semantics do not implement this protocol.

The fixed mdtest settings are:

| Setting | Value | Rationale |
|---|---:|---|
| Items per rank | 100,000 | Long enough for single-rank phases while bounding peak namespace size |
| mdtest iterations | 1 | External repetitions remain separately identifiable and resumable |
| Bytes written per file | 0 | Namespace baseline without storage data writes |
| Bytes read per file | 0 | Measures open/read/close path without data transfer |
| Stat stride | 0 | Each rank stats its own objects |
| Phase barriers | Enabled | Preserve comparable phase boundaries |
| Sync after phase | Disabled | No durability claim; avoids global sync semantics |
| Stonewall timer | Disabled | Every rank performs the same fixed operation count |
| Unique-directory overhead | Excluded | Layout setup is not mixed into operation rates |

The implementation should generate the equivalent of:

```bash
mdtest -d <attempt-workdir> -n 100000 -i 1 -w 0 -e 0 -N 0 -P
```

The per-rank layout additionally uses `-u`. The runner must derive accepted flags
from the reviewed installed version and save the final argument vector. It must
not use `-B`, `-t`, `-W`, `-Y`, `-S`, `-c`, `-R`, `-F`, or `-D` in this experiment.
Shell command strings are provenance only; execution uses argument arrays.

### Directory layouts

| Layout ID | mdtest behavior | Interpretation |
|---|---|---|
| `flat` | All rank-owned objects are placed below one attempt work directory; omit `-u` | Contention in one shared parent directory |
| `per_rank` | Use `-u`; each rank receives a unique working directory below the attempt root; omit `-t` | Shallow hierarchy with namespace contention divided by rank |

mdtest must generate collision-free rank-specific object names in both layouts.
The pilot verifies this across hosts. The per-rank case is not a deep-tree study;
branching-factor and tree-depth experiments can be added as a separate extension.

### Cache interpretation

Every attempt uses a new namespace, but that does not make metadata-server or
client caches cold. Within an invocation, later phases intentionally observe
objects touched by earlier phases. Report the phase sequence and describe results
as fresh-namespace operation rates under the observed cache state, not cold-cache
rates. A failed or interrupted phase invalidates the whole invocation; later
phase output is never combined with a retry.

## 3. Experiment matrix

### Client placement

| Placement ID | Participating clients | Meaning |
|---|---|---|
| `anjuna2` | `anjuna2` | Remote client-to-metadata path |
| `anjuna3` | `anjuna3` | Client colocated with management and metadata services |
| `dual` | `anjuna2`, `anjuna3` | Concurrent requests from both clients |

### Concurrency

Use **1, 4, and 16 MPI ranks per participating client**. Therefore single-client
cases have 1, 4, or 16 total ranks, while dual-client cases have 2, 8, or 32 total
ranks. Dual-client cases always place equal rank counts on both clients.

Preflight must verify at least 16 allowed CPU cores on each client and refuse
oversubscription. Pin each rank to one distinct core using syntax validated for
the installed MPI implementation. Save the MPI host map and binding report. CPU
binding controls rank placement; it does not isolate `anjuna3` ranks from its
metadata daemon, so colocated resource contention is part of that labelled case.

### Counts

```text
3 client placements
x 3 rank counts per client
x 2 directory layouts
x 5 repetitions
= 90 restartable mdtest invocations
```

Each successful invocation emits seven primary phase measurements when supported
by the installed version: directory create/stat/remove and file
create/stat/read/remove. This yields up to 630 phase rows, but the invocation is
the atomic validity and retry unit.

Across the full matrix, the nominal primary work is 588 million object operations
before mdtest scaffold operations and retries. This is a planning count, not a
wall-time estimate; use the pilot's observed rates and cleanup time for allocation
planning.

At maximum concurrency, mdtest requests 3,200,000 files and 3,200,000 directories
across 32 ranks over its phase sequence. Before admission, check free metadata
bytes and inodes using pilot-observed peak consumption plus a configurable safety
reserve. Fail rather than reduce `-n` or the rank count. Record the actual maximum
simultaneously allocated objects observed by the pilot because mdtest versions may
sequence file and directory phases differently.

### Ordering

Generate the 18 placement/rank/layout configurations from a recorded random seed.
Shuffle them once, then rotate that order by seven positions for each repetition.
Seven is coprime to 18, so each configuration occupies five distinct order
positions. This balances position partially without claiming a complete Latin
square. Persist the canonical plan before execution and follow it on resume.

Do not run independent mdtest coordinators on both clients. Invoke one MPI job
from `anjuna3` for each case so one process owns barriers, rank mapping, output,
and exit status. Only one matrix case runs at a time.

## 4. Namespace ownership and safety

The authorized base is a reviewed BeeGFS path, initially proposed as:

```text
/mnt/beegfs/pfs/.metadata-mdtest/<run-id>/
```

Both clients must resolve this as the same BeeGFS mount and observe the same run
marker before any measurement. The implementation must not assume the path is
writable; the pilot confirms authorization and mount identity first.

Use this structure:

```text
<run-id>/
  owner.json
  attempts/
    <case-id>/
      attempt-<number>/
        owner.json
        work/
```

The owner file contains the run ID, case ID, attempt ID, configuration fingerprint,
creator host, and creation timestamp. Cleanup may recursively delete only the
recorded attempt directory after all of these checks pass:

1. The configured base is the exact expected BeeGFS mount and filesystem type.
2. The normalized candidate is a strict descendant of the configured run root.
3. No path component is a symlink.
4. The attempt owner file exactly matches the manifest identity.
5. The path depth and basename match the generated attempt path.
6. The run lock is held by the active coordinator.

Never issue an unscoped wildcard delete, delete the configured base, follow a
symlink during cleanup, or clean another run's directory. Preserve local raw
artifacts even after remote/shared namespace cleanup. A cleanup failure is a
separate resumable state and does not turn valid benchmark evidence into failure.

The full run root is removed only after all cases are valid and all attempt
cleanup is complete. Failed and interrupted attempt namespaces are cleaned before
their cases are retried. Pre-existing paths or mismatched markers cause a hard
failure rather than adoption.

## 5. Inventory and preflight

Create a reviewed `metadata_inventory.json` before the pilot. It records:

- Client hostnames, SSH destinations, BeeGFS client IDs, and mount paths.
- Management and metadata node IDs, host, service ports, and metadata target ID.
- Metadata target mount, backing filesystem/device, capacity, and inode source.
- Active client-to-metadata transport, source/destination addresses, interfaces,
  MTU, speed, and evidence timestamp.
- BeeGFS client cache mode and relevant client/meta configuration file checksums.
- mdtest, MPI launcher, BeeGFS client, server, kernel, and OS versions.
- Allowed CPU sets and physical/logical core counts on both clients.
- The authorized benchmark base and its owner/group/mode.

Read-only preflight runs before a new session and again after allocation changes:

1. Verify hostname identity, synchronized clocks, non-interactive SSH, and tools.
2. Verify the exact same BeeGFS filesystem and benchmark root from both clients.
3. Verify management and metadata node identity and service health.
4. Verify NetBench is disabled on every participating client.
5. Verify active transport and routes match the reviewed inventory.
6. Verify mdtest and MPI versions/help hashes match the run fingerprint.
7. Verify rank counts fit each host's allowed CPU set without oversubscription.
8. Verify no owned process or namespace from an incomplete attempt remains before
   recovery, and no unrelated process is targeted by cleanup.
9. Verify free metadata bytes/inodes exceed the pilot-derived requirement.
10. Record inherited pool/stripe pattern and storage membership without changing
    them; reject a change within a run.

The runner must also perform a tiny unmeasured cross-client visibility probe in a
run-specific preflight directory: create on one client, stat on the other, reverse
the direction, and remove it. Save the commands and output. This validates shared
namespace access but is not benchmark data.

## 6. Execution and process lifecycle

Run one coordinator on `anjuna3`. For each case it:

1. Checks the mutable allocation deadline and admission budget.
2. Creates and verifies a new owned attempt directory.
3. Captures pre-attempt filesystem, service, process, network, and device state.
4. Writes a durable `running` attempt record with the exact MPI/mdtest arguments.
5. Launches one MPI job with explicit hosts, slots, rank count, and core binding.
6. Captures stdout and stderr directly to attempt-specific local files.
7. Enforces the reservation deadline and a separate pilot-derived hard timeout.
8. Reaps the MPI process group and any verified run-owned remote launch wrappers.
9. Captures post-attempt telemetry immediately after mdtest exits.
10. Validates native output, rank placement, phase evidence, and namespace cleanup.
11. Durably marks the attempt complete only after all required artifacts validate.

The launcher should be equivalent to the following, with exact options adapted
to the captured MPI implementation:

```text
mpirun -np <total-ranks>
  --host <host:slots,...>
  --map-by ppr:<ranks-per-client>:node
  --bind-to core
  mdtest <fixed arguments>
```

Do not use an implicit host allocation or default binding. The runner records the
actual rank-to-host/core map and rejects an unexpected distribution. A launch on
`anjuna3` is local; remote ranks use MPI's non-interactive launch mechanism.

Every local and remote process receives a run/attempt identity in its environment
and command. Cleanup sends TERM, waits, then KILL only to a process whose PID,
start time, executable, command, and attempt identity all match recorded state.
Never use `pkill mdtest`, `killall`, or broad MPI cleanup.

## 7. Native evidence validation

Success requires all of the following:

- MPI and mdtest exit zero without signal or timeout.
- Native output identifies the expected mdtest version, total task count, test
  path, one iteration, and object count.
- The rank map contains exactly the planned ranks on exactly the planned clients.
- Each required directory and file phase appears exactly once with finite,
  positive elapsed time and finite, positive operation rate where operations
  were performed.
- Known summary rows for deliberately disabled operations, such as rename, are
  permitted only with the installed version's documented zero or not-run value;
  they are retained but are not primary phase rows.
- Reported operation counts agree with ranks multiplied by items per rank, subject
  only to explicitly documented mdtest scaffold operations.
- Any output-provided rate agrees with count divided by phase time within a small
  configured floating-point/printing tolerance.
- No mdtest/MPI error, skipped phase, early stonewall completion, or unknown output
  schema is present.
- Required before/after telemetry is complete and internally ordered.
- The attempt work directory is gone after owned cleanup, or cleanup is recorded
  as separately pending after otherwise valid evidence.

Do not infer success from exit status alone. Preserve unfamiliar native output and
mark the attempt failed validation; never guess columns or silently emit partial
derived rows. A parser failure after validated measurement evidence changes only
analysis state and must not repeat mdtest.

## 8. Telemetry

Capture snapshots immediately before launch and immediately after mdtest exits.
Collection must be lightweight and identical across comparable cases.

### Clients

- UTC and monotonic timestamps, hostname, load, allowed CPUs, and memory state.
- `/proc/stat`, `/proc/meminfo`, `/proc/vmstat`, and relevant interface counters.
- BeeGFS mount/options, cache mode, connection/transport state, and client identity.
- MPI rank/process map and per-process CPU statistics where available.

### Metadata host and target

- Metadata service PID identity, CPU time, RSS, context switches, and `/proc/<pid>/io`.
- Metadata target free bytes/inodes and `statfs` identity.
- Backing-device diskstats and host VM/writeback counters.
- Metadata-service interface counters and errors.

Because `anjuna3` can be both client and metadata host, collect one host snapshot
and label both roles rather than double-counting its counters.

### Storage servers

Capture interface and target-device counters on `colva1` through `colva4` before
and after each attempt. Zero-byte cases are expected to produce little bulk data
traffic, but target-selection or implementation details can still create service
work. These deltas qualify the path; they are not subtracted from mdtest rates.

Background activity may contribute to host-wide deltas. Record it as a limitation
and use process-specific metadata-daemon counters where permissions allow. Missing
required telemetry fails preflight or the attempt according to whether it is known
before or after launch; optional unsupported counters are declared in inventory.

## 9. Progress, resume, and time budget

Use one durable manifest and lock in a persistent results directory. Give every
case and retry stable IDs. Attempt states are `pending`, `running`, `completed`,
`failed`, or `interrupted`; cleanup and analysis have separate states.

The configuration fingerprint covers at least:

- Protocol/schema versions and canonical plan.
- mdtest/MPI paths, versions, help hashes, and fixed options.
- Inventory and topology evidence.
- Client placements, ranks, layouts, object counts, and repetitions.
- Namespace root, cleanup policy, instrumentation, and ordering seed.
- Cache interpretation and frozen BeeGFS settings.

On resume:

1. Reject incompatible configuration, inventory, tool, or canonical-plan changes.
2. Convert abandoned `running` attempts to `interrupted` while preserving evidence.
3. Reap only verified owned processes and clean only verified owned namespaces.
4. Revalidate every completed attempt before skipping it.
5. Retry failed/interrupted cases with a new attempt ID from a fresh namespace.
6. Retry pending cleanup independently without rerunning valid measurements.
7. Re-run analysis only when derived outputs are absent or stale.

The runner accepts `--time-limit`, `--deadline`, and an atomic deadline extension,
following the local FIO and network runner contract. Admission uses a
configuration-specific observed wall-time estimate, a safety multiplier, fixed
launch/telemetry overhead, and a cleanup reserve. The hard timeout is exceptional
failure protection, not an expected duration. At reservation expiry, terminate
and reap the active attempt, save evidence, clean owned namespace state, mark it
interrupted, flush the manifest, and exit as a resumable stop.

One mdtest invocation is the maximum restartable measurement duration. It cannot
be paused or resumed between phases because create/stat/read/remove depend on one
another and cache/namespace state is not durable benchmark progress.

## 10. Result layout and analysis

```text
<results-dir>/
  manifest.json
  deadline.json
  metadata_config.json
  metadata_inventory.json
  environment/
  attempts/<case-id>/attempt-<number>/
    command.json
    rank_map.txt
    stdout.txt
    stderr.txt
    exit.json
    telemetry_before.json
    telemetry_after.json
    validation.json
    cleanup.json
  analysis/
    phases.csv
    attempts.csv
    telemetry.csv
    summary.md
    parse_report.json
```

`phases.csv` contains one row per validated phase with placement, participating
clients, ranks per client, total ranks, layout, repetition, attempt, object type,
operation, requested and reported operation counts, elapsed seconds, native
operations/second, command wall time, and provenance paths.

Primary reporting uses native aggregate operations/second and phase elapsed time.
For every placement/layout/rank/phase report all five values, median, mean,
standard deviation, coefficient of variation, minimum, and maximum. Five runs are
too few to characterize extreme tails; do not manufacture per-operation latency
percentiles from aggregate phase timing.

Derived comparisons are:

- **Within-placement scaling efficiency:** `rate_N / (N * rate_1)`, calculated
  separately for each layout and phase.
- **Dual-client concurrency efficiency:** dual-client rate divided by the sum of
  corresponding `anjuna2` and `anjuna3` single-client rates at the same ranks per
  client, layout, phase, and repetition block.
- **Layout ratio:** per-rank-layout rate divided by flat-layout rate at fixed
  placement, concurrency, phase, and repetition block.
- **Colocation ratio:** `anjuna3` rate divided by `anjuna2` rate at fixed factors.
  Label this as an observed whole-path difference, not pure network overhead.

Present directory and file operations separately. Never average unlike operations
into one metadata score. Do not subtract network, local-storage, or end-to-end
throughput results to infer metadata overhead.

## 11. Pilot and acceptance gates

Run a separate four-case pilot with one repetition and the production 100,000-item
count:

| Case | Placement | Ranks/client | Layout | Purpose |
|---|---|---:|---|---|
| 1 | `anjuna2` | 1 | `flat` | Remote launch, baseline output, and path evidence |
| 2 | `anjuna3` | 1 | `flat` | Colocated execution and role-aware telemetry |
| 3 | `dual` | 1 | `per_rank` | Cross-host MPI launch and shared namespace visibility |
| 4 | `dual` | 16 | `flat` | Maximum rank/object count and worst planned parent contention |

Before the full run, the pilot must establish:

- Correct mdtest phase semantics and parseable native output.
- Exact rank placement and binding with no oversubscription.
- Collision-free multi-host naming and expected object counts.
- No data-bearing file I/O despite the read phase.
- Every primary one-rank phase lasts at least one second. If any is shorter,
  increase one common item count, revise the protocol/fingerprint, and repeat the
  entire pilot before starting the full run.
- Metadata inode/byte peak, wall time, cleanup time, and safety reserve.
- Complete process cleanup after success, interruption, and forced launch failure.
- Safe namespace cleanup and successful retry with a new attempt ID.
- Complete required telemetry and plausible counter ordering.
- Parser rejection of truncated output, wrong task counts, missing phases, nonfinite
  rates, stale artifacts, and incomplete cleanup.

Pilot results remain labelled pilot evidence and are not inserted into the five
full repetitions. Replace provisional admission and hard-timeout values with
pilot-observed values before full execution.

## 12. Deferred extensions

Add these only as separate protocols with their own fingerprints and analysis:

- Data-bearing small files, initially a fixed 4-KiB write/read size, with storage
  traffic and persistence semantics stated explicitly.
- Deeper directory trees using fixed branching factor and depth.
- Neighbor-rank or random stat to study non-owner lookup behavior.
- Shared-file contention and collective creation.
- Metadata durability with explicit synchronization included in measured time.
- A different metadata target layout after the cluster gains multiple metadata
  nodes or mirroring.

Do not append any extension to an existing run or reinterpret baseline zero-byte
results as data-bearing small-file performance.
