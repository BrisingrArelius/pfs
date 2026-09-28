# BeeGFS Cache-Effects Benchmark Design

**Status: offline plan and raw-to-plot visualizer implemented; live feasibility
capture, benchmark execution, restoration watchdog, and pilot not implemented.**

## 1. Research question

Measure the same normal BeeGFS read on an HDD and an SSD when data is served
predominantly from:

1. Backend storage after client and server cache invalidation.
2. Storage-server RAM after client cache invalidation.
3. Client RAM after an explicit client warm-up.

```text
client miss/server miss: client -> network -> OSS -> target device
client miss/server hit:  client -> network -> OSS page cache
client hit:              client page cache -> application
```

These are achieved-state classifications supported by traffic evidence, not names
assigned merely from command order. This design implements section 6 of
[`MicroBenchmarks.md`](../../../docs/specs/MicroBenchmarks.md).

## 2. Narrow fixed scope

Use only `anjuna2` as the measured client. `anjuna3` is excluded as a client
because it hosts management and metadata services and BeeGFS 7.4.4 colocated
roles constrain client cache modes. It remains the coordinator and metadata host.

Use one MPI rank, one 8-GiB file per medium, sequential 1-MiB POSIX reads,
one stripe, a 512-KiB chunk size, and five repetitions per medium/state. Select
one HDD and one SSD target on the same OSS. The full matrix is:

```text
2 media x 3 cache states x 5 independent repetitions = 30 measured units
```

This intentionally does not multiply by D/S/H, chooser, stripe, concurrency,
random access, or capacity. Cache location and physical medium are the only
experimental factors. Compare states within a medium first; a cross-media
comparison also changes the device, not just cache state.

Excluded:

- NetBench and synthetic server buffers.
- Direct I/O during measured reads; it would bypass the client cache being tested.
- Application-level prefetch, mmap, random access, write-back, or durability tests.
- Whole-cluster cache claims; only the owned file and observed participating paths
  are classified.
- A client-hit claim under BeeGFS `buffered` mode if the complete working set
  cannot reside in an identifiable client cache.

## 3. Feasibility gate and cache mode

Client-hit testing requires BeeGFS `native` client cache mode on `anjuna2`, a
supported mount configuration, and enough available RAM for the complete 8-GiB
file plus a configured 8-GiB safety reserve. The one participating OSS must also
have at least 16 GiB available: 8 GiB for one complete file and 8 GiB reserve.
Each case starts from cache invalidation; never warm both files at once. The
observed default `/mnt/beegfs` mount is `buffered`. Prefer a separate `native`
mount on `anjuna2` inside a private mount namespace, with a run-owned copy of
the client config under the project results directory. Keep the existing mount
and `/etc/beegfs/beegfs-client.conf` unchanged. `probe_native_mount.py` checks
whether a second mount works and `/proc` reports an independent native client.
If it fails, stop and revise the protocol rather than label buffered reads hits.

If `anjuna2` cannot safely use native mode, do not silently substitute buffered
mode and call a repeated read a client hit. Record `client_hit` as infeasible and
run a separately labelled two-state server/backend study only after approving a
revised protocol. Full three-state completion requires verified native mode.

The measured IOR and all client-side preparation must run inside the same
private mount namespace. Its extra mount disappears when the namespace ends,
including coordinator loss; still unmount and verify the original buffered
mount on normal exit. Pool changes and client/server cache drops remain
privileged global cluster state. Acquire an exclusive lock and use an independent
watchdog to restore pool membership on coordinator loss; `finally` is insufficient.

## 4. Dataset and placement

Create a reviewed inventory and two temporary single-target pools containing one
confirmed HDD and one confirmed SSD on the same OSS. Targets `101` and `104` on
`colva1` are examples from a dated inventory, not validated live selections.
Record and restore their exact original pool memberships.

Create one fresh 8-GiB file per medium beneath an authorized run root. Before
creation, set each parent pattern to its single-target pool, one target, and
512-KiB chunks. Prepare each file with normal BeeGFS POSIX direct sequential
writes and final fsync outside all measured/cache-preparation windows. Verify:

- Exact size, entry identity, owner marker, and no sparse holes.
- Desired and actual single-target layout matching the reviewed medium and target.
- Successful durable setup and sufficient free bytes/inodes.
- A deterministic payload signature sampled after preparation.

Reuse each immutable file across its 15 measurements. Every pre-measurement check
rejects identity, size, layout, pool, or modification-time changes. Cleanup and
pool restoration occur only after all valid measurements; cleanup failure never
reruns valid reads. Record the medium, file identity, and actual target ID with
every attempt; do not accept an HDD file or target as an SSD measurement.

## 5. Measured read

Use native MPI-enabled IOR directly:

```text
mpirun -np 1 <binding> ior
  -a POSIX -r -E -k -g -t 1m -b 8g -s 1 -i 1
  -o <owned-file-for-this-medium>
  -O summaryFormat=JSON -O summaryFile=<native-summary>
```

Do not pass `--posix.odirect`, `-D`, or a write option. Every measurement reads
exactly 8 GiB once. IOR's earliest-open to latest-close elapsed time and native
aggregate bandwidth are primary. Record open, transfer, close, bytes, operation
count, CPU, and command wall time separately.

Pin IOR/MPI versions and output schema. The pilot must prove IOR does not recreate,
truncate, remove, or modify the input file and that the measured operation reaches
the expected POSIX/BeeGFS cache path.

## 6. State preparation

All cache-control commands require a dedicated exclusive reservation and
non-interactive narrowly authorized privilege. Before any drop, run `sync`, wait
for writeback to settle, and verify no benchmark-owned dirty data. Dropping cache
affects unrelated processes, so fail if exclusive access cannot be established.

Each state preparation and its measured read form one indivisible restartable
unit. Warm state is never checkpointed or resumed.

### Client miss, server miss

1. Drop page/dentry/inode caches on `anjuna2` and the participating storage host.
2. Wait the fixed pilot-derived settling interval.
3. Verify low client residency for the file and capture zero-point counters.
4. Run the measured read immediately.
5. Expect client/OSS network traffic and target-device reads.

### Client miss, server hit

1. Drop caches on client and the participating storage host.
2. Perform one complete unmeasured buffered read from `anjuna2` to populate server
   and client caches; verify backend and network traffic during this warm-up.
3. Drop caches on `anjuna2` only. Do not run `sync` or cache-control commands on
   storage hosts afterward.
4. Verify low client residency, reset telemetry baselines, and measure immediately.
5. Expect network traffic but little corresponding target-device read activity.

### Client hit

1. Drop caches on client and the participating storage host.
2. Perform one complete unmeasured buffered read from `anjuna2`.
3. Do not invalidate client or server cache.
4. Verify at least 95% of file pages resident from the client using a reviewed
   `fincore`/`mincore` method, reset telemetry baselines, and measure immediately.
5. Expect little bulk network and little target-device read activity.

Warm-up output, timing, residency, and counters are setup evidence, never included
in measured throughput. No unrelated command may access the file between final
state verification and IOR launch.

## 7. Achieved-state classification

Compute ratios against IOR logical bytes using client/OSS interface deltas and
backing-device read-byte deltas for the actual target. Initial acceptance
thresholds are:

| Intended state | Bulk network ratio | Backend read ratio | Client residency before measure |
|---|---:|---:|---:|
| Client miss/server miss | >= 0.80 | >= 0.80 | <= 0.10 |
| Client miss/server hit | >= 0.80 | <= 0.20 | <= 0.10 |
| Client hit | <= 0.20 | <= 0.20 | >= 0.95 |

Ratios are evidence thresholds, not bandwidth corrections. Network protocol
overhead can make its ratio exceed 1; readahead can make backend bytes exceed
logical bytes. The pilot may tighten thresholds, but changing them afterward
requires a new protocol/fingerprint. A case outside its thresholds is retained as
`cache_influenced_unverified` and is not included in the intended-state comparison.

Host-wide counters include background activity. Require quiescence checks and
process/service telemetry; if noise cannot be bounded, fail rather than infer a
cache state. A repeated read alone is never sufficient evidence.

## 8. Inventory, preflight, and telemetry

`cache_inventory.json` records the client, shared OSS, both targets, exact mount,
cache mode, RAM, cache-control helpers, page-residency tool, backing devices,
interfaces/routes, pool baseline/restoration, IOR/MPI versions, and authorized
namespace.

Preflight verifies:

- NetBench disabled on both clients.
- `anjuna2` effective native mode on the private measured mount, unchanged
  buffered mode on the original mount, and at least 16 GiB available RAM.
- Both targets Online/Good with exact expected devices, single-target layouts,
  and at least 16 GiB available on the participating OSS.
- Non-interactive cache drop, private mount cleanup, MPI, SSH, and telemetry access.
- No competing benchmark/state lock and no pending dirty/writeback activity.
- File identities/layouts or sufficient preparation capacity.

Capture timestamps plus client/meta/OSS CPU, memory, VM, writeback, service PID
I/O, BeeGFS connection state, interfaces, target diskstats, filesystem bytes/inodes,
and page residency. Use snapshots around warm-up separately from snapshots around
the measured read.

Darshan is supplementary and enabled for all full measured reads only if a pilot
proves one-rank POSIX coverage and acceptable overhead. Save logs by explicit
attempt ID. Native IOR remains authoritative; Darshan cannot prove residency.

## 9. Ordering, progress, and recovery

Generate five repetition blocks. Within each block, use a seeded rotation of
the six medium/state combinations; do not always run HDD before SSD or cold
before warm. Each state preparation starts from cache drops as specified,
so no state inherits another measured state's cache condition.

The manifest fingerprints dataset/layout, mode, commands, thresholds, telemetry,
order seed, inventory, and tool versions. Track measurement, dataset preparation,
private-mount cleanup, pool restoration, namespace cleanup, and analysis
separately with atomic updates.

On interruption:

1. Reap only verified owned IOR/MPI processes.
2. End the private mount and restore original pool state before releasing locks.
3. Preserve partial preparation/measurement evidence.
4. Mark the whole cache-state unit interrupted.
5. On resume, revalidate cluster state and dataset, then repeat cache preparation
   and measurement under a new attempt ID.

Completed units are skipped only after native output, state evidence, thresholds,
identity, and cleanup records revalidate. Parsing never triggers a measurement.
Allocation admission uses pilot-observed drop, warm-up, measure, telemetry, and
restoration time plus a cleanup reserve.

## 10. Implementation shape

```text
cache/
  run_cache.py              plan, dataset lifecycle, state preparation, IOR run
  probe_native_mount.py     bounded private-native-mount feasibility probe
  cache_config.json         HDD/SSD workload, thresholds, order
  cache_inventory.json      RAM, modes, paired targets, devices, paths, baseline
  visualize_results.py      validate raw evidence and plot achieved cache states
  tests/                    fake counters/IOR, restoration, interruption fixtures
```

Live execution still requires `prepare_dataset()` for each medium,
`capture_state_evidence()`, `execute_unit()`, and `restore_cluster_state()`.
The existing offline code has `plan_units()`, `prepare_cache_state()`, and
`classify_achieved_state()`. Keep privileged mutations isolated and idempotent.
The classifier consumes saved evidence only; it never reruns cache-control
commands or IOR.

## 11. Validation, results, and pilot

Require exact medium/target/layout and rank/command, zero IOR exit, 8 GiB read,
finite positive native time/rate, unchanged file, complete telemetry windows,
achieved-state thresholds, NetBench-off, and restored cluster state. Store raw
commands, stdout/stderr, IOR JSON, Darshan if enabled, residency, cache-control
logs, counters, identity/layout, and restoration evidence.

Visual output generated directly from raw manifests, IOR JSON and telemetry:

```text
plots/generations/<id>/cache_throughput.png       achieved-state read bandwidth
plots/generations/<id>/cache_path_evidence.png    network/backend ratios
plots/plot_manifest.json                         current generation and exclusions
```

Report all repetitions by medium, median/mean/stddev/CV/min/max,
first-versus-warm ratios, network/backend ratios, and achieved state. Do not
subtract local, iperf3, or NetBench rates.

Pilot one full preparation of each file plus one measurement of each of the six
medium/state cases, then repeat the client-hit case on each medium (eight pilot
measurements). Inject interruption during warm-up and measurement. The pilot must
fit within 20 minutes including preparation and restoration; if a bounded
pre-pilot shows it cannot, revise and document the pilot before production. Full
execution is blocked until the pilot proves:

- The private native mount works and the original buffered mount remains unchanged.
- The 8-GiB file fits client cache and reaches >=95% measured residency.
- Cold, server-hit, and client-hit telemetry separates under the fixed thresholds.
- Cache drops and writeback settling are effective on all required hosts.
- Both datasets/layouts, visualizer rejection, resume, owned cleanup, and timing.

The 30-case wall-time estimate must use observed pilot measurements: two file
preparations plus ten cold reads, ten server-hit reads, ten client-hit reads,
twenty full-file warm-ups, thirty sets of cache-state transitions, telemetry,
pool/mode restoration and cleanup. Do not promise an allocation duration from
unit counts alone.

If any intended state cannot be verified, revise the protocol or report it
infeasible; never weaken the state label to make the matrix appear complete.
