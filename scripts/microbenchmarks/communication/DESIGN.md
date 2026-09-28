# BeeGFS Communication Benchmark Design

**Status: offline plan and raw-to-plot visualizer implemented; live runner,
reviewed inventory, privileged restoration, and cluster pilot not implemented.**

## 1. Purpose and interpretation

Measure BeeGFS client-to-storage-service communication with BeeGFS 7.4.4
NetBench enabled. IOR generates normal BeeGFS requests, but storage servers
discard write payloads and synthesize read payloads from memory rather than
performing normal target-filesystem data I/O.

```text
IOR rank -> POSIX -> BeeGFS client -> TCP -> BeeGFS storage service
                                      NetBench: no normal backend data transfer
```

This measures BeeGFS protocol, client/server request handling, memory copies,
CPU, and transport together. It is not pure TCP bandwidth, normal cached-file
performance, a storage-device result, or an overhead that can be subtracted
from another benchmark.

This design implements section 5 of
[`MicroBenchmarks.md`](../../../docs/specs/MicroBenchmarks.md), the invariants in
[`Global.md`](../../../docs/specs/Global.md), and the observed topology in
[`CLUSTER_TOPOLOGY.md`](../../../docs/CLUSTER_TOPOLOGY.md).

## 2. Included and excluded scope

Included:

- `anjuna2`, `anjuna3`, and equal-rank dual-client placements.
- Sequential synthetic writes and reads.
- File-per-process (`N-N`) and shared-file (`N-1`) organizations.
- One and four MPI ranks per participating client.
- One-target and four-target stripe fan-out.
- Five independent measured repetitions.
- Native IOR output, rank maps, actual layouts, network/service CPU, memory,
  interface, and backend-device evidence.

Excluded:

- Random I/O; NetBench is a streaming communication characterization.
- D/S/H, capacity, and stripe-size sweeps. Backend data I/O is bypassed.
- Cache-state claims. POSIX direct I/O is used to prevent client cache hits from
  replacing the intended BeeGFS requests.
- Data verification in measured NetBench mode. Writes are discarded and reads
  are synthetic by definition.
- Darshan in the initial matrix. It is optional only after a pilot demonstrates
  complete rank coverage and acceptable overhead, then remains fixed for a run.
- Simultaneous read/write traffic.

## 3. Fixed IOR protocol

Use the installed MPI-enabled IOR binary directly, not
`scripts/workloads/posix_synthetic_workload_IOR.py`. The application wrapper is
single-rank and does not own NetBench state or multi-client lifecycle.

| Setting | Value |
|---|---:|
| API | POSIX |
| Transfer size | 1 MiB |
| Block size | 16 GiB per rank |
| Segments | 1 |
| Measured stonewall | 30 seconds |
| IOR iterations | 1; repetitions are external |
| Intra-test barriers | Enabled (`-g`) |
| Direct I/O | Enabled (`--posix.odirect`) |
| Final fsync | Disabled in NetBench mode |
| Files | Kept until validation and owned cleanup |

The 16-GiB block exceeds the maximum payload one 2.5-Gbit/s client link can
move in 30 seconds. The pilot must confirm every measured rank reaches the
stonewall rather than exhausting its block; otherwise increase one common block
size, revise the protocol, and repeat the pilot.

The measured command is equivalent to:

```text
mpirun <explicit hosts/ranks/binding> ior
  -a POSIX -t 1m -b 16g -s 1 -i 1 -g -D 30 -k
  --posix.odirect
  <-w OR -r -E>
  <-F for file-per-process>
  -o <owned-path>
  -O summaryFormat=JSON -O summaryFile=<native-summary>
```

Exact option spelling is pinned to captured `ior -h` output. Preflight must
prove that the installed build supports POSIX, BeeGFS, stonewalling, direct I/O,
and machine-readable summaries. Unknown output schemas are rejected rather than
parsed heuristically.

### Write protocol

Create a fresh path while NetBench is enabled and run IOR `-w`. Servers discard
payloads, so file length may remain zero. The pilot records the exact IOR warning
and exit semantics for this version. Validation accepts only that reviewed
zero-length warning; unrelated size, short-transfer, or MPI errors fail.

### Read protocol

A read cannot be prepared by a NetBench write because that file can remain length
zero. With NetBench disabled everywhere, prepare the complete IOR file(s) using:

```text
ior -a POSIX -t 1m -b 16g -s 1 -i 1 -g -w -k -e --posix.odirect ...
```

Verify size, layout, target eligibility, IOR exit, and synchronization before
enabling NetBench. Prepared read files may be reused across the five repetitions
of an otherwise identical configuration after identity/layout revalidation.
Preparation is never included in measured communication throughput.

## 4. Fixed matrix

| Factor | Values |
|---|---|
| Client placement | `anjuna2`, `anjuna3`, `dual` |
| Ranks per participating client | 1, 4 |
| Direction | synthetic write, synthetic read |
| Organization | file per process, shared file |
| Desired stripe count | 1, 4 |
| Repetition | 1-5 |

```text
3 placements x 2 rank levels x 2 directions x 2 organizations
x 2 stripe counts x 5 repetitions = 240 measured units
```

Single-client totals are one or four ranks. Dual-client totals are two or eight
ranks, split equally. `anjuna3` cases are labelled colocated because that host
also runs management and metadata services.

One IOR invocation is one restartable unit. A dual-client MPI invocation is one
unit; ranks are never launched by separate coordinators or combined across
attempts.

Generate a seeded canonical plan. Block read units by prepared-dataset identity,
shuffle configurations inside each block, and rotate the base order between
repetitions. Persist the complete plan before traffic.

## 5. Controlled placement

Create a reviewed communication inventory with one eligible confirmed target on
each storage server. The initial proposed four-target set is `104,205,305,404`,
subject to fresh device/media verification. Place those targets in a dedicated
experimental pool during this domain and restore every target to its exact prior
pool afterward. Never consume the historical hard-coded pool scripts as live
inventory.

Use a fixed 512-KiB chunk size and desired stripe count 1 or 4. Apply the pattern
to a fresh configuration directory before creating files. Query every file with
`beegfs-ctl --getentryinfo --verbose` and record desired and actual target IDs.
A four-stripe file must include one eligible target from each OSS. A one-stripe
file records its selected OSS; file-per-process cases record every rank file.

Freeze `tuneTargetChooser=roundrobin` for this experiment to avoid uncontrolled
random geometric imbalance. Changing it requires privileged metadata-service
configuration and restart; capture the original value, change it only in an
exclusive maintenance allocation, verify the effective value, and restore it
after the domain. A failure to restore blocks all later normal-I/O experiments.

## 6. NetBench state transaction

NetBench is controlled independently on each client through:

```text
/proc/fs/beegfs/<client-instance>/netbench_mode
```

The path is discovered from the mounted client instance, never hard-coded from
an old capture. Every transition uses non-interactive, narrowly authorized
privilege. Interactive `sudo` during a measurement is prohibited.

For each unit:

1. Verify NetBench is `0` on both clients.
2. Prepare/revalidate read files while it is `0`.
3. Capture pre-measurement telemetry.
4. Enable (`1`) only on participating clients and read it back.
5. Launch the synchronized MPI/IOR invocation.
6. Disable (`0`) on participating clients immediately after IOR exits.
7. Verify `0` on both clients before parsing, file cleanup, or any normal I/O.
8. Capture post-measurement telemetry and validate evidence.

Install a run-owned privileged lease helper on each participating client before
full execution. It must restore `0` if its coordinator heartbeat expires or its
supervised MPI process exits. The Python runner also disables mode in `finally`,
on signals, at startup, before resume recovery, and at normal completion. These
layers are complementary; a manifest is not proof that runtime state was reset.

If mode cannot be proven disabled, stop. Do not start another benchmark domain.

## 7. Inventory and preflight

`communication_inventory.json` records:

- Host roles, client IDs, mount identity, MPI/IOR/BeeGFS versions, and CPU sets.
- NetBench proc paths and a tested non-interactive enable/disable mechanism.
- Management, metadata, storage nodes, targets, media, pools, and service ports.
- Exact experimental targets and baseline pool membership restoration map.
- Client-to-OSS interfaces, addresses, transport, MTU, and link speed.
- Metadata chooser path/value and service control mechanism.
- Authorized BeeGFS namespace and persistent result location.

Preflight verifies all fields live, all target/node states Online/Good, the exact
mount from both clients, sufficient free bytes/inodes for read preparation,
NetBench disabled, no conflicting runner lock, non-interactive MPI launch, and
no unsupported oversubscription. It runs a tiny unmeasured write/read semantic
probe to establish this IOR version's NetBench behavior and cleans it safely.

## 8. Execution and telemetry

Run one coordinator on `anjuna3`. It creates owned namespace directories and a
single MPI hostfile/rank map per attempt. Pin one rank per allowed core. Save the
resolved executable paths, environment, complete argument vectors, stdout,
stderr, machine-readable IOR summary, rank map, exit status, and process identity.

Capture immediately before enabling and after disabling NetBench:

- Client CPU, memory, VM, BeeGFS connection state, and interface counters.
- Storage-service PID CPU/RSS/I/O and per-OSS interface counters.
- Target/device diskstats, free space, and writeback state.
- Metadata-service process and metadata-target counters.

NetBench measurements should show BeeGFS storage-service network traffic with
little corresponding target-device data activity. Background traffic prevents
perfect equality, so retain raw deltas and establish validation tolerances in
the pilot. A result with substantial unexplained backend activity is labelled
invalid NetBench evidence, not normal communication performance.

## 9. Native validation and metrics

Require:

- Zero MPI/IOR exit status and exact expected ranks/hosts.
- Expected API, operation, organization, transfer/block size, barriers, direct
  I/O, stripe count, and 30-second stonewall in native output.
- Positive bytes, operations, and finite phase times/rates for every rank.
- Every measured rank reached the stonewall within configured tolerance.
- No short-transfer, EOF, data-check, or unexpected file-size error. Only the
  pilot-approved NetBench zero-length write warning is allowed.
- NetBench was verified `1` during traffic and `0` afterward.
- Actual layouts and eligible targets match the configuration.
- Required telemetry and complete cleanup are present.

Primary metrics are IOR aggregate MiB/s, bytes, operation count, transfer time,
open/close time, and rank completion skew. Also report client/OSS CPU, aggregate
network bytes, backend-device bytes, and five-repetition variation. Preserve
native units and state exactly how IOR computed each rate. Do not subtract
iperf3 or normal-IOR throughput to estimate overhead.

## 10. Implementation shape

```text
communication/
  run_communication.py       coordinator, plan, NetBench transaction, IOR lifecycle
  communication_config.json fixed scientific settings and planning estimates
  communication_inventory.json reviewed hosts, targets, paths, and restoration map
  visualize_results.py       validate native evidence and plot synthetic rates
  tests/                     fake native IOR and visualization fixtures
```

The code currently supplies `plan_units()`, command builders, saved-state checks,
and native-evidence validation. Live execution still requires read-only **live**
`preflight()`, normal-mode `prepare_read_dataset()`, watchdog-backed
`netbench_transaction()`, `execute_unit()`, and `restore_cluster_state()`.
The visualizer reads saved evidence and never starts MPI.

## 11. Progress, cleanup, and artifacts

Use the same durable contract as the FIO/network designs: one locked manifest,
scientific fingerprint, canonical plan, session records, stable unit IDs, unique
attempt IDs, atomic writes, mutable allocation deadline, cleanup reserve, and
separate measurement/cleanup/analysis state.

An interrupted unit is retried in full after NetBench is disabled and its setup
is restored. Completed evidence is revalidated before skip. Read preparation is
checkpointed independently but reused only after path, identity, size, layout,
pool, chooser, and NetBench-off validation. Cleanup removes only marker-owned
paths below the authorized root.

Results contain:

```text
manifest.json, deadline.json, configuration/inventory snapshots
state-transitions/, preparations/, attempts/, environment/
plots/generations/<id>/synthetic_read.png, synthetic_write.png, backend_ratio.png
plots/plot_manifest.json (current complete generation)
```

The visualizer reads raw attempts, checks native metrics and recorded state, then
creates plots directly. Plotting failure never reruns IOR.

## 12. Pilot gates

Pilot one repetition of eight cases: one and four ranks, both directions, both
organizations, covering single and dual clients and stripe counts 1 and 4. Include
at least one maximum eight-rank dual-client case.

Before the full run prove:

- NetBench enable/disable and watchdog restoration under success, SIGTERM, and
  forced coordinator loss.
- Write zero-length semantics and read preparation compatibility.
- All ranks reach 30-second stonewall with 16-GiB blocks.
- Machine-readable IOR parsing and rank placement.
- Stripe/layout verification and pool/chooser restoration.
- Network activity with negligible backend data activity under pilot tolerances.
- Attempt cleanup, resume, evidence corruption rejection, and realistic timing.

Do not schedule the full matrix until the original pool membership and chooser
can also be restored in a failure-injection test.
