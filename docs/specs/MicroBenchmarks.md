 # BeeGFS Microbenchmark Suite

## 1. Purpose and scope

Evaluate BeeGFS 7.4.4 storage placement and explain observed performance using
six **benchmark domains**. These are complementary experiments, not independently
measurable serial stages. Throughput differences cannot be subtracted to obtain
	per-component overheads.

All microbenchmark tools belong under `scripts/microbenchmarks/`. This is the
experimental specification, not a claim that the legacy scripts implement it.
Application validation is a separate suite. Darshan is instrumentation, not a
seventh benchmark domain. Shared factors, repetition and resume rules apply
across the experiment design.

| Benchmark domain | Primary tool | Question |
|---|---|---|
| Local storage | FIO | What can individual target filesystems and concurrent target groups deliver locally? |
| Network transport | iperf3 or RDMA benchmark | What can the actual transport and relevant network paths sustain? |
| BeeGFS communication | IOR with BeeGFS NetBench enabled | What can synthetic BeeGFS requests deliver without normal storage data operations? |
| End-to-end placement | IOR with normal BeeGFS operation | What does the complete filesystem deliver under the full placement matrix? |
| Cache effects | Controlled reads through normal BeeGFS | How does serving data from client RAM, server RAM or backend storage change performance? |
| Metadata operations | mdtest | How do namespace and file-management operations scale? |

These six groups are the planned microbenchmark suite. StorageBench is an
optional local-storage alternative/cross-check, not a second mandatory matrix.
The full end-to-end matrix is retained; pilots estimate its cost, not reduce it.
Capacity meaning and numerical levels remain unresolved.

## 2. Actual operation paths

These paths describe ordinary non-mirrored operation. Record the real topology,
including colocated services and any mirroring; replication adds traffic/work.

### File creation and opening: metadata and placement

```text
Application open/create
  -> Linux VFS / BeeGFS client
  -> metadata request over the network where needed
  -> BeeGFS metadata service -> metadata filesystem/storage
  <- metadata/layout reply to client

For new files: storage-pool eligibility + capacity classification + target chooser
              determine the target/stripe layout used for subsequent data I/O.
```

Existing files retain their layout. Bulk file contents pass directly between
clients and storage servers, not through the metadata server. Management provides
registry/control information; it is not an inline hop on every read or write.

### Reading file contents after obtaining the layout

```text
Application read -> VFS / BeeGFS client
  |-- usable client-RAM data -------------------------> return to application
  `-- request routed according to file stripe layout
        -> TCP/RDMA, NICs and network
        -> BeeGFS storage service -> target filesystem
             |-- usable server-RAM data -------------> return via network
             `-- block/storage stack -> controller/device
                                                       -> return via network
```

There may still be metadata/control traffic when file contents are cached.
Cache hits change how far a data request travels; caching is not a serial hop
after the device. Client RAM, server RAM and backend traffic must be distinguished.

### Writing file contents

```text
Application write -> VFS / BeeGFS client (possible buffering)
  -> stripe routing -> network -> BeeGFS storage service
  -> target filesystem (possible server buffering)
  -> block/storage stack -> controller/device
```

A write return can mean buffered acceptance. Persistent-write timing includes
synchronization completion under the recorded filesystem/device configuration.
Client `O_DIRECT` is not equivalent to `fsync`, does not by itself establish
durability, and does not prove server caches were bypassed. Record relevant
server synchronization and device-cache settings before making persistence claims.

## 3. Local storage

**Path:** local FIO -> target filesystem -> local storage stack/device. This is
a filesystem-backed target measurement, not pure physical-media isolation.

Use dedicated local benchmark files and direct I/O for device-oriented results.
Measure individual targets and relevant simultaneous target groups, including
targets sharing a server/controller. Do not sum isolated bandwidths and present
that sum as measured aggregate performance. Record host/mount/device identity,
filesystem, request size, job count, queue depth and dataset geometry.

| Operation | Access | Request size |
|---|---|---|
| Read | Sequential | 1 MiB |
| Write | Sequential | 1 MiB |
| Read | Random | 4 KiB |
| Write | Random | 4 KiB |
| Read | Random | 128 KiB |

The 128-KiB random read was present in the original specification. It characterizes
larger random requests alongside the 4-KiB small-request test; it is not a special
BeeGFS boundary or a demonstrated application request size. Retain it with that
explicit rationale. Record throughput, IOPS, mean/tail latency and target variation.

Reuse compatible historical FIO measurements. Missing percentiles or the 128-KiB
case mean those measurements are unavailable, not that existing throughput is
invalid. Unresolved target identities, aggregate concurrency and capacity state
limit reuse; identify the missing coverage before scheduling supplementary tests.

### Why StorageBench is optional

StorageBench generates streaming I/O locally on BeeGFS storage servers, excluding
clients/network and distributed file striping. It overlaps FIO's local-storage
scope. Use it to investigate a target, cross-check streaming results or substitute
when no suitable local baseline exists. A complete FIO matrix plus a complete
StorageBench matrix is not required. Their difference is not isolated backend
overhead; neither reproduces all real client-request handling.

## 4. Network transport

**Path:** benchmark endpoint -> transport/NIC/network -> remote endpoint.
No BeeGFS request processing or target-storage data I/O is exercised.

- Determine actual BeeGFS connections/interfaces; RDMA enabled in a configuration
  does not establish the transport used by a particular connection.
- Use iperf3 for TCP or an appropriate RDMA bandwidth benchmark for RDMA.
- Test both directions, relevant client/server pairs, single and multiple streams,
  and relevant simultaneous paths to expose shared NIC/switch limits.
- Record throughput, stream/process count, CPU load, topology, tool versions and
  relevant network counters. Report isolated and simultaneous results separately.

This matrix concerns transport paths and concurrency, not media or physical
fullness. Reuse compatible transport evidence; do not repeat identical paths for
every storage-pool label. A TCP result alone does not validate active RDMA traffic.

## 5. BeeGFS communication

**Path:** IOR -> BeeGFS client -> network -> storage-service synthetic handling.
Normal target-filesystem data reads/writes are bypassed.

NetBench is a BeeGFS-provided mode, not an IOR feature. In 7.4.4 its client runtime
interface is `/proc/fs/beegfs/<clientID>/netbench_mode`. With the mode enabled,
servers discard data writes and supply data reads from memory buffers. File
creation/unlink still operate normally. Written files can remain length zero;
the future runner must validate compatible read preparation/IOR options rather
than assume a normal write-then-read or data-verification workflow works.

Use IOR to generate sequential read/write traffic with file-per-process and
shared-file organization. Define request sizes, client/server participation,
concurrency and stripe fan-out; record actual target destinations. Match these
settings with relevant normal-IOR comparisons. Check that requested traffic
actually reaches storage servers rather than being satisfied by client caches.

This is a distinct communication characterization experiment in the suite, not
only an emergency diagnostic. It does not need a duplicate physical-fullness
matrix: normal backend data I/O is bypassed. If placement changes participating
servers/fan-out, record and measure those distinct communication configurations.

Report repeated synthetic throughput separately. CPU, request handling,
concurrency and client settings affect it. It is not pure wire bandwidth, normal
cache-hit performance, a universal storage upper bound, or subtractable overhead.
Use ordinary IOR and telemetry to investigate differences. NetBench is not a
replacement for either network tests or cache-effects tests.

Use dedicated test files. Record mode state on every participating client and
disable/verify it after the test, on controlled interruption, and before normal
I/O. An aborted allocation requires checking mode state on reacquisition too.

## 6. End-to-end placement

**Path:** normal IOR through the complete BeeGFS data path described above,
including actual storage operations and the declared cache treatment.

| Operation | File organization |
|---|---|
| Sequential read | File per process (N-N) |
| Sequential write | File per process (N-N) |
| Random read | File per process (N-N) |
| Random write | File per process (N-N) |
| Sequential read | Shared file (N-1) |
| Sequential write | Shared file (N-1) |

Retain the full experiment matrix: D/S/H, randomized/roundrobin, stripe
counts 1/2/4, stripe sizes 256 KiB/512 KiB/1 MiB, workload concurrency and five
repetitions. Capacity remains an unresolved factor; the proposed two free-space
conditions are not silently treated as agreed. Define IOR API, transfer sizes,
dataset sizes, client/rank placement and timing options before execution.

For every measured run:

1. Verify inventory, membership, effective chooser, cache configuration and
   NetBench-disabled state. Record the actual transport and session identity.
2. Apply pool/stripe pattern before creating new files. Recreate files when
   placement changes; preserve logical dataset contents across comparisons.
3. Verify sample file layouts, desired/actual stripe count and participating
   targets. Record eligibility limits rather than silently changing the matrix.
4. Apply the same declared preparation/cache protocol to comparable runs.
5. Include synchronization in persistent-write phase time/throughput. Size the
   working set using actual per-server placement and memory, not total RAM alone.
6. Collect benchmark output plus CPU, network, target/backend and memory/writeback
   telemetry. Target-service traffic alone does not demonstrate device access.
7. Retain raw outputs and checkpoint progress under the global completion rules.

Collect throughput, random IOPS, phase times, per-process completion skew where
available, bytes/utilization per target and variation across repetitions. State
how each metric is obtained; Darshan summaries alone do not supply every metric.

If the capacity experiment is confirmed as same-device high/low free space,
apply the global provisioning/headroom rules. Observe every target, not just
pool-average fullness; prepare filler outside timing and restore conditions.
A mixed Normal/Low target-allocation study is a different experiment. Changing
reported free space tests policy, not physical filesystem fullness.

## 7. Cache effects

### Cache location, mode and state

- **Client cache:** RAM on the application/client machine. BeeGFS `buffered` uses
  small read-ahead/write-back buffers; `native` uses the Linux page cache and can
  retain a larger working set. This is not a client-local SSD cache.
- **Server cache:** RAM used by the target filesystem on storage servers,
  independently of the BeeGFS client cache mode.
- Application prefetch/buffering and controller/device caches may also matter.
  Record relevant settings; do not invent a separate matrix for each by default.

Cache **mode** specifies caching behavior; cache **state** describes whether the
particular data is resident. There is no single whole-system cache on/off switch.
On colocated client/server nodes, account for shared resources and BeeGFS 7.4.4's
restriction to buffered client mode; do not prescribe native mode there.

### Dedicated normal-read experiment

Use normal BeeGFS reads with identical logical workloads and placement for each
comparison. Define a working set and mode that can realize each requested state.

| Intended condition | Expected bulk-data path | Corroborating evidence |
|---|---|---|
| Client miss, server miss | Client -> network -> server -> backend storage | Network traffic and backend read activity |
| Client miss, server hit | Client -> network -> server RAM | Network traffic with little corresponding backend read activity |
| Client hit | Client RAM -> application | Little corresponding bulk network/backend activity |

Specify preparation commands, scope and timing for clients and servers. Flush
pending writes before any cold-read preparation. Warm server data with a known
read, then invalidate client data without invalidating server data where feasible;
verify rather than assuming a cache state. Client-hit tests require data that
fits the effective client cache; a large fully-warm working set is not generally
possible in buffered mode. Readahead, background traffic and eviction complicate
attribution; retain telemetry and qualify states that cannot be established.

Record first/repeated read time, throughput, network/backend bytes, per-server
data volume, RAM, cache mode and achieved state. Use five independent repetitions,
recreating the intended state each time. A repeated read alone is not proof of a
hit. Label unverified cases cache-influenced, not verified cold or fully warm.

### Protocol shared by all tests

The main placement matrix uses a fixed declared cache mode and preparation
protocol for direct comparisons. Do not automatically multiply every workload
by a supposed on/off switch. Explicitly add mode as a factor only for a declared
mode-comparison study. Use direct local FIO for device-oriented results; normal
IOR needs working-set/persistence controls and backend evidence. Capacity limits
may prevent sustained working sets; report infeasibility or cache influence.

This dedicated cache experiment is part of the microbenchmark suite. Application
validation retains realistic caching and reports first/subsequent epoch behavior.
NetBench memory responses do not represent ordinary cached file data.

## 8. Metadata operations

Use mdtest for create, stat, open/read/close and remove operations. Match selected
application concurrency points, compare flat/hierarchical directory organization,
and use five repetitions under a fixed declared storage/metadata configuration.
Specify files per process, file sizes, directories and phase timing before running.

**Path:** client -> metadata service/storage, with operation-dependent storage-
server work. This is not pure metadata-daemon CPU isolation. Distinguish zero-byte
namespace tests from data-bearing small-file operations; record any data traffic.
Pools restrict data targets, not metadata-server placement, so do not mechanically
repeat the full media/stripe matrix as a metadata-server experiment.

## 9. Instrumentation and interpretation

Benchmark output reports performance; Darshan reports application-issued I/O;
system monitoring reports the actual client/server/network/backend activity.
Darshan does not identify cache residency, device traffic or network saturation.
Its usual logs are summaries; DXT detailed tracing is a separately declared option.

| Benchmark domain | Darshan policy |
|---|---|
| Local storage | FIO native metrics; no routine Darshan |
| Network transport | Transport-tool metrics; no Darshan |
| BeeGFS communication | Optional IOR instrumentation, explicitly synthetic |
| End-to-end placement | Instrument measured IOR runs after integration validation |
| Cache effects | Instrument supported measured reads to check application I/O consistency; use system telemetry for state |
| Metadata operations | mdtest native metrics primary; Darshan supplementary where useful |

Keep instrumentation identical across direct comparisons. Check enabled modules,
process/rank coverage and overhead in a pilot. Preserve raw logs with unique run
association, independently of derived CSVs; parsing may resume without rerunning
the measurement. Exclude setup I/O from measured-phase statistics or label it
separately. Global instrumentation/resume rules also apply to application suites.

During normal IOR collect per-core/client/server CPU, interface traffic and
relevant error/retransmission counters, per-target/device activity, memory/cache/
writeback indicators and file layouts. This covers interactions no isolated tool
reproduces completely, including real storage-service filesystem-request handling.
Use comparisons to narrow hypotheses, never subtract unrelated throughput values
to claim component overhead. Equal pool throughput does not imply equal device speed.

## 10. Historical coverage and remaining work

| Existing evidence | Reusable scope | Important gaps |
|---|---|---|
| April-4 local FIO | Historical sequential and 4-KiB random target-path behavior | Host/device mapping, concurrent target groups, 128-KiB random reads, tail latency and capacity provenance |
| Historical iperf3 | Measured TCP paths/directions | Actual BeeGFS transport, current topology and relevant simultaneous paths |
| April-2/16 BeeGFS FIO | Genuine preliminary end-to-end HDD/SSD comparison | Full D/S/H × chooser × stripe × concurrency × capacity design and cache/placement evidence |
| OST logs/plots | Historical allocation/capacity observations | Complete run association and traffic/state verification |

No verified dedicated NetBench, controlled three-state cache or mdtest result set
has been identified in the preserved results. Existing synthetic-workload Darshan
results belong to the application evidence, not a completed DLIO experiment.
Old FIO through BeeGFS is not invalid because it used FIO rather than IOR; reuse
is limited by configuration coverage and provenance. Compatible results may be
reused per benchmark domain without treating unknown fields as current machine state.

## 11. Execution, checkpointing and duration

1. Inventory topology, transport, memory, target membership, chooser and settings.
2. Audit/reuse local and transport evidence; fill coverage gaps.
3. Characterize synthetic BeeGFS communication, then verify NetBench is disabled.
4. Execute normal-IOR placement runs under the declared cache protocol.
5. Execute the dedicated cache-effects and metadata groups.
6. Continue with separately specified application validation.

Every domain and the whole suite must implement persistent progress. Completed
measurements and analysis stages are distinct; an interrupted measurement is
retried after restoring preparation. Warm caches are not durable checkpoints. On
allocation changes revalidate settings, restore appropriate state and record the
session boundary. Legacy scripts do not yet implement this suite-wide contract.

| Group | Wall-time model (include repetitions) |
|---|---|
| Local FIO | Invocations × (runtime + ramp) + file preparation/cleanup |
| Network | Path/direction/stream/concurrency cases × duration + setup |
| NetBench | Communication configurations × duration + preparation/mode transitions |
| Normal IOR | Preparation + read bytes/rate + write bytes/rate + synchronization + cleanup |
| Cache effects | State preparation + measured reads, repeated for each feasible condition |
| Metadata | Counts/rates by phase + directory setup/cleanup |

Illustrations, not cluster wall-time promises:

- The preserved FIO runner uses 60 s runtime plus 5 s ramp. Its 840 local cases
  imply about 15.2 h of scheduled FIO time; 240 pool cases imply 4.3 h, both before
  preparation and other overhead. Concurrent targets require separate estimates.
- With two proposed capacity levels, full IOR coverage is
  `3 × 2 × 2 × 3 × 3 × 5 × 6 = 3,240` measured cases per concurrency point.
  This is 54 h at one minute per case or 270 h at five minutes, excluding setup.
  Read/write phases may share a command but still consume time. Capacity remains
  unresolved; these numbers are conditional planning estimates.
- Reading 1 TiB at 280 MiB/s takes about 62 minutes before other work.

Use short pilots to estimate each group's preparation, measurement and cleanup
costs; retain uncertainty and update estimates from completed cases. Schedule
the full matrix across allocations. Define the maximum duration of one restartable
measurement; if it exceeds allocation length, obtain a longer allocation or
declare scientifically valid segmentation before measurement, not arbitrary pause.

## 12. Run record and unresolved execution parameters

Retain exact commands/tool versions, configuration fingerprint, topology,
transport evidence, membership/chooser, requested/actual stripes, capacity,
cache preparation/state evidence, data geometry, instrumentation, repetition,
block/allocation order, exit status, raw output/logs, telemetry and derived data.
Use `results/microbenchmarks/runs/<run-id>/` with a manifest, progress records and
separate raw/derived artifacts. Record reuse provenance and unknown fields.

Before execution specify client/process placement, IOR API/request sizes/dataset
sizes/timing, target concurrency groups, NetBench-compatible read setup, mdtest
counts/file sizes, feasible cache modes/states and exact monitoring sources.
Resolve capacity only when the research definition/site bands are agreed.

## 13. Why the old eight-item design changed

The original L0–L7 table is in Obsidian commit `d74790e`. It already rejected
additive-overhead arithmetic; that was not a new correction. The intermediate
B/E/S/V/D grouping hid scope changes and is superseded by the six plain groups.

| Original item | Current treatment |
|---|---|
| L0 physical storage / FIO | Local storage; accurately includes the local filesystem |
| L1 StorageBench | Optional local-storage cross-check; no duplicate mandatory matrix |
| L2 network | Network transport |
| L3 NetBench + IOR | Explicit BeeGFS communication benchmark domain |
| L4 normal IOR + pooling | End-to-end placement, full matrix retained |
| L5 cold/warm cache study | Explicit cache-effects group plus a protocol shared by all tests |
| L6 mdtest | Metadata operations |
| L7 DLIO/applications | Separate application experiment specification |

Eight becomes six by consolidating the overlapping local-storage tools and
separating application validation. No distinct communication or cache experiment
is hidden in a generic diagnostic bucket. Tests cover paths and branches, not
every internal function; end-to-end telemetry investigates their interactions.

## 14. References

- [BeeGFS 7.4.4 architecture](https://doc.beegfs.io/7.4.4/architecture/overview.html)
- [Benchmarking: FIO alternatives, StorageBench, NetBench, IOR and mdtest](https://doc.beegfs.io/7.4.4/advanced_topics/benchmark.html)
- [Client caching modes](https://doc.beegfs.io/7.4.4/advanced_topics/client_caching.html)
- [Storage pools](https://doc.beegfs.io/7.4.4/advanced_topics/storage_pools.html)
- [Target management/capacity classes](https://doc.beegfs.io/7.4.4/advanced_topics/target_management.html)
- Herold and Breuner, *An Introduction to BeeGFS*, ThinkParQ, version 2.0, June 2018.
- Boito, Pallez and Teylo, *The role of storage target allocation in applications' I/O performance with BeeGFS*, Cluster 2022, DOI `10.1109/CLUSTER51413.2022.00039`.
- Chowdhury et al., *I/O Characterization and Performance Evaluation of BeeGFS for Deep Learning*, ICPP 2019, DOI `10.1145/3337821.3337902`.
- Oeste et al., *ADA-FS — Advanced Data Placement via Ad hoc File Systems at Extreme Scales*, 2020, DOI `10.1007/978-3-030-47956-5_4`.
