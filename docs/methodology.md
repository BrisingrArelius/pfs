# Experiment methodology

This repository documents the experiment design and implementation status. Its
documentation is standalone; no local external reference path is required.

## Factors

| Factor | Definition |
|---|---|
| Storage configuration | D: all candidates in Default; S: SSD-only; H: HDD-only |
| Target chooser | `randomized` (random) and `roundrobin` |
| Stripe count | 1, 2, 4; record actual count and eligibility limitations |
| Stripe size | 256 KiB, 512 KiB, 1 MiB |
| Capacity | Proposed high/low free space on the same devices; meaning/bands pending confirmation |
| Concurrency | Workload-specific scaling points |
| Repetitions | Five independent measured executions per configuration |

“Pooling off” means D, not an unpooled BeeGFS state. Every target belongs to one
storage pool; D cannot coexist with S/H using the same candidate targets.
There is no separate named-mixed M experiment. D/S/H compare eligible target
sets/media, potentially with different target counts; not isolated pool overhead.

Capacity comparisons within each configuration preserve devices, eligible target
count and working set. Record free bytes/percent, free inodes, installed capacity,
and internal Normal/Low/Emergency class **per target**, before and after each run.
Prepare filler outside timing, let preparation activity settle and restore bands
between repetitions. Account for input and checkpoint growth. Installed capacity,
physical fullness and internal capacity classification are distinct variables.
No numerical bands or artificial free-space overrides are prescribed here.

Use recorded, counterbalanced blocks for expensive state changes, with feasible
randomization within blocks. Membership and measurement conditions remain fixed
during a run; verify them at every transition. Create fresh files after changing
placement, chooser or stripe pattern and verify actual placement and traffic.

## Measurements and feedback resolution

| Role | Purpose |
|---|---|
| Local storage — FIO | Individual targets and relevant concurrent target groups |
| Network transport — TCP/RDMA tools | Actual transport, paths/directions and simultaneous traffic |
| BeeGFS communication — IOR + NetBench | Synthetic requests across relevant clients/servers and stripe fan-out |
| End-to-end placement — normal IOR | Full pool/chooser/stripe/capacity/concurrency comparison matrix |
| Cache effects — controlled normal reads | Feasible client-miss/server-miss, client-miss/server-hit and client-hit states |
| Metadata operations — mdtest | Explicit operations, directory layouts and concurrency |

These are six benchmark domains, not additive stages. Application validation through
DLIO/synthetic workloads is separately specified. StorageBench is an optional
local-storage alternative/cross-check, not a second mandatory matrix. Component
domains use their applicable factors; the full end-to-end matrix remains intact.

NetBench discards server writes and returns memory-backed reads. It measures a
synthetic BeeGFS communication path affected by request handling, CPU, concurrency
and client configuration. It neither predicts storage performance nor measures
normal caching, pure wire bandwidth or additive overhead. It is an explicit suite
experiment: match relevant request sizes, concurrency and participating servers,
verify server traffic, and report it separately. Disable and verify mode on every
participating client before normal I/O, including after interrupted allocations.
Physical-fullness combinations do not characterize its bypassed backend path.
It is not a universal performance upper bound.

For sustained IOR, include write synchronization in timing, size the working set
for actual placement and per-server memory, and collect backend device traffic
alongside target/network traffic. Client cache dropping or direct I/O does not
prove cold server caches. Label results sustained-storage, cache-influenced or
verified cold according to evidence. For DLIO preserve realistic caching and
report epoch 1 separately from later epochs; first access is not automatically cold.

### Operation paths and cache controls

Create/open obtains namespace and layout information through the metadata service.
Bulk data flows directly between clients and storage servers. A read can return
from client RAM, or traverse the network to server RAM, or continue to backend
storage. Writes may be buffered on clients/servers; direct I/O is not durability.
Metadata tests can also trigger storage-server work, depending on the operation.
Record mirroring if enabled; the simple paths describe non-mirrored operation.

Client `buffered` mode uses small read-ahead/write-back buffers; `native` uses the
Linux page cache. Cache mode differs from whether particular data is resident.
The dedicated cache domain recreates feasible states and checks bulk network and
backend traffic. Main placement comparisons share a declared mode/preparation
protocol rather than automatically multiplying every workload by cache on/off.

### Darshan, progress and runtime

Darshan records application I/O, not the cache/device source of bytes. Validate
rank/worker coverage and overhead; retain native metrics and system telemetry.
Normal IOR and supported application measurements use validated instrumentation;
NetBench instrumentation is optional and synthetic, mdtest supplementary, and
local FIO/network benchmarks use native output. Standard summaries are not DXT
traces and do not automatically supply per-epoch metrics. Retain raw logs with
explicit run association independently of parsed output.

Every domain needs durable suite progress: configuration fingerprints, stable
configuration/repetition/phase IDs, distinct attempts, atomic updates and validated
completion. Resume skips completed measurements, retries interrupted units after
preparation, and retries failed parsing without repeating successful I/O. Record
allocation boundaries and revalidate topology/settings; warm caches cannot be
checkpointed across allocations. A dependent epoch sequence must restart as a
unit. This contract exceeds the legacy workload runner's current resume support.

Pilot preparation/measurement/cleanup to schedule the full matrix. With two
proposed capacity levels, IOR has 3,240 operation/layout cases per concurrency
point: 54 h at one minute each or 270 h at five, before overhead. The existing
60 s + 5 s FIO timing implies about 15.2 h for 840 local invocations and 4.3 h
for 240 pool invocations before preparation. These are conditional estimates,
not observed total runtimes. Fit restartable measurements within allocations.

Future functionality is tracked in the [implementation backlog](IMPLEMENTATION_BACKLOG.md).

## Historical interpretation

The approximately 280–282 MiB/s sequential pool results are close to the historical
2.35 Gbit/s TCP measurement (280.14 MiB/s). This suggests a shared/network-path
ceiling, but configured RDMA support means the actual BeeGFS transport remains
unproven. Historical target-local plots do not measure client-to-OSS transport.
Missing run metadata and ambiguous FIO size labels must remain explicit.

## Sources

- [BeeGFS 7.4.4 benchmarking](https://doc.beegfs.io/7.4.4/advanced_topics/benchmark.html)
- [Target management and capacity classes](https://doc.beegfs.io/7.4.4/advanced_topics/target_management.html)
- [Storage pools](https://doc.beegfs.io/7.4.4/advanced_topics/storage_pools.html)
- [Client caching](https://doc.beegfs.io/7.4.4/advanced_topics/client_caching.html)
