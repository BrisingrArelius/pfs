# Experiment methodology status

This document describes the methodology currently represented by repository code
and evidence.

## Implemented tools

| Area | Present implementation |
|---|---|
| Local storage | Per-target FIO runner with native JSON, deadlines and measurement-level resume; protocol-5 full run completed |
| Network transport | Six-host iperf3 runner with native endpoint JSON, synchronized concurrent epochs, telemetry, and resume; 190-unit/320-path full raw run available |
| Microbenchmark figures | FIO, iperf3, and cache visualizers plot completed native run artifacts directly; metadata has no plotter yet |
| Cache effects | Separate read (`run_cache_read.py`) and write (`run_cache_write.py`) runners share `cache_common.py` preparation; buffered/native full read and four full write runs are available with combined plots |
| Metadata operations | `run_mdtest.py` completed its four-case, 1,000-item-per-rank pilot from anjuna3; the 90-case full plan is configured for 10,000 items per rank and five repetitions, but has not run |
| Placement administration | Shell helpers with historical hard-coded target inventories |
| Application workloads | Single-process IOR wrapper, two current contiguous read-only profiles, Darshan invocation, and run-specific logs/checkpoints |
| Darshan parsing | Aggregate POSIX/MPI-IO/STDIO rows in `global.csv` |
| Workload analysis | Aggregate statistics, derived POSIX metrics, heatmaps, PCA, and HDD/SSD comparisons |
| Trace characterization | Contiguity and operation-rate analyses over preserved external Darshan-derived CSVs |

The previous local FIO implementation is archived. The replacement follows the
[local-storage design](../scripts/microbenchmarks/fio/DESIGN.md): one prepared
file per target, five workloads with five repetitions, reservation deadlines and
measurement-level resume. Protocol 5 uses four sustained direct-I/O jobs and has
completed a 700-measurement four-host run. The user-directed scope is per-target
only, with no simultaneous-target experiment.
Historical local FIO and BeeGFS FIO datasets exist under
`results/microbenchmarks/legacy/`. Historical network measurements also exist,
and the current iperf3 full run has completed. FIO and iperf3 visualizers read
native run artifacts directly and write version-aware/epoch-aware plots without
an intermediate CSV parser. The completed FIO run includes both FIO 3.28 and
3.36, so figures and per-media reference lines are separated by tool version.

## Missing experiment coverage

Local FIO and iperf3 have documented protocols and completed cluster runs.
BeeGFS communication remains offline-only. Metadata has a completed functional
pilot and a defined full raw-evidence path, but no full cluster run, automated
native-output parser, or plots. The cache domain
has completed buffered and native full **read** runs, with three early native
investigation pilots preserved separately. The write runner implements
server-cache/disk fsync-policy cases and native client write acceptance; all
four buffered/native × fsync-policy full runs are available. The current workload
pipeline is application-level tooling, not a completed DLIO experiment.
At the owner's request, the metadata runner is narrowly scoped: it has fixed
per-case timeouts and skips intact completed cases on rerun, but no
allocation-wide deadline extension or automatic recovery of a failed MPI case.

The whole-workload D/S/H, chooser, stripe, capacity, and concurrency matrix
belongs to the separate DLIO experiment; it is not implemented. Numerical
capacity bands are not defined. Suite-wide durable progress, allocation recovery, complete
provenance capture, mutable allocation deadlines and clean time-budget shutdown
across the whole suite are also missing. Buffered cache states were verified
with client/server traffic and device-read counters. Native pilot `-06` and
full `-01` also verified their intended paths with raw IOR and traffic evidence.

## Cache investigation and current protocol

Buffered pilot `cache-buffered-pilot-04` achieved 4/4 intended states and
buffered full `cache-buffered-full-01` achieved 20/20. The full-run mean IOR
rates were 182.9 MiB/s for HDD backend, 239.0 for HDD server RAM, 237.8 for
SSD backend, and 240.5 for SSD server RAM. The HDD server-RAM gain over its
backend state was about 31%.

Native pilots `cache-native-pilot-01` through `-03` had unverified client-RAM
cases. Manual tests with `tuneFileCacheBufSize=2097152` established that native
client caching can hold the 8-GiB BeeGFS file: an IOR read after a separate
`dd` warm-up process exited reached 9053 MiB/s, and other warm reads reached
11425–11863 MiB/s. The benchmark uses 1-MiB POSIX transfers; the original
524288-byte threshold is below that size, while 2097152 permits the native
page-cache path. The runner now checks the effective threshold on the live
mount before a native run.

Native pilot `cache-native-pilot-06` achieved 8/8 intended states after the
mapping probe was removed. Its four client-RAM IOR reads reached 9537–9616
MiB/s with client-received and server-sent traffic ratios below 0.00003 and
zero target-device read bytes. The other four cases' raw counters also match
their backend or server-RAM labels; the IOR summaries and commands match the
one-rank, 8-GiB, 1-MiB POSIX protocol. This pilot's result schema predates the
explicit verification marker and saved pre-read idle counter sample, so the
recorded quiet-window flag cannot be independently recomputed. The live cache
threshold during the pilot was not recorded in its artifacts.

Native full `cache-native-full-01` achieved 30/30 intended states, five reads
per medium/state. Mean IOR rates in MiB/s were HDD backend 163.4, HDD server
RAM 235.9, HDD client RAM 9586.3, SSD backend 244.4, SSD server RAM 243.9,
and SSD client RAM 9550.0. Its ten client-RAM reads ranged from 9515.3 to
9626.5 MiB/s; client-received traffic was at most 0.0000312 of the 8-GiB
logical read, server-sent traffic at most 0.0000017, and target-device reads
were zero. The visualizer revalidated all 30 raw IOR summaries, commands,
counters and state labels, and produced throughput and traffic-evidence plots.
The full run also used the transitional format without a saved idle-window
sample, live configuration snapshot, pool-membership listing or raw BeeGFS
file-entry layout. Target labels relied on runner checks but cannot be
independently reconstructed from the preserved files. Shortly after the run, before
restoration, the live mount showed `native` with `tuneFileCacheBufSize=2097152`;
that observation does not prove the exact setting throughout the run. After
the experiment, the live mount was verified as `buffered` with `524288`.
The current runner records the idle sample and rejects a native mount below
2097152 bytes. Both current cache runners now look up each target's live
storage pool, request one target within that pool on a run-owned directory,
verify the exact target ID and save native placement evidence. This fixed a
write-pilot preparation failure: the inherited Default pool excluded HDD
target 101; no write workload ran in that attempt.

The prior mapping-based `residency()` probe was the reproducible runner failure.
The writable version coincided with system `Cached` falling from 9,007,872 to
624,860 kB. After it was changed to read-only `libc.mmap` plus `mincore`, the
actual imported runner returned 0.0 and `Cached` fell from 8,978,276 to
596,132 kB. The exact BeeGFS/kernel cause is unknown. The current runner does
not map the file between warm-up and IOR. It classifies paths from independent
client-received, server-sent and backing-device-read counters, and labels a
case `unverified` when the observed ratios or pre-read quiet check fail.
These host-wide counters do not quantify resident pages; older probe values
remain diagnostics only. Earlier roughly 240-MiB/s native readings
do not prove a staging-buffer or network ceiling; same-path iperf3 reached
280.5 MiB/s and the native pilot rates varied.

## Current operation paths

BeeGFS namespace and layout lookup uses the metadata service. Bulk file data moves
between clients and selected storage servers. Reads can be served from client
memory, server memory, or backend storage. The existing evidence does not identify
which cache level served each historical read.

The observed `anjuna3` client uses BeeGFS `buffered` cache mode. Its configuration
enables RDMA with TCP fallback, but it had no RDMA link at the documented topology
snapshot; its selected BeeGFS routes were TCP. See
[CLUSTER_TOPOLOGY.md](CLUSTER_TOPOLOGY.md).

Local FIO uses files on mounted target filesystems. With `direct=1`, file data
bypasses the normal page cache, but the filesystem, block mapping, RAID/LVM,
controller, and device firmware remain in the path. This represents the backend
filesystem path used by BeeGFS targets.

Raw-device FIO names a block device or logical volume directly and removes
filesystem behavior. Raw write measurements are destructive. The repository has
no verified disposable raw-device mapping and no recorded raw-device benchmark.

## Current workload and trace limitations

`run_workloads.py` invokes the Python IOR wrapper for every profile. The wrapper
supports contiguous/sequential and random access. It rejects `strided` and
`nd_strided`; the runner does not delegate those profiles to the standalone C
implementation. The current profile file contains only two contiguous read-only
profiles.

The Darshan parser writes aggregate rows only. Per-file and per-rank records are
not retained, so sharing classes such as single-shared-file, file-per-process, and
partial-shared cannot be derived from its current CSV output. The parser also does
not extract HDF5/PnetCDF dimensional counters.

Historical trace studies cover a limited Polaris corpus. Their derived
contiguity and frequency distributions are corpus-specific. No profile thresholds
or HDD/SSD placement classifier have been implemented from those studies.

## Historical interpretation

Historical BeeGFS pool measurements around 280-282 MiB/s are close to a preserved
2.35-Gbit/s TCP measurement of 280.14 MiB/s. This is evidence of a possible
network-path ceiling, not proof of causation. Those runs lack current transport
verification and complete run metadata.

The preserved local FIO datasets contain ambiguous size labels and incomplete
environment provenance. Their raw measurements remain available, but unsupported
labels are not treated as verified facts.

## Sources

- [BeeGFS 7.4.4 benchmarking](https://doc.beegfs.io/7.4.4/advanced_topics/benchmark.html)
- [Target management and capacity classes](https://doc.beegfs.io/7.4.4/advanced_topics/target_management.html)
- [Storage pools](https://doc.beegfs.io/7.4.4/advanced_topics/storage_pools.html)
- [Client caching](https://doc.beegfs.io/7.4.4/advanced_topics/client_caching.html)
- [Current client caching description](https://doc.beegfs.io/latest/advanced_topics/client_caching.html)
- [Qian et al., FAST 2024: buffered and direct I/O in distributed file systems](https://www.usenix.org/system/files/fast24-qian.pdf)
