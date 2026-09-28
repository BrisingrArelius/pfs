# Experiment methodology status

This document describes the methodology currently represented by repository code
and evidence. Outstanding experiment work is summarized in the
[implementation backlog](IMPLEMENTATION_BACKLOG.md).

## Implemented tools

| Area | Present implementation |
|---|---|
| Local storage | Per-target FIO runner with native JSON, deadlines and measurement-level resume; protocol-5 full run completed |
| Network transport | Six-host iperf3 runner with native endpoint JSON, synchronized concurrent epochs, telemetry, and resume; 190-unit/320-path full raw run available |
| Microbenchmark figures | FIO and iperf3 visualizers plot their completed native runs directly; four other domains have raw-to-plot code tested only with synthetic evidence |
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

All six domains have implementation designs under `scripts/microbenchmarks/`.
BeeGFS communication, end-to-end placement, cache effects, and metadata operations
have fixed offline plans, command builders, and raw-evidence visualizers, but
**not** live benchmark runners or validated cluster pilots.
Reviewed inventories, installed-tool output schemas, privileged-state restoration,
owned cleanup, and remote recovery remain gates. The current workload
pipeline is application-level tooling rather than the missing placement runner.

The complete D/S/H, target-chooser, stripe-count, stripe-size, capacity,
concurrency, and repetition matrix is not implemented. Numerical capacity bands
are not defined. Suite-wide durable progress, allocation recovery, complete
provenance capture, mutable allocation deadlines, clean time-budget shutdown, and
cache-state verification are also missing.

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
