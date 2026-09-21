# Results index

Historical results are preserved byte-for-byte. The experiment design does
not retroactively establish missing transport, capacity, placement or cache metadata.

- [Microbenchmark evidence](microbenchmarks/legacy/README.md): local FIO,
  BeeGFS-pool FIO, network snapshots and placement/capacity evidence.
- [Workload evidence](workloads/legacy/README.md): Darshan CSVs/figures and
  historical execution records.
- [Trace-characterization evidence](trace_analysis/legacy/README.md): contiguity
  and frequency distributions from external Polaris Darshan logs.
- [`cluster-inventory/20260921/`](cluster-inventory/20260921/): normalized
  read-only storage-node inventory output referenced by the cluster topology.

Current tools place new outputs under `microbenchmarks/runs/<run-id>/` or
`workloads/runs/<run-id>/` (or `trace_analysis/runs/<run-id>/` for trace characterization),
with exact commands/configuration, raw output, derived metrics/plots and
provenance. Tools default to these run-specific areas.
