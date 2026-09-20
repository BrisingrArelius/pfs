# Results index

Historical artifacts are preserved byte-for-byte. The new experiment design does
not retroactively establish missing transport, capacity, placement or cache metadata.

- [Microbenchmark evidence](microbenchmarks/legacy/README.md): local FIO,
  BeeGFS-pool FIO, network snapshots and placement/capacity evidence.
- [Workload evidence](workloads/legacy/README.md): Darshan CSVs/figures and
  historical execution records.
- [Trace-characterization evidence](trace_analysis/legacy/README.md): contiguity
  and frequency distributions from external Polaris Darshan logs.

Future outputs belong under `microbenchmarks/runs/<run-id>/` or
`workloads/runs/<run-id>/` (or `trace_analysis/runs/<run-id>/` for trace research),
with exact commands/configuration, raw output, derived
metrics/plots and provenance. Existing scripts still have their original output
defaults; see [deferred path repairs](../TODOS_SCRIPT_CHANGES.md).
