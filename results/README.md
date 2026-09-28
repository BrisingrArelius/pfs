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

Imported microbenchmark raw runs and their plots belong under
`microbenchmarks/runs/<run-id>/`. The current FIO and iperf3 full runs contain
`plots/plot_manifest.json`, which lists the PNGs in their current complete plot
generation. The existing cluster runners retain their documented host-side raw
result paths before retrieval. Workload and trace outputs use
`workloads/runs/<run-id>/` and `trace_analysis/runs/<run-id>/`, respectively.
Cache, communication, placement, and metadata have no live raw runs yet.
