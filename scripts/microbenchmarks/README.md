# Microbenchmarks

This directory contains the available microbenchmark and placement utilities.
Retrieved local results and generated plots belong under
`results/microbenchmarks/runs/`; FIO and iperf3 retain their documented
on-cluster raw-result paths. Verify live inventory and environment settings
before execution.

| Location | Existing contents | Status |
|---|---|---|
| `fio/` | Local-target FIO runner and raw-to-plot visualizer | Protocol-5 run completed |
| `network/` | iperf3 runner and raw-to-plot visualizer | Full raw run available |
| `communication/` | NetBench IOR plan/command and raw-to-plot visualizer | Cluster execution and pilot pending |
| `cache/` | Controlled cache plan/command and raw-to-plot visualizer | Cluster execution and pilot pending |
| `metadata/` | mdtest plan/command and raw-to-plot visualizer | Cluster execution and pilot pending |
| `placement/` | End-to-end IOR plan/command and raw-to-plot visualizer | Cluster execution and pilot pending; historical pool scripts are not live authority |
| `analysis/parse_ost_logs.py` | OST usage log heatmaps | Legacy input default; new run output |
| `analysis/parse_du.py` | Capacity text parser | Explicit input path |
| `analysis/generate_heatmap.py` | Placement heatmap utility | Requires BeeGFS files and `beegfs-ctl` |

The IOR synthetic-workload adapter is in `../workloads/`; it is an
application-workload component, not a multi-process placement runner.

Every storage microbenchmark domain now has a design document:

| Domain | Design |
|---|---|
| Local storage | [`fio/DESIGN.md`](fio/DESIGN.md) |
| Network transport | [`network/DESIGN.md`](network/DESIGN.md) |
| BeeGFS communication | [`communication/DESIGN.md`](communication/DESIGN.md) |
| End-to-end placement | [`placement/DESIGN.md`](placement/DESIGN.md) |
| Cache effects | [`cache/DESIGN.md`](cache/DESIGN.md) |
| Metadata operations | [`metadata/DESIGN.md`](metadata/DESIGN.md) |

FIO and iperf3 runners capture native results. Every domain has a visualizer that
consumes raw manifests and native files directly; no intermediate CSV parser is
needed. The other four domains still require reviewed cluster inventories,
privileged state restoration, and timed pilots before running benchmarks.
Their plot tests use synthetic native fixtures; installed IOR/mdtest and MPI
output schemas must be confirmed by those pilots before interpreting real figures.
Application validation belongs in `scripts/workloads/`.

See [methodology](../../docs/methodology.md) and the
[implementation backlog](../../docs/IMPLEMENTATION_BACKLOG.md) for suite-wide
resume, instrumentation, cache preparation and allocation-planning requirements.

Historical outputs live in [results/microbenchmarks/legacy](../../results/microbenchmarks/legacy/README.md).
The FIO and plotting tools that create files default beneath
`results/microbenchmarks/runs/<run-id>/`; `parse_du.py` prints to standard output.
A suite-wide artifact contract is missing.
