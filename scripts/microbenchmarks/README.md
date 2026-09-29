# Microbenchmarks

This directory contains the five layer-effects microbenchmark domains and
historical supporting utilities.
Retrieved local results and generated plots belong under
`results/microbenchmarks/runs/`; FIO and iperf3 retain their documented
on-cluster raw-result paths. Verify live inventory and environment settings
before execution.

| Location | Existing contents | Status |
|---|---|---|
| `fio/` | Local-target FIO runner and raw-to-plot visualizer | Protocol-5 run completed |
| `network/` | iperf3 runner and raw-to-plot visualizer | Full raw run available |
| `communication/` | NetBench IOR plan/command and raw-to-plot visualizer | No live runner or cluster results |
| `cache/` | Separate read and write runners with shared preparation; read raw-to-plot visualizer | Buffered/native read full runs completed; write runner unpiloted |
| `metadata/` | One mdtest runner with native raw-result capture | Runnable four-case pilot; 90-case full plan awaits its pilot-derived item count |
| `placement/` | Historical pool-management scripts only | Retained for provenance; not live authority or a microbenchmark domain |
| `analysis/parse_ost_logs.py` | OST usage log heatmaps | Legacy input default; new run output |
| `analysis/parse_du.py` | Capacity text parser | Explicit input path |
| `analysis/generate_heatmap.py` | Placement heatmap utility | Requires BeeGFS files and `beegfs-ctl` |

The IOR synthetic-workload adapter is in `../workloads/`; it is an
application-workload component, not a layer benchmark.

Current domain documentation:

| Domain | Document |
|---|---|
| Local storage | [`fio/DESIGN.md`](fio/DESIGN.md) |
| Network transport | [`network/DESIGN.md`](network/DESIGN.md) |
| BeeGFS communication | [`communication/DESIGN.md`](communication/DESIGN.md) |
| Cache effects | [`cache/README.md`](cache/README.md) |
| Metadata operations | [`metadata/README.md`](metadata/README.md) |

FIO and iperf3 runners capture native results, and their visualizers consume
those raw artifacts directly. Communication has no live runner or cluster result.
Metadata has a raw-capture runner but no cluster result or validated
visualizer yet. Cache has completed buffered and native full **read** runs on
`anjuna2`; write cases have not been piloted there.
Application validation, including the D/S/H whole-workload comparison, is
specified separately in [`DL_Experiment.md`](../../docs/specs/DL_Experiment.md).

See [methodology](../../docs/methodology.md) for current implementation coverage
and observed limitations.

Historical outputs live in [results/microbenchmarks/legacy](../../results/microbenchmarks/legacy/README.md).
The FIO and plotting tools that create files default beneath
`results/microbenchmarks/runs/<run-id>/`; `parse_du.py` prints to standard output.
A suite-wide artifact contract is missing.
