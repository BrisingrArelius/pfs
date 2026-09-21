# Microbenchmarks

This directory contains the available microbenchmark and placement utilities.
New output defaults to run-specific directories under
`results/microbenchmarks/runs/`. Verify live inventory and environment settings
before execution.

| Location | Existing contents | Status |
|---|---|---|
| `fio/` | Matrix runner, JSON config, analyzer, visualizer | Available local/BeeGFS FIO tooling |
| `placement/` | Configure/reset pool shell scripts | Historical hard-coded inventory; not a verified live setup |
| `analysis/parse_ost_logs.py` | OST usage log heatmaps | Legacy input default; new run output |
| `analysis/parse_du.py` | Capacity text parser | Explicit input path |
| `analysis/generate_heatmap.py` | Placement heatmap utility | Requires BeeGFS files and `beegfs-ctl` |

The IOR synthetic-workload adapter is in `../workloads/`; it is an
application-workload component, not a multi-process placement runner.

Dedicated runners for network transport, BeeGFS communication, end-to-end
placement, cache effects, and metadata operations are missing.
Application validation belongs in `scripts/workloads/`.

See [methodology](../../docs/methodology.md) and the
[implementation backlog](../../docs/IMPLEMENTATION_BACKLOG.md) for suite-wide
resume, instrumentation, cache preparation and allocation-planning requirements.

Historical outputs live in [results/microbenchmarks/legacy](../../results/microbenchmarks/legacy/README.md).
The FIO and plotting tools that create files default beneath
`results/microbenchmarks/runs/<run-id>/`; `parse_du.py` prints to standard output.
A suite-wide artifact contract is missing.
