# Microbenchmarks

All existing microbenchmark tooling is grouped here. This refactor moves files
without modifying executable logic, configuration values, or embedded paths.
See [deferred path fixes](../../TODOS_SCRIPT_CHANGES.md) before execution.

| Location | Existing contents | Status |
|---|---|---|
| `fio/` | Matrix runner, JSON config, analyzer, visualizer | Legacy FIO; path fixes deferred |
| `placement/` | Configure/reset pool shell scripts | Historical hard-coded inventory; not a verified live setup |
| `analysis/parse_ost_logs.py` | OST usage log heatmaps | Input/output paths deferred |
| `analysis/parse_du.py` | Capacity text parser | Explicit input path; historical usage text |
| `analysis/generate_heatmap.py` | Placement heatmap utility | Preserved source; inspect site paths before use |

The IOR synthetic-workload adapter remains in `../workloads/`; it is an
application-workload component, not the planned multi-process placement runner.

Future tooling for all six domains belongs beneath this directory: local storage,
network transport, BeeGFS communication (IOR + NetBench), end-to-end placement
(normal IOR), cache effects and metadata operations (mdtest). Dedicated runners
for the latter five domains are not implemented by this reorganization.
Application validation belongs in `scripts/workloads/`.

See [methodology](../../docs/methodology.md) and the
[implementation backlog](../../docs/IMPLEMENTATION_BACKLOG.md) for suite-wide
resume, instrumentation, cache preparation and allocation-planning requirements.

Historical outputs live in [results/microbenchmarks/legacy](../../results/microbenchmarks/legacy/README.md).
Future run outputs should use `results/microbenchmarks/runs/<run-id>/` with
configuration, raw output, derived metrics, plots and provenance. That output
contract requires both mechanical path repairs and new runner functionality.
