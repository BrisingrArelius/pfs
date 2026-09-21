# Trace-characterization tools

These tools characterize existing Darshan traces to inform workload-profile
thresholds. They support the same research project, but do not benchmark BeeGFS.

- [Contiguity](contiguity/README.md): per-file consecutive-operation ratios,
  distributions and minimum-operation filtering.
- [Frequency](frequency/README.md): per-file I/O rates and empirical
  seldom/frequent split comparisons.

Each area has its own parser and analyzer. Their formulas, filters, thresholds,
and plots are documented in the area-specific guides.
External Darshan logs and a suitable `darshan-parser` are required to rerun them.

Preserved outputs are indexed in
[results/trace_analysis/legacy](../../results/trace_analysis/legacy/README.md).
Explicit paths remain supported; omitted output paths use timestamped directories
under `results/trace_analysis/runs/`.

Research context: [literature review](../../docs/research/LitReview_task1.md) and
[profile setup](../../docs/research/Task%201%20-%20Profiles%20Setup.md).
