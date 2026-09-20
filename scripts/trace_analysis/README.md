# Trace-analysis tools

These tools characterize existing Darshan traces to inform workload-profile
thresholds. They support the same research project, but do not benchmark BeeGFS.

- [Contiguity](contiguity/README.md): per-file consecutive-operation ratios,
  distributions and minimum-operation filtering.
- [Frequency](frequency/README.md): per-file I/O rates and empirical
  seldom/frequent split comparisons.

Python sources are unchanged from the former `CONTIG_TESTING_CLAUDE/` and
`FREQ_TESTING_CLAUDE/` directories. Each retains its own parser and analyzer.
External Darshan logs and a suitable `darshan-parser` are required to rerun them.

Preserved outputs are indexed in
[results/trace_analysis/legacy](../../results/trace_analysis/legacy/README.md).
Use explicit paths and new output directories when rerunning; deferred embedded
path/help cleanup is recorded in [the script TODOs](../../TODOS_SCRIPT_CHANGES.md).

Research context: [literature review](../../docs/research/LitReview_task1.md) and
[profile setup](../../docs/research/Task%201%20-%20Profiles%20Setup.md).
