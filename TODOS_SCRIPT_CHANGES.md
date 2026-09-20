# Deferred script path changes

These changes are **not implemented**. This checklist covers mechanical repairs
needed after file organization; it does not authorize changes to benchmark logic,
pool membership, workload parameters, cache handling, or measurement semantics.
Use [the migration map](docs/MIGRATION.md) for original locations.

## Microbenchmark tools

- [ ] `scripts/microbenchmarks/fio/matrix_benchmark.py`: resolve the existing
  `fio_config.json` independently of the caller's working directory; route new
  outputs beneath `results/microbenchmarks/runs/<run-id>/`. Review both normal
  `--results-dir` handling and OST-only hard-coded `ost_results`. Preserve FIO
  settings and CLI behavior except the explicitly agreed path repair.
- [ ] `scripts/microbenchmarks/fio/analyze_matrix.py`: replace adjacent
  `results`/`ost_results` auto-discovery with the relocated legacy datasets or an
  explicit input path. Do not change calculations.
- [ ] `scripts/microbenchmarks/fio/visualize_matrix.py`: fix repository-root
  traversal after the extra directory level, input auto-discovery and fixed
  `results/ost_plots` output. Require a destination separate from legacy evidence.
  The previously discussed `--output-dir`/`--dataset-label` edits were not present
  in the recovered source; any reintroduction is a later script edit.
- [ ] `scripts/microbenchmarks/analysis/parse_ost_logs.py`: update absolute
  `DEFAULT_LOG` and script-relative `DEFAULT_OUTPUT`; historical inputs now live
  under `results/microbenchmarks/legacy/placement/{from-logs,from-scripts}/`.
  Direct new plots to a new run/analysis directory.
- [ ] `scripts/microbenchmarks/analysis/parse_du.py`: update embedded usage text
  to its new path. The parser accepts an explicit input; do not alter parsing.
- [ ] `scripts/microbenchmarks/analysis/generate_heatmap.py`: audit site-specific
  paths and output destinations after moving from `scripts/analysis/`.
- [ ] `scripts/microbenchmarks/placement/*.sh`: update callers/help references
  from `scripts/pooling_scripts/`. Hardware inventory/pool-ID corrections are a
  separate future task, not a path-only repair.

## Workload orchestration and analysis

- [ ] `scripts/workloads/run_pipeline.py`: repair `BASE_DIR`, `SCRIPTS_DIR`,
  workload/parser/analysis script lookup and all output paths. The old
  `scripts/analysis.py` lookup was already broken; actual analysis is
  `scripts/analysis/analysis.py`. Keep numeric pool IDs and execution logic
  unchanged in a path-only patch; they require a separate live-inventory review.
- [ ] `scripts/workloads/run_workloads.py`: repair script-relative profile,
  IOR-wrapper and C-binary lookups (they now share its directory), parser lookup
  (`scripts/analysis/parse_darshan.py`), output/log/checkpoint locations and resume
  paths. Historical checkpoints are evidence, not defaults for new runs.
- [ ] `scripts/analysis/parse_darshan.py`: audit relative input/output defaults
  and callers after relocation; preserve counter extraction and CSV schema.
- [ ] `scripts/analysis/analysis.py`: audit default outputs and embedded examples;
  historical input CSVs are now under `results/workloads/legacy/darshan/`.
  New derived outputs should not overwrite the historical figures/statistics.
- [ ] Update embedded script docstrings/help, shell invocation strings and any
  repository-root/working-directory assumptions using the migration map.
- [ ] Leave `archive/legacy-code/old_scripts/` historical; do not silently revive
  its hard-coded paths as supported entry points.

## Trace-analysis paths

- [ ] `scripts/trace_analysis/{contiguity,frequency}/parse_logs.py`: update
  embedded usage examples that refer to the old top-level folders or adjacent
  `output/`; review the existing external `darshan-parser` path. Preserve parser
  behavior and metrics. Raw corpus paths remain external, not repository data.
- [ ] `scripts/trace_analysis/{contiguity,frequency}/analyze_distribution.py`:
  audit embedded input/output examples after relocation. Historical inputs now
  live under `results/trace_analysis/legacy/{contiguity,frequency}/`; select new
  outputs beneath `results/trace_analysis/runs/<run-id>/`. Keep rate formulas,
  filters, thresholds and plots unchanged in any path-only repair.
- [ ] These tools accept explicit input/output paths; verify parent-directory
  creation and working-directory assumptions before relying on the new examples.

## Verification for the later path-only patch

- [ ] Review diff to exclude algorithm, workload, pool and measurement changes.
- [ ] Check argument/config resolution from repository root and another directory.
- [ ] Check generated commands with mocked external commands, without running
  FIO, IOR, pool changes, cache dropping or cluster workloads.
- [ ] Check historical input selection and isolated output destinations.
- [ ] Check resume/log/parser handoffs with small fixtures where needed.

New IOR/mdtest/network/DLIO implementations and scientific fixes belong to a
separate future implementation phase described in the experiment specifications
and [implementation backlog](docs/IMPLEMENTATION_BACKLOG.md).
