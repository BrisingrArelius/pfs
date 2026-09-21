# Completed relocation path fixes

These changes are implemented. They make the moved scripts find their files and
save new results in the correct directories. They do not change
benchmark behavior, workload settings, pool membership, cache behavior, parsing
calculations, or measurement meaning.

See [the migration map](MIGRATION.md) for the old and new file locations.

## Microbenchmark tools

- [x] Make `matrix_benchmark.py` find `fio_config.json` next to the script, no
  matter which directory the command is run from.
- [x] Save new FIO results under `results/microbenchmarks/runs/<run-id>/` instead
  of the old `results/` or `ost_results/` directories. Keep all FIO settings and
  command-line behavior unchanged.
- [x] Make `analyze_matrix.py` use an explicit input file or the correct new
  location. Do not change any calculations.
- [x] Fix `visualize_matrix.py` so it finds the repository correctly and writes
  new plots outside `results/microbenchmarks/legacy/`. Do not overwrite historical
  plots. New CLI features such as `--dataset-label` are outside this task.
- [x] Replace the old absolute/default paths in `parse_ost_logs.py`. Historical
  logs remain under `results/microbenchmarks/legacy/placement/`; new plots go in
  the current run's analysis directory.
- [x] Update the usage text in `parse_du.py` to show its new location. Do not
  change how it parses data.
- [x] Check `generate_heatmap.py` for paths that became wrong when it moved.
- [x] Update old `scripts/pooling_scripts/` references in the placement scripts
  and their callers. Do not change target IDs, pool IDs, or pool operations.

## Workload orchestration and analysis

- [x] Fix `run_pipeline.py` so it finds `run_workloads.py`, the Darshan parser,
  and `scripts/analysis/analysis.py` in their new locations. Fix where it writes
  output. Do not change pool IDs or execution behavior.
- [x] Fix `run_workloads.py` so it finds `profiles.json`, the IOR wrapper, the C
  workload, and the Darshan parser in their new locations.
- [x] Give new workload runs new output, log, error, and checkpoint locations.
  Never use historical checkpoints as the starting state of a new run.
- [x] Check the default paths and callers for `parse_darshan.py`. Keep its counter
  extraction and CSV format unchanged.
- [x] Check the default paths and examples for `analysis.py`. New analysis must
  not overwrite historical CSVs, figures, or statistics.
- [x] Update old paths in script help text, docstrings, and generated commands.
- [x] Do not make anything under `archive/legacy-code/` runnable again.

## Trace-analysis paths

- [x] Update old folder names and output paths in both `parse_logs.py` scripts.
  Check the configured external `darshan-parser` path. Do not change parsing or
  metrics; the raw Polaris log collection remains outside this repository.
- [x] Update paths and examples in both `analyze_distribution.py` scripts.
  Historical outputs remain under `results/trace_analysis/legacy/`; new outputs
  go under `results/trace_analysis/runs/<run-id>/`.
- [x] Keep the existing formulas, filters, thresholds, and plots unchanged.
- [x] Verify that these tools create output directories and work when launched
  from a directory other than the repository root.

## Verification performed

- [x] Confirm the diff contains path/help/output-location changes only.
- [x] Run path-resolution tests from the repository root and another directory.
- [x] Test generated commands with mocks. Do not run FIO, IOR, pool changes,
  cache dropping, or cluster workloads during this repair.
- [x] Confirm historical files are read-only inputs and new output goes elsewhere.
- [x] Test checkpoint, log, and parser handoffs with small fake files.

This file does not cover new IOR, mdtest, network, or DLIO runners, scientific
changes, improved resume design, new instrumentation, or live pool inventory.
Those tasks are in the [implementation backlog](IMPLEMENTATION_BACKLOG.md).
