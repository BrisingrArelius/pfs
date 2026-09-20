# BeeGFS Storage Pool Experiments

Existing benchmark and workload tools, historical measurements, and the experimental
design for evaluating BeeGFS target placement.

## Refactor status

This is a file-organization and documentation refactor. Existing scripts and
configuration contents are preserved. Their internal paths have **not** been
updated; follow [TODOS_SCRIPT_CHANGES.md](TODOS_SCRIPT_CHANGES.md) before treating
the relocated tools as a working pipeline. No new benchmark runners were implemented.

## Repository layout

```text
scripts/
  microbenchmarks/
    fio/                 existing FIO runner, configuration, analysis and visualizer
    placement/           existing pool-management helpers
    analysis/            OST-log, capacity and placement utilities
  workloads/             synthetic workloads, profiles and legacy orchestration
  analysis/              Darshan parsing and workload analysis
  trace_analysis/        contiguity and frequency characterization of Darshan traces
results/
  microbenchmarks/legacy/ fio-local, fio-beegfs, network and placement evidence
  workloads/legacy/      Darshan outputs and execution records
  trace_analysis/legacy/ preserved contiguity/frequency CSVs and plots
docs/                    methodology, migration map, changelog and historical guide
  research/              historical context, profile notes and literature review
  references/            existing reference PDFs
archive/                 old runner versions retained for historical reference
```

The former `CONTIG_TESTING_CLAUDE/` and `FREQ_TESTING_CLAUDE/` projects now live
under `scripts/trace_analysis/contiguity/` and `scripts/trace_analysis/frequency/`.
They support workload-profile research using external Polaris Darshan logs;
they are not BeeGFS microbenchmark runners.

## Experimental design

- **D:** all candidate targets in Default (the experimental “pooling off” baseline).
- **S:** SSD-only target eligibility.
- **H:** HDD-only target eligibility.
- Target selection: **randomized** and **roundrobin**.
- Capacity: proposed high/low free space on the same devices; interpretation and
  site-specific bands remain pending confirmation.
- Six benchmark domains: local storage, network transport, BeeGFS communication
  (IOR + NetBench), end-to-end placement (normal IOR), cache effects and metadata.
- The full placement matrix is retained; each domain requires durable progress
  across allocations and its applicable preparation and instrumentation.
- DLIO and additional application workloads provide separately specified validation.

These are specification requirements, not features already implemented by the
preserved scripts. See [methodology](docs/methodology.md).

## Documentation

- [Microbenchmark inventory](scripts/microbenchmarks/README.md)
- [Historical results and provenance](results/README.md)
- [Migration map and recovery record](docs/MIGRATION.md)
- [Deferred script path changes](TODOS_SCRIPT_CHANGES.md)
- [Future implementation backlog](docs/IMPLEMENTATION_BACKLOG.md)
- [Workload documentation](scripts/workloads/README.md)
- [Darshan parser](scripts/analysis/parse_darshan_README.md)
- [Workload analysis](scripts/analysis/analysis_README.md)
- [Trace-analysis tools](scripts/trace_analysis/README.md)
- [Research notes](docs/research/CONTEXT.md)
- [Changelog](docs/CHANGELOG.md)
- [Historical pipeline guide](docs/legacy-pipeline.md)
