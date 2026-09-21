# BeeGFS Storage Pool Experiments

Existing benchmark and workload tools, historical measurements, and the experimental
design for evaluating BeeGFS target placement.

## Status

Existing FIO, workload, Darshan-analysis, placement, and trace-analysis tools are
available under `scripts/`. New outputs use run-specific directories under
`results/`; historical evidence is kept under `results/*/legacy/`. The complete
six-domain experiment suite is not implemented yet; remaining work is listed in
the [implementation backlog](docs/IMPLEMENTATION_BACKLOG.md).

## Repository layout

```text
scripts/
  microbenchmarks/
    fio/                 existing FIO runner, configuration, analysis and visualizer
    placement/           existing pool-management helpers
    analysis/            OST-log, capacity and placement utilities
  workloads/             synthetic workloads, profiles and orchestration
  analysis/              Darshan parsing and workload analysis
  trace_analysis/        contiguity and frequency characterization of Darshan traces
results/
  microbenchmarks/legacy/ fio-local, fio-beegfs, network and placement evidence
  workloads/legacy/      Darshan outputs and execution records
  trace_analysis/legacy/ preserved contiguity/frequency CSVs and plots
docs/                    methodology, implementation backlog, references and history
  research/              historical context, profile notes and literature review
  references/            existing reference PDFs
archive/                 old runner versions retained for historical reference
```

The tools under `scripts/trace_analysis/` characterize external Polaris Darshan
logs for workload-profile research; they are not BeeGFS microbenchmark runners.

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

These are design requirements, not all implemented features. See
[methodology](docs/methodology.md).

## Documentation

- [Microbenchmark inventory](scripts/microbenchmarks/README.md)
- [Historical results and provenance](results/README.md)
- [Future implementation backlog](docs/IMPLEMENTATION_BACKLOG.md)
- [Workload documentation](scripts/workloads/README.md)
- [Darshan parser](scripts/analysis/parse_darshan_README.md)
- [Workload analysis](scripts/analysis/analysis_README.md)
- [Trace-analysis tools](scripts/trace_analysis/README.md)
- [Research notes](docs/research/CONTEXT.md)

## History

- [Migration and recovery record](docs/MIGRATION.md)
- [Completed path repairs](docs/COMPLETED_PATH_REPAIRS.md)
- [Changelog](docs/CHANGELOG.md)
- [Historical pipeline guide](docs/legacy-pipeline.md)
