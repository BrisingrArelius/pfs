# BeeGFS Storage Pool Experiments

Existing benchmark and workload tools, historical measurements, and current
implementation status for BeeGFS target-placement experiments.

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
    analysis/            Darshan parsing and workload-result analysis
  trace_characterization/ contiguity and frequency studies of Darshan traces
results/
  microbenchmarks/legacy/ fio-local, fio-beegfs, network and placement evidence
  workloads/legacy/      Darshan outputs and execution records
  trace_analysis/legacy/ preserved contiguity/frequency CSVs and plots
docs/                    methodology, implementation backlog, references and history
  research/              historical context, profile notes and literature review
  references/            existing reference PDFs
archive/                 old runner versions retained for historical reference
```

The tools under `scripts/trace_characterization/` characterize external Polaris Darshan
logs for workload-profile research; they are not BeeGFS microbenchmark runners.

## Missing functionality

- The complete D/S/H placement matrix is missing.
- Dedicated network, BeeGFS communication, cache-state, and metadata runners are
  missing.
- Durable suite-wide progress and resume support is missing.
- Controlled capacity conditions and application-validation coverage are missing.

The detailed open work is in the
[implementation backlog](docs/IMPLEMENTATION_BACKLOG.md). Current behavior and
limitations are described in [methodology](docs/methodology.md).

## Documentation

Current-state documentation describes only implemented or observed behavior.
Missing behavior is stated as missing. The implementation backlog is kept in
`docs/IMPLEMENTATION_BACKLOG.md`; changelog and research notes are historical
records rather than active specifications.

- [Microbenchmark inventory](scripts/microbenchmarks/README.md)
- [Historical results and provenance](results/README.md)
- [Implementation backlog](docs/IMPLEMENTATION_BACKLOG.md)
- [Observed cluster topology](docs/CLUSTER_TOPOLOGY.md)
- [Workload documentation](scripts/workloads/README.md)
- [Darshan parsing and workload analysis](scripts/workloads/analysis/README.md)
- [Trace-characterization tools](scripts/trace_characterization/README.md)
- [Research notes](docs/research/CONTEXT.md)

## History

- [Migration and recovery record](docs/MIGRATION.md)
- [Completed path repairs](docs/COMPLETED_PATH_REPAIRS.md)
- [Changelog](docs/CHANGELOG.md)
- [Historical pipeline guide](docs/legacy-pipeline.md)
