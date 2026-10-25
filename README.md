# BeeGFS Storage Pool Experiments

Existing benchmark and workload tools, historical measurements, and current
implementation status for BeeGFS target-placement experiments.

## Status

Existing FIO, network, workload, Darshan-analysis, and trace-analysis tools are
available under `scripts/`. New outputs use run-specific directories under
`results/`; historical evidence is kept under `results/*/legacy/`. FIO,
iperf3, and cache reads have completed runs; cache writes, communication, and
metadata have no completed cluster runs.

## Repository layout

```text
scripts/
  microbenchmarks/
    fio/                 local-target FIO runner and raw-to-plot visualizer
    network/             iperf3 runner and raw-to-plot visualizer
    communication/       offline NetBench IOR plan and raw-to-plot visualizer
    cache/               separate read/write runners, shared prep and read visualizer
    metadata/            one native mdtest pilot/full runner
    placement/           historical pool-management helpers only
    analysis/            OST-log, capacity and placement utilities
  workloads/             synthetic workloads, profiles and orchestration
    analysis/            Darshan parsing and workload-result analysis
  trace_characterization/ contiguity and frequency studies of Darshan traces
results/
  microbenchmarks/legacy/ fio-local, fio-beegfs, network and placement evidence
  workloads/legacy/      Darshan outputs and execution records
  trace_analysis/legacy/ preserved contiguity/frequency CSVs and plots
docs/                    methodology, references and history
  research/              historical context, profile notes and literature review
  references/            existing reference PDFs
archive/                 old runner versions retained for historical reference
```

The tools under `scripts/trace_characterization/` characterize external Polaris Darshan
logs for workload-profile research; they are not BeeGFS microbenchmark runners.

## Current coverage

- The separate DLIO whole-workload D/S/H comparison is not implemented.
- BeeGFS communication has no live runner. Metadata has an unpiloted raw-capture
  runner; cache has completed buffered/native read runs and an unpiloted write runner.
- Durable suite-wide progress and resume support is missing.
- Controlled capacity conditions and application-validation coverage are missing.

Current behavior and limitations are described in [methodology](docs/methodology.md).

## Documentation

The changelog and research notes preserve historical observations.

- [Microbenchmark inventory](scripts/microbenchmarks/README.md)
- [Historical results and provenance](results/README.md)
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
