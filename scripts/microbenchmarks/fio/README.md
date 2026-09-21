# FIO Benchmark Matrix Suite

This is the available FIO matrix tool, not the complete six-domain experiment
runner. New results use run-specific output directories. Historical datasets are indexed in
[results](../../../results/microbenchmarks/legacy/README.md).

A benchmark framework that replaces static `.fio` jobs with a configurable matrix runner.
It runs FIO workloads across BeeGFS directories or locally mounted target
filesystems, captures repeated measurements, and produces aggregated JSON results.

## Configuration

Edit `fio_config.json` to control the benchmark matrix:

- `runs_per_test`: number of repetitions per workload configuration
- `file_sizes`: total file sizes to test
- `modes`: enable or disable workload types
- `block_size_seq`: block size for sequential workloads
- `block_size_rand`: block size for random workloads
- `io_depth`: FIO I/O depth
- `num_jobs`: number of FIO jobs per test

Example:

```json
{
    "runs_per_test": 5,
    "file_sizes": ["1g", "10g"],
    "modes": {
        "seq_read": true,
        "seq_write": true,
        "rand_read": true,
        "rand_write": true,
        "seq_rw": true,
        "rand_rw": true
    },
    "block_size_seq": "1m",
    "block_size_rand": "4k",
    "io_depth": 32,
    "num_jobs": 4
}
```

## Running the benchmark

Invocation from the repository root:

```bash
python3 scripts/microbenchmarks/fio/matrix_benchmark.py --beegfs
```

### BeeGFS mode
- `--beegfs`: run against BeeGFS mountpoints
- `--pool hdd|ssd|all`: choose target pools
- `--custom-dir`: specify a custom mount directory

Examples:

```bash
python3 scripts/microbenchmarks/fio/matrix_benchmark.py --beegfs --pool all
python3 scripts/microbenchmarks/fio/matrix_benchmark.py --beegfs --pool hdd
python3 scripts/microbenchmarks/fio/matrix_benchmark.py --beegfs --pool ssd
python3 scripts/microbenchmarks/fio/matrix_benchmark.py --beegfs --pool custom --custom-dir /existing/beegfs/directory
```

### Local target-filesystem mode
- `--ost`: run against locally mounted target paths; this is not raw block-device I/O
- `--pool hdd|ssd|all`: choose target directories

Example:

```bash
python3 scripts/microbenchmarks/fio/matrix_benchmark.py --ost --pool hdd
```

### Other useful options
- `--results-dir`: output JSON directory; defaults to a new directory under `results/microbenchmarks/runs/`
- `--no-drop-cache`: skip dropping page cache between runs

## Output

The benchmark writes aggregated JSON results to the configured results directory, for example:

The default is `results/microbenchmarks/runs/fio-<timestamp>/matrix_results_<timestamp>.json`.
An explicit `--results-dir` overrides it.

Each entry includes FIO bandwidth, IOPS, latency, and optional BeeGFS OST hit information.

## Current local-storage evidence

The main preserved April 4 dataset contains 840 rows:

- seven runner labels: `HDD_OST1..4` and `SSD_OST1..3`
- one and ten files
- `1g` and `10g` FIO `size` values
- sequential read/write/mixed and 4-KiB random read/write/mixed modes
- five repetitions per combination

The labels were generated from local path suffixes `/mnt/hdd1..4` and
`/mnt/nvme1..3`. The result rows do not record those paths, hostnames, BeeGFS
target IDs, filesystems, block devices, device models, or controllers. Therefore
the dataset does not establish which OSS or physical device produced a label.

The current runner retains the same generic local labels and default mount paths.
It also retains stale BeeGFS defaults under `/mnt/beegfs/advay`, which are absent
from the observed 2026-09-21 namespace. No verified current target-to-mount mapping
is configured in the runner.

## Analyze benchmark results

Run the analyzer to summarize the JSON output:

```bash
python3 analyze_matrix.py
```

Or target a specific run file:

```bash
python3 analyze_matrix.py ../../../results/microbenchmarks/legacy/fio-local/20260404/matrix_results_20260404_220033.json
```

The analyzer prints a table of average metrics per pool, mode, file count, and size.

No-argument analysis/visualization selects the newest result under
`results/microbenchmarks/runs/`; visualization writes beside it under `plots/`
unless `--output-dir` is provided.
Historical size labels represent the configured FIO `size` with `nrfiles`; do not
interpret them as per-file sizes without checking geometry. This script
does not implement the primary IOR experiment in the updated specification.
