# FIO Benchmark Matrix Suite

This is the available FIO matrix tool, not the complete six-domain experiment
runner. New results use run-specific output directories. Historical datasets are indexed in
[results](../../../results/microbenchmarks/legacy/README.md).

A benchmark framework that replaces static `.fio` jobs with a configurable matrix runner.
It runs FIO workloads across BeeGFS pools or raw OST targets, captures repeated measurements, and produces aggregated JSON results.

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

Invocation from any working directory:

```bash
python3 matrix_benchmark.py --beegfs
```

### BeeGFS mode
- `--beegfs`: run against BeeGFS mountpoints
- `--pool hdd|ssd|all`: choose target pools
- `--custom-dir`: specify a custom mount directory

Examples:

```bash
python3 matrix_benchmark.py --beegfs --pool all
python3 matrix_benchmark.py --beegfs --pool hdd
python3 matrix_benchmark.py --beegfs --pool ssd
python3 matrix_benchmark.py --beegfs --pool nvme-fast --custom-dir /mnt/beegfs/advay/nvme-fast
```

### Raw OST mode
- `--ost`: run directly against OST-mounted paths
- `--pool hdd|ssd|all`: choose target OST directories

Example:

```bash
python3 matrix_benchmark.py --ost --pool hdd
```

### Other useful options
- `--results-dir`: output JSON directory; defaults to a new directory under `results/microbenchmarks/runs/`
- `--no-drop-cache`: skip dropping page cache between runs

## Output

The benchmark writes aggregated JSON results to the configured results directory, for example:

The default is `results/microbenchmarks/runs/fio-<timestamp>/matrix_results_<timestamp>.json`.
An explicit `--results-dir` overrides it.

Each entry includes FIO bandwidth, IOPS, latency, and optional BeeGFS OST hit information.

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
