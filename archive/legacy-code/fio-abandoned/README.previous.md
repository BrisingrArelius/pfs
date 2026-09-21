# Local-storage FIO Matrix (superseded documentation)

This tool measures the local XFS filesystems backing the verified storage
targets. It does not benchmark through the BeeGFS client path and does not
support arbitrary pools or mount paths. Historical datasets are indexed in
[results](../../../results/microbenchmarks/legacy/README.md).

`local_storage_fio.py` contains only FIO job generation, read preparation, raw
artifact capture, and data-file cleanup. `run_local_storage_fio.py` supplies the
host inventory, safety checks, deadlines, manifests, and target-level resume.

## Configuration

Edit `fio_config.json` to control the benchmark matrix:

- `runs_per_test`: number of repetitions per workload configuration
- `file_sizes`: I/O region size per FIO worker
- `workloads`: explicit FIO operation, block size, read preparation, and timing policy
- `io_depth`: FIO I/O depth
- `num_jobs`: number of FIO jobs per test
- `free_space_buffer`: free space required in addition to the active dataset

Example:

```json
{
    "runs_per_test": 5,
    "num_files": [1],
    "file_sizes": ["10g"],
    "workloads": [
        {"name": "seq_read", "rw": "read", "block_size": "1m", "prepare": true, "time_based": false},
        {"name": "seq_write", "rw": "write", "block_size": "1m", "prepare": false, "time_based": false},
        {"name": "rand_read", "rw": "randread", "block_size": "4k", "prepare": true, "prepare_block_size": "1m", "time_based": true, "runtime_seconds": 60, "ramp_time_seconds": 5},
        {"name": "rand_read_128k", "rw": "randread", "block_size": "128k", "prepare": true, "prepare_block_size": "1m", "time_based": true, "runtime_seconds": 60, "ramp_time_seconds": 5}
    ],
    "io_depth": 32,
    "num_jobs": 1,
    "free_space_buffer": "5g"
}
```

## Running the benchmark

Run this command on each of `colva1` through `colva4`, choosing a result
directory that can be reused for resume:

```bash
sudo python3 scripts/microbenchmarks/fio/run_local_storage_fio.py \
  --results-dir "results/microbenchmarks/runs/$(hostname -s)-local-fio" \
  --time-limit 5h
```

The runner selects that host's verified target IDs and mount paths from
`target_inventory.json`. It completes the full matrix on one target before
moving to the next target. The four hosts may run concurrently because each
process uses only its host-local filesystems; targets within one host are
intentionally not tested simultaneously so each result describes one physical
device rather than aggregate OSS throughput. A missing or unmounted inventory
target is marked failed in the run manifest.
For inventory targets, preflight also requires `findmnt` to report the expected
inventory device as the mount source and XFS as the filesystem type.

Each FIO invocation currently uses one worker, one file, `iodepth=32`, and a
10-GiB I/O region. FIO `size` is assigned per worker and distributed across that
worker's `nrfiles`; with the current single worker and file it is 10 GiB total.
Sequential runs stop when that amount of I/O is complete. Random-read runs use a
5-second ramp and 60 measured seconds because completing 10 GiB of 4-KiB random
I/O on an HDD could otherwise take hours. For every read-only
configuration, the runner first performs an unmeasured
sequential write using the same worker count, file count, size, and filename
layout. Those files remain in place for all read repetitions and are deleted
afterward, including when the repetition loop is interrupted.

The current matrix has one 10-GiB region per worker, four modes, and five
repetitions: 20 measured FIO runs per target. The modes are 1-MiB sequential
read, 1-MiB sequential write, 4-KiB random read, and 128-KiB random read. The
10-GiB region, five repetitions, 1-MiB sequential block size, and queue depth 32
are engineering baseline choices inherited from the existing suite, not
literature-derived thresholds. Both 4-KiB and 128-KiB random reads are explicitly
required by the current implementation backlog.

Mounted XFS filesystems are used because these disks are active BeeGFS target
filesystems and the measurement is intended to include the deployed filesystem
stack. Raw-device writes would bypass XFS and overwrite live target data and
metadata, so this runner never passes inventory `device` paths to FIO.

### Other useful options
- `--results-dir`: output JSON directory; defaults to a new directory under `results/microbenchmarks/runs/`
- `--time-limit 5h`: stop before a relative reservation budget expires
- `--deadline 2026-09-22T04:00:00+00:00`: use an absolute reservation deadline
- `--extend-deadline 2h`: atomically extend the active deadline in an existing `--results-dir`
- `--cleanup-buffer 5m`: reserve time for termination, cleanup, and state writes
- `--resume`: skip completed targets and restart an interrupted or failed target from its first configuration; requires the original `--results-dir`

Cache dropping requires root. When the runner is already root it writes
`drop_caches` directly; otherwise it requires `sudo -n true` to succeed during
preflight. Running the complete command with `sudo` is reliable for a long run,
but creates root-owned result files. A non-root run requires an appropriate
non-interactive sudo policy because a normal cached sudo credential may expire.

Resume uses the whole target as its checkpoint. If a reservation ends after run
2 of 5, that target is marked interrupted. A later invocation with the same
configuration, target selection, result directory, and `--resume` discards that
target's partial rows and restarts it; previously completed targets are skipped.

Extend a running reservation from another shell with:

```bash
sudo python3 scripts/microbenchmarks/fio/run_local_storage_fio.py \
  --results-dir "results/microbenchmarks/runs/$(hostname -s)-local-fio" \
  --extend-deadline 2h
```

The active runner rereads `deadline_state.json` at most every five seconds while
FIO is running.

## Output

The benchmark writes aggregated JSON results and `run_manifest.json` to the
configured results directory, for example:

The default is `results/microbenchmarks/runs/fio-<timestamp>/matrix_results_<timestamp>.json`.
An explicit `--results-dir` overrides it.

Each result entry includes FIO bandwidth, IOPS, and latency. The manifest records
the command, configuration snapshot,
inventory targets, FIO/Python/kernel versions, deadline, cache policy, per-target
status, failures, and filesystem capacity before and after each target.

Raw artifacts are retained under
`raw/<target>/<file-count-and-size>/<mode>/`. Each directory contains the exact
generated `.fio` job files and native FIO JSON for preparation and measured
runs. Failed commands retain captured standard output and error when available.

Result and manifest updates use a temporary file followed by `os.replace`. This
prevents an interruption during JSON serialization from replacing the last valid
checkpoint with a truncated file. It is not a multi-writer synchronization
mechanism; concurrent hosts must not share one result directory.

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

The current local-storage runner instead labels new rows by numeric target ID,
records the host and inventory metadata, and uses the verified 2026-09-21
target-to-mount mapping. The historical generic labels remain historical only.

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
