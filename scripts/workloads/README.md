# Workloads

This directory contains workload definitions, the IOR wrapper, the C workload
implementation, and the workload/pipeline runners.

The runners use run-specific output locations. This is an application-workload
pipeline rather than the complete six-domain suite.
Historical outputs live under
[results/workloads/legacy](../../results/workloads/legacy/README.md).

## What this directory contains

- `profiles.json` — workload profile definitions
- `posix_synthetic_workload_IOR.py` — Python wrapper that turns profiles into IOR commands
- `posix_synthetic_workload.c` — standalone C workload implementation
- `run_workloads.py` — workload runner with run-specific output/log paths
- `run_pipeline.py` — HDD/SSD workload orchestration
- `analysis/` — Darshan parsing and workload-result analysis

## Profile execution model

`run_workloads.py` reads `profiles.json` and expands profiles that define `file_size_gb` into size variants.
Each generated variant is named like `profile_100mb`, `profile_1gb`, or `profile_10gb`.

The script recalculates `num_ops` from `file_size_gb` and `op_size`, so the `num_ops` value in `profiles.json` is overwritten at runtime.

### Deterministic file naming

Workload files are written to the configured `workload_dir` using a deterministic pattern:

```
{work_dir}/workload_{profile_name}_run{run_index}_f{file_index}
```

This allows setup and workload phases to access the same files without extra coordination.

## Setup vs workload phases

- **Setup mode (`mode=0`)**
  - Writes files only
  - Runs without Darshan instrumentation
  - Used for pure-read profiles to prepare input files
- **Workload mode (`mode=1`)**
  - Runs under `mpirun -np 1`
  - Darshan is attached via `LD_PRELOAD`
  - Generates the measured `.darshan` log

Pure-read profiles require a setup pass, while pure-write and mixed profiles run directly in workload mode.

## Access patterns

`posix_synthetic_workload_IOR.py` implements sequential/contiguous and random
profiles. `run_workloads.py` always invokes this Python IOR wrapper. Although
the profile loader accepts `strided` and `nd_strided`, the wrapper rejects them;
the runner does not delegate those patterns to `posix_synthetic_workload.c`.
Metadata-heavy special handling also exists only in the standalone C program.

The current `profiles.json` contains two contiguous, read-only profiles:
`small_contiguous_read_heavy_freq_1` and
`small_contiguous_read_heavy_freq_2`.

## Profile fields

| Field | Description |
|---|---|
| `read_ratio` | Fraction of operations that are reads |
| `access_pattern` | I/O access pattern |
| `stride_size` | Distance between offsets for strided access |
| `op_size` | Size of each I/O operation (bytes) |
| `num_ops` | Total number of operations (recomputed from `file_size_gb`) |
| `num_files` | Number of files involved in the workload |
| `num_phases` | Number of alternating read/write phases |
| `fsync_interval` | Call `fsync()` every N writes |
| `file_size_gb` | Target workload sizes used to auto-generate variants |

## Output

Historical Darshan summary outputs are preserved at:

- `results/workloads/legacy/darshan/hdd/global.csv`
- `results/workloads/legacy/darshan/ssd/global.csv`

New outputs, logs, errors, and checkpoints go under
`results/workloads/runs/<run-id>/`. Historical OST logs remain under
`results/microbenchmarks/legacy/placement/` and are not resume state.

## Usage

Run the full project pipeline:

```bash
python3 scripts/workloads/run_pipeline.py --runs 5
```

Run a specific profile:

```bash
python3 scripts/workloads/run_workloads.py --only small_contiguous_read_heavy_freq_1 --runs 5 --storage-type hdd
```

## Adding a new profile

Add an entry to `profiles.json` with the desired parameter values. Example:

```json
"my_workload": {
    "read_ratio":      0.3,
    "access_pattern":  "random",
    "stride_size":     0,
    "op_size":         8192,
    "num_ops":         20000,
    "num_files":       4,
    "num_phases":      2,
    "fsync_interval":  0,
    "file_size_gb":    [0.1, 1, 10]
}
```

Then run:

```bash
python3 scripts/workloads/run_pipeline.py --runs 5
```
