# Workloads

This directory contains workload definitions, the IOR wrapper, the original C
implementation and the relocated `run_workloads.py` / `run_pipeline.py`.

Scripts were moved without content changes. Internal paths still require
[deferred repairs](../../TODOS_SCRIPT_CHANGES.md); the usage below describes the
legacy workflow. Historical outputs now live under
[results/workloads/legacy](../../results/workloads/legacy/README.md).

## What this directory contains

- `profiles.json` — workload profile definitions
- `posix_synthetic_workload_IOR.py` — Python wrapper that turns profiles into IOR commands
- `posix_synthetic_workload.c` — original C workload implementation used for some stride patterns
- `run_workloads.py` — relocated workload runner; internal paths pending repair
- `run_pipeline.py` — relocated legacy orchestration; internal paths pending repair

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

The `metadata_heavy` profile uses a special path that creates and unlinks many small files, and it does not require a separate setup phase.

## Access patterns

The available access patterns are:

- `sequential` / `contiguous`
- `random`
- `strided`
- `nd_strided`

`posix_synthetic_workload_IOR.py` handles sequential and random profiles.
`strided` and `nd_strided` profiles are delegated to the C implementation in `posix_synthetic_workload.c` when available.

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

The unchanged runner still derives outputs and logs from its old project-root
assumptions. Historical OST logs now live under
`results/microbenchmarks/legacy/placement/`; future destinations need path repairs.

## Legacy CLI (after deferred path repairs)

Run the full project pipeline:

```bash
python3 scripts/workloads/run_pipeline.py --runs 5
```

Run a specific profile:

```bash
python3 scripts/workloads/run_workloads.py --only read_heavy --runs 5 --storage-type hdd
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
