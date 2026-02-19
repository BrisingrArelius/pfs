# workloads/

This directory contains everything needed to simulate I/O workloads for Darshan instrumentation.
Workloads are driven by a single C program (`synthetic_workload.c`) and a JSON profile file (`profiles.json`).
The runner script `run_workloads.py` (in the project root) compiles the binary, iterates over profiles, and invokes the Darshan parser automatically.

---

## Contents

| File | Purpose |
|---|---|
| `synthetic_workload.c` | Single C program that executes any workload profile |
| `profiles.json` | Defines all workload classes and their parameters |
| `tmp/` | Scratch directory — workload files written and deleted here at runtime |

---

## synthetic_workload.c

A single C program that simulates any I/O workload given a set of parameters passed as CLI args.
It is compiled once by `run_workloads.py` before any profiles are run.

### How it works

All behavior is determined by the parameters passed in. The same binary executes every profile — the only variable is the parameter values. This eliminates implementation variance between workload classes.

The binary has two run modes, controlled by the final CLI argument:

### Run Modes

| Mode | Value | Darshan attached | Purpose |
|---|---|---|---|
| Setup | `0` | No | Writes files to disk and exits. No reads, no cleanup. Run by `run_workloads.py` **without** `LD_PRELOAD`. |
| Workload | `1` | Yes | The measured run. For read profiles: opens pre-existing files and reads only. For write/mixed: writes and reads normally. Cleans up files on exit. |

Any profile with `read_ratio > 0` requires a setup pass before the measured run. This is handled automatically by `run_workloads.py` — setup runs without `LD_PRELOAD` so Darshan cannot attach, keeping the Darshan log clean.

`metadata_heavy` is an exception — it always runs in mode `1` directly, creating and deleting its own files as part of the workload.

### File Naming — How Workload Mode Finds Its Files

Both setup and workload mode use the same deterministic path:

```
{work_dir}/workload_{profile_name}_f{file_index}
```

For example, `read_heavy` with `num_files=1`:
```
./workloads/tmp/workload_read_heavy_f0
```

Setup writes this file. Workload mode opens it by the same name. No coordination needed — the path is fully determined by the parameters, which are identical in both invocations.

Running the same profile multiple times (`--runs N`) overwrites the setup file each run, which is intentional — it ensures the data being read is consistent and reproducible across runs.

### Access Patterns

| Value | Pattern | Behavior |
|---|---|---|
| `0` | `sequential` | Offset advances linearly by `op_size` each operation |
| `1` | `random` | **Reads:** `lseek` to a random aligned offset within the file before each `read`. **Writes:** `lseek` to a random offset within the already-written extent before each `write` |
| `2` | `strided` | Offset advances by `stride_size` each operation, wrapping within the file |

The setup write (mode `0`) is always sequential regardless of the profile's access pattern — only the measured run uses the configured pattern.

### Phases

`num_phases` controls how many alternating write/read blocks the workload is divided into.
Phase 1 is always a write phase for mixed profiles.

| `num_phases` | Behavior | `RW_SWITCHES` in Darshan |
|---|---|---|
| `1` | Single write pass | 0 |
| `2` | Write half, read half | 1 |
| `4` | W → R → W → R | 3 |

For pure read profiles (`read_ratio = 1.0`) in workload mode, all phases are read phases — writes are handled entirely by the separate setup pass.

### Metadata Workload

Profiles named `metadata_heavy` trigger a special code path:
- Creates `num_files` individual files in the scratch directory
- Writes `op_size` bytes to each
- `stat()`s each file
- `unlink()`s each file

This produces high `POSIX_OPENS`, `POSIX_STATS`, and `POSIX_FSYNCS` with minimal `POSIX_BYTES_WRITTEN`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `read_ratio` | float 0.0–1.0 | Fraction of total ops that are reads |
| `access_pattern` | `sequential` \| `random` \| `strided` | Access pattern (see above) |
| `stride_size` | bytes | Distance between operation offsets (strided only) |
| `op_size` | bytes | Size of each individual read or write |
| `num_ops` | integer | Total number of I/O operations |
| `num_files` | integer | Spread ops across this many files |
| `num_phases` | integer ≥ 1 | Number of alternating write/read phases |
| `fsync_interval` | integer | Call `fsync()` every N writes (0 = never) |

Total data volume = `num_ops × op_size`.

---

## profiles.json

Defines all workload classes. Each entry is a named profile with the parameters above.
`run_workloads.py` reads this file and passes each profile's values directly to the binary as CLI args.

### Defined Profiles

| Profile | Pattern | `op_size` | `num_ops` | `read_ratio` | `num_phases` | Notes |
|---|---|---|---|---|---|---|
| `write_heavy` | sequential | 64 KB | 10,000 | 0.0 | 1 | Pure sequential writes, no setup needed |
| `read_heavy` | sequential | 64 KB | 10,000 | 1.0 | 2 | Setup writes file, workload reads only |
| `random_read` | random | 4 KB | 10,000 | 1.0 | 2 | Setup writes file, workload reads at random offsets |
| `random_write` | random | 4 KB | 10,000 | 0.0 | 1 | Random offset writes, no setup needed |
| `mixed_rw` | sequential | 64 KB | 10,000 | 0.5 | 8 | 8 alternating phases, high `RW_SWITCHES` |
| `strided_read` | strided | 64 KB | 10,000 | 1.0 | 2 | Setup writes file, workload reads at 512 KB stride |
| `strided_write` | strided | 64 KB | 10,000 | 0.0 | 1 | Strided writes at 512 KB stride, no setup needed |
| `large_io` | sequential | 10 MB | 100 | 0.5 | 2 | Few large ops, setup + workload |
| `small_io` | random | 512 B | 50,000 | 0.5 | 2 | Many tiny ops, setup + workload |
| `metadata_heavy` | sequential | 4 KB | 1,000 | 0.0 | 1 | 1,000 files — create/stat/delete, no setup needed |
| `sync_heavy` | sequential | 64 KB | 10,000 | 0.0 | 1 | `fsync` after every write, no setup needed |

### Adding a New Profile

Add an entry to `profiles.json`:

```json
"my_workload": {
    "read_ratio":      0.3,
    "access_pattern":  "random",
    "stride_size":     0,
    "op_size":         8192,
    "num_ops":         20000,
    "num_files":       4,
    "num_phases":      2,
    "fsync_interval":  0
}
```

No code changes needed. Run with:

```bash
python run_workloads.py --only my_workload --runs 5
```
