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

### Access Patterns

| Value | Pattern | Behavior |
|---|---|---|
| `0` | `sequential` | Offset advances linearly by `op_size` each operation |
| `1` | `random` | **Reads:** `lseek` to a random aligned offset within the file before each `read`. **Writes:** `lseek` to a random offset within the already-written extent before each `write` |
| `2` | `strided` | Offset advances by `stride_size` each operation, wrapping within the file |

### Phases

`num_phases` controls how many alternating write/read blocks the workload is divided into.
Phase 1 is always a write phase.

| `num_phases` | Behavior | `RW_SWITCHES` in Darshan |
|---|---|---|
| `1` | Single write pass | 0 |
| `2` | Write half, read half | 1 |
| `4` | W → R → W → R | 3 |

For pure read profiles (`read_ratio = 1.0`), phase 1 is always a silent setup write to create the file. All remaining phases are reads.

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
| `write_heavy` | sequential | 64 KB | 10,000 | 0.0 | 1 | Pure sequential writes |
| `read_heavy` | sequential | 64 KB | 10,000 | 1.0 | 2 | Phase 1 writes, phase 2 reads |
| `random_read` | random | 4 KB | 10,000 | 1.0 | 2 | Random offset reads |
| `random_write` | random | 4 KB | 10,000 | 0.0 | 1 | Random offset writes |
| `mixed_rw` | sequential | 64 KB | 10,000 | 0.5 | 8 | 8 alternating phases, high `RW_SWITCHES` |
| `strided_read` | strided | 64 KB | 10,000 | 1.0 | 2 | Stride = 512 KB (BeeGFS chunk boundary) |
| `strided_write` | strided | 64 KB | 10,000 | 0.0 | 1 | Stride = 512 KB |
| `large_io` | sequential | 10 MB | 100 | 0.5 | 2 | Few large ops |
| `small_io` | random | 512 B | 50,000 | 0.5 | 2 | Many tiny ops |
| `metadata_heavy` | sequential | 4 KB | 1,000 | 0.0 | 1 | 1,000 files — create/stat/delete |
| `sync_heavy` | sequential | 64 KB | 10,000 | 0.0 | 1 | `fsync` after every write |

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
