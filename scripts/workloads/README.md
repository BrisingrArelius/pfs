# workloads/

This directory contains everything needed to simulate POSIX I/O workloads for Darshan instrumentation.
Workloads are driven by a single C program (`posix_synthetic_workload.c`) and a JSON profile file (`profiles.json`).
The runner script `run_workloads.py` (in the project root) compiles the binary, iterates over profiles, and invokes the Darshan parser automatically.

> **Note:** `posix_synthetic_workload.c` uses raw POSIX syscalls (`open`/`read`/`write`/`lseek`/`fsync`).
> It only generates **POSIX module** data in Darshan logs. MPI-IO and STDIO workloads are not yet implemented.

---

## Contents

| File | Purpose |
|---|---|
| `posix_synthetic_workload.c` | C program that executes any POSIX workload profile |
| `profiles.json` | Defines all workload classes and their parameters |
| `tmp/` | Scratch directory — workload files written and deleted here at runtime |

---

## posix_synthetic_workload.c

A single C program that simulates any POSIX I/O workload given a set of parameters passed as CLI args.
It is compiled once by `run_workloads.py` before any profiles are run.

### How it works

All behavior is determined by the parameters passed in. The same binary executes every profile — the only variable is the parameter values. This eliminates implementation variance between workload classes.

The binary has two run modes, controlled by the final CLI argument:

### Run Modes

| Mode | Value | Darshan attached | Purpose |
|---|---|---|---|
| Setup | `0` | No | Writes files to disk and exits. No reads, no cleanup. Run by `run_workloads.py` **without** `LD_PRELOAD`. |
| Workload | `1` | Yes | The measured run. For read profiles: opens pre-existing files and reads only. For write/mixed: writes and reads normally. Cleans up files on exit. |

Only profiles with `read_ratio == 1.0` (pure-read) require a setup pass before the measured run. This is handled automatically by `run_workloads.py` — setup runs without `LD_PRELOAD` so Darshan cannot attach, keeping the Darshan log clean. Mixed profiles (`read_ratio > 0` and `< 1.0`) create their own files during the workload run.

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
| `0` | `sequential` / `contiguous` | Offset advances linearly by `op_size` each operation |
| `1` | `random` | Offset is a uniformly random `op_size`-aligned block within `[0, file_size − op_size]` — guarantees no out-of-bounds access |
| `2` | `strided` | Offset = `(global_op_index × stride_size) % file_size`, rounded down to the nearest `op_size` boundary to maintain block alignment |
| `3` | `nd_strided` | Multi-dimensional strided access simulating 2D array traversal. Alternates between row-major and column-major access patterns to create complex cache behavior. Uses `stride_size` as row stride. |

The setup write (mode `0`) is always sequential regardless of the profile's access pattern — only the measured workload run uses the configured pattern.

For strided and nd_strided workloads, the stride cursor is global across phases — each phase continues from where the previous one left off, so the stride sequence is never reset.

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

### Size Variants

**All profiles now use `file_size_gb` to automatically generate multiple size variants.**

Profiles include a `file_size_gb` field to automatically generate multiple size variants:

```json
"my_profile": {
    "read_ratio": 1.0,
    "access_pattern": "contiguous",
    "op_size": 4096,
    "num_ops": 50000,              // IGNORED - overwritten by calculation
    "file_size_gb": [1, 10]
}
```

This will generate two variants:
- `my_profile_1gb` — `num_ops` calculated to produce ~1 GB total I/O
- `my_profile_10gb` — `num_ops` calculated to produce ~10 GB total I/O

**The `num_ops` field in the JSON is ignored and overwritten.** It's only kept for backward compatibility.

**Calculation:** `num_ops = (target_size_bytes) / op_size`

For example:
- **Large ops** (4 MB): 1 GB → 256 ops, 10 GB → 2,560 ops, 100 GB → 25,600 ops
- **Small ops** (4 KB): 1 GB → 262,144 ops, 10 GB → 2,621,440 ops, 100 GB → 26,214,400 ops

**Execution Order:** For each profile variant, all runs complete on HDD, then SSD, before moving to the next profile:
```
profile1_1gb: HDD run1-5, then SSD run1-5
profile2_1gb: HDD run1-5, then SSD run1-5
...
profile1_10gb: HDD run1-5, then SSD run1-5
profile2_10gb: HDD run1-5, then SSD run1-5
```

HDD runs append to `./output/hdd/global.csv`, SSD runs append to `./output/ssd/global.csv`.

### Defined Profiles

**Base profiles: 19**  
**Size variants per profile: 2** (1GB, 10GB)  
**Total profile variants: 38**

Current profiles include various I/O patterns (sequential, random, strided, nd_strided) with different read/write ratios, operation sizes, and phase configurations. See `profiles.json` for the complete list.

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
