# parse_darshan.py

Parses a `.darshan` binary log file and extracts a configurable set of I/O counters into CSV files.

---

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--log <path>` | Yes | Path to the `.darshan` log file to parse |
| `--label <name>` | Yes | Workload label — used in the output filename and as a column value |
| `--posix` | At least one | Extract POSIX module counters |
| `--mpi` | At least one | Extract MPI-IO module counters |
| `--stdio` | At least one | Extract STDIO module counters |
| `--output-dir <path>` | No | Override the default output directory (`./darshan_output`) |

Multiple module flags can be combined:
```bash
python parse_darshan.py --log run.darshan --label checkpoint --posix --mpi
```

---

## Counter Modules

| Flag | Darshan Module | Covers |
|---|---|---|
| `--posix` | `POSIX` | Low-level POSIX I/O: `read()`, `write()`, `open()`, `seek()`, etc. |
| `--mpi` | `MPI-IO` | MPI-IO calls: independent, collective, split, non-blocking operations |
| `--stdio` | `STDIO` | Buffered C stdio: `fopen()`, `fread()`, `fwrite()`, `fflush()` |

> **Note:** Your workload determines which modules will have data. The synthetic workload (`posix_synthetic_workload.c`) uses raw POSIX syscalls and only produces POSIX module records. A program using `fopen`/`fprintf` would produce STDIO records instead.

---

## Output

**Per-run CSV** — `{label}_{modules}.csv`

One row per file accessed during the run. Columns are the counters for the requested modules only. If a file with the same name already exists, a numeric suffix is appended (`_1`, `_2`, etc.) — existing files are never overwritten.

```
timestamp, label, file_id, STDIO_OPENS, STDIO_READS, STDIO_WRITES, ...
```

**Global CSV** — `global.csv`

One row per parser invocation, appended on every run. All counters from all modules are always present as columns. Counters for modules not requested in a given run are filled with `NaN`.

```
timestamp, label, modules_used, POSIX_OPENS, ..., MPIIO_INDEP_OPENS, ..., STDIO_OPENS, ...
```

---

## Counter Configuration

The counters tracked are defined as dicts at the top of the script:

```python
POSIX_COUNTERS = {
    "POSIX_READS": "sum",
    "POSIX_BYTES_READ": "sum",
    "POSIX_F_READ_TIME": "sum",
    "POSIX_F_MAX_READ_TIME": "max",
    ...
}
```

Each counter maps to an aggregation strategy used when collapsing per-file rows into a single run-level value in `global.csv`:

| Strategy | Used for |
|---|---|
| `sum` | Counts, bytes, histogram buckets, access pattern counters |
| `max` | Peak times, maximum operation sizes, end timestamps |
| `min` | Start timestamps (earliest across all files) |
| `first` | Alignment values (a property of the system, same for all files) |

To add or remove counters, edit the relevant dict at the top of the script. Counter names must match exactly what `darshan-parser` reports — verify against a real log with:

```bash
darshan-parser --all <logfile.darshan> | grep -E "^POSIX|^MPI|^STDIO"
```

---

## Examples

```bash
# STDIO-only workload
python parse_darshan.py --log ./logs/run1.darshan --label sequential_write --stdio

# POSIX + MPI workload with custom output directory
python parse_darshan.py --log ./logs/run2.darshan --label checkpoint --posix --mpi --output-dir /scratch/results

# All three modules
python parse_darshan.py --log ./logs/run3.darshan --label random_read --posix --mpi --stdio
```
