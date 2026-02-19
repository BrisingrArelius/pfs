# pfs — Darshan I/O Counter Parser

A Python script for extracting and logging Darshan I/O counters from workload runs into structured CSV files, intended for workload characterization and analysis.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Workflow](#workflow)
- [Disclaimer](#disclaimer)

---

## Overview

This project instruments workloads with [Darshan](https://www.mcs.anl.gov/research/projects/darshan/), a lightweight I/O characterization library, and parses the resulting binary log files into CSV format for analysis.

Each run of the parser produces:
- A **per-run CSV** with one row per file accessed, containing only the counters for the requested modules
- A row appended to a **global CSV** with all counters present across all modules, using `NaN` for modules not collected in that run

This makes the global CSV immediately usable as a feature matrix for downstream analysis across many runs and workload types.

---

## Directory Structure

```
pfs/
├── run_workloads.py          # Compiles workload binary, runs profiles, invokes parser
├── parse_darshan.py          # Parses .darshan logs → CSV
├── parse_darshan_README.md   # Full reference for parse_darshan.py
├── darshan_output/           # Created automatically — CSV output lives here
│   ├── global.csv            # One row per run, all counters across all modules
│   └── <label>_<modules>.csv # Per-run, per-file counters for each run
├── workloads/
│   ├── posix_synthetic_workload.c  # Single C POSIX program that simulates all workload types
│   ├── profiles.json               # Defines all workload classes and their parameters
│   ├── tmp/                        # Scratch dir — workload files written and deleted here
│   └── README.md                   # How posix_synthetic_workload.c works, profile reference
└── README.md
```

---

## Dependencies

- Python 3.8+
- [pydarshan](https://github.com/darshan-hpc/darshan) — Python bindings for reading Darshan logs
- [pandas](https://pandas.pydata.org/)
- Darshan runtime and `darshan-parser` available on the system

Install Python dependencies:
```bash
pip install darshan pandas
```

---

## Workflow

### Option A — Run all workload profiles automatically

`run_workloads.py` handles compilation, execution, and parsing in one step:

```bash
# Run all profiles once
python run_workloads.py

# Run all profiles 5 times each
python run_workloads.py --runs 5

# Run specific profiles only
python run_workloads.py --only read_heavy write_heavy --runs 10

# Dry run — see what would execute without running anything
python run_workloads.py --dry-run
```

See [`workloads/README.md`](workloads/README.md) for how profiles are defined and how to add new ones.

### Option B — Run and parse a single log manually

```bash
# Compile the workload binary
gcc -O2 -o workloads/posix_synthetic_workload workloads/posix_synthetic_workload.c

# Run under Darshan instrumentation (mode 1 = workload)
LD_PRELOAD=/path/to/libdarshan.so ./workloads/posix_synthetic_workload \
    write_heavy 0.0 0 0 65536 10000 1 1 0 ./workloads/tmp 1

# Parse the resulting log
python parse_darshan.py --log /tmp/<logfile>.darshan --label write_heavy --posix
```

### Inspect the output

```
darshan_output/
├── global.csv                  ← one row appended per run, all counters
└── write_heavy_posix.csv       ← per-file breakdown for that run
```

See [`parse_darshan_README.md`](parse_darshan_README.md) for the full parser reference.

---

## Disclaimer

This project was developed with the assistance of **GitHub Copilot (powered by Claude)**. The architecture, counter selection, aggregation logic, and implementation were designed collaboratively through an iterative conversation.