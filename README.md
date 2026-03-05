# BeeGFS Storage Pool Analysis

A comprehensive toolkit for analyzing I/O workload behavior on BeeGFS storage pools using Darshan instrumentation. Compare HDD vs SSD performance, identify discriminative I/O counters, and inform storage tiering decisions.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Detailed Documentation](#detailed-documentation)
- [Workflow](#workflow)
- [Analysis](#analysis)
- [Disclaimer](#disclaimer)

---

## Overview

This project:
1. **Runs synthetic workloads** on BeeGFS HDD and SSD storage pools
2. **Instruments with Darshan** to capture I/O counters
3. **Parses binary logs** to CSV format
4. **Analyzes and visualizes** counter data to identify patterns

**Use case:** Determine which I/O characteristics (sequential access, operation size, seek rate, etc.) predict whether a workload should be placed on HDD or SSD storage.

---

## Quick Start

```bash
# Run workloads on both HDD and SSD storage pools (recommended)
python3 run_pipeline.py --runs 5

# Resume if interrupted
python3 run_pipeline.py --runs 5 --resume

# Analyze existing data only
python3 run_pipeline.py --analyze-only
```

**Results:**
- Darshan data: `output/hdd/darshan/global.csv` and `output/ssd/darshan/global.csv`
- Analysis output: `output/hdd/analysis/` and `output/ssd/analysis/`

---

## Directory Structure

```
pfs/
├── README.md
├── run_pipeline.py              # Main entry point — runs full pipeline
├── output/                      # All results (created by scripts)
│   ├── hdd/
│   │   ├── darshan/
│   │   │   └── global.csv      # HDD workload runs (one row per run)
│   │   └── analysis/           # HDD analysis outputs
│   └── ssd/
│       ├── darshan/
│       │   └── global.csv      # SSD workload runs (one row per run)
│       └── analysis/           # SSD analysis outputs
└── scripts/
    ├── run_workloads.py         # Compile, run, and parse workloads
    ├── parse_darshan.py         # Extract counters from Darshan logs
    ├── analysis.py              # Statistical analysis & visualization
    ├── checkpoint.json          # Resume state (auto-created)
    ├── parse_darshan_README.md  # Parser documentation
    ├── analysis_README.md       # Analysis documentation
    ├── workloads/
    │   ├── posix_synthetic_workload.c  # C workload simulator
    │   ├── profiles.json               # 19 workload profiles (57 variants)
    │   └── README.md                   # Workload documentation
    └── pooling_scripts/         # BeeGFS pool management
        ├── configure_pools.sh   # Create HDD/SSD pools
        └── reset_pools.sh       # Reset to default pool
```

---

## Dependencies

**Required:**
- Python 3.8+
- [pydarshan](https://github.com/darshan-hpc/darshan) — Darshan log parser
- [pandas](https://pandas.pydata.org/) — Data manipulation
- [NumPy](https://numpy.org/) — Numerical computing
- [Matplotlib](https://matplotlib.org/) — Plotting
- [Seaborn](https://seaborn.pydata.org/) — Statistical visualization
- [scikit-learn](https://scikit-learn.org/) — Machine learning (PCA)
- Darshan runtime library (MPI-enabled)
- MPI compiler (`mpicc`, `mpirun`)

```bash
pip install darshan pandas numpy matplotlib seaborn scikit-learn
```

---

## Detailed Documentation

- **`scripts/workloads/README.md`** — Workload definitions and parameters
- **`scripts/parse_darshan_README.md`** — Parser usage and counter details
- **`scripts/analysis_README.md`** — Analysis methodology and interpretation

---

## Workflow

### Recommended Workflow

**1. Configure BeeGFS pools** (one-time setup):
```bash
cd scripts/pooling_scripts
sudo ./configure_pools.sh
```

**2. Run workloads on both storage pools:**
```bash
python3 run_pipeline.py --runs 5
```

This automatically:
- Compiles the workload binary (with `O_DIRECT` for cache-bypass I/O)
- Runs all 19 profiles × 3 size variants (100MB, 1GB, 10GB) × 5 runs on HDD, then SSD
- Clears system caches before every run (2 minute stabilization wait)
- Parses Darshan logs and appends to `output/hdd/darshan/global.csv` and `output/ssd/darshan/global.csv`

**Resume if interrupted:**
```bash
python3 run_pipeline.py --runs 5 --resume
```

**3. Analyze and compare results:**
```bash
python3 run_pipeline.py --analyze-only
```

### Manual Control (Advanced)

```bash
# Run only specific profiles
python3 scripts/run_workloads.py --runs 5 --only large_contiguous_write_heavy_freq --storage-type hdd

# Run fewer iterations for testing
python3 run_pipeline.py --runs 3

# HDD or SSD only
python3 run_pipeline.py --runs 5 --hdd-only
python3 run_pipeline.py --runs 5 --ssd-only
```
---

## Analysis

The analysis tool (`scripts/analysis.py`) processes Darshan counter data to identify patterns and discriminative features.

**Key outputs:**
- **Heatmaps**: Visualize counter patterns across workload types
- **Bar charts**: Show top discriminative counters with error bars
- **PCA plot**: 2D projection showing workload clustering
- **Statistics**: Mean, std, CV, min, max for each counter

**Example results:**

Discriminative counters for HDD vs SSD placement:

| Counter | HDD Friendly | SSD Friendly |
|---------|-------------|-------------|
| `POSIX_SEQ_READ_RATIO` | High (>0.8) | Low (<0.5) |
| `POSIX_RW_SWITCHES` | Low (<2) | High (>5) |
| `POSIX_MEAN_WRITE_SIZE` | Large (>1MB) | Small (<64KB) |
| `POSIX_SEEK_RATE` | Low (<0.1) | High (>0.5) |
| `POSIX_WRITE_DURATION` | Long (sustained) | Short (bursty) |

**For detailed methodology and interpretation**, see `scripts/analysis_README.md`.
```

---

## Disclaimer

This project was developed with the assistance of **GitHub Copilot (powered by Claude)**. The architecture, counter selection, aggregation logic, and implementation were designed collaboratively through an iterative conversation.
