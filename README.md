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
# Full pipeline: Run workloads on both HDD and SSD pools, then analyze
python3 run_pipeline.py --runs 5

# HDD only (for testing)
python3 run_pipeline.py --runs 3 --hdd-only

# SSD only (for testing)
python3 run_pipeline.py --runs 3 --ssd-only

# Analyze existing data without re-running workloads
python3 run_pipeline.py --analyze-only
```

**Results:** All outputs in `output/hdd/` and `output/ssd/` directories.

---

## Directory Structure

```
pfs/
├── run_pipeline.py               # Main orchestration script
├── README.md                     # This file
├── output/                       # All results (created by pipeline)
│   ├── hdd/
│   │   ├── darshan/             # HDD pool Darshan CSVs
│   │   │   ├── global.csv       # One row per run, all counters
│   │   │   └── *.csv            # Per-run, per-file details
│   │   └── analysis/            # HDD pool analysis results
│   │       ├── heatmap_all_counters.png
│   │       ├── heatmap_stable_counters.png
│   │       ├── bar_charts_discriminative.png
│   │       ├── pca_clustering.png
│   │       ├── statistics.csv
│   │       └── means_only.csv
│   └── ssd/
│       ├── darshan/             # SSD pool Darshan CSVs
│       └── analysis/            # SSD pool analysis results
└── scripts/
    ├── run_workloads.py         # Compile, run, and parse workloads
    ├── parse_darshan.py         # Extract counters from Darshan logs
    ├── analysis.py              # Statistical analysis & visualization
    ├── parse_darshan_README.md  # Parser documentation
    ├── analysis_README.md       # Analysis documentation
    ├── workloads/
    │   ├── posix_synthetic_workload.c  # C workload simulator
    │   ├── profiles.json               # 11 workload definitions
    │   └── README.md                   # Workload documentation
    └── pooling_scripts/         # BeeGFS pool management
        ├── configure_pools.sh   # Create HDD/SSD pools
        └── reset_pools.sh       # Reset to default pool
```

---

## Dependencies

**Core tools:**
- Python 3.8+
- [pydarshan](https://github.com/darshan-hpc/darshan) — Darshan log parser
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

**Installation:**

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

The pipeline handles everything automatically. For manual control, see individual script documentation.

### Automated (Recommended)

```bash
# Full pipeline: HDD + SSD pools, 5 runs each
python3 run_pipeline.py --runs 5
```

### Manual Steps

If you need fine-grained control:

**1. Configure BeeGFS pools** (one-time setup):
```bash
cd scripts/pooling_scripts
sudo ./configure_pools.sh
```

**2. Run workloads on HDD pool:**
```bash
python3 scripts/run_workloads.py --runs 5 \
    --workload-dir /mnt/beegfs/advay/hdd/workloads/tmp \
    --output output/hdd/darshan
```

**3. Run workloads on SSD pool:**
```bash
python3 scripts/run_workloads.py --runs 5 \
    --workload-dir /mnt/beegfs/advay/ssd/workloads/tmp \
    --output output/ssd/darshan
```

**4. Analyze results:**
```bash
python3 scripts/analysis.py --input output/hdd/darshan/global.csv \
    --output-dir output/hdd/analysis

python3 scripts/analysis.py --input output/ssd/darshan/global.csv \
    --output-dir output/ssd/analysis
```

# Run (mode 1 = workload with Darshan instrumentation)
mpirun -np 1 ./workloads/posix_synthetic_workload \
    write_heavy 0.0 0 0 65536 10000 1 1 0 ./workloads/tmp 1

# Parse
python parse_darshan.py --log /path/to/log.darshan --label write_heavy --posix
```

**Output:**
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
