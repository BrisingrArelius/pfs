# pfs — Darshan I/O Counter Parser

A Python toolkit for extracting and analyzing Darshan I/O counters from workload runs to characterize workload behavior and inform storage tiering decisions.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Workflow](#workflow)
- [Analysis](#analysis)
- [Disclaimer](#disclaimer)

---

## Overview

This project instruments workloads with [Darshan](https://www.mcs.anl.gov/research/projects/darshan/), a lightweight I/O characterization library, and parses the resulting binary log files into CSV format for analysis.

**Key components:**
- **Synthetic workloads** (`posix_synthetic_workload.c`) — Single C program that simulates 11 different workload types
- **Parser** (`parse_darshan.py`) — Extracts counters from `.darshan` logs into CSV
- **Analysis** (`analysis.py`) — Statistical analysis and visualizations to identify discriminative features

**Output:**
- **Per-run CSV**: One row per file accessed, counters for requested modules
- **Global CSV**: One row per run with all counters (NaN for modules not collected)
- **Analysis outputs**: Heatmaps, PCA plots, statistics

---

## Directory Structure

```
pfs/
├── run_workloads.py              # Orchestrator: compiles, runs profiles, parses logs
├── parse_darshan.py              # Extracts counters from .darshan logs → CSV
├── analysis.py                   # Statistical analysis & visualizations
├── parse_darshan_README.md       # Full parser reference
├── darshan_output/               # CSV outputs (created automatically)
│   ├── global.csv                # One row per run, all counters
│   └── <label>_<modules>.csv     # Per-run, per-file counters
├── analysis_output/              # Analysis results (created by analysis.py)
│   ├── heatmap_all_counters.png
│   ├── heatmap_stable_counters.png
│   ├── bar_charts_discriminative.png
│   ├── pca_clustering.png
│   ├── statistics.csv
│   └── means_only.csv
├── workloads/
│   ├── posix_synthetic_workload.c  # POSIX I/O workload simulator
│   ├── profiles.json               # Workload definitions
│   ├── tmp/                        # Scratch directory for workload files
│   └── README.md                   # Workload documentation
├── pooling_scripts/              # BeeGFS storage pool management
│   ├── configure_pools.sh
│   └── reset_pools.sh
└── README.md
```

---

## Dependencies

**Core tools:**
- Python 3.8+
- [pydarshan](https://github.com/darshan-hpc/darshan) — Darshan log parser
- [pandas](https://pandas.pydata.org/) — Data manipulation
- Darshan runtime (for instrumentation)

**Analysis tools (for `analysis.py`):**
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

**Installation:**

```bash
# Core
pip install darshan pandas

# Analysis
pip install numpy matplotlib seaborn scikit-learn

# Or all at once
pip install darshan pandas numpy matplotlib seaborn scikit-learn
```

---

## Workflow

### Quick Start: Full Pipeline

**Run everything automatically** (workloads on both HDD and SSD pools, parse, analyze):

```bash
# Full pipeline: 5 runs per profile on both HDD and SSD
python3 run_pipeline.py --runs 5

# Test with fewer runs
python3 run_pipeline.py --runs 3

# Only HDD pool (for testing)
python3 run_pipeline.py --runs 3 --hdd-only

# Only SSD pool (for testing)
python3 run_pipeline.py --runs 3 --ssd-only

# Skip workloads, analyze existing data
python3 run_pipeline.py --analyze-only
```

**What it does:**
1. Configures BeeGFS storage pools (sets pool ID for directories)
2. Runs workloads on HDD pool → `darshan_output_hdd/global.csv`
3. Runs workloads on SSD pool → `darshan_output_ssd/global.csv`
4. Analyzes both datasets → `analysis_output_hdd/` and `analysis_output_ssd/`
5. Compares results to identify HDD vs SSD behavior differences

**Outputs:**
- `darshan_output_hdd/global.csv` — HDD pool counter data
- `darshan_output_ssd/global.csv` — SSD pool counter data
- `analysis_output_hdd/` — HDD visualizations and statistics
- `analysis_output_ssd/` — SSD visualizations and statistics

---

### Manual Workflow (Step-by-Step)

### Step 1: Run workloads and generate logs

**Option A — Automated (all profiles):**

```bash
# Run all profiles once
python run_workloads.py

# Run all profiles 5 times each
python run_workloads.py --runs 5

# Run specific profiles
python run_workloads.py --only read_heavy write_heavy --runs 10
```

**Option B — Manual (single profile):**

```bash
# Compile
mpicc -O2 -o workloads/posix_synthetic_workload workloads/posix_synthetic_workload.c \
    -L/usr/local/lib -ldarshan -lpthread -lrt -lz

# Run (mode 1 = workload with Darshan instrumentation)
mpirun -np 1 ./workloads/posix_synthetic_workload \
    write_heavy 0.0 0 0 65536 10000 1 1 0 ./workloads/tmp 1

# Parse
python parse_darshan.py --log /path/to/log.darshan --label write_heavy --posix
```

**Output:**
```
darshan_output/
├── global.csv                     # Aggregated counters (one row per run)
└── write_heavy_run1_posix.csv     # Per-file breakdown
```

See [`workloads/README.md`](workloads/README.md) for profile definitions and [`parse_darshan_README.md`](parse_darshan_README.md) for parser details.

### Step 2: Analyze counter data

Once you have multiple runs (5+ recommended):

```bash
python3 analysis.py --input ./darshan_output/global.csv
```

**Output:**
```
analysis_output/
├── heatmap_all_counters.png        # Normalized heatmap (all counters)
├── heatmap_stable_counters.png     # Stable counters only (low CV)
├── bar_charts_discriminative.png   # Top N discriminative counters
├── pca_clustering.png              # 2D PCA projection
├── statistics.csv                  # Full statistics (mean/std/min/max/cv)
└── means_only.csv                  # Summary (mean values only)
```

---

## Analysis

### What `analysis.py` does

Performs statistical analysis across multiple runs to identify:

1. **Stable counters** — Low coefficient of variation (CV < threshold), consistent across runs → reliable for classification
2. **Discriminative counters** — High variance across workload types → good predictors for storage placement

**Key visualizations:**
- **Heatmaps**: Normalized (0-1 scale) counter values across workloads. Normalization needed because counters have vastly different scales (bytes: millions, switches: <10).
- **Bar charts**: Top N counters that best distinguish workload types (mean ± std)
- **PCA plot**: 2D projection showing natural clustering of workloads

### Options

```bash
python3 analysis.py --input ./darshan_output/global.csv \
    --cv-threshold 0.15 \     # Stability threshold (default: 0.2)
    --top-n 15 \              # Number of discriminative counters to plot (default: 10)
    --output-dir ./results    # Output directory (default: ./analysis_output)
```

### Example: Storage Tiering Rules

Use discriminative counters to build HDD vs SSD placement rules:

| Counter | HDD Friendly | SSD Friendly |
|---------|-------------|-------------|
| `POSIX_SEQ_READS/WRITES` | High (sequential) | Low (random) |
| `POSIX_RW_SWITCHES` | Low (single-pass) | High (mixed I/O) |
| `POSIX_SIZE_*_1M_4M` | High (large I/O) | Low (small I/O) |
| `POSIX_SEEKS` | Low | High (random access) |
| Timestamp duration | Long (sustained) | Short (bursty) |

**Example rules:**
```python
if POSIX_SEQ_READS > 80%:
    placement = "HDD"  # Sequential read-heavy
elif POSIX_RW_SWITCHES > 5:
    placement = "SSD"  # Mixed random I/O
elif avg_op_size > 1MB:
    placement = "HDD"  # Large block I/O
```

---

## Disclaimer

This project was developed with the assistance of **GitHub Copilot (powered by Claude)**. The architecture, counter selection, aggregation logic, and implementation were designed collaboratively through an iterative conversation.
