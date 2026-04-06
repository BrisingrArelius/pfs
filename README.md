# BeeGFS Storage Pool Analysis

A toolkit for running synthetic BeeGFS workloads, collecting Darshan I/O counters, and comparing HDD vs SSD storage behavior.

---

## Quick Start

```bash
# Run the full pipeline on HDD and SSD pools
python3 run_pipeline.py --runs 5

# Resume if interrupted
python3 run_pipeline.py --runs 5 --resume

# Analyze existing data only
python3 run_pipeline.py --analyze-only
```

**Key outputs:**
- `output/hdd/global.csv`
- `output/ssd/global.csv`
- `output/hdd/analysis/`
- `output/ssd/analysis/`

---

## What this repository contains

- `run_pipeline.py` — orchestrates pool setup, workload execution, Darshan parsing, and analysis
- `scripts/run_workloads.py` — workload runner for profile execution and Darshan parsing
- `scripts/parse_darshan.py` — extracts Darshan counters into structured CSV output
- `scripts/analysis/analysis.py` — computes statistics and generates visualizations
- `scripts/workloads/` — workload profile definitions and implementation helpers
- `scripts/fio/` — FIO benchmark matrix suite
- `scripts/pooling_scripts/` — BeeGFS pool management helpers
- `scripts/parse_ost_logs.py` — OST usage heatmap generation
- `scripts/parse_du.py` — disk usage summary utility

---

## Documentation

- `scripts/workloads/README.md` — workload profile and execution details
- `scripts/fio/README.md` — FIO benchmark matrix documentation
- `scripts/parse_darshan_README.md` — Darshan parser documentation
- `scripts/analysis/analysis_README.md` — analysis methodology and interpretation
- `scripts/pooling_scripts/README.md` — BeeGFS pooling helper docs
- `scripts/parse_ost_logs_README.md` — OST log heatmap documentation
- `scripts/parse_du_README.md` — `du` output summary helper

---

## Directory structure

```
pfs/
├── README.md
├── run_pipeline.py
├── output/
│   ├── hdd/
│   │   ├── global.csv
│   │   └── analysis/
│   └── ssd/
│       ├── global.csv
│       └── analysis/
├── scripts/
│   ├── run_workloads.py
│   ├── parse_darshan.py
│   ├── analysis/
│   │   └── analysis.py
│   ├── workloads/
│   │   ├── profiles.json
│   │   ├── posix_synthetic_workload.c
│   │   ├── posix_synthetic_workload_IOR.py
│   │   └── README.md
│   ├── fio/
│   │   ├── matrix_benchmark.py
│   │   ├── analyze_matrix.py
│   │   ├── fio_config.json
│   │   └── README.md
│   ├── pooling_scripts/
│   │   ├── configure_pools.sh
│   │   ├── reset_pools.sh
│   │   └── README.md
│   ├── parse_ost_logs.py
│   ├── parse_darshan_README.md
│   ├── analysis_README.md
│   └── parse_du.py
```

---

## Dependencies

Install the Python requirements:

```bash
pip install darshan pandas numpy matplotlib seaborn scikit-learn
```

System dependencies:
- `mpicc`, `mpirun`
- Darshan runtime and parser support
- `beegfs-ctl` access on BeeGFS client nodes
- `fio` for the FIO benchmark suite

---

## Recommended workflow

1. Configure BeeGFS pools:

```bash
cd scripts/pooling_scripts
sudo ./configure_pools.sh
```

2. Run the workload pipeline:

```bash
python3 run_pipeline.py --runs 5
```

3. Inspect results:
- `output/hdd/global.csv`
- `output/ssd/global.csv`
- `output/hdd/analysis/`
- `output/ssd/analysis/`

4. Analyze existing data only:

```bash
python3 run_pipeline.py --analyze-only
```

---

## Manual control

Run workloads directly:

```bash
python3 scripts/run_workloads.py --runs 5 --storage-type hdd
python3 scripts/run_workloads.py --runs 5 --storage-type ssd
python3 scripts/run_workloads.py --runs 5 --only read_heavy
```

Analyze data directly:

```bash
python3 scripts/analysis/analysis.py --input output/ssd/global.csv --output-dir output/ssd/analysis
python3 scripts/analysis/analysis.py --hdd output/hdd/global.csv --ssd output/ssd/global.csv --output-dir output/comparison
```

---

## Notes

- `run_workloads.py` uses `/mnt/beegfs/advay` and `/mnt/nfs_shared/darshan-logs` by default.
- `run_workloads.py` attaches Darshan with `LD_PRELOAD` for measured runs and removes temporary workload files after completion.
- `scripts/parse_ost_logs.py` consumes `scripts/ost_space_and_usage.log` and can generate OST heatmaps.

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
