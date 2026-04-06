# Analysis Pipeline Documentation

This document explains how `scripts/analysis/analysis.py` consumes Darshan counter data, computes derived metrics, and generates visual reports.

## Overview

`analysis.py` supports two modes:

- **Single-file mode**: analyze one `global.csv` and generate plots and statistics
- **Comparison mode**: analyze HDD and SSD `global.csv` files together

## Usage

### Single-file analysis

```bash
python3 scripts/analysis/analysis.py --input output/ssd/global.csv --output-dir output/ssd/analysis
```

### HDD vs SSD comparison

```bash
python3 scripts/analysis/analysis.py --hdd output/hdd/global.csv --ssd output/ssd/global.csv --output-dir output/comparison
```

## Outputs

Single-file analysis writes:

- `heatmap_all_counters.png`
- `heatmap_stable_counters.png`
- `bar_charts_discriminative.png`
- `pca_clustering.png`
- `statistics.csv`
- `mean_values.csv`

Comparison mode additionally writes:

- `heatmap_hdd_ssd_interleaved.png`
- `bandwidth_comparison.png`
- `latency_comparison.png`
- `performance_gains.png`
- `ssd_stats/statistics.csv`
- `ssd_stats/mean_values.csv`

## What gets analyzed

The script loads `global.csv` and computes:

- profile-level statistics for each counter
- derived metrics such as bandwidth, latency, mean I/O size, sequential ratios, and seek rate
- coefficient of variation (CV) to identify stable counters
- discriminative counter rankings across profiles

## Derived metrics

When the required Darshan counters are present, the following derived values are computed:

- `POSIX_READ_BW_MBps`
- `POSIX_WRITE_BW_MBps`
- `POSIX_TOTAL_BW_MBps`
- `POSIX_READ_LATENCY_ms`
- `POSIX_WRITE_LATENCY_ms`
- `POSIX_OPEN_LATENCY_ms`
- `POSIX_CLOSE_LATENCY_ms`
- `POSIX_META_LATENCY_ms`
- `POSIX_READ_DENSITY`
- `POSIX_WRITE_DENSITY`
- `POSIX_MEAN_READ_SIZE`
- `POSIX_MEAN_WRITE_SIZE`
- `POSIX_SEQ_READ_RATIO`
- `POSIX_SEQ_WRITE_RATIO`
- `POSIX_SEEK_RATE`

## Counter filtering

The analysis ignores counters that are:

- all `NaN`
- all zeros
- timestamp markers such as `*_START_TIMESTAMP` and `*_END_TIMESTAMP`

This makes visual output easier to interpret.

## Stable vs discriminative counters

- **Stable counters** are those with low run-to-run variability (CV below the selected threshold).
- **Discriminative counters** are those whose mean values differ most across workload profiles.

Use stable counters for reliable feature selection and discriminative counters for workload separation.

## Interpretation tips

- The all-counters heatmap shows the full workload signature.
- The stable-counters heatmap highlights the most repeatable metrics.
- Bar charts show which counters vary most across workloads.
- PCA plots show whether workloads cluster by behavior.

## Requirements

This script requires:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install via:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
  write_heavy: 0 ± 0
  mixed_rw:    7 ± 0.5
  read_heavy:  0 ± 0
```
→ Perfect discriminator for mixed workloads!

### 4. PCA Clustering

**File**: `pca_clustering.png`

**What it shows**:
- 2D scatter plot where each point = one workload profile
- X/Y axes = Principal components (PC1, PC2)
- Distance between points = Similarity in I/O behavior

**What is PCA?**
Principal Component Analysis reduces 100+ counters to 2 dimensions while preserving maximum variance.

**How to read it**:
- **Tight clusters** = Workloads with similar I/O patterns
- **Separated groups** = Distinct I/O behaviors
- **Outliers** = Unique workload characteristics

**Example interpretation**:
- Sequential workloads (write_heavy, read_heavy) cluster together
- Random workloads (random_read, random_write) cluster separately
- Mixed workloads (mixed_rw) may be in between

**Validation**:
If your synthetic workloads cluster into distinct groups, they successfully represent different I/O classes!

---

## Interpretation Guide

### Using Statistics to Select Features

1. **Check CV in `statistics.csv`**:
   ```csv
   profile,POSIX_READS_mean,POSIX_READS_std,POSIX_READS_cv
   write_heavy,0,0,nan
   read_heavy,10000,50,0.005
   ```
   → `POSIX_READS` has low CV (0.005) for read_heavy → stable counter

2. **Check discriminative scores** (bar charts):
   - Counters with large differences between profiles are good predictors

3. **Check PCA plot**:
   - If workloads don't separate, you may need more discriminative counters

### Example: Building HDD/SSD Placement Rules

From the analysis outputs, identify counters that correlate with storage preferences:

**HDD-friendly indicators**:
- High `POSIX_SEQ_READ_RATIO` / `POSIX_SEQ_WRITE_RATIO` (sequential access > 80%)
- High `POSIX_MEAN_WRITE_SIZE` (large I/O operations, e.g., >1MB)
- Low `POSIX_SEEK_RATE` (few random accesses, <0.1 seeks per op)
- Long `POSIX_WRITE_DURATION` (sustained I/O, >5 seconds)
- High `POSIX_SIZE_*_1M_4M` histogram bins

**Example workload (large_io)**:
```
POSIX_MEAN_WRITE_SIZE = 10,485,760 (10 MB)
POSIX_SEQ_WRITE_RATIO = 1.0 (100% sequential)
POSIX_WRITE_DURATION = 1.8 seconds
POSIX_WRITE_DENSITY = 285 MB/s
→ Recommendation: HDD (sustained sequential throughput)
```

**SSD-friendly indicators**:
- High `POSIX_RW_SWITCHES` (mixed read/write, >5 switches)
- High `POSIX_SEEK_RATE` (random access, >0.5 seeks per op)
- Low `POSIX_MEAN_WRITE_SIZE` (small I/O operations, <64KB)
- Short `POSIX_WRITE_DURATION` (bursty I/O, <1 second)
- High `POSIX_SIZE_*_0_100` or `POSIX_SIZE_*_100_1K` histogram bins

**Example workload (metadata_heavy)**:
```
POSIX_MEAN_WRITE_SIZE = 4,096 (4 KB)
POSIX_OPENS = 1,000 (many file operations)
POSIX_WRITE_DURATION = 12 seconds
POSIX_WRITE_DENSITY = 340 KB/s
→ Recommendation: SSD (metadata-intensive, small random I/O)
```

**Build decision tree**:
```python
# Simple rule-based classifier
def recommend_storage(counters):
    mean_write_size = counters['POSIX_MEAN_WRITE_SIZE']
    seq_write_ratio = counters['POSIX_SEQ_WRITE_RATIO']
    seek_rate = counters['POSIX_SEEK_RATE']
    rw_switches = counters['POSIX_RW_SWITCHES']
    
    # Strong HDD indicators
    if seq_write_ratio > 0.8 and mean_write_size > 1_000_000:
        return "HDD"
    
    # Strong SSD indicators
    if rw_switches > 5 or seek_rate > 0.5:
        return "SSD"
    
    # Small I/O → SSD
    if mean_write_size < 64_000:
        return "SSD"
    
    # Large sequential I/O → HDD
    if mean_write_size > 1_000_000:
        return "HDD"
    
    # Default
    return "SSD"
```

---

## Usage Examples

### Single Storage Type Analysis

```bash
python3 analysis.py --input ./output/hdd/global.csv --output-dir ./output/hdd/analysis
```

Output in `./output/hdd/analysis/`

### HDD vs SSD Comparison Mode

```bash
python3 analysis.py --hdd ./output/hdd/global.csv --ssd ./output/ssd/global.csv \
    --output-dir ./analysis_output/comparison
```

Generates comparison visualizations:
- **`heatmap_hdd_ssd_interleaved.png`** — Interleaved heatmap with HDD/SSD profiles side-by-side for direct comparison
- **`bandwidth_comparison.png`** — 3-panel figure: Read BW, Write BW, Total BW (HDD orange, SSD metric-color)
- **`latency_comparison.png`** — 4-panel figure: Read, Write, Open, Close latencies (auto log-scale)
- **`performance_gains.png`** — Speedup factors (SSD/HDD for bandwidth) and latency reductions (HDD/SSD)

Also generates individual analysis for both storage types in separate subdirectories.

### Custom Thresholds

```bash
# Stricter stability requirement (CV < 0.1 instead of 0.2)
python3 analysis.py --input ./darshan_output/global.csv --cv-threshold 0.1

# Show top 20 discriminative counters instead of 10
python3 analysis.py --input ./darshan_output/global.csv --top-n 20

# Custom output directory
python3 analysis.py --input ./darshan_output/global.csv --output-dir ./my_analysis
```

### Workflow

1. **Run workloads** (5+ runs per profile recommended):
   ```bash
   python3 run_workloads.py --runs 5
   ```

2. **Analyze**:
   ```bash
   python3 analysis.py --input ./darshan_output/global.csv
   ```

3. **Inspect outputs**:
   - Look at `heatmap_all_counters.png` for overview
   - Check `bar_charts_discriminative.png` for top predictors
   - Review `statistics.csv` for detailed numbers
   - Use `pca_clustering.png` to validate workload separation

4. **Build rules**:
   - Identify stable, discriminative counters
   - Create threshold-based rules for storage placement

---

## Output Files Reference

| File | Content | Use Case |
|------|---------|----------|
| `heatmap_all_counters.png` | Normalized heatmap of all counters | Overview of workload signatures |
| `heatmap_stable_counters.png` | Heatmap of only reliable counters | Focus on trustworthy metrics |
| `bar_charts_discriminative.png` | Top N discriminators with error bars | Identify best predictors |
| `pca_clustering.png` | 2D PCA projection | Validate workload separation |
| `statistics.csv` | Full stats (mean/std/min/max/cv/count) | Detailed numerical analysis |
| `means_only.csv` | Summary (just mean values) | Quick reference table |

---

## Troubleshooting

### "No stable counters to plot"

**Cause**: All counters have CV > threshold  
**Solution**:
- Increase `--cv-threshold` (e.g., 0.3 instead of 0.2)
- Or: Run more iterations to reduce variance

### "Valid counters: 0"

**Cause**: All counters are NaN or zero  
**Solution**:
- Check that `global.csv` has actual data
- Verify Darshan logs were created correctly

### PCA plot shows no separation

**Cause**: Workloads have very similar I/O patterns  
**Solution**:
- Check if your workload definitions are actually different
- Look at discriminative counters to see what varies
- May need to design more diverse workloads

