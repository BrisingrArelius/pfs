# Analysis Pipeline Documentation

This document explains how `analysis.py` works, what it analyzes, and how to interpret the results.

---

## Table of Contents

- [Overview](#overview)
- [What Gets Analyzed](#what-gets-analyzed)
- [Statistical Metrics](#statistical-metrics)
- [Visualizations](#visualizations)
- [Interpretation Guide](#interpretation-guide)
- [Usage Examples](#usage-examples)

---

## Overview

`analysis.py` performs statistical analysis on Darshan counter data to identify:
1. Which counters are **stable** (consistent across runs)
2. Which counters are **discriminative** (vary significantly across workload types)
3. Natural clustering patterns in workload behavior

**Input**: `global.csv` (one row per run, columns = Darshan counters)  
**Output**: Statistics CSVs + visualizations (heatmaps, bar charts, PCA plot)

---

## What Gets Analyzed

### Counters Included

**Raw Darshan counters**:
- **Operation counts**: `POSIX_READS`, `POSIX_WRITES`, `POSIX_SEEKS`, `POSIX_FSYNCS`, etc.
- **Byte counts**: `POSIX_BYTES_READ`, `POSIX_BYTES_WRITTEN`
- **Access patterns**: `POSIX_SEQ_READS`, `POSIX_CONSEC_WRITES`, `POSIX_RW_SWITCHES`
- **Size histograms**: `POSIX_SIZE_READ_0_100`, `POSIX_SIZE_WRITE_1M_4M`, etc.
- **Alignment**: `POSIX_MEM_ALIGNMENT`, `POSIX_FILE_ALIGNMENT`
- **Timing**: `POSIX_F_READ_TIME`, `POSIX_F_WRITE_TIME`, `POSIX_F_META_TIME`
- **Stride patterns**: Darshan tracks the 4 most common access strides:
  - `POSIX_STRIDE1_STRIDE` / `POSIX_STRIDE1_COUNT`: Most frequent stride (e.g., 65536 bytes, 9999 times)
  - `POSIX_STRIDE2_STRIDE` / `POSIX_STRIDE2_COUNT`: Second most common stride
  - `POSIX_STRIDE3_STRIDE` / `POSIX_STRIDE3_COUNT`: Third most common stride
  - `POSIX_STRIDE4_STRIDE` / `POSIX_STRIDE4_COUNT`: Fourth most common stride
  - **Why useful?** Reveals mixed access patterns (e.g., mostly sequential with periodic random seeks)

**Derived metrics** (computed automatically):
- **Duration windows**: 
  - `POSIX_OPEN_DURATION` = `END_TIMESTAMP - START_TIMESTAMP` for open operations
  - `POSIX_READ_DURATION` = Duration of all read operations (seconds)
  - `POSIX_WRITE_DURATION` = Duration of all write operations (seconds)
  - `POSIX_CLOSE_DURATION`, `POSIX_META_DURATION`, etc.

- **I/O density** (throughput):
  - `POSIX_READ_DENSITY` = `BYTES_READ / READ_DURATION` (bytes/second)
  - `POSIX_WRITE_DENSITY` = `BYTES_WRITTEN / WRITE_DURATION` (bytes/second)

- **Mean operation sizes**:
  - `POSIX_MEAN_READ_SIZE` = `BYTES_READ / READS` (bytes per read)
  - `POSIX_MEAN_WRITE_SIZE` = `BYTES_WRITTEN / WRITES` (bytes per write)

- **Access pattern ratios**:
  - `POSIX_SEQ_READ_RATIO` = `SEQ_READS / READS` (sequential read percentage)
  - `POSIX_SEQ_WRITE_RATIO` = `SEQ_WRITES / WRITES` (sequential write percentage)
  - `POSIX_SEEK_RATE` = `SEEKS / (READS + WRITES)` (seeks per I/O operation)

### Counters Excluded

**Metadata** (parser-added, not from Darshan):
- `timestamp` — When the log was parsed
- `label` — Run label (e.g., "write_heavy_run1")
- `modules_used` — Which Darshan modules were parsed

**Raw values without context** (replaced by derived metrics):
- `*_START_TIMESTAMP`, `*_END_TIMESTAMP` — Use `*_DURATION` instead
- `STRIDE2_*`, `STRIDE3_*`, `STRIDE4_*` — Use `STRIDE1_COUNT` (aggregated)

### Counters Filtered Out

During analysis, counters are automatically removed if:
- **All NaN** — Counter not present in any log
- **All zeros** — Counter never triggered (e.g., `POSIX_MMAPS` if no mmaps occurred)

---

## Statistical Metrics

For each counter, across all runs of each workload profile, we compute:

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **Mean** | `Σ(values) / n` | Typical value for this counter in this workload |
| **Std** | Standard deviation | How much the counter varies across runs |
| **Min** | Minimum value | Lowest observed value |
| **Max** | Maximum value | Highest observed value |
| **CV** | `std / mean` | **Coefficient of Variation** — relative variability |
| **Count** | Number of runs | How many data points contributed |

### Why Coefficient of Variation (CV)?

**CV measures reliability**:
- **Low CV (<0.1)**: Counter is stable across runs → reliable for classification
- **High CV (>0.2)**: Counter is noisy → unreliable, don't trust it

**Example**:
```
Scenario A: [7, 7, 7, 7, 7]     → mean=7, std=0,   CV=0.0   (perfectly stable)
Scenario B: [0, 5, 10, 2, 18]   → mean=7, std=7.2, CV=1.03  (highly variable)
```

Both have the same mean, but only Scenario A is reliable!

### Counter Classification

**Stable Counters** (CV < threshold):
- Used in the "stable counters heatmap"
- Preferred for building classification rules

**Discriminative Counters** (high variance across profiles):
- Computed as: `std(mean_per_profile) / mean(mean_per_profile)`
- High score = counter values differ significantly between workload types
- Top N are shown in bar charts

---

## Visualizations

### 1. Heatmap (All Counters)

**File**: `heatmap_all_counters.png`

**What it shows**:
- Rows = Workload profiles (write_heavy, read_heavy, mixed_rw, etc.)
- Columns = All valid Darshan counters
- Colors = Normalized values (0-1 scale)

**Normalization**:
Each counter is scaled independently to [0, 1]:
```
normalized_value = (value - min_of_counter) / (max_of_counter - min_of_counter)
```

**Why normalize?**
Counters have vastly different scales:
- `POSIX_BYTES_WRITTEN`: 0 to 655,360,000
- `POSIX_RW_SWITCHES`: 0 to 7

Without normalization, the heatmap would be dominated by large-scale counters and small counters would be invisible.

**How to read it**:
- **Red/hot** = High value for that counter in that workload
- **Blue/cool** = Low value
- **Patterns** = Visual "signature" of each workload

**Example patterns**:
- `write_heavy` has red `POSIX_WRITES`, blue `POSIX_READS`
- `mixed_rw` has red `POSIX_RW_SWITCHES`, others have blue

### 2. Heatmap (Stable Counters Only)

**File**: `heatmap_stable_counters.png`

**What it shows**:
Same as above, but **only counters with CV < threshold** (default: 0.2)

**Purpose**:
Focus on reliable metrics. If building a classifier, use these counters — they're consistent across runs.

**Color map**: Uses blue palette (YlGnBu) to distinguish from the all-counters heatmap

### 3. Bar Charts (Discriminative Counters)

**File**: `bar_charts_discriminative.png`

**What it shows**:
- Top N counters (default: 10) that best distinguish workload types
- Each subplot = one counter
- Bars = Mean value per profile
- Error bars = ± Standard deviation

**How to read it**:
- **Large differences between profiles** = Good discriminator
- **Small error bars** = Stable (low std)
- **Large error bars** = Noisy (high std)

**Example**:
```
POSIX_RW_SWITCHES:
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

### Basic Usage

```bash
python3 analysis.py --input ./darshan_output/global.csv
```

Output in `./analysis_output/`

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

