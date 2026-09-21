# Analysis pipeline

## Darshan parsing

`parse_darshan.py` extracts POSIX, MPI-IO, and STDIO counters from Darshan logs.
It appends one aggregate row per invocation to `global.csv`; per-file output is
missing. `--log` selects one log and `--logs` accepts a comma-separated list.
At least one of `--posix`, `--mpi`, and `--stdio` is required. Aggregation uses
sum, minimum, maximum, or first value according to each counter's definition.

```bash
python3 scripts/workloads/analysis/parse_darshan.py --log logs/run.darshan --label example --posix --output-dir results/workloads/runs/example/metrics
```

Without `--output-dir`, output uses a timestamped directory under
`results/workloads/runs/`. Repeated invocations append to the selected CSV.

## Workload analysis

`analyze_darshan.py` consumes aggregated Darshan `global.csv` data, computes derived
metrics, and writes statistics and plots. Historical inputs and figures are under
`results/workloads/legacy/darshan/`. New analysis output defaults to a timestamped
directory under `results/workloads/runs/`.

## Modes

Single-file mode analyzes one aggregate CSV:

```bash
python3 scripts/workloads/analysis/analyze_darshan.py \
  --input results/workloads/legacy/darshan/ssd/global.csv \
  --output-dir results/workloads/runs/example/ssd-analysis
```

Comparison mode analyzes HDD and SSD aggregate CSVs together:

```bash
python3 scripts/workloads/analysis/analyze_darshan.py \
  --hdd results/workloads/legacy/darshan/hdd/global.csv \
  --ssd results/workloads/legacy/darshan/ssd/global.csv \
  --output-dir results/workloads/runs/example/comparison
```

## Single-file output

- `heatmap_all_counters.png`
- `heatmap_stable_counters.png`
- `bar_charts_discriminative.png`
- `pca_clustering.png`
- `statistics.csv`
- `means_only.csv`

The analysis removes all-`NaN`, all-zero, and timestamp-marker counters. It
computes profile-level statistics, coefficient of variation, discriminative
counter rankings, PCA, and derived POSIX metrics when their source counters are
available.

Derived metrics include bandwidth, operation latency, metadata latency, I/O
density, mean access size, sequential-operation ratios, and seek rate.

`--cv-threshold` controls the stable-counter threshold. `--top-n` controls the
number of discriminative counters shown.

## Comparison output

- `heatmap_hdd_ssd_interleaved.png`
- `bandwidth_comparison.png`
- `latency_comparison.png`
- `performance_gains_bandwidth.png`, when matching bandwidth metrics exist
- `performance_gains_latency.png`, when matching latency metrics exist
- `statistics.csv` and `means_only.csv` for the HDD input
- `ssd_stats/statistics.csv` and `ssd_stats/means_only.csv` for the SSD input

Comparison mode exports statistics for both inputs but does not produce the full
single-file plot set separately for each storage type.

## Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Missing functionality

The script does not implement an HDD/SSD placement classifier or validated
placement thresholds. It only exposes statistics and visual comparisons from
which such rules could be studied.
