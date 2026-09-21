# Microbenchmark analysis utilities

## `parse_du.py`

Groups saved `du -h` entries by displayed suffix (`G`, `M`, `K`, or other).
It sorts the G and M groups numerically, prints all G entries and up to 15 M
entries, and counts the remaining entries. It does not apply numeric size
thresholds or determine whether paths are directories.

```bash
python3 scripts/microbenchmarks/analysis/parse_du.py <du_output.txt>
```

## `generate_heatmap.py`

Recursively queries files in a directory with `beegfs-ctl --getentryinfo` and
plots file-to-target membership. It requires a mounted BeeGFS namespace.
`--output` selects the PNG destination; the default is a new microbenchmark
run directory.

```bash
python3 scripts/microbenchmarks/analysis/generate_heatmap.py <beegfs-directory>
```

## `parse_ost_logs.py`

Parses historical OST usage logs and generates heatmaps for workload runs.
The default input is the preserved June 28 log. New heatmaps default to a new
directory under `results/microbenchmarks/runs/`; explicit paths are also accepted.

## Usage

```bash
python3 scripts/microbenchmarks/analysis/parse_ost_logs.py --log results/microbenchmarks/legacy/placement/from-scripts/ost_space_and_usage.log --output results/microbenchmarks/runs/example/plots/ost_heatmap.png
```

To specify a custom log or output file:

```bash
python3 scripts/microbenchmarks/analysis/parse_ost_logs.py --log path/to/input.log --output results/microbenchmarks/runs/example/plots/ost_heatmap.png
```

## Options

- `--log` — path to the OST space log file
- `--output` — path to save the generated heatmap PNG
- `--active-only` — omit OSTs that show zero bytes written across all profiles
- `--all-nodes` — include all nodes, even offline or inactive storage nodes

## What it does

The script:
- reads before/after OST space snapshots for each workload run
- computes per-OST bytes written by each workload
- aggregates and pivots results into a profile-by-OST matrix
- draws a heatmap showing average GiB written to each OST

## Output

The default output is a timestamped run directory. Historical logs and plots are
never used as output destinations.
