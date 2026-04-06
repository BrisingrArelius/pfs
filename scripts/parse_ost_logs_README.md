# parse_ost_logs.py

Parses `scripts/ost_space_and_usage.log` and generates OST usage heatmaps for workload runs.

## Usage

```bash
python3 scripts/parse_ost_logs.py
```

To specify a custom log or output file:

```bash
python3 scripts/parse_ost_logs.py --log scripts/ost_space_and_usage.log --output output/ost_heatmap.png
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

The default output path is `output/ost_heatmap.png`.
