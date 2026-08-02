"""
Read the per-file rate CSV produced by parse_logs.py, show the log10-scale
distribution of rate_active / rate_wallclock (ops/sec), and report three candidate
seldom/frequent split methods (median, tercile, quartile) for each metric.

Standalone script, parallel to but independent of CONTIG_TESTING_CLAUDE/analyze_distribution.py.

Usage:
    python3 analyze_distribution.py <in_csv> <out_dir> [--min-ops N]
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BAR_COLOR = "#3B6FA0"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#333333"
SPLIT_COLORS = {"median": "#C0392B", "tercile": "#8E44AD", "quartile": "#1E8449"}

METRICS = ["rate_active", "rate_wallclock"]

# log10-scale bucket edges in ops/sec: <1, 1-10, 10-100, ..., 100K-1M, >1M
LOG_EDGES = [0, 1, 2, 3, 4, 5, 6]  # log10(rate) edges; below 0 and above 6 are open-ended
LOG_LABELS = ["<1", "1-10", "10-100", "100-1K", "1K-10K", "10K-100K", "100K-1M", ">1M"]


def load_rates(csv_path, min_ops=0):
    out = {m: [] for m in METRICS}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["total_ops"]) < min_ops:
                continue
            for m in METRICS:
                v = row[m]
                if v == "":
                    continue
                out[m].append(float(v))
    return out


def log_bucket_index(rate):
    if rate <= 0:
        return 0  # shouldn't happen (guarded upstream), fold into <1 bucket defensively
    exp = math.log10(rate)
    idx = int(math.floor(exp)) + 1  # exp<0 -> bucket 0 ("<1"), 0<=exp<1 -> bucket 1 ("1-10"), etc.
    if idx < 0:
        idx = 0
    if idx >= len(LOG_LABELS):
        idx = len(LOG_LABELS) - 1
    return idx


def percentile(sorted_vals, pct):
    """Nearest-rank percentile, 0<=pct<=100."""
    n = len(sorted_vals)
    if n == 0:
        return None
    k = max(0, min(n - 1, int(round(pct / 100.0 * (n - 1)))))
    return sorted_vals[k]


def compute_splits(sorted_vals):
    return {
        "median": {"cuts": [percentile(sorted_vals, 50)],
                   "bands": ["seldom (<=p50)", "frequent (>p50)"]},
        "tercile": {"cuts": [percentile(sorted_vals, 33.33), percentile(sorted_vals, 66.67)],
                    "bands": ["seldom (<=p33)", "moderate (p33-p67)", "frequent (>p67)"]},
        "quartile": {"cuts": [percentile(sorted_vals, 25), percentile(sorted_vals, 75)],
                     "bands": ["seldom (<=p25)", "unclassified (p25-p75)", "frequent (>p75)"]},
    }


def band_counts(sorted_vals, cuts):
    n = len(sorted_vals)
    counts = []
    prev_idx = 0
    for c in cuts:
        idx = 0
        while idx < n and sorted_vals[idx] <= c:
            idx += 1
        counts.append(idx - prev_idx)
        prev_idx = idx
    counts.append(n - prev_idx)
    return counts


def plot_log_histogram(rates, metric, out_dir, suffix, n_total, min_ops):
    counts = [0] * len(LOG_LABELS)
    for r in rates:
        counts[log_bucket_index(r)] += 1
    pct = [100.0 * c / n_total for c in counts]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    x = range(len(LOG_LABELS))
    bars = ax.bar(x, pct, color=BAR_COLOR, width=0.75, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(LOG_LABELS, fontsize=9, color=TEXT_COLOR)
    ax.set_ylabel("% of files", fontsize=11, color=TEXT_COLOR)
    ax.set_xlabel(f"{metric} (ops/sec, log10-scale buckets)", fontsize=11, color=TEXT_COLOR)
    title = f"Distribution of per-file {metric}  (n = {n_total} files"
    title += f", total_ops >= {min_ops})" if min_ops > 0 else ")"
    ax.set_title(title, fontsize=13, color=TEXT_COLOR, pad=14)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#B0B0B0")
    for rect, p in zip(bars, pct):
        if p >= 1.0:
            ax.annotate(f"{p:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=TEXT_COLOR)
    fig.tight_layout()
    png_path = out_dir / f"{metric}_distribution{suffix}.png"
    fig.savefig(png_path)
    plt.close(fig)
    return png_path, counts, pct


def plot_split_overlay(rates, metric, method, cuts, out_dir, suffix, n_total, min_ops):
    counts = [0] * len(LOG_LABELS)
    for r in rates:
        counts[log_bucket_index(r)] += 1
    pct = [100.0 * c / n_total for c in counts]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    x = range(len(LOG_LABELS))
    ax.bar(x, pct, color=BAR_COLOR, width=0.75, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(LOG_LABELS, fontsize=9, color=TEXT_COLOR)
    ax.set_ylabel("% of files", fontsize=11, color=TEXT_COLOR)
    ax.set_xlabel(f"{metric} (ops/sec, log10-scale buckets)", fontsize=11, color=TEXT_COLOR)
    title = f"{metric} -- {method} split  (n = {n_total} files"
    title += f", total_ops >= {min_ops})" if min_ops > 0 else ")"
    ax.set_title(title, fontsize=13, color=TEXT_COLOR, pad=14)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#B0B0B0")

    color = SPLIT_COLORS[method]
    for c in cuts:
        if c is None or c <= 0:
            continue
        # map the raw cut value onto the continuous log-x position
        xpos = math.log10(c) + 1
        ax.axvline(xpos, color=color, linestyle="--", linewidth=1.6, zorder=4)
        ax.annotate(f"{c:.1f} ops/s", xy=(xpos, ax.get_ylim()[1]), xytext=(3, -3),
                    textcoords="offset points", fontsize=8, color=color, va="top")

    fig.tight_layout()
    png_path = out_dir / f"{metric}_{method}_split{suffix}.png"
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--min-ops", type=int, default=0,
                     help="exclude files with fewer than this many total_ops before "
                          "computing rate distributions/splits")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_minops{args.min_ops}" if args.min_ops > 0 else ""

    rates_by_metric = load_rates(args.in_csv, min_ops=args.min_ops)

    split_rows = []
    for metric in METRICS:
        rates = rates_by_metric[metric]
        n_total = len(rates)
        print(f"\n=== {metric} (n={n_total}, total_ops>={args.min_ops}) ===")
        if n_total == 0:
            print("  no rows with a defined rate for this metric/floor -- skipping")
            continue

        png_path, counts, pct = plot_log_histogram(rates, metric, out_dir, suffix, n_total, args.min_ops)
        print(f"  Wrote raw distribution plot: {png_path}")
        print(f"  {'Bucket':<10}{'# files':>10}{'% of files':>14}")
        for label, c, p in zip(LOG_LABELS, counts, pct):
            print(f"  {label:<10}{c:>10}{p:>13.2f}%")

        sorted_rates = sorted(rates)
        mean_r = sum(rates) / n_total
        median_r = percentile(sorted_rates, 50)
        print(f"  Mean:   {mean_r:.2f} ops/sec")
        print(f"  Median: {median_r:.2f} ops/sec")

        splits = compute_splits(sorted_rates)
        for method, spec in splits.items():
            cuts = spec["cuts"]
            counts_b = band_counts(sorted_rates, cuts)
            bands = spec["bands"]
            print(f"\n  -- {method} split --")
            for band_name, cnt, cut in zip(bands, counts_b, list(cuts) + [None]):
                pct_b = 100.0 * cnt / n_total
                print(f"    {band_name:<28}{cnt:>10} files  ({pct_b:5.2f}%)")
            print(f"    cut points: {[f'{c:.2f}' for c in cuts]} ops/sec")

            plot_p = plot_split_overlay(rates, metric, method, cuts, out_dir, suffix, n_total, args.min_ops)
            print(f"    Wrote split overlay plot: {plot_p}")

            row = {
                "metric": metric, "split_method": method, "n_total": n_total,
                "cut_points_ops_per_sec": ";".join(f"{c:.4f}" for c in cuts),
            }
            for band_name, cnt in zip(bands, counts_b):
                row[f"band__{band_name}"] = cnt
            split_rows.append(row)

    if split_rows:
        all_band_keys = sorted({k for row in split_rows for k in row if k.startswith("band__")})
        fieldnames = ["metric", "split_method", "n_total", "cut_points_ops_per_sec"] + all_band_keys
        splits_csv = out_dir / f"rate_splits{suffix}.csv"
        with open(splits_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"\nWrote combined split summary: {splits_csv}")


if __name__ == "__main__":
    main()
