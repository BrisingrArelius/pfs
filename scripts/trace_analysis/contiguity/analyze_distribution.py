"""
Read the per-file contig_ratio CSV produced by parse_logs.py, bin contig_ratio
into 5%-wide buckets, print the distribution table, and plot it.

Usage:
    python3 analyze_distribution.py <in_csv> <out_dir>
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BAR_COLOR = "#3B6FA0"      # single hue, sequential-magnitude use (one series)
GRID_COLOR = "#D9D9D9"     # recessive gridlines
TEXT_COLOR = "#333333"


def load_ratios(csv_path, min_ops=0):
    ratios = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["total_ops"]) < min_ops:
                continue
            ratios.append(float(row["contig_ratio"]))
    return ratios


def bucket_index(ratio, n_bins=20):
    # ratio in [0, 1]; bucket width = 0.05. ratio == 1.0 goes in the last bucket.
    idx = int(ratio * n_bins)
    if idx >= n_bins:
        idx = n_bins - 1
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--bin-width-pct", type=int, default=5)
    ap.add_argument("--min-ops", type=int, default=0,
                     help="exclude files with fewer than this many total_ops "
                          "(low op counts quantize the ratio, e.g. 1 op -> always 0%%, "
                          "2 ops -> only 0%% or 50%% possible)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios = load_ratios(args.in_csv, min_ops=args.min_ops)
    n_total = len(ratios)
    if n_total == 0:
        raise SystemExit("No rows found in input CSV (after --min-ops filter)")

    n_bins = 100 // args.bin_width_pct
    counts = [0] * n_bins
    for r in ratios:
        counts[bucket_index(r, n_bins)] += 1

    labels = [f"{i*args.bin_width_pct}-{(i+1)*args.bin_width_pct}%" for i in range(n_bins)]
    pct_of_files = [100.0 * c / n_total for c in counts]

    # ---- print table ----
    print(f"Total files (records) analyzed: {n_total}\n")
    print(f"{'Bucket':<12}{'# files':>10}{'% of files':>14}")
    for label, c, p in zip(labels, counts, pct_of_files):
        print(f"{label:<12}{c:>10}{p:>13.2f}%")

    # basic stats
    ratios_sorted = sorted(ratios)
    mean_r = sum(ratios) / n_total
    median_r = ratios_sorted[n_total // 2]
    print(f"\nMean contiguous ratio:   {mean_r*100:.2f}%")
    print(f"Median contiguous ratio: {median_r*100:.2f}%")

    suffix = f"_minops{args.min_ops}" if args.min_ops > 0 else ""
    stats_path = out_dir / f"distribution_table{suffix}.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket", "num_files", "pct_of_files"])
        for label, c, p in zip(labels, counts, pct_of_files):
            writer.writerow([label, c, f"{p:.4f}"])
    print(f"\nWrote distribution table: {stats_path}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    x = range(n_bins)
    bars = ax.bar(x, pct_of_files, color=BAR_COLOR, width=0.75, zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8, color=TEXT_COLOR)
    ax.set_ylabel("% of files", fontsize=11, color=TEXT_COLOR)
    ax.set_xlabel("Contiguous ratio  =  (CONSEC_READS + CONSEC_WRITES) / total_ops", fontsize=11, color=TEXT_COLOR)
    title = f"Distribution of per-file contiguous I/O ratio  (n = {n_total} files"
    title += f", total_ops ≥ {args.min_ops})" if args.min_ops > 0 else ")"
    ax.set_title(title, fontsize=13, color=TEXT_COLOR, pad=14)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#B0B0B0")

    # direct labels on bars that carry >=1% of files, to keep it readable
    for rect, pct in zip(bars, pct_of_files):
        if pct >= 1.0:
            ax.annotate(
                f"{pct:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color=TEXT_COLOR,
            )

    fig.tight_layout()
    png_path = out_dir / f"contig_ratio_distribution{suffix}.png"
    fig.savefig(png_path)
    print(f"Wrote plot: {png_path}")


if __name__ == "__main__":
    main()
