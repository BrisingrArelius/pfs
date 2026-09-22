#!/usr/bin/env python3
"""Create readable research plots from normalized local-FIO measurements."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError as error:
    raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from error


COLORS = {"hdd": "#B86B25", "nvme": "#16858C"}
WORKLOAD_ORDER = ["seq_read", "seq_write", "rand_read_4k", "rand_write_4k", "rand_read_128k"]
WORKLOAD_LABELS = {
    "seq_read": "Sequential read\n1 MiB",
    "seq_write": "Sequential write\n1 MiB",
    "rand_read_4k": "Random read\n4 KiB",
    "rand_write_4k": "Random write\n4 KiB",
    "rand_read_128k": "Random read\n128 KiB",
}
def parse_args(argv=None):
    """Parse a normalized measurement CSV and explicit plot destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def load_rows(path):
    """Load and type-check the columns required for plotting."""
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"host", "target_id", "media", "workload", "repetition", "bw_mib_s", "iops"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"measurement CSV is empty or missing columns: {sorted(required)}")
    for row in rows:
        row["target_id"] = int(row["target_id"])
        row["repetition"] = int(row["repetition"])
        row["bw_mib_s"] = float(row["bw_mib_s"])
        row["iops"] = float(row["iops"])
        if row["media"] not in COLORS or row["workload"] not in WORKLOAD_LABELS:
            raise ValueError(f"unsupported media/workload: {row['media']}/{row['workload']}")
        if row["bw_mib_s"] <= 0 or row["iops"] <= 0:
            raise ValueError("plot metrics must be positive")
    return rows


def groups(rows, keys):
    """Group dictionaries by a tuple of named fields."""
    result = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        result.setdefault(key, []).append(row)
    return result


def style_axis(axis, ylabel, log=False):
    """Apply one restrained treatment to a vertical metric axis."""
    axis.set_ylabel(ylabel)
    if log:
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure, path):
    """Save a plot deterministically and release its matplotlib resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def use_log(values):
    """Use a logarithmic axis only when linear scale would hide a category."""
    return min(values) > 0 and max(values) / min(values) >= 20


def plot_workload_by_ost(rows, workload, output_dir):
    """Plot one access pattern's bandwidth while keeping every OST separate."""
    selected = [row for row in rows if row["workload"] == workload]
    grouped = groups(selected, ("host", "target_id", "media"))
    targets = sorted(grouped)
    labels = [str(target) for host, target, media in targets]
    all_values = []
    media_medians = {media: [] for media in COLORS}
    figure, axis = plt.subplots(figsize=(max(14, len(targets) * 0.52), 7))
    for position, target in enumerate(targets):
        values = [row["bw_mib_s"] for row in grouped[target]]
        all_values.extend(values)
        color = COLORS[target[2]]
        median = statistics.median(values)
        media_medians[target[2]].append(median)
        axis.bar(position, median, width=0.68, color=color, edgecolor="#29333A",
                 linewidth=0.6, alpha=0.85, zorder=2)
        axis.errorbar(position, median,
                      yerr=[[median - min(values)], [max(values) - median]],
                      fmt="none", ecolor="#29333A", elinewidth=1.2, capsize=3, zorder=4)
        axis.scatter([position] * len(values), values, marker="_", s=65,
                     color="#29333A", linewidths=1.1, alpha=0.8, zorder=5)
    axis.set_xticks(range(len(targets)), labels, rotation=45, ha="right")
    axis.set_xlabel("OST ID")
    axis.set_title(f"{WORKLOAD_LABELS[workload].replace(chr(10), ' ')} bandwidth by OST")
    style_axis(axis, "Bandwidth (MiB/s)", use_log(all_values))
    for media in ("hdd", "nvme"):
        if not media_medians[media]:
            continue
        average = statistics.mean(media_medians[media])
        axis.axhline(average, color=COLORS[media], linestyle=":", linewidth=1.8, zorder=1)
        axis.text(0.995, average, f"{media.upper()} average: {average:,.1f} MiB/s",
                  transform=axis.get_yaxis_transform(), color=COLORS[media],
                  ha="right", va="bottom", fontweight="bold")
    legend = [Patch(facecolor=COLORS[media], label=media.upper()) for media in ("hdd", "nvme")]
    legend.append(Line2D([], [], color="#29333A", marker="_", linestyle="-",
                         label="Repetitions / min-max"))
    axis.legend(handles=legend, frameon=False, ncol=4)
    figure.tight_layout()
    relative = f"by_access_pattern/{workload}.png"
    save_figure(figure, output_dir / relative)
    return relative


def remove_obsolete_plots(output_dir, current_files):
    """Remove only plot files named by the previous generated manifest."""
    manifest = output_dir / "plot_manifest.json"
    if not manifest.is_file():
        return
    previous = json.loads(manifest.read_text()).get("plots", [])
    root = output_dir.resolve()
    for relative in previous:
        candidate = (output_dir / relative).resolve()
        if candidate.is_relative_to(root) and candidate.suffix == ".png" and relative not in current_files:
            candidate.unlink(missing_ok=True)
    for directory in (output_dir / "by_workload", output_dir / "by_storage_type", output_dir / "by_ost"):
        try:
            directory.rmdir()
        except (FileNotFoundError, OSError):
            pass


def main(argv=None):
    """Generate both grouping views and record exactly what each plot represents."""
    args = parse_args(argv)
    try:
        rows = load_rows(args.measurements)
        files = [plot_workload_by_ost(rows, workload, args.output_dir)
                 for workload in WORKLOAD_ORDER]
        remove_obsolete_plots(args.output_dir, files)
        metadata = {
            "source": str(args.measurements.resolve()), "measurements": len(rows), "plots": files,
            "semantics": {
                "sampling_unit": "each row is one OST; OST measurements are never combined",
                "figures": "one bandwidth figure per access pattern",
                "marks": "bars show OST medians, ticks show repetitions, and whiskers show min-max",
                "media": "HDD and NVMe are identified by color only, not aggregated",
                "axes": "OSTs are horizontal; bandwidth in MiB/s is vertical",
                "references": "dotted lines show the mean of the per-OST medians for each media type",
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "plot_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Created {len(files)} plots under {args.output_dir}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot visualize results: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
