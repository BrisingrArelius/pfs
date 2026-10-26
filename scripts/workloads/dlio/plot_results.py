#!/usr/bin/env python3
"""Plot validated DLIO ResNet pool results from parse_results.py tables."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


POOLS = (1, 3, 6)
LABELS = {1: "Default (colva4)", 3: "HDD (colva1–3)", 6: "SSD (colva1–3)"}
COLORS = {1: "#335F87", 3: "#D08734", 6: "#33836B"}
REPETITIONS = (1, 2, 3)
EPOCHS = (1, 2, 3, 4, 5)


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def require_complete(data, expected, key):
    observed = [key(item) for item in data]
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(observed):
        raise ValueError("Incomplete, duplicated or unexpected parsed rows")


def summary(values):
    if len(values) != 3:
        raise ValueError(f"Expected three independent repetitions, found {len(values)}")
    return statistics.mean(values), statistics.stdev(values)


def decorate(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.23)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training(epochs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    columns = (("nominal_training_MiB_s_DLIO", "DLIO-modeled training I/O (MiB/s)"),
               ("training_block_wall_s", "Training block wall time (s)"))
    for ax, (column, ylabel) in zip(axes, columns):
        for pool in POOLS:
            for repetition in REPETITIONS:
                points = [next(float(row[column]) for row in epochs if
                               int(row["pool"]) == pool and int(row["repetition"]) == repetition
                               and int(row["epoch"]) == epoch) for epoch in EPOCHS]
                ax.plot(EPOCHS, points, color=COLORS[pool], alpha=.19, linewidth=1)
            means, deviations = [], []
            for epoch in EPOCHS:
                values = [float(row[column]) for row in epochs if int(row["pool"]) == pool
                          and int(row["epoch"]) == epoch]
                mean, sd = summary(values)
                means.append(mean)
                deviations.append(sd)
            ax.errorbar(EPOCHS, means, yerr=deviations, fmt="o-", capsize=3,
                        linewidth=2.1, markersize=5, color=COLORS[pool], label=LABELS[pool])
        decorate(ax, ylabel)
        ax.set_xticks(EPOCHS)
        ax.set_xlabel("Training epoch")
    anomaly = next(row for row in epochs if (int(row["pool"]), int(row["repetition"]),
                        int(row["epoch"])) == (6, 1, 2))
    axes[0].scatter([2], [float(anomaly["nominal_training_MiB_s_DLIO"])],
                    marker="*", s=125, facecolor="#B23C48", edgecolor="white", zorder=5)
    axes[0].annotate("SSD rep 1: 60 s gap before checkpoint\nDLIO rate vs block time differs by 10.7%",
                     xy=(2, float(anomaly["nominal_training_MiB_s_DLIO"])),
                     xytext=(2.35, 236), fontsize=8.5, color="#772D3A",
                     arrowprops={"arrowstyle": "->", "color": "#B23C48"})
    axes[0].set_ylim(bottom=225)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("ResNet-50 training by epoch · 8 ranks, three runs per pool", fontweight="bold")
    fig.text(.5, -.01, "Lines: run-level observations; bold: mean ± sample SD. "
             "Modeled rate is not measured device traffic.", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, out / "training_epochs.png")


def plot_checkpoints(checkpoints, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, column, ylabel in (
        (axes[0], "rank0_save_s_DLIO", "Rank-0 save duration (s)"),
        (axes[1], "rank0_MiB_s_no_fsync", "Rank-0 save-completion rate (MiB/s)"),
    ):
        for offset, pool in zip((-.19, 0, .19), POOLS):
            for epoch in (2, 4):
                values = [float(row[column]) for row in checkpoints
                          if int(row["pool"]) == pool and int(row["epoch"]) == epoch]
                mean, sd = summary(values)
                x = epoch + offset
                ax.errorbar(x, mean, yerr=sd, fmt="o", markersize=8, capsize=4,
                            color=COLORS[pool], label=LABELS[pool] if epoch == 2 else None)
                ax.scatter([x - .075, x, x + .075], values, marker="o", s=15,
                           color=COLORS[pool], alpha=.68, zorder=4)
        decorate(ax, ylabel)
        ax.set_xticks((2, 4), ("After epoch 2", "After epoch 4"))
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Checkpoint saves · one ~100-MB rank-0 file per event", fontweight="bold")
    fig.text(.5, -.01, "Mean ± sample SD and three run values per pool. No fsync; "
             "not eight-rank aggregate or durable throughput.", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, out / "checkpoint_rank0.png")


def plot_run_order(runs, out):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for pool in POOLS:
        values = [float(next(row["training_mean_MiB_s_DLIO"] for row in runs if
                        int(row["pool"]) == pool and int(row["repetition"]) == repetition))
                  for repetition in REPETITIONS]
        ax.plot(REPETITIONS, values, "o-", linewidth=2, markersize=6,
                color=COLORS[pool], label=LABELS[pool])
    decorate(ax, "DLIO-modeled mean training I/O (MiB/s)")
    ax.set_xticks(REPETITIONS)
    ax.set_xlabel("Repetition · each block ran Default → HDD → SSD")
    ax.legend(frameon=False)
    ax.set_title("Run-to-run training comparison", fontweight="bold")
    fig.tight_layout()
    save(fig, out / "training_repetitions.png")


def plot_load_tail(rank_epochs, out):
    # Each run contributes one mean of its eight rank-level p95 values, so the
    # error bar reflects three runs, not 24 incorrectly independent ranks.
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for pool in POOLS:
        run_values = {}
        for repetition in REPETITIONS:
            run_values[repetition] = [statistics.mean(float(row["load_p95_s"])
                for row in rank_epochs if int(row["pool"]) == pool
                and int(row["repetition"]) == repetition and int(row["epoch"]) == epoch)
                for epoch in EPOCHS]
            ax.plot(EPOCHS, run_values[repetition], color=COLORS[pool], alpha=.18)
        means, errors = zip(*(summary([run_values[rep][epoch - 1] for rep in REPETITIONS])
                              for epoch in EPOCHS))
        ax.errorbar(EPOCHS, means, yerr=errors, fmt="o-", color=COLORS[pool],
                    markersize=5, linewidth=2, capsize=3, label=LABELS[pool])
    decorate(ax, "Mean per-run rank p95 load-step time (s)")
    ax.set_xticks(EPOCHS)
    ax.set_xlabel("Training epoch")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Application-level loading tail · not POSIX read latency", fontweight="bold")
    fig.tight_layout()
    save(fig, out / "rank_load_p95.png")


def plot_placement(placements, out):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.3), sharey=False)
    for ax, pool in zip(axes, POOLS):
        records = sorted((row for row in placements if int(row["pool"]) == pool),
                         key=lambda row: int(row["ost"]))
        counts = [int(row["dataset_files_assigned"]) for row in records]
        ax.bar([row["ost"] for row in records], counts, color=COLORS[pool], alpha=.83)
        ax.axhline(1024 * 4 / len(records), color="#526278", linestyle="--",
                   linewidth=1)
        decorate(ax, "File assignments")
        ax.set_title(LABELS[pool], loc="left", fontsize=10)
    axes[-1].set_xlabel("BeeGFS storage target ID")
    fig.suptitle("ResNet input placement · each of 1,024 files selects four OSTs",
                 fontweight="bold")
    fig.text(.5, -.01, "Dashed line: even assignment. Counts are file membership, "
             "not allocated bytes or device traffic.",
             ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, out / "dataset_ost_assignments.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Copied runs-... directory after parse_results.py")
    args = parser.parse_args()
    analysis = args.run_directory / "analysis"
    epochs = rows(analysis / "epochs.csv")
    ranks = rows(analysis / "rank_epochs.csv")
    checkpoints = rows(analysis / "checkpoints.csv")
    runs = rows(analysis / "runs.csv")
    placement = rows(analysis / "dataset_ost_counts.csv")
    expected_run = {(p, r) for p in POOLS for r in REPETITIONS}
    expected_epoch = {(p, r, e) for p, r in expected_run for e in EPOCHS}
    require_complete(runs, expected_run,
                     lambda row: (int(row["pool"]), int(row["repetition"])))
    require_complete(epochs, expected_epoch,
                     lambda row: (int(row["pool"]), int(row["repetition"]), int(row["epoch"])))
    require_complete(ranks, {(p, r, e, rank) for p, r, e in expected_epoch for rank in range(8)},
                     lambda row: (int(row["pool"]), int(row["repetition"]), int(row["epoch"]), int(row["rank"])))
    require_complete(checkpoints, {(p, r, e) for p, r in expected_run for e in (2, 4)},
                     lambda row: (int(row["pool"]), int(row["repetition"]), int(row["epoch"])))
    if not placement or set(int(row["pool"]) for row in placement) != set(POOLS):
        raise ValueError("Incomplete dataset placement table")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "figure.facecolor": "white"})
    out = analysis / "plots"
    out.mkdir(exist_ok=True)
    plot_training(epochs, out)
    plot_checkpoints(checkpoints, out)
    plot_run_order(runs, out)
    plot_load_tail(ranks, out)
    plot_placement(placement, out)
    captions = {
        "training_epochs.png": "Per-epoch DLIO modeled rate and measured training-block duration; the timing anomaly is marked.",
        "checkpoint_rank0.png": "Rank-0 checkpoint save completion only; no fsync and no eight-rank aggregate rate.",
        "training_repetitions.png": "DLIO training summary for each serial repetition and pool.",
        "rank_load_p95.png": "Per-rank application-step load p95, summarized first within each run.",
        "dataset_ost_assignments.png": "Number of input files assigned to each OST; not bytes or traffic.",
    }
    (out / "README.md").write_text("# ResNet DLIO figures\n\n" + "\n\n".join(
        f"## {name.removesuffix('.png').replace('_', ' ').title()}\n\n![{caption}]({name})\n\n{caption}"
        for name, caption in captions.items()) + "\n")
    (out / "plot_manifest.json").write_text(json.dumps({
        "source": str(args.run_directory), "inputs": ["epochs.csv", "rank_epochs.csv",
        "checkpoints.csv", "runs.csv", "dataset_ost_counts.csv"], "figures": captions,
    }, indent=2) + "\n")
    print(f"Wrote {len(captions)} plots to {out}")


if __name__ == "__main__":
    main()
