#!/usr/bin/env python3
"""Plot copied three-run throttled-SSD DLIO sweeps; accept pools as they finish."""

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from pool_common import FAST, RANKS, WORKLOADS


ROOT = Path(__file__).resolve().parents[4]
COLORS = {14: "#335F87", 15: "#497E9B", 16: "#9B8A51", 17: "#D08734", 18: "#B44F4F"}
EPOCHS = (1, 2, 3, 4, 5)
PLOTS = ("pool_performance.png", "training_throughput_by_epoch.png",
         "checkpoint_write_throughput.png", "training_throughput_by_run.png",
         "data_loading_time.png", "training_throughput_inferred.png")


def label(pool):
    return f"{len(FAST[pool])} fast / {4 - len(FAST[pool])} slow"


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    if len(values) != 3:
        raise ValueError(f"Expected three independent runs, found {len(values)}")
    return statistics.mean(values), statistics.stdev(values)


def p95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def parse_sweep(root, workload):
    profile = WORKLOADS[workload]
    condition = json.loads((root / "condition.json").read_text())
    pool = condition["pool"]
    if (pool not in FAST or condition["profile"] != profile.profile
            or set(condition["fast"]) != set(FAST[pool])):
        raise ValueError(f"Wrong workload or SSD condition: {root}")
    metrics = read_csv(root / "metrics.csv")
    if len(metrics) != 24:
        raise ValueError(f"Expected 24 runner metrics (3 runs × 8), got {len(metrics)}: {root}")
    indexed = {}
    for row in metrics:
        key = (int(row["repetition"]), row["phase"], row["epoch"])
        if int(row["pool"]) != pool or key in indexed:
            raise ValueError(f"Duplicate or wrong-pool metric: {root}: {key}")
        indexed[key] = row
    runs, epochs, checkpoints, loading = [], [], [], []
    for repetition in (1, 2, 3):
        directory = root / f"rep{repetition}-pool{pool}"
        config = yaml.safe_load((directory / ".hydra/config.yaml").read_text())["workload"]
        summary = json.loads((directory / "summary.json").read_text())
        native = json.loads((directory / "per_epoch_stats.json").read_text())
        if (config["dataset"]["num_files_train"] != profile.files
                or config["reader"]["batch_size"] != profile.batch_size
                or config["train"]["epochs"] != 5 or sorted(map(int, native)) != list(EPOCHS)):
            raise ValueError(f"Incomplete or wrong workload: {directory}")
        record_bytes = float(config["dataset"]["record_length"])
        batch_size = int(config["reader"]["batch_size"])
        steps = (profile.files * config["dataset"]["num_samples_per_file"]
                 // (batch_size * RANKS))
        rank_data = [json.loads((directory / f"{rank}_output.json").read_text())
                     for rank in range(RANKS)]
        input_bytes = input_seconds = save_bytes = save_seconds = 0.0
        native_rates = summary["metric"]["train_throughput_samples_per_second"]
        if len(native_rates) != len(EPOCHS) or not any(value > 0 for value in native_rates):
            raise ValueError(f"Missing positive DLIO throughput: {directory}")
        for epoch in EPOCHS:
            native_epoch = native[str(epoch)]
            block = native_epoch.get("block1", {})
            phase = "training_input_block" if "duration" in block else "training_epoch_wall_only"
            row = indexed[repetition, phase, str(epoch)]
            samples = [data[str(epoch)]["load"]["block1"] for data in rank_data]
            if any(len(values) != steps for values in samples):
                raise ValueError(f"Missing rank input steps: {directory}, epoch {epoch}")
            if workload == "unet":
                seconds = float(row["duration_s"])
                if (phase == "training_epoch_wall_only" and row["throughput"]
                        or phase == "training_input_block" and epoch not in (2, 4)):
                    raise ValueError(f"Unexpected U-Net native rate: {directory}, epoch {epoch}")
                byte_count = steps * batch_size * RANKS * record_bytes
                rate = byte_count / seconds / 1048576
                input_bytes += byte_count
                input_seconds += seconds
            else:
                if phase != "training_input_block":
                    raise ValueError(f"Missing ResNet native rate: {directory}, epoch {epoch}")
                rate = float(row["throughput"]) * record_bytes / 1048576
            epochs.append({"pool": pool, "repetition": repetition, "epoch": epoch,
                           "configured_record_bytes": record_bytes,
                           "processed_samples": steps * batch_size * RANKS,
                           "estimated_logical_bytes": steps * batch_size * RANKS * record_bytes,
                           "timing_phase": phase,
                           "timing_duration_s": float(row["duration_s"]),
                           "epoch_completion_s": float(native_epoch["duration"]),
                           "native_samples_s_reported": native_rates[epoch - 1],
                           "native_samples_s": native_rates[epoch - 1] if native_rates[epoch - 1] > 0 else math.nan,
                           "training_input_MiB_s": rate,
                           "rate_kind": "estimated from batches and time" if workload == "unet"
                           else "DLIO modeled samples/s × record bytes"})
            loading.append({"pool": pool, "repetition": repetition, "epoch": epoch,
                            "rank_load_p95_s": statistics.mean(p95(values) for values in samples)})
            if epoch in (2, 4):
                checkpoint = indexed[repetition, "checkpoint_save_rank0", str(epoch)]
                size = int(checkpoint["bytes_written"])
                duration = float(checkpoint["duration_s"])
                if size <= 0 or duration <= 0:
                    raise ValueError(f"Invalid checkpoint: {directory}, epoch {epoch}")
                save_bytes += size
                save_seconds += duration
                checkpoints.append({"pool": pool, "repetition": repetition, "epoch": epoch,
                                    "checkpoint_rank0_MiB_s_no_fsync": size / duration / 1048576})
        summary_phase = "training_summary_incomplete" if workload == "unet" else "training_summary"
        if (repetition, summary_phase, "") not in indexed:
            raise ValueError(f"Missing DLIO run summary: {directory}")
        elapsed = (datetime.fromisoformat(summary["end"])
                   - datetime.fromisoformat(summary["start"])).total_seconds()
        if elapsed <= 0:
            raise ValueError(f"Invalid full-run duration: {directory}")
        total_samples = steps * batch_size * RANKS * len(EPOCHS)
        runs.append({"pool": pool, "repetition": repetition, "source": str(directory),
                     "total_processed_samples": total_samples,
                     "whole_run_seconds": elapsed,
                     "end_to_end_samples_s": total_samples / elapsed,
                     "native_samples_s": statistics.mean(value for value in native_rates if value > 0),
                     "native_epochs_used": ",".join(str(epoch) for epoch, value in zip(EPOCHS, native_rates) if value > 0),
                     "native_summary_samples_s_reported": summary["metric"]["train_throughput_mean_samples_per_second"],
                     "training_input_MiB_s": input_bytes / input_seconds / 1048576
                     if workload == "unet" else float(summary["metric"]["train_io_mean_MB_per_second"]),
                     "checkpoint_rank0_MiB_s_no_fsync": save_bytes / save_seconds / 1048576,
                     "whole_run_minutes": elapsed / 60})
    return pool, runs, epochs, checkpoints, loading


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, output, name, bottom=0):
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(output / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary(workload, pools, groups, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, metric, title, unit in zip(axes,
            ("end_to_end_samples_s", "checkpoint_rank0_MiB_s_no_fsync"),
            ("End-to-end throughput", "Checkpoint write throughput"),
            ("Samples/s · total processed / full run time", "Rank-0 MiB/s; no fsync")):
        records = {row["pool"]: row for row in groups if row["metric"] == metric}
        ceiling = max(records[pool]["mean"] + records[pool]["stddev"] for pool in pools) * 1.35
        for position, pool in enumerate(pools):
            row = records[pool]
            ax.barh(position, row["mean"], color=COLORS[pool], alpha=.85)
            ax.errorbar(row["mean"], position, xerr=row["stddev"], fmt="none",
                        color="#243447", capsize=4)
            ax.text(row["mean"] - ceiling * .015, position,
                    f"{row['mean']:.2f} ± {row['stddev']:.2f}", ha="right", va="center",
                    fontsize=8.5, bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none"})
        ax.set_yticks(range(len(pools)), [label(pool) for pool in pools])
        ax.invert_yaxis()
        ax.set_xlim(0, ceiling)
        ax.set_xlabel(unit)
        ax.set_title(title, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{workload}: configured SSD speed mix"
                 " · mean ± SD (3 runs)", fontweight="bold")
    fig.text(.5, .02, "End-to-end rate is derived from processed sample count and DLIO start/end timestamps;\n"
             "includes all epochs, checkpointing and overhead within those timestamps, not dataset staging or cache preparation.",
             ha="center", fontsize=8)
    save(fig, output, "pool_performance.png", bottom=.12)


def plot_epochs(workload, pools, rows, field, ylabel, title, filename, output, inferred=False):
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    for pool in pools:
        for repetition in (1, 2, 3):
            values = [next(row[field] for row in rows if row["pool"] == pool
                           and row["repetition"] == repetition and row["epoch"] == epoch)
                      for epoch in EPOCHS]
            ax.plot(EPOCHS, values, color=COLORS[pool], alpha=.2, linewidth=1,
                    linestyle=":" if inferred else "-")
        means, errors = zip(*(mean_sd([row[field] for row in rows
                            if row["pool"] == pool and row["epoch"] == epoch])
                            if all(math.isfinite(row[field]) for row in rows
                                   if row["pool"] == pool and row["epoch"] == epoch)
                            else (math.nan, math.nan)
                               for epoch in EPOCHS))
        ax.plot(EPOCHS, means, "o:" if inferred else "o-",
                color=COLORS[pool], linewidth=2, label=label(pool))
        measured = [(epoch, mean, error) for epoch, mean, error in zip(EPOCHS, means, errors)
                    if math.isfinite(mean)]
        if measured:
            xs, ys, deviations = zip(*measured)
            ax.errorbar(xs, ys, yerr=deviations, fmt="none", capsize=3, color=COLORS[pool])
    ax.set_xticks(EPOCHS)
    ax.set_xlabel("Epoch")
    if field == "native_samples_s":
        ax.set_xlabel("Epoch · zero-reported DLIO rates omitted; gaps are not interpolated")
    style(ax, ylabel)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{workload}: {title}", fontweight="bold")
    save(fig, output, filename)


def plot_checkpoints(workload, pools, rows, output):
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    offsets = [0] if len(pools) == 1 else [(.35 * i / (len(pools) - 1) - .175) for i in range(len(pools))]
    for offset, pool in zip(offsets, pools):
        for epoch in (2, 4):
            values = [row["checkpoint_rank0_MiB_s_no_fsync"] for row in rows
                      if row["pool"] == pool and row["epoch"] == epoch]
            avg, sd = mean_sd(values)
            ax.errorbar(epoch + offset, avg, yerr=sd, fmt="o", markersize=7,
                        capsize=4, color=COLORS[pool], label=label(pool) if epoch == 2 else None)
            ax.scatter([epoch + offset + (i-1)*.03 for i in range(3)], values,
                       color=COLORS[pool], s=15, alpha=.6)
    ax.set_xticks((2, 4), ("After epoch 2", "After epoch 4"))
    style(ax, "Checkpoint write throughput (MiB/s, rank 0; no fsync)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{workload}: checkpoint write throughput", fontweight="bold")
    save(fig, output, "checkpoint_write_throughput.png")


def plot_runs(workload, pools, rows, output):
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    for pool in pools:
        values = [next(row["end_to_end_samples_s"] for row in rows
                       if row["pool"] == pool and row["repetition"] == repetition)
                  for repetition in (1, 2, 3)]
        ax.plot((1, 2, 3), values, "o-", linewidth=2, color=COLORS[pool], label=label(pool))
    ax.set_xticks((1, 2, 3))
    ax.set_xlabel("Repetition")
    style(ax, "End-to-end throughput (samples/s; full run including checkpoints)")
    ax.legend(frameon=False)
    ax.set_title(f"{workload}: end-to-end throughput by run", fontweight="bold")
    save(fig, output, "training_throughput_by_run.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--unet", action="store_const", dest="workload", const="unet")
    choice.add_argument("--resnet", action="store_const", dest="workload", const="resnet")
    choice.add_argument("--cosmoflow", action="store_const", dest="workload", const="cosmoflow")
    parser.add_argument("run_directories", type=Path, nargs="+", help="Completed runs-pool14-... etc. directories")
    parser.add_argument("--output", type=Path, help="Default: results/workloads/plots/dlio/throttled_ssd/WORKLOAD")
    args = parser.parse_args()
    all_runs, all_epochs, all_checkpoints, all_loading = [], [], [], []
    pools = set()
    for directory in args.run_directories:
        pool, runs, epochs, checkpoints, loading = parse_sweep(directory, args.workload)
        if pool in pools:
            parser.error(f"More than one sweep for pool {pool}")
        pools.add(pool)
        all_runs.extend(runs)
        all_epochs.extend(epochs)
        all_checkpoints.extend(checkpoints)
        all_loading.extend(loading)
    pools = sorted(pools)
    output = args.output or ROOT / "results/workloads/plots/dlio/throttled_ssd" / args.workload
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    for metric, unit in (("end_to_end_samples_s", "samples/s (full run)"),
                         ("training_input_MiB_s", "MiB/s (inferred for U-Net)"),
                         ("checkpoint_rank0_MiB_s_no_fsync", "MiB/s"),
                         ("whole_run_minutes", "min")):
        for pool in pools:
            avg, sd = mean_sd([row[metric] for row in all_runs if row["pool"] == pool])
            groups.append({"metric": metric, "pool": pool, "n_runs": 3,
                           "mean": avg, "stddev": sd, "unit": unit})
    write_csv(output / "comparison.csv", groups)
    write_csv(output / "comparison_runs.csv", sorted(all_runs, key=lambda row: (row["pool"], row["repetition"])))
    write_csv(output / "epochs.csv", sorted(all_epochs, key=lambda row: (row["pool"], row["repetition"], row["epoch"])))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12})
    plot_summary(args.workload, pools, groups, output)
    (output / "pool_performance_inferred.png").unlink(missing_ok=True)
    plot_epochs(args.workload, pools, all_epochs, "native_samples_s",
                "DLIO training throughput (samples/s)",
                "DLIO training throughput by epoch", "training_throughput_by_epoch.png", output)
    plot_epochs(args.workload, pools, all_epochs, "training_input_MiB_s",
                "Estimated input throughput (MiB/s; NOT a DLIO-reported byte rate)",
                "inferred input throughput · dotted = calculated",
                "training_throughput_inferred.png", output, inferred=True)
    plot_checkpoints(args.workload, pools, all_checkpoints, output)
    plot_runs(args.workload, pools, all_runs, output)
    plot_epochs(args.workload, pools, all_loading, "rank_load_p95_s",
                "Data loading time per step (s, rank p95)",
                "data loading time by epoch", "data_loading_time.png", output)
    print(f"Validated {len(pools)} pool(s), {len(all_runs)} runs; "
          f"six PNGs and CSVs: {output}")


if __name__ == "__main__":
    main()
