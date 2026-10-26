#!/usr/bin/env python3
"""Validate copied DLIO 2.0.0 pool runs; emit tables, a report, and plots.

Use --resnet, --unet, or --cosmoflow. No DLIO workload is launched here.
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from dlio_pool_profiles import POOLS, RANKS, WORKLOADS, select_workload


REPETITIONS = (1, 2, 3)
EPOCHS = (1, 2, 3, 4, 5)
LABELS = {1: "Default (colva4)", 3: "HDD (colva1–3)", 6: "SSD (colva1–3)"}
COLORS = {1: "#335F87", 3: "#D08734", 6: "#33836B"}
TF_DATA = re.compile(r"model-(2|4)-\d+-(\d+)\.pb-\d+\.data-\d+-of-\d+$")
PT_DATA = re.compile(r"model-(2|4)-\d+-(\d+)\.pt$")


def read_json(path):
    with path.open() as file:
        return json.load(file)


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, items):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(items[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(items)


def p95(items):
    if not items:
        raise ValueError("Missing per-step timing samples")
    return sorted(items)[math.ceil(.95 * len(items)) - 1]


def mean_std(items):
    if len(items) != 3:
        raise ValueError(f"Expected three independent runs, found {len(items)}")
    return statistics.mean(items), statistics.stdev(items)


def placements(path):
    records = []
    with path.open() as file:
        for line in file:
            filename, text = line.rstrip("\n").split("\t")
            targets = tuple(map(int, text.split(",")))
            if len(targets) != 4 or len(set(targets)) != 4:
                raise ValueError(f"Expected four distinct OSTs: {filename}")
            records.append((Path(filename).name, targets))
    return records


def checkpoint_placements(records, profile):
    pattern = TF_DATA if profile.tensorflow else PT_DATA
    result = defaultdict(dict)
    for filename, targets in records:
        match = pattern.fullmatch(filename)
        if match:
            epoch, rank = map(int, match.groups())
            if rank in result[epoch]:
                raise ValueError(f"Duplicate rank checkpoint: {filename}")
            result[epoch][rank] = targets
    if any(set(result[epoch]) != set(range(RANKS)) for epoch in (2, 4)):
        raise ValueError("Missing eight-rank checkpoint placement evidence")
    return result


def validate_runner_csv(root):
    items = read_csv(root / "metrics.csv")
    if len(items) != 72:
        raise ValueError(f"Expected 72 runner rows, found {len(items)}")
    indexed = {}
    for item in items:
        key = (int(item["pool"]), int(item["repetition"]), item["phase"], item["epoch"])
        if key in indexed:
            raise ValueError(f"Duplicate runner CSV row: {key}")
        indexed[key] = item
    return indexed


def parse(root, profile):
    indexed = validate_runner_csv(root)
    source_osts = {}
    for pool in POOLS:
        entries = placements(root / f"pool{pool}-dataset-osts.tsv")
        expected = min(profile.files, 65) if profile.files > 1024 else profile.files
        if len(entries) != expected:
            raise ValueError(f"Pool {pool}: expected {expected} input placement entries, got {len(entries)}")
        source_osts[pool] = entries

    epoch_rows, rank_rows, ckpt_rows, run_rows = [], [], [], []
    gaps, host_discrepancies = [], []
    for repetition in REPETITIONS:
        for pool in POOLS:
            raw = root / f"rep{repetition}-pool{pool}"
            config = yaml.safe_load((raw / ".hydra/config.yaml").read_text())["workload"]
            summary = read_json(raw / "summary.json")
            epochs = read_json(raw / "per_epoch_stats.json")
            if (config["model"] != {"resnet50_v100": "resnet50", "unet3d_v100": "unet3d",
                                    "cosmoflow_v100": "cosmoflow"}[profile.profile]
                    or config["train"]["epochs"] != 5
                    or config["dataset"]["num_files_train"] != profile.files
                    or config["reader"]["batch_size"] != profile.batch_size
                    or config["checkpoint"]["type"] != "all_ranks"
                    or summary["num_accelerators"] != RANKS
                    or sorted(map(int, epochs)) != list(EPOCHS)
                    or f"/pool{pool}/dataset" not in config["dataset"]["data_folder"]):
                raise ValueError(f"Workload/epoch mismatch: {raw}")
            native_rates = summary["metric"]["train_throughput_samples_per_second"]
            if len(native_rates) != len(EPOCHS):
                raise ValueError(f"Incomplete DLIO rate series: {raw}")
            record_bytes = float(config["dataset"]["record_length"])
            native_mean = float(summary["metric"]["train_io_mean_MB_per_second"])
            if not math.isclose(native_mean, statistics.mean(native_rates) * record_bytes / 1048576,
                                rel_tol=1e-6):
                raise ValueError(f"DLIO summary does not match its per-epoch rates: {raw}")
            rank_data = {rank: read_json(raw / f"{rank}_output.json") for rank in range(RANKS)}
            hosts = sorted({data["hostname"] for data in rank_data.values()})
            if len(hosts) != 1:
                raise ValueError(f"Different client hosts in one run: {raw}: {hosts}")
            if summary["num_hosts"] != len(hosts):
                host_discrepancies.append((pool, repetition, summary["num_hosts"], len(hosts)))
            ckpt_targets = checkpoint_placements(placements(raw / "checkpoint-osts.tsv"), profile)
            missing_epochs = []
            run_epoch_wall = []
            for epoch in EPOCHS:
                native = epochs[str(epoch)]
                block = native.get("block1", {})
                available = "duration" in block
                consolidated_phase = "training_input_block" if available else "training_epoch_wall_only"
                original = indexed[(pool, repetition, consolidated_phase, str(epoch))]
                if available:
                    if not math.isclose(float(original["throughput"]), native_rates[epoch - 1], rel_tol=1e-9):
                        raise ValueError(f"Training rate mismatch: {raw}, epoch {epoch}")
                    if not math.isclose(float(original["duration_s"]), float(block["duration"]), abs_tol=.01):
                        raise ValueError(f"Training duration mismatch: {raw}, epoch {epoch}")
                else:
                    missing_epochs.append(epoch)
                    if (native_rates[epoch - 1] != 0 or original["throughput"]
                            or not math.isclose(float(original["duration_s"]), float(native["duration"]), abs_tol=.01)):
                        raise ValueError(f"Unmarked missing training metric: {raw}, epoch {epoch}")
                load_samples = []
                for rank, data in rank_data.items():
                    step = data[str(epoch)]
                    loading, processing = step["load"]["block1"], step["proc"]["block1"]
                    if len(loading) != len(processing) or not loading:
                        raise ValueError(f"Incomplete rank step timings: {raw}, epoch {epoch}, rank {rank}")
                    load_samples.extend(loading)
                    rank_rows.append({"pool": pool, "repetition": repetition, "epoch": epoch,
                                      "rank": rank, "host": data["hostname"], "steps": len(loading),
                                      "load_mean_s": statistics.mean(loading), "load_p95_s": p95(loading),
                                      "load_max_s": max(loading), "proc_mean_s": statistics.mean(processing),
                                      "proc_p95_s": p95(processing),
                                      "rank_samples_per_s_DLIO": step["throughput"]["block1"]})
                expected_steps = (profile.files * config["dataset"]["num_samples_per_file"]
                                  // (profile.batch_size * RANKS))
                if any(len(rank_data[r][str(epoch)]["load"]["block1"]) != expected_steps
                       for r in range(RANKS)):
                    raise ValueError(f"Incomplete input steps: {raw}, epoch {epoch}")
                duration = float(native["duration"])
                run_epoch_wall.append(duration)
                gap = ""
                if epoch in (2, 4):
                    if not available:
                        raise ValueError(f"Checkpoint epoch missing training block: {raw}, epoch {epoch}")
                    event = native["ckpt1"]
                    gap = (datetime.fromisoformat(event["start"])
                           - datetime.fromisoformat(block["end"])).total_seconds()
                    runner_ckpt = indexed[(pool, repetition, "checkpoint_save_rank0", str(epoch))]
                    ckpt_time = float(event["duration"])
                    bytes_rank0 = int(runner_ckpt["bytes_written"])
                    if (not math.isclose(ckpt_time, float(runner_ckpt["duration_s"]), abs_tol=.01)
                            or bytes_rank0 < .9 * config["checkpoint"]["model_size"]):
                        raise ValueError(f"Invalid checkpoint timing or payload: {raw}, epoch {epoch}")
                    rank0_rate = bytes_rank0 / 1048576 / ckpt_time
                    if not math.isclose(rank0_rate, float(runner_ckpt["throughput"]), rel_tol=1e-6):
                        raise ValueError(f"Rank-0 checkpoint rate mismatch: {raw}, epoch {epoch}")
                    ckpt_rows.append({"pool": pool, "repetition": repetition, "epoch": epoch,
                                      "rank0_save_s_DLIO": ckpt_time, "rank0_bytes": bytes_rank0,
                                      "rank0_MiB_s_no_fsync": rank0_rate,
                                      "rank0_data_OSTs": ",".join(map(str, ckpt_targets[epoch][0])),
                                      "all_data_OSTs": ",".join(map(str, sorted({t for ids in
                                                                                  ckpt_targets[epoch].values() for t in ids})))})
                rate = native_rates[epoch - 1] if available else ""
                nominal = rate * record_bytes / 1048576 if available else ""
                wall_consistency = ""
                if available:
                    wall_consistency = 100 * (rate * float(block["duration"])
                                              / (expected_steps * profile.batch_size * RANKS) - 1)
                if (gap != "" and gap > 10) or (wall_consistency != "" and abs(wall_consistency) > 2):
                    gaps.append((pool, repetition, epoch, gap, wall_consistency))
                epoch_rows.append({"pool": pool, "repetition": repetition, "epoch": epoch,
                                   "epoch_wall_s": duration,
                                   "training_block_wall_s": block.get("duration", ""),
                                   "training_samples_per_s_DLIO": rate,
                                   "nominal_training_MiB_s_DLIO": nominal,
                                   "training_rate_status": "native" if available else "unavailable",
                                   "block_to_checkpoint_gap_s": gap,
                                   "DLIO_rate_vs_wall_pct": wall_consistency,
                                   "rank_step_load_p95_s": p95(load_samples)})
            summary_phase = "training_summary_incomplete" if missing_epochs else "training_summary"
            runner_summary = indexed[(pool, repetition, summary_phase, "")]
            if not math.isclose(float(runner_summary["throughput"]), native_mean, rel_tol=1e-9):
                raise ValueError(f"Runner/native summary mismatch: {raw}")
            run_rows.append({"pool": pool, "repetition": repetition,
                             "observed_client": ",".join(hosts), "rank_count": RANKS,
                             "reported_num_hosts": summary["num_hosts"],
                             "observed_num_hosts": len(hosts), "start": summary["start"],
                             "end": summary["end"], "mean_epoch_wall_s": statistics.mean(run_epoch_wall),
                             "native_training_mean_MiB_s": native_mean,
                             "native_training_mean_status": "incomplete" if missing_epochs else "valid",
                             "epochs_missing_training_rate": ",".join(map(str, missing_epochs))})

    counts = []
    for pool in POOLS:
        tally = Counter(target for _, ids in source_osts[pool] for target in ids)
        for ost, count in sorted(tally.items()):
            counts.append({"pool": pool, "ost": ost, "input_files_assigned": count})
    return epoch_rows, rank_rows, ckpt_rows, run_rows, counts, gaps, host_discrepancies


def pool_summary(profile, epochs, checkpoints, runs):
    result = []
    for pool in POOLS:
        pool_runs = [r for r in runs if r["pool"] == pool]
        pool_epochs = [r for r in epochs if r["pool"] == pool]
        pool_ckpts = [r for r in checkpoints if r["pool"] == pool]
        metrics = {
            "epoch_mean_wall_s": [r["mean_epoch_wall_s"] for r in pool_runs],
            "epoch1_wall_s": [r["epoch_wall_s"] for r in pool_epochs if r["epoch"] == 1],
            "checkpoint2_rank0_MiB_s": [r["rank0_MiB_s_no_fsync"] for r in pool_ckpts if r["epoch"] == 2],
            "checkpoint4_rank0_MiB_s": [r["rank0_MiB_s_no_fsync"] for r in pool_ckpts if r["epoch"] == 4],
        }
        if all(r["native_training_mean_status"] == "valid" for r in pool_runs):
            metrics["native_training_mean_MiB_s"] = [r["native_training_mean_MiB_s"] for r in pool_runs]
        else:
            # U-Net's five-epoch DLIO mean includes three unmeasured zero placeholders.
            metrics["native_epoch2_4_samples_per_s"] = [statistics.mean(
                r["training_samples_per_s_DLIO"] for r in pool_epochs
                if r["repetition"] == repetition and r["epoch"] in (2, 4))
                for repetition in REPETITIONS]
        for name, values in metrics.items():
            mean, sd = mean_std(values)
            result.append({"pool": pool, "metric": name, "n_runs": 3,
                           "mean": mean, "stddev": sd})
    return result


def comparison(profile, root, epochs, ranks, checkpoints, runs):
    """Keep each repetition as the unit of comparison, not its five epochs."""
    values = []
    for run in runs:
        pool, repetition = run["pool"], run["repetition"]
        native_epochs = [row for row in epochs if row["pool"] == pool
                         and row["repetition"] == repetition]
        if len(native_epochs) != 5:
            raise ValueError(f"Missing comparison epochs: pool {pool}, repetition {repetition}")
        if run["native_training_mean_status"] == "valid":
            input_rate = run["native_training_mean_MiB_s"]
            source = "DLIO nominal rate across 5 epochs"
        else:
            config = yaml.safe_load((root / f"rep{repetition}-pool{pool}/.hydra/config.yaml").read_text())["workload"]
            record_bytes = float(config["dataset"]["record_length"])
            batch_size = int(config["reader"]["batch_size"])
            input_bytes, input_seconds = 0.0, 0.0
            for epoch in native_epochs:
                step_counts = {row["steps"] for row in ranks if row["pool"] == pool
                               and row["repetition"] == repetition and row["epoch"] == epoch["epoch"]}
                if len(step_counts) != 1:
                    raise ValueError("Different or missing rank step counts")
                input_bytes += step_counts.pop() * batch_size * RANKS * record_bytes
                input_seconds += float(epoch["training_block_wall_s"] or epoch["epoch_wall_s"])
            input_rate = input_bytes / input_seconds / 1048576
            source = "nominal samples × mean record bytes / block-or-epoch wall time"
        saves = [row for row in checkpoints if row["pool"] == pool
                 and row["repetition"] == repetition]
        if len(saves) != 2:
            raise ValueError(f"Missing paired checkpoint events in pool {pool}, repetition {repetition}")
        save_rate = sum(row["rank0_bytes"] for row in saves) / (
            1048576 * sum(row["rank0_save_s_DLIO"] for row in saves))
        elapsed = (datetime.fromisoformat(run["end"]) - datetime.fromisoformat(run["start"])).total_seconds() / 60
        values.append({"pool": pool, "repetition": repetition, "training_input_MiB_s": input_rate,
                       "training_input_source": source, "checkpoint_rank0_MiB_s_no_fsync": save_rate,
                       "whole_run_minutes": elapsed})
    metrics = (("training_input_MiB_s", "MiB/s", "higher"),
               ("checkpoint_rank0_MiB_s_no_fsync", "MiB/s", "higher"),
               ("whole_run_minutes", "min", "lower"))
    aggregates = []
    for metric, unit, direction in metrics:
        for pool in POOLS:
            group = [row[metric] for row in values if row["pool"] == pool]
            avg, sd = mean_std(group)
            aggregates.append({"metric": metric, "pool": pool, "n_runs": 3,
                               "mean": avg, "stddev": sd, "unit": unit,
                               "better": direction})
    return values, aggregates


def make_report(profile, epochs, runs, groups, gaps, host_discrepancies):
    by_key = {(r["pool"], r["metric"]): r for r in groups}
    metric = ("native_training_mean_MiB_s" if profile.profile != "unet3d_v100"
              else "native_epoch2_4_samples_per_s")
    label = ("DLIO training MiB/s" if profile.profile != "unet3d_v100"
             else "Native epochs 2/4 samples/s")
    def fmt(pool, key):
        item = by_key[pool, key]
        return f"{item['mean']:.2f} ± {item['stddev']:.2f}"
    lines = [f"# {profile.profile} · DLIO pool comparison", "",
             "Nine runs: three per pool, eight MPI ranks per run.", "",
             f"| Pool | Mean epoch wall (s) | Epoch 1 wall (s) | {label} | "
             "Checkpoint 2 rank-0 (MiB/s) | Checkpoint 4 rank-0 (MiB/s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for pool in POOLS:
        lines.append("| " + str(pool) + " | " + " | ".join(fmt(pool, key) for key in (
            "epoch_mean_wall_s", "epoch1_wall_s", metric,
            "checkpoint2_rank0_MiB_s", "checkpoint4_rank0_MiB_s")) + " |")
    lines.extend(["", "Mean ± sample SD across three independent runs. Epoch wall time includes",
                  "checkpoints in epochs 2 and 4; checkpoint rate is rank-0 save completion",
                  "without fsync, not durable or eight-rank aggregate throughput.",
                  "Modeled training MiB/s is samples/s × configured mean record length, not",
                  "measured POSIX read bytes or backend device traffic.",
                  "The pool order was Default → HDD → SSD each repetition; Default uses colva4",
                  "while HDD/SSD span colva1–3. These are configured-pool comparisons.", ""])
    if profile.profile == "unet3d_v100":
        lines.extend(["**U-Net DLIO 2.0.0 reporting gap:** training-block duration and throughput",
                      "are absent in epochs 1, 3, and 5 although every rank loaded the expected",
                      "number of batches. Its native five-epoch mean includes zeros for those",
                      "epochs and is not used here. Epoch wall times cover all five epochs; the",
                      "native rate column uses only epochs 2 and 4, not a five-epoch mean.",
                      "NPZ record lengths vary; mean-size-based MiB/s is only nominal.", ""])
        for pool in POOLS:
            first = [row for row in epochs if row["pool"] == pool and row["epoch"] == 1]
            typical = statistics.median(row["epoch_wall_s"] for row in first)
            for row in first:
                if row["epoch_wall_s"] - typical > 10:
                    lines.append(f"Epoch-1 wall-time outlier: pool {pool}, repetition "
                                 f"{row['repetition']} took {row['epoch_wall_s']:.2f} s "
                                 f"({row['epoch_wall_s'] - typical:.2f} s above that pool's median). "
                                 "Its cause is not established.")
                    lines.append("")
    else:
        lines.extend(["All five native per-epoch training rates are present.", ""])
    if gaps:
        lines.extend(["## Timing observations", ""])
        for pool, repetition, epoch, gap, mismatch in gaps:
            text = f"- Pool {pool}, repetition {repetition}, epoch {epoch}:"
            if gap != "":
                text += f" {gap:.2f} s between training block end and checkpoint start;"
            if mismatch != "":
                text += f" DLIO rate differs from block-wall-implied rate by {mismatch:+.2f}%."
            lines.append(text)
        lines.extend(["", "Gaps before checkpoint start are not checkpoint-save time.", ""])
    if host_discrepancies:
        lines.append(f"Per-rank hostnames indicate one client; DLIO summary reported eight hosts "
                     f"in {len(host_discrepancies)}/9 runs. Use per-rank hostnames for topology.")
    lines.append("")
    return "\n".join(lines)


def comparison_report(profile, aggregates):
    lines = ["## Pool comparison at a glance", "",
             "| Metric | Highest-performing observed mean | Default | HDD | SSD |",
             "|---|---|---:|---:|---:|"]
    descriptions = (
        ("training_input_MiB_s", "Training input (MiB/s)"),
        ("checkpoint_rank0_MiB_s_no_fsync", "Checkpoint rank 0 (MiB/s)"),
        ("whole_run_minutes", "Whole-run time (min; lower is better)"),
    )
    for metric, label in descriptions:
        values = {row["pool"]: row for row in aggregates if row["metric"] == metric}
        winner = (min if metric == "whole_run_minutes" else max)(
            POOLS, key=lambda pool: values[pool]["mean"])
        cells = [f"{values[pool]['mean']:.2f} ± {values[pool]['stddev']:.2f}" for pool in POOLS]
        lines.append(f"| {label} | {LABELS[winner]} | " + " | ".join(cells) + " |")
    lines.extend(["", "Winners here mean only the numerically best mean across **three** runs;",
                  "they are not significance claims or isolated HDD/SSD effects.",
                  "Input and checkpoint are different I/O phases and must not be added as throughput.",
                  "The run timer includes startup, reads, and checkpoint saves."])
    if profile.profile == "unet3d_v100":
        lines.extend(["U-Net training input MiB/s is a **nominal proxy**, not DLIO's invalid",
                      "five-epoch mean: validated batches × configured mean NPZ record size",
                      "divided by the sum of training-block times where available and epoch",
                      "wall times for checkpoint-free epochs 1, 3, and 5. The latter can",
                      "include non-read overhead; individual NPZ file sizes vary."])
    else:
        lines.append("Training input MiB/s is DLIO's modeled sample-based five-epoch mean.")
    lines.extend(["Checkpoint MiB/s is rank-0 file bytes / rank-0 DLIO save time across",
                  "two dependent saves within each run; it has no fsync or eight-rank barrier.", ""])
    return "\n".join(lines)


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_comparison(profile, aggregates, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    entries = (
        ("training_input_MiB_s", "Training input", "Nominal MiB/s · higher is better"),
        ("checkpoint_rank0_MiB_s_no_fsync", "Checkpoint save", "Rank-0 MiB/s · higher is better"),
        ("whole_run_minutes", "Whole run", "Minutes · lower is better"),
    )
    for ax, (metric, title, xlabel) in zip(axes, entries):
        records = {row["pool"]: row for row in aggregates if row["metric"] == metric}
        winner = (min if metric == "whole_run_minutes" else max)(
            POOLS, key=lambda pool: records[pool]["mean"])
        ceiling = max(records[pool]["mean"] + records[pool]["stddev"] for pool in POOLS) * 1.3
        for position, pool in enumerate(POOLS):
            item = records[pool]
            ax.barh(position, item["mean"], color=COLORS[pool], alpha=.85,
                    edgecolor="#18212b" if pool == winner else "none",
                    linewidth=2 if pool == winner else 0)
            ax.errorbar(item["mean"], position, xerr=item["stddev"], fmt="none",
                        color="#243447", capsize=4)
            ax.text(item["mean"] - ceiling * .015, position,
                    f"{item['mean']:.2f} ± {item['stddev']:.2f}" + ("  ★" if pool == winner else ""),
                    ha="right", va="center", fontsize=8.5, color="#162334",
                    bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none", "pad": 1})
        ax.set_yticks(range(3), [LABELS[pool] for pool in POOLS])
        ax.invert_yaxis()
        ax.set_xlim(0, ceiling)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{profile.profile}: which configured pool leads for each metric?",
                 fontweight="bold")
    note = ("U-Net input is a nominal wall-time proxy; " if profile.profile == "unet3d_v100"
            else "DLIO-modeled input rate; ")
    fig.text(.5, -.015, "Bars = mean ± SD across 3 runs; ★ = best observed mean. " + note +
             "rank-0 checkpoint saves have no fsync.", ha="center", fontsize=9)
    save(fig, out / "pool_comparison.png")


def plot_training(profile, epochs, out):
    unet = profile.profile == "unet3d_v100"
    fields = (("epoch_wall_s", "Epoch completion time (s)") if unet else
              ("nominal_training_MiB_s_DLIO", "DLIO-modeled training I/O (MiB/s)"))
    second = (("training_samples_per_s_DLIO", "Available native training rate (samples/s)") if unet
              else ("training_block_wall_s", "Training block wall time (s)"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (column, ylabel) in zip(axes, (fields, second)):
        valid_epochs = (2, 4) if unet and column == "training_samples_per_s_DLIO" else EPOCHS
        for pool in POOLS:
            for repetition in REPETITIONS:
                ys = [next(float(row[column]) for row in epochs if
                           row["pool"] == pool and row["repetition"] == repetition
                           and row["epoch"] == epoch) for epoch in valid_epochs]
                ax.plot(valid_epochs, ys, color=COLORS[pool], alpha=.2, linewidth=1)
            means, errors = zip(*(mean_std([float(row[column]) for row in epochs if
                row["pool"] == pool and row["epoch"] == epoch]) for epoch in valid_epochs))
            ax.errorbar(valid_epochs, means, yerr=errors, fmt="o-", capsize=3,
                        color=COLORS[pool], linewidth=2, label=LABELS[pool])
        ax.set_xticks(EPOCHS)
        ax.set_xlabel("Epoch")
        style(ax, ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    if unet:
        axes[1].set_ylim(bottom=0)
        axes[1].text(.5, .04, "Epochs 1, 3, 5: DLIO rate unavailable, not zero I/O",
                     transform=axes[1].transAxes, ha="center", fontsize=8, color="#7d3641")
    else:
        flagged = [row for row in epochs if row["block_to_checkpoint_gap_s"] != ""
                   and float(row["block_to_checkpoint_gap_s"]) > 10]
        for row in flagged:
            axes[0].scatter(row["epoch"], row["nominal_training_MiB_s_DLIO"],
                            marker="*", s=130, color="#B23C48", edgecolor="white", zorder=5)
        if flagged:
            axes[0].text(.98, .04, "★ pre-checkpoint gap flagged in report",
                         transform=axes[0].transAxes, ha="right", fontsize=8, color="#7d3641")
    fig.suptitle(f"{profile.profile}: training by epoch · three runs per pool", fontweight="bold")
    save(fig, out / "training_epochs.png")


def plot_checkpoints(profile, checkpoints, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, ylabel in ((axes[0], "rank0_save_s_DLIO", "Rank-0 save duration (s)"),
                            (axes[1], "rank0_MiB_s_no_fsync", "Rank-0 save rate (MiB/s)")):
        for offset, pool in zip((-.19, 0, .19), POOLS):
            for epoch in (2, 4):
                values = [row[key] for row in checkpoints if row["pool"] == pool and row["epoch"] == epoch]
                mean, deviation = mean_std(values)
                x = epoch + offset
                ax.errorbar(x, mean, yerr=deviation, fmt="o", markersize=7, capsize=4,
                            color=COLORS[pool], label=LABELS[pool] if epoch == 2 else None)
                ax.scatter([x - .065, x, x + .065], values, color=COLORS[pool], s=15, alpha=.6)
        ax.set_xticks((2, 4), ("After epoch 2", "After epoch 4"))
        style(ax, ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{profile.profile}: rank-0 checkpoint save · no fsync", fontweight="bold")
    fig.text(.5, -.01, "Three runs per pool; not a synchronized eight-rank rate or durable-write result.",
             ha="center", fontsize=9)
    save(fig, out / "checkpoint_rank0.png")


def plot_repetitions(profile, runs, out):
    unet = profile.profile == "unet3d_v100"
    column = "mean_epoch_wall_s" if unet else "native_training_mean_MiB_s"
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    for pool in POOLS:
        vals = [next(row[column] for row in runs if row["pool"] == pool and
                     row["repetition"] == repetition) for repetition in REPETITIONS]
        ax.plot(REPETITIONS, vals, "o-", color=COLORS[pool], label=LABELS[pool], linewidth=2)
    ax.set_xticks(REPETITIONS)
    ax.set_xlabel("Repetition · pool order: Default → HDD → SSD")
    style(ax, "Mean epoch wall (s; lower is faster)" if unet else
          "DLIO-modeled mean training I/O (MiB/s)")
    ax.legend(frameon=False)
    ax.set_title(f"{profile.profile}: run-to-run comparison", fontweight="bold")
    save(fig, out / "training_repetitions.png")


def plot_load_tail(profile, ranks, out):
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    for pool in POOLS:
        run_values = {}
        for repetition in REPETITIONS:
            run_values[repetition] = [statistics.mean(row["load_p95_s"] for row in ranks if
                row["pool"] == pool and row["repetition"] == repetition and row["epoch"] == epoch)
                for epoch in EPOCHS]
            ax.plot(EPOCHS, run_values[repetition], color=COLORS[pool], alpha=.2)
        means, errors = zip(*(mean_std([run_values[rep][epoch - 1] for rep in REPETITIONS])
                              for epoch in EPOCHS))
        ax.errorbar(EPOCHS, means, yerr=errors, fmt="o-", color=COLORS[pool],
                    linewidth=2, capsize=3, label=LABELS[pool])
    ax.set_xticks(EPOCHS)
    ax.set_xlabel("Epoch")
    style(ax, "Mean within-run rank p95 load-step time (s)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Application-level load timing · not POSIX read latency", fontweight="bold")
    save(fig, out / "rank_load_p95.png")


def plot_placement(profile, counts, out):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2))
    input_count = min(profile.files, 65) if profile.files > 1024 else profile.files
    for ax, pool in zip(axes, POOLS):
        records = sorted((row for row in counts if row["pool"] == pool), key=lambda row: row["ost"])
        positions = list(range(len(records)))
        ax.bar(positions, [r["input_files_assigned"] for r in records],
               color=COLORS[pool], alpha=.83)
        ax.axhline(input_count * 4 / len(records), color="#526278", linestyle="--", linewidth=1)
        ax.set_xticks(positions, [str(r["ost"]) for r in records])
        style(ax, "File assignments")
        ax.set_title(LABELS[pool], loc="left", fontsize=10)
    axes[-1].set_xlabel("BeeGFS storage target ID")
    fig.suptitle(f"{profile.profile}: {input_count} input files × four OSTs"
                 + (" (placement sample)" if profile.files > 1024 else ""), fontweight="bold")
    fig.text(.5, -.01, "Dashed: even file assignment; counts are not byte or device traffic.",
             ha="center", fontsize=9)
    save(fig, out / "dataset_ost_assignments.png")


def analyze(root, profile, output, overwrite=False):
    epochs, ranks, checkpoints, runs, counts, gaps, hosts = parse(root, profile)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; choose --output or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    tables = {"epochs.csv": epochs, "rank_epochs.csv": ranks,
              "checkpoints.csv": checkpoints, "runs.csv": runs,
              "dataset_ost_counts.csv": counts}
    for filename, items in tables.items():
        write_csv(output / filename, items)
    groups = pool_summary(profile, epochs, checkpoints, runs)
    write_csv(output / "pool_summary.csv", groups)
    run_values, comparison_groups = comparison(profile, root, epochs, ranks, checkpoints, runs)
    write_csv(output / "comparison_runs.csv", run_values)
    write_csv(output / "comparison.csv", comparison_groups)
    (output / "report.md").write_text(make_report(profile, epochs, runs, groups, gaps, hosts)
                                      + "\n" + comparison_report(profile, comparison_groups))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "figure.facecolor": "white"})
    figures = output / "plots"
    figures.mkdir(exist_ok=True)
    plot_comparison(profile, comparison_groups, figures)
    plot_training(profile, epochs, figures)
    plot_checkpoints(profile, checkpoints, figures)
    plot_repetitions(profile, runs, figures)
    plot_load_tail(profile, ranks, figures)
    plot_placement(profile, counts, figures)
    captions = {
        "pool_comparison.png": "Pool ranking by input, checkpoint, and whole-run metrics; read the phase-specific caveats.",
        "training_epochs.png": "Per-epoch training measurements; absent U-Net native rates are not plotted as zeros.",
        "checkpoint_rank0.png": "Rank-0 checkpoint save completion without fsync.",
        "training_repetitions.png": "Run-to-run comparison across the fixed pool order.",
        "rank_load_p95.png": "Application-level per-rank loading-step p95, summarized by run.",
        "dataset_ost_assignments.png": "Input file assignments per OST; not bytes or traffic.",
    }
    (figures / "README.md").write_text(f"# {profile.profile} figures\n\n" + "\n\n".join(
        f"## {name.removesuffix('.png').replace('_', ' ').title()}\n\n![{caption}]({name})\n\n{caption}"
        for name, caption in captions.items()) + "\n")
    (figures / "plot_manifest.json").write_text(json.dumps({
        "source": str(root), "inputs": list(tables), "figures": captions,
    }, indent=2) + "\n")
    print(f"Validated {len(runs)} runs, {len(epochs)} epochs, {len(ranks)} rank-epochs, "
          f"{len(checkpoints)} checkpoints; tables, report, and plots: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    select_workload(parser)
    parser.add_argument("run_directory", type=Path, help="Copied runs-... directory")
    parser.add_argument("--output", type=Path, help="Destination (default: RUN/analysis)")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs in existing destination")
    args = parser.parse_args()
    analyze(args.run_directory, WORKLOADS[args.workload],
            args.output or args.run_directory / "analysis", args.overwrite)


if __name__ == "__main__":
    main()
