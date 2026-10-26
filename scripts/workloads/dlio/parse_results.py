#!/usr/bin/env python3
"""Parse a copied DLIO 2.0.0 ResNet pool sweep without replaying I/O.

Native training rates are modeled samples/second; rank load/proc values are
application-step timings, not POSIX counters or backend read/write traffic.
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

import yaml


POOLS = (1, 3, 6)
REPETITIONS = (1, 2, 3)
EPOCHS = (1, 2, 3, 4, 5)
RANKS = range(8)
DATA_FILE = re.compile(r"model-(2|4)-\d+-(\d+)\.pb-\d+\.data-\d+-of-\d+$")


def load_json(path):
    with path.open() as file:
        return json.load(file)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize empty step timings")
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def write_csv(path, rows, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def placement(path):
    records = []
    with path.open() as file:
        for line in file:
            name, targets = line.rstrip("\n").split("\t")
            ids = tuple(map(int, targets.split(",")))
            if len(ids) != 4 or len(set(ids)) != 4:
                raise ValueError(f"Expected four distinct OSTs for {name}")
            records.append((Path(name).name, ids))
    return records


def checkpoint_targets(records):
    by_epoch = defaultdict(dict)
    for name, ids in records:
        match = DATA_FILE.fullmatch(name)
        if match:
            epoch, rank = map(int, match.groups())
            if rank in by_epoch[epoch]:
                raise ValueError(f"Duplicate checkpoint data file: {name}")
            by_epoch[epoch][rank] = ids
    for epoch in (2, 4):
        if set(by_epoch[epoch]) != set(RANKS):
            raise ValueError(f"Incomplete epoch-{epoch} checkpoint data placement")
    return by_epoch


def parse(root):
    consolidated = list(csv.DictReader((root / "metrics.csv").open(newline="")))
    if len(consolidated) != 72:
        raise ValueError(f"Expected 72 runner CSV rows, found {len(consolidated)}")
    indexed = {}
    for row in consolidated:
        key = (int(row["pool"]), int(row["repetition"]), row["phase"], row["epoch"])
        if key in indexed:
            raise ValueError(f"Duplicate runner metric {key}")
        indexed[key] = row

    datasets = {}
    for pool in POOLS:
        records = placement(root / f"pool{pool}-dataset-osts.tsv")
        if len(records) != 1024:
            raise ValueError(f"Pool {pool}: expected 1024 recorded TFRecords, found {len(records)}")
        datasets[pool] = records

    epochs_out, rank_out, checkpoints_out, run_out = [], [], [], []
    host_mismatches = []
    timing_flags = []
    for repetition in REPETITIONS:
        for pool in POOLS:
            raw = root / f"rep{repetition}-pool{pool}"
            config = yaml.safe_load((raw / ".hydra/config.yaml").read_text())["workload"]
            summary = load_json(raw / "summary.json")
            native_epochs = load_json(raw / "per_epoch_stats.json")
            if (config["model"] != "resnet50" or config["train"]["epochs"] != 5
                    or config["dataset"]["num_files_train"] != 1024
                    or config["reader"]["batch_size"] != 64
                    or config["checkpoint"]["type"] != "all_ranks"
                    or summary["num_accelerators"] != 8
                    or sorted(map(int, native_epochs)) != list(EPOCHS)):
                raise ValueError(f"Unexpected workload configuration or missing epochs: {raw}")
            if f"/pool{pool}/dataset" not in config["dataset"]["data_folder"]:
                raise ValueError(f"Dataset path does not match pool {pool}: {raw}")
            rates = summary["metric"]["train_throughput_samples_per_second"]
            if len(rates) != 5:
                raise ValueError(f"Expected five DLIO training rates: {raw}")
            steps = (config["dataset"]["num_files_train"] * config["dataset"]["num_samples_per_file"]
                     // (config["reader"]["batch_size"] * len(RANKS)))
            expected_samples = steps * config["reader"]["batch_size"] * len(RANKS)
            record_bytes = float(config["dataset"]["record_length"])
            native_mean = float(summary["metric"]["train_io_mean_MB_per_second"])
            calculated_mean = statistics.mean(rates) * record_bytes / 1048576
            if not math.isclose(native_mean, calculated_mean, rel_tol=1e-6):
                raise ValueError(f"Native training-rate formula did not match: {raw}")

            hosts = set()
            ranks = {}
            for rank in RANKS:
                data = load_json(raw / f"{rank}_output.json")
                hosts.add(data["hostname"])
                ranks[rank] = data
            if len(hosts) != 1 or next(iter(hosts)) != "anjuna2":
                raise ValueError(f"Unexpected observed client hosts for {raw}: {hosts}")
            if summary["num_hosts"] != len(hosts):
                host_mismatches.append((pool, repetition, summary["num_hosts"], len(hosts)))

            ckpt_placements = checkpoint_targets(placement(raw / "checkpoint-osts.tsv"))
            run_out.append({"pool": pool, "repetition": repetition, "observed_client": "anjuna2",
                            "rank_count": 8, "reported_num_hosts": summary["num_hosts"],
                            "observed_num_hosts": len(hosts),
                            "training_mean_MiB_s_DLIO": native_mean,
                            "start": summary["start"], "end": summary["end"]})
            for epoch in EPOCHS:
                event = native_epochs[str(epoch)]
                load_all = []
                for rank, data in ranks.items():
                    step = data[str(epoch)]
                    load = step["load"]["block1"]
                    proc = step["proc"]["block1"]
                    if len(load) != len(proc) or not load:
                        raise ValueError(f"Incomplete per-rank step arrays in {raw}, epoch {epoch}, rank {rank}")
                    load_all.extend(load)
                    rank_out.append({"pool": pool, "repetition": repetition,
                                     "epoch": epoch, "rank": rank, "hostname": data["hostname"],
                                     "steps": len(load), "load_mean_s": statistics.mean(load),
                                     "load_p95_s": percentile(load, .95), "load_max_s": max(load),
                                     "proc_mean_s": statistics.mean(proc),
                                     "proc_p95_s": percentile(proc, .95),
                                     "rank_samples_per_s_DLIO": step["throughput"]["block1"]})
                nominal_rate = float(rates[epoch - 1]) * record_bytes / 1048576
                block_s = float(event["block1"]["duration"])
                wall_implied_rate = expected_samples / block_s
                rate_vs_wall_pct = 100 * (float(rates[epoch - 1]) / wall_implied_rate - 1)
                gap = ""
                if epoch in (2, 4):
                    gap = (datetime.fromisoformat(event["ckpt1"]["start"])
                           - datetime.fromisoformat(event["block1"]["end"])).total_seconds()
                if abs(rate_vs_wall_pct) > 2 or gap != "" and gap > 10:
                    timing_flags.append((pool, repetition, epoch, rate_vs_wall_pct, gap))
                training_csv = indexed[(pool, repetition, "training_input_block", str(epoch))]
                if not math.isclose(float(training_csv["throughput"]), float(rates[epoch - 1]), rel_tol=1e-9):
                    raise ValueError(f"Runner/native training mismatch: {raw}, epoch {epoch}")
                epochs_out.append({"pool": pool, "repetition": repetition, "epoch": epoch,
                                   "epoch_wall_s": event["duration"],
                                   "training_block_wall_s": event["block1"]["duration"],
                                   "training_samples_per_s_DLIO": rates[epoch - 1],
                                   "nominal_training_MiB_s_DLIO": nominal_rate,
                                   "wall_implied_samples_per_s": wall_implied_rate,
                                   "DLIO_rate_vs_wall_pct": rate_vs_wall_pct,
                                   "block_to_checkpoint_gap_s": gap,
                                   "rank_step_load_p95_s": percentile(load_all, .95),
                                   "rank_step_load_max_s": max(load_all)})
                if epoch in (2, 4):
                    native = event["ckpt1"]
                    runner = indexed[(pool, repetition, "checkpoint_save_rank0", str(epoch))]
                    duration = float(native["duration"])
                    actual_bytes = int(runner["bytes_written"])
                    if not math.isclose(duration, float(runner["duration_s"]), abs_tol=.01):
                        raise ValueError(f"Runner/native checkpoint mismatch: {raw}, epoch {epoch}")
                    if not 90000000 <= actual_bytes <= 110000000:
                        raise ValueError(f"Unexpected rank-0 checkpoint bytes in {raw}: {actual_bytes}")
                    rank0_targets = ckpt_placements[epoch][0]
                    event_targets = sorted({t for ids in ckpt_placements[epoch].values() for t in ids})
                    checkpoints_out.append({"pool": pool, "repetition": repetition,
                                            "epoch": epoch, "rank0_save_s_DLIO": duration,
                                            "rank0_bytes_from_runner": actual_bytes,
                                            "rank0_MiB_s_no_fsync": actual_bytes / 1048576 / duration,
                                            "rank0_data_OSTs": ",".join(map(str, rank0_targets)),
                                            "all_data_OSTs": ",".join(map(str, event_targets))})
            runner_summary = indexed[(pool, repetition, "training_summary", "")]
            if not math.isclose(float(runner_summary["throughput"]), native_mean, rel_tol=1e-9):
                raise ValueError(f"Runner/native training summary mismatch: {raw}")

    return datasets, epochs_out, rank_out, checkpoints_out, run_out, host_mismatches, timing_flags


def report(root, destination, overwrite=False):
    datasets, epochs, ranks, checkpoints, runs, mismatches, flags = parse(root)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; choose another --output or pass --overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "epochs.csv", epochs, list(epochs[0]))
    write_csv(destination / "rank_epochs.csv", ranks, list(ranks[0]))
    write_csv(destination / "checkpoints.csv", checkpoints, list(checkpoints[0]))
    write_csv(destination / "runs.csv", runs, list(runs[0]))

    placements = []
    for pool in POOLS:
        counter = Counter(t for _, ids in datasets[pool] for t in ids)
        for ost, count in sorted(counter.items()):
            placements.append({"pool": pool, "ost": ost, "dataset_files_assigned": count})
    write_csv(destination / "dataset_ost_counts.csv", placements, list(placements[0]))

    groups = []
    for pool in POOLS:
        pool_runs = [row for row in runs if row["pool"] == pool]
        pool_epochs = [row for row in epochs if row["pool"] == pool]
        pool_checkpoints = [row for row in checkpoints if row["pool"] == pool]
        measurements = {
            "training_mean_MiB_s": [row["training_mean_MiB_s_DLIO"] for row in pool_runs],
            "epoch1_nominal_MiB_s": [row["nominal_training_MiB_s_DLIO"] for row in pool_epochs if row["epoch"] == 1],
            "epochs2_5_nominal_MiB_s": [statistics.mean(row["nominal_training_MiB_s_DLIO"]
                                            for row in pool_epochs if row["repetition"] == repetition and row["epoch"] > 1)
                                         for repetition in REPETITIONS],
            "checkpoint2_rank0_MiB_s": [row["rank0_MiB_s_no_fsync"] for row in pool_checkpoints if row["epoch"] == 2],
            "checkpoint4_rank0_MiB_s": [row["rank0_MiB_s_no_fsync"] for row in pool_checkpoints if row["epoch"] == 4],
        }
        for metric, values in measurements.items():
            if len(values) != 3:
                raise ValueError(f"Expected three independent observations for {metric}, pool {pool}")
            groups.append({"pool": pool, "metric": metric, "n_runs": 3,
                           "mean": statistics.mean(values), "stddev": statistics.stdev(values)})
    write_csv(destination / "pool_summary.csv", groups, list(groups[0]))

    by_metric = {(row["pool"], row["metric"]): row for row in groups}
    def cell(pool, metric):
        value = by_metric[(pool, metric)]
        return f"{value['mean']:.2f} ± {value['stddev']:.2f}"
    lines = ["# ResNet-50 DLIO pool comparison", "",
             "Nine completed runs: three repetitions per pool, eight MPI ranks per run.",
             "", "| Pool | DLIO mean training MiB/s | Epoch 1 nominal MiB/s | "
             "Epochs 2–5 nominal MiB/s | Checkpoint 2 rank-0 MiB/s | Checkpoint 4 rank-0 MiB/s |",
             "|---|---:|---:|---:|---:|---:|"]
    for pool in POOLS:
        lines.append("| " + str(pool) + " | " + " | ".join(cell(pool, key) for key in (
            "training_mean_MiB_s", "epoch1_nominal_MiB_s", "epochs2_5_nominal_MiB_s",
            "checkpoint2_rank0_MiB_s", "checkpoint4_rank0_MiB_s")) + " |")
    def relative(pool, metric):
        return 100 * (by_metric[(pool, metric)]["mean"] / by_metric[(1, metric)]["mean"] - 1)
    lines.extend(["", "Values are mean ± sample standard deviation across three independent runs.",
                  f"Observed training mean versus pool 1: pool 3 {relative(3, 'training_mean_MiB_s'):+.1f}%, "
                  f"pool 6 {relative(6, 'training_mean_MiB_s'):+.1f}%.",
                  f"Pool 3 versus pool 1: epoch 1 {relative(3, 'epoch1_nominal_MiB_s'):+.1f}%, "
                  f"epochs 2–5 {relative(3, 'epochs2_5_nominal_MiB_s'):+.1f}%. "
                  "These are descriptive comparisons, not media-isolated effects.",
                  "Training MiB/s is DLIO's modeled sample rate × configured bytes/record; it is not a measured",
                  "POSIX read-byte counter or device throughput. Per-rank load/proc times are application-step",
                  "timings, not separate read/write throughput; do not sum them across concurrent ranks.",
                  "Checkpoint MiB/s is actual rank-0 file bytes / DLIO's rank-0 save duration; no fsync,",
                  "and no synchronized eight-rank completion time. Checkpoint data files were removed after",
                  "the runner saved their sizes and placement; other ranks' measured byte totals are unavailable.",
                  "Client caches were dropped before each run, but server cache residency was not verified.",
                  "Default resides on colva4; HDD/SSD pools span colva1–3, so media and server topology differ.",
                  "The run order was pool 1 → 3 → 6 in each repetition; time/order effects are not eliminated.",
                  "", f"Actual rank output hostnames: anjuna2 only. DLIO summary num_hosts reports 8 in "
                  f"{len(mismatches)}/9 runs; use observed per-rank hostnames instead.", ""])
    if flags:
        lines.extend(["## Timing discrepancy to investigate", "",
                      "Expected samples per epoch are calculated from the fixed profile's file, record,",
                      "batch and rank counts. The wall-implied rate is expected samples / DLIO's training",
                      "block duration; it is a consistency check, not a measured byte rate. Flagged cases:", ""])
        for pool, repetition, epoch, mismatch, gap in flags:
            lines.append(f"- Pool {pool}, repetition {repetition}, epoch {epoch}: DLIO samples/s versus "
                         f"block wall time {mismatch:+.2f}%; block-end to checkpoint-start "
                         f"{gap:.2f} s." if gap != "" else
                         f"- Pool {pool}, repetition {repetition}, epoch {epoch}: DLIO samples/s versus "
                         f"block wall time {mismatch:+.2f}%.")
        lines.extend(["", "Do not attribute this gap to checkpoint write time: it occurs **before** DLIO's",
                      "checkpoint timer starts. Keep the native metric; investigate its attribution.", ""])
    (destination / "report.md").write_text("\n".join(lines))
    print(f"Validated {len(runs)} runs, {len(epochs)} epochs, {len(ranks)} rank-epochs, "
          f"{len(checkpoints)} checkpoints; report: {destination / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Copied runs-... directory containing metrics.csv")
    parser.add_argument("--output", type=Path, help="New output directory (default: RUN/analysis)")
    parser.add_argument("--overwrite", action="store_true", help="Replace parser outputs at --output")
    args = parser.parse_args()
    report(args.run_directory, args.output or args.run_directory / "analysis", args.overwrite)


if __name__ == "__main__":
    main()
