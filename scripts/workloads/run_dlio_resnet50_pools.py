#!/usr/bin/env python3
"""Compare ResNet-50 training and checkpoint saves on three prepared BeeGFS pools.

Uses the installed DLIO 2.0.0 and existing TFRecord datasets. Checkpoint rates
are application-observed save rates, NOT fsync-inclusive or disk-drain rates.
"""

import argparse
import csv
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


POOLS = (1, 3, 6)
RANKS = 8
REPETITIONS = 3
DATA_ROOT = Path("/mnt/beegfs/pfs/tejas/dlio/resnet50")
RESULT_ROOT = Path.home() / "tejas/dlio/resnet50"
ELIGIBLE = {
    1: set(range(401, 408)),
    3: {101, 102, 103, 201, 202, 203, 204, 301, 302, 303, 304},
    6: {104, 105, 106, 107, 205, 206, 305, 306, 307},
}
TARGET = re.compile(r"\+ (\d+) @ ")
CKPT_FILE = re.compile(r"-(\d+)-(\d+)-(\d+)\.pb-\d+\.(?:index|data-\d+-of-\d+)$")
FIELDS = ("pool", "repetition", "phase", "epoch", "duration_s", "bytes_written",
          "throughput", "throughput_unit", "ost_ids", "raw_output")


def placement(files, output, pool):
    """Record per-file actual targets; return the complete observed target set."""
    used = set()
    with output.open("w") as record:
        for file in files:
            result = subprocess.run(
                ("sudo", "beegfs-ctl", "--getentryinfo", "--verbose", str(file)),
                check=True, capture_output=True, text=True,
            )
            targets = {int(value) for value in TARGET.findall(result.stdout)}
            if len(targets) != 4 or not targets <= ELIGIBLE[pool]:
                raise RuntimeError(f"Unexpected pool or stripe targets for {file}: {sorted(targets)}")
            used.update(targets)
            record.write(f"{file}\t{','.join(map(str, sorted(targets)))}\n")
    return used


def checkpoint_sizes(files, model_size_bytes):
    """Group TensorFlow .index/.data files by epoch and rank."""
    sizes = {}
    ranks = {}
    data_sizes = {}
    for file in files:
        match = CKPT_FILE.search(file.name)
        if not match:
            if file.name == "checkpoint":  # TensorFlow's latest-checkpoint pointer.
                continue
            raise RuntimeError(f"Unrecognized checkpoint file: {file}")
        epoch, _, rank = map(int, match.groups())
        sizes[epoch, rank] = sizes.get((epoch, rank), 0) + file.stat().st_size
        if ".data-" in file.name:
            ranks.setdefault(epoch, set()).add(rank)
            data_sizes[epoch, rank] = data_sizes.get((epoch, rank), 0) + file.stat().st_size
    if set(sizes) != {(epoch, rank) for epoch in (2, 4) for rank in range(RANKS)} or any(
        ranks.get(epoch) != set(range(RANKS)) for epoch in (2, 4)
    ):
        raise RuntimeError(f"Expected rank-sharded checkpoints at epochs 2 and 4: {ranks}")
    for (epoch, rank), size in data_sizes.items():
        if size < model_size_bytes * 0.9:
            raise RuntimeError(
                f"Checkpoint epoch {epoch} rank {rank} is only {size} bytes; "
                f"expected approximately {model_size_bytes} bytes"
            )
    return sizes


def prepare_client_cache():
    subprocess.run(("sync",), check=True)
    subprocess.run(("sudo", "tee", "/proc/sys/vm/drop_caches"),
                   input="3\n", text=True, stdout=subprocess.DEVNULL, check=True)


def metrics(run_dir, pool, repetition, dataset_targets, checkpoint_targets, sizes):
    summary = json.loads((run_dir / "summary.json").read_text())
    epochs = json.loads((run_dir / "per_epoch_stats.json").read_text())
    samples = summary["metric"]["train_throughput_samples_per_second"]
    if sorted(map(int, epochs)) != [1, 2, 3, 4, 5] or len(samples) != 5:
        raise RuntimeError(f"Expected five completed epochs in {run_dir}")
    rows = []
    for epoch in range(1, 6):
        data = epochs[str(epoch)]
        rows.append((pool, repetition, "training_input_block", epoch,
                     data["block1"]["duration"], "",
                     samples[epoch - 1], "samples/s", dataset_targets, str(run_dir)))
        events = [value for name, value in data.items() if name.startswith("ckpt")]
        if len(events) != (1 if epoch in (2, 4) else 0):
            raise RuntimeError(f"Unexpected checkpoint events at epoch {epoch}: {data}")
        for event in events:
            duration = float(event["duration"])
            if duration <= 0 or sizes[epoch, 0] <= 0:
                raise RuntimeError(f"Invalid checkpoint duration or size at epoch {epoch}")
            # DLIO 2.0.0 times only rank 0's save. Never divide all-rank bytes
            # by that rank-local duration to claim aggregate throughput.
            rows.append((pool, repetition, "checkpoint_save_rank0", epoch,
                         duration, sizes[epoch, 0],
                         sizes[epoch, 0] / 1048576 / duration, "MiB/s (no fsync)",
                         checkpoint_targets, str(run_dir)))
    rows.append((pool, repetition, "training_summary", "", "", "",
                 summary["metric"]["train_io_mean_MB_per_second"],
                 "MiB/s (DLIO mean)", dataset_targets, str(run_dir)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size-bytes", type=int, required=True,
                        help="Checkpoint model size per rank; ResNet profile has no value")
    args = parser.parse_args()
    if args.model_size_bytes <= 0:
        parser.error("--model-size-bytes must be positive")
    if importlib.metadata.version("dlio-benchmark") != "2.0.0":
        parser.error("This runner expects the installed DLIO 2.0.0 output and options")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + environment.get("PYTHONPATH", "")

    results = RESULT_ROOT / f"runs-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    results.mkdir(parents=True, exist_ok=False)
    osts = {}
    for pool in POOLS:
        files = sorted((DATA_ROOT / f"pool{pool}/dataset/train").glob("*.tfrecord"))
        if len(files) != 1024:
            raise RuntimeError(f"Pool {pool}: expected 1024 training TFRecords, got {len(files)}")
        osts[pool] = placement(files, results / f"pool{pool}-dataset-osts.tsv", pool)

    with (results / "metrics.csv").open("w", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(FIELDS)
        for repetition in range(1, REPETITIONS + 1):
            for pool in POOLS:
                run_dir = results / f"rep{repetition}-pool{pool}"
                checkpoint_dir = DATA_ROOT / f"pool{pool}/checkpoints/rep{repetition}-{results.name}"
                checkpoint_dir.mkdir()  # Inherits its prepared pool's stripe pattern.
                run_dir.mkdir()
                dlio_args = (
                    "mpirun", "-np", str(RANKS), "dlio_benchmark", "workload=resnet50_v100",
                    "++workload.workflow.generate_data=False",
                    "++workload.workflow.train=True",
                    "++workload.workflow.checkpoint=True",
                    f"++workload.dataset.data_folder={DATA_ROOT / f'pool{pool}/dataset'}",
                    f"++workload.checkpoint.checkpoint_folder={checkpoint_dir}",
                    "++workload.checkpoint.type=all_ranks",
                    "++workload.checkpoint.checkpoint_mechanism_classname=dlio_resnet_checkpoint.ResNetCheckpointing",
                    "++workload.checkpoint.checkpoint_after_epoch=2",
                    "++workload.checkpoint.epochs_between_checkpoints=2",
                    f"++workload.checkpoint.model_size={args.model_size_bytes}",
                    "++workload.checkpoint.num_layers=0",
                    "++workload.checkpoint.layer_parameters=[]",
                    "++workload.reader.batch_size=64",
                    "++workload.reader.file_shuffle=seed",
                    "++workload.reader.sample_shuffle=seed",
                    "++workload.reader.prefetch_size=2",
                    "++workload.train.epochs=5",
                    "++workload.train.computation_time=0",
                    f"hydra.run.dir={run_dir}",
                )
                print(f"Running repetition {repetition}, pool {pool}", flush=True)
                prepare_client_cache()
                with (run_dir / "console.log").open("w") as log:
                    subprocess.run(dlio_args, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
                if not files:
                    raise RuntimeError(f"No checkpoint files in {checkpoint_dir}")
                sizes = checkpoint_sizes(files, args.model_size_bytes)
                checkpoint_osts = placement(files, run_dir / "checkpoint-osts.tsv", pool)
                rows = metrics(run_dir, pool, repetition,
                               ",".join(map(str, sorted(osts[pool]))),
                               ",".join(map(str, sorted(checkpoint_osts))), sizes)
                writer.writerows(rows)
                report.flush()
                # Only delete this run's checkpoints after metrics and OSTs are saved.
                # Keep files on failure so an interrupted run can be inspected.
                shutil.rmtree(checkpoint_dir)
    print(f"Results: {results / 'metrics.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(1)
