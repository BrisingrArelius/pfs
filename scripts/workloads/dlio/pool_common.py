"""Shared, non-CLI mechanics for the three separate DLIO pool runners."""

import csv
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from dlio_pool_profiles import DATA_ROOT, ELIGIBLE, POOLS, RANKS, RESULT_ROOT, data_files


TARGET = re.compile(r"\+ (\d+) @ ")
TF_FILE = re.compile(r"-(\d+)-(\d+)-(\d+)\.pb-\d+\.(?:index|data-\d+-of-\d+)$")
PT_FILE = re.compile(r"-(\d+)-(\d+)-(\d+)\.pt$")
FIELDS = ("pool", "repetition", "phase", "epoch", "duration_s", "bytes_written",
          "throughput", "throughput_unit", "ost_ids", "raw_output")


def placement(files, output, pool):
    observed = set()
    with output.open("w") as record:
        for file in files:
            info = subprocess.run(("sudo", "beegfs-ctl", "--getentryinfo", "--verbose", str(file)),
                                  capture_output=True, text=True, check=True).stdout
            targets = {int(value) for value in TARGET.findall(info)}
            if len(targets) != 4 or not targets <= ELIGIBLE[pool]:
                raise RuntimeError(f"Unexpected stripe targets for {file}: {sorted(targets)}")
            observed.update(targets)
            record.write(f"{file}\t{','.join(map(str, sorted(targets)))}\n")
    return ",".join(map(str, sorted(observed)))


def checkpoint_sizes(files, profile, epochs, model_bytes):
    sizes, payload, ranks = {}, {}, {}
    for file in files:
        if profile.tensorflow and file.name == "checkpoint":
            continue  # TensorFlow's overwritten latest-checkpoint pointer.
        match = (TF_FILE if profile.tensorflow else PT_FILE).search(file.name)
        if not match:
            raise RuntimeError(f"Unrecognized checkpoint file: {file}")
        epoch, _, rank = map(int, match.groups())
        sizes[epoch, rank] = sizes.get((epoch, rank), 0) + file.stat().st_size
        if not profile.tensorflow or ".data-" in file.name:
            payload[epoch, rank] = payload.get((epoch, rank), 0) + file.stat().st_size
            ranks.setdefault(epoch, set()).add(rank)
    expected = {(epoch, rank) for epoch in epochs for rank in range(RANKS)}
    if set(sizes) != expected or set(payload) != expected or any(
        ranks[epoch] != set(range(RANKS)) for epoch in epochs
    ):
        raise RuntimeError(f"Expected complete eight-rank checkpoints in epochs {epochs}: {ranks}")
    for (epoch, rank), size in payload.items():
        if size < model_bytes * 0.9:
            raise RuntimeError(f"Epoch {epoch} rank {rank}: only {size} of ~{model_bytes} bytes saved")
    return sizes


def metrics(output, pool, repetition, dataset_osts, checkpoint_osts, sizes):
    summary = json.loads((output / "summary.json").read_text())
    epochs = json.loads((output / "per_epoch_stats.json").read_text())
    samples = summary["metric"]["train_throughput_samples_per_second"]
    if sorted(map(int, epochs)) != [1, 2, 3, 4, 5] or len(samples) != 5:
        raise RuntimeError(f"Expected five completed epochs in {output}")
    rows = []
    for epoch in range(1, 6):
        data = epochs[str(epoch)]
        rows.append((pool, repetition, "training_input_block", epoch,
                     data["block1"]["duration"], "", samples[epoch - 1],
                     "samples/s", dataset_osts, str(output)))
        events = [event for name, event in data.items() if name.startswith("ckpt")]
        if len(events) != (1 if epoch in (2, 4) else 0):
            raise RuntimeError(f"Unexpected checkpoint events for epoch {epoch} in {output}")
        for event in events:
            duration = float(event["duration"])
            if duration <= 0:
                raise RuntimeError(f"Invalid checkpoint duration in {output}")
            # DLIO 2.0.0 times rank 0 only; do not claim eight-rank throughput.
            size = sizes[epoch, 0]
            rows.append((pool, repetition, "checkpoint_save_rank0", epoch,
                         duration, size, size / 1048576 / duration,
                         "MiB/s (no fsync)", checkpoint_osts, str(output)))
    rows.append((pool, repetition, "training_summary", "", "", "",
                 summary["metric"]["train_io_mean_MB_per_second"],
                 "MiB/s (DLIO mean)", dataset_osts, str(output)))
    return rows


def run(profile, checkpoint_class, model_bytes, pilot=False):
    if importlib.metadata.version("dlio-benchmark") != "2.0.0":
        raise RuntimeError("These runners require the installed DLIO 2.0.0")
    root = DATA_ROOT / profile.folder
    output = RESULT_ROOT / profile.folder / (
        ("pilot-" if pilot else "runs-") + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    pools = (1,) if pilot else POOLS
    targets = {}
    for pool in pools:
        files = data_files(profile, pool)
        if len(files) != profile.files:
            raise RuntimeError(f"Pool {pool}: expected {profile.files} input files; found {len(files)}")
        # 524,288 placement commands per CosmoFlow pool would dominate the run.
        if len(files) > 1024:
            stride = max(1, len(files) // 64)
            files = list(dict.fromkeys(files[::stride][:64] + files[-1:]))
        targets[pool] = placement(files, output / f"pool{pool}-dataset-osts.tsv", pool)

    with (output / "metrics.csv").open("w", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(FIELDS)
        for repetition in range(1, 2 if pilot else 4):
            for pool in pools:
                run_dir = output / f"rep{repetition}-pool{pool}"
                checkpoint_dir = root / f"pool{pool}/checkpoints/{output.name}-rep{repetition}"
                checkpoint_dir.mkdir()
                run_dir.mkdir()
                size = min(model_bytes, 16 * 1024 * 1024) if pilot else model_bytes
                command = [
                    "mpirun", "-np", str(RANKS), "dlio_benchmark", f"workload={profile.profile}",
                    "++workload.workflow.generate_data=False",
                    "++workload.workflow.train=True",
                    "++workload.workflow.checkpoint=True",
                    f"++workload.dataset.data_folder={root / f'pool{pool}/dataset'}",
                    f"++workload.checkpoint.checkpoint_folder={checkpoint_dir}",
                    "++workload.checkpoint.type=all_ranks",
                    f"++workload.checkpoint.checkpoint_mechanism_classname={checkpoint_class}",
                    "++workload.checkpoint.checkpoint_after_epoch=2",
                    "++workload.checkpoint.epochs_between_checkpoints=2",
                    f"++workload.checkpoint.model_size={size}",
                    "++workload.checkpoint.num_layers=0",
                    "++workload.checkpoint.layer_parameters=[]",
                    f"++workload.reader.batch_size={profile.batch_size}",
                    f"++workload.reader.read_threads={profile.reader_threads}",
                    "++workload.reader.file_shuffle=seed",
                    "++workload.reader.sample_shuffle=seed",
                    "++workload.reader.prefetch_size=2",
                    f"++workload.train.epochs={2 if pilot else 5}",
                    "++workload.train.computation_time=0",
                    f"hydra.run.dir={run_dir}",
                ]
                if pilot:
                    command.insert(-1, "++workload.train.total_training_steps=2")
                print(f"{profile.profile}: repetition {repetition}, pool {pool}", flush=True)
                subprocess.run(("sync",), check=True)
                subprocess.run(("sudo", "tee", "/proc/sys/vm/drop_caches"),
                               input="3\n", text=True, stdout=subprocess.DEVNULL, check=True)
                with (run_dir / "console.log").open("w") as log:
                    subprocess.run(command, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
                sizes = checkpoint_sizes(files, profile, (2,) if pilot else (2, 4), size)
                observed = placement(files, run_dir / "checkpoint-osts.tsv", pool)
                if pilot:
                    stats = json.loads((run_dir / "per_epoch_stats.json").read_text())
                    if len(stats) != 2 or "ckpt1" not in stats["2"]:
                        raise RuntimeError("Pilot did not record the epoch-2 checkpoint")
                    print(f"Pilot: {run_dir}; checkpoint files retained at {checkpoint_dir}")
                else:
                    writer.writerows(metrics(run_dir, pool, repetition,
                                             targets[pool], observed, sizes))
                    report.flush()
                    shutil.rmtree(checkpoint_dir)
    print(f"Output: {output}")
