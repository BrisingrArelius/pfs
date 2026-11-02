"""Shared mechanics for the two separate DLIO SSD-speed-mix runners."""

import csv
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NORMAL = Path(__file__).resolve().parents[1] / "normal"
sys.path.insert(0, str(NORMAL))
sys.path.insert(0, str(NORMAL.parent))
from dlio_pool_profiles import DATA_ROOT, RANKS, RESULT_ROOT, WORKLOADS  # noqa: E402
from normal.pool_common import FIELDS, checkpoint_sizes, metrics  # noqa: E402

TARGETS = (104, 205, 305, 306)
FAST = {14: TARGETS, 15: (104, 205, 305), 16: (104, 205), 17: (104,), 18: ()}
DEVICES = {104: ("colva1", "/dev/nvme1n1p1"),
           205: ("colva2", "/dev/nvme1n1p1"),
           305: ("colva3", "/dev/nvme1n1p1"),
           306: ("colva3", "/dev/nvme2n1p1")}
CLASSES = {"resnet": "normal.run_resnet50_pools.ResNetCheckpointing",
           "unet": "normal.run_unet3d_pools.SeededPyTorchCheckpointing"}


def beegfs(*args):
    return subprocess.run(["sudo", "beegfs-ctl", *map(str, args)], check=True,
                          text=True, capture_output=True).stdout


def verify(pool):
    listing = beegfs("--liststoragepools")
    fast = len(FAST[pool])
    match = re.search(rf"^\s*{pool}\s+dlio_ssd_f{fast}_s{4-fast}\s+([\d,]+)", listing, re.M)
    members = set(map(int, match[1].split(","))) if match else set()
    if members != set(TARGETS):
        raise RuntimeError(f"Pool {pool} must contain {TARGETS}; found {members}")
    for host in ("colva1", "colva2", "colva3"):
        text = subprocess.run(["ssh", host, shlex.join(("systemctl", "show",
            "beegfs-storage.service", "-p", "IOReadBandwidthMax", "-p", "IOWriteBandwidthMax"))],
            check=True, capture_output=True, text=True).stdout
        for direction in ("IOReadBandwidthMax", "IOWriteBandwidthMax"):
            seen = {}
            for line in text.splitlines():
                if line.startswith(direction + "="):
                    device, rate = line.split("=", 1)[1].split()
                    seen[device] = int(rate)
            expected = {device: (45 if target in FAST[pool] else 15) * 1048576
                        for target, (node, device) in DEVICES.items() if node == host}
            if seen != expected:
                raise RuntimeError(f"{host}: {direction}={seen}, expected {expected}")


def pattern(path, pool):
    if not path.exists():
        path.mkdir()
        beegfs("--setpattern", "--pattern=raid0", f"--storagepoolid={pool}",
               "--numtargets=4", "--chunksize=512k", path)
    info = beegfs("--getentryinfo", path)
    if (not re.search(rf"Storage Pool:\s*{pool}\b", info, re.I)
            or "Chunksize: 512K" not in info
            or not re.search(r"Number of storage targets: desired: 4\b", info)):
        raise RuntimeError(f"Unexpected BeeGFS pattern: {path}\n{info}")


def placement(files, output):
    with output.open("w") as record:
        for file in files:
            info = beegfs("--getentryinfo", "--verbose", file)
            targets = set(map(int, re.findall(r"\+ (\d+) @ ", info)))
            if targets != set(TARGETS):
                raise RuntimeError(f"Unexpected actual stripe targets: {file}: {targets}")
            record.write(f"{file}\t{','.join(map(str, sorted(targets)))}\n")
    return ",".join(map(str, TARGETS))


def prepare(profile, pool):
    source = DATA_ROOT / profile.folder / "pool1/dataset/train"
    inputs = sorted(source.glob(f"*.{profile.extension}"))
    if len(inputs) != profile.files:
        raise RuntimeError(f"Expected {profile.files} normal input files, got {len(inputs)}")
    root = DATA_ROOT / "throttled_ssd" / profile.folder / f"pool{pool}"
    root.mkdir(parents=True, exist_ok=True)
    dataset, checkpoints = root / "dataset", root / "checkpoints"
    pattern(dataset, pool)
    pattern(checkpoints, pool)
    train = dataset / "train"
    pattern(train, pool)
    (dataset / "valid").mkdir(exist_ok=True)
    present = list(train.iterdir())
    if present and ({p.name for p in present} != {p.name for p in inputs}
                    or any((train / file.name).stat().st_size != file.stat().st_size for file in inputs)):
        raise RuntimeError(f"Partial or changed dataset: {train}")
    if not present:
        for file in inputs:
            shutil.copyfile(file, train / file.name)
    if any((train / file.name).stat().st_size != file.stat().st_size for file in inputs):
        raise RuntimeError(f"Incomplete dataset copy: {train}")
    placement((train / file.name for file in inputs), root / "dataset-osts.tsv")
    return root


def run(name, pool, pilot=False):
    if pool not in FAST or name not in CLASSES:
        raise ValueError("Expected ResNet/U-Net and a condition pool 14–18")
    if importlib.metadata.version("dlio-benchmark") != "2.0.0":
        raise RuntimeError("DLIO 2.0.0 required")
    verify(pool)
    profile = WORKLOADS[name]
    root = prepare(profile, pool)
    output = RESULT_ROOT / "throttled_ssd" / profile.folder / (
        ("pilot-" if pilot else "runs-") + f"pool{pool}-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True)
    (output / "condition.json").write_text(json.dumps({"pool": pool, "fast": FAST[pool],
        "slow": sorted(set(TARGETS) - set(FAST[pool])), "targets": TARGETS,
        "fast_MiB_s": 45, "slow_MiB_s": 15, "profile": profile.profile}, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(NORMAL.parent), str(NORMAL),
                                         env.get("PYTHONPATH", "")))
    with (output / "metrics.csv").open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(FIELDS)
        for repetition in range(1, 2 if pilot else 4):
            directory = output / f"rep{repetition}-pool{pool}"
            directory.mkdir()
            checkpoint_dir = root / "checkpoints" / f"{output.name}-rep{repetition}"
            checkpoint_dir.mkdir()
            size = min(profile.model_bytes, 16 * 1048576) if pilot else profile.model_bytes
            command = ["mpirun", "-np", str(RANKS), "dlio_benchmark", f"workload={profile.profile}",
                       f"++workload.dataset.num_files_train={profile.files}",
                       "++workload.workflow.generate_data=False", "++workload.workflow.train=True",
                       "++workload.workflow.checkpoint=True",
                       f"++workload.dataset.data_folder={root / 'dataset'}",
                       f"++workload.checkpoint.checkpoint_folder={checkpoint_dir}",
                       "++workload.checkpoint.type=all_ranks",
                       f"++workload.checkpoint.checkpoint_mechanism_classname={CLASSES[name]}",
                       "++workload.checkpoint.checkpoint_after_epoch=2",
                       "++workload.checkpoint.epochs_between_checkpoints=2",
                       f"++workload.checkpoint.model_size={size}",
                       "++workload.checkpoint.num_layers=0", "++workload.checkpoint.layer_parameters=[]",
                       f"++workload.reader.batch_size={profile.batch_size}",
                       f"++workload.reader.read_threads={profile.reader_threads}",
                       "++workload.reader.file_shuffle=seed", "++workload.reader.sample_shuffle=seed",
                       "++workload.reader.prefetch_size=2",
                       f"++workload.train.epochs={2 if pilot else 5}",
                       "++workload.train.computation_time=0", f"hydra.run.dir={directory}"]
            if pilot:
                command.insert(-1, "++workload.train.total_training_steps=2")
            subprocess.run(["sync"], check=True)
            subprocess.run(["sudo", "tee", "/proc/sys/vm/drop_caches"], input="3\n",
                           text=True, stdout=subprocess.DEVNULL, check=True)
            with (directory / "console.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            files = sorted(file for file in checkpoint_dir.rglob("*") if file.is_file())
            sizes = checkpoint_sizes(files, profile, (2,) if pilot else (2, 4), size)
            checkpoint_osts = placement(files, directory / "checkpoint-osts.tsv")
            if pilot:
                epochs = json.loads((directory / "per_epoch_stats.json").read_text())
                if len(epochs) != 2 or "ckpt1" not in epochs["2"]:
                    raise RuntimeError("Pilot checkpoint missing")
                print(f"Pilot checkpoint retained: {checkpoint_dir}")
            else:
                writer.writerows(metrics(directory, pool, repetition,
                                         ",".join(map(str, TARGETS)), checkpoint_osts, sizes))
                csvfile.flush()
                shutil.rmtree(checkpoint_dir)
    print(f"Output: {output}")
