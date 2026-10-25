#!/usr/bin/env python3
"""Generate one complete DLIO dataset in each existing BeeGFS pool."""

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime

from dlio_pool_profiles import (DATA_ROOT, POOLS, RANKS, RESULT_ROOT, WORKLOADS,
                                check_version, select_workload)


def layout(directory, pool):
    info = subprocess.run(("sudo", "beegfs-ctl", "--getentryinfo", str(directory)),
                          capture_output=True, text=True, check=True).stdout
    found = re.search(r"Storage Pool:\s*(\d+)\b", info, re.IGNORECASE)
    if not found or int(found.group(1)) != pool or "Chunksize: 512K" not in info or not re.search(
        r"Number of storage targets: desired: 4\b", info
    ):
        raise RuntimeError(f"Unexpected BeeGFS pattern on {directory}:\n{info}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    select_workload(parser)
    args = parser.parse_args()
    check_version(parser)
    profile = WORKLOADS[args.workload]
    root = DATA_ROOT / profile.folder
    root.mkdir(parents=True, exist_ok=True)
    for pool in POOLS:
        pool_dir = root / f"pool{pool}"
        if not pool_dir.exists():
            pool_dir.mkdir()
            subprocess.run(("sudo", "beegfs-ctl", "--setpattern", "--pattern=raid0",
                            f"--storagepoolid={pool}", "--numtargets=4", "--chunksize=512k",
                            str(pool_dir)), check=True)
        layout(pool_dir, pool)
        dataset = pool_dir / "dataset"
        checkpoints = pool_dir / "checkpoints"
        dataset.mkdir(exist_ok=True)
        checkpoints.mkdir(exist_ok=True)
        layout(dataset, pool)
        layout(checkpoints, pool)

        train = dataset / "train"
        count = sum(1 for _ in train.glob(f"*.{profile.extension}"))
        if count == profile.files:
            print(f"Pool {pool}: reusing {count} existing training files", flush=True)
            continue
        if count or (train.exists() and any(train.iterdir())):
            raise RuntimeError(f"Pool {pool}: partial dataset ({count}/{profile.files}); inspect before retrying")

        if pool == 1:
            output = RESULT_ROOT / profile.folder / "generation" / (
                f"pool1-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
            output.mkdir(parents=True)
            command = ("mpirun", "-np", str(RANKS), "dlio_benchmark",
                       f"workload={profile.profile}",
                       "++workload.workflow.generate_data=True",
                       "++workload.workflow.train=False",
                       "++workload.workflow.checkpoint=False",
                       f"++workload.dataset.data_folder={dataset}",
                       f"++workload.checkpoint.checkpoint_folder={checkpoints}",
                       "++workload.checkpoint.model_size=0",
                       "++workload.checkpoint.num_layers=0",
                       "++workload.checkpoint.layer_parameters=[]",
                       f"hydra.run.dir={output}")
            print(f"Pool 1: generating {profile.files} {profile.extension} files", flush=True)
            with (output / "console.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            source = root / "pool1/dataset/train"
            if sum(1 for _ in source.glob(f"*.{profile.extension}")) != profile.files:
                raise RuntimeError("Pool 1 dataset must be complete before making pool copies")
            train.mkdir(exist_ok=True)
            print(f"Pool {pool}: copying pool 1's dataset byte-for-byte", flush=True)
            for file in source.glob(f"*.{profile.extension}"):
                shutil.copyfile(file, train / file.name)
        count = sum(1 for _ in train.glob(f"*.{profile.extension}"))
        if count != profile.files:
            raise RuntimeError(f"Pool {pool}: DLIO exited but wrote {count}/{profile.files} files")
        print(f"Pool {pool}: prepared {count} files", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(1)
