"""Fixed DLIO 2.0.0 profiles for the three-pool application comparison."""

import argparse
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path


POOLS = (1, 3, 6)
RANKS = 8
DATA_ROOT = Path("/mnt/beegfs/pfs/tejas/dlio")
RESULT_ROOT = Path.home() / "tejas/dlio"
ELIGIBLE = {
    1: set(range(401, 408)),
    3: {101, 102, 103, 201, 202, 203, 204, 301, 302, 303, 304},
    6: {104, 105, 106, 107, 205, 206, 305, 306, 307},
}


@dataclass(frozen=True)
class Workload:
    folder: str
    profile: str
    extension: str
    files: int
    model_bytes: int  # Explicit per-rank checkpoint override for this pilot.
    batch_size: int
    reader_threads: int
    tensorflow: bool


WORKLOADS = {
    "resnet": Workload("resnet50", "resnet50_v100", "tfrecord", 1024,
                       100000000, 64, 8, True),
    "unet": Workload("unet3d", "unet3d_v100", "npz", 168,
                     499153191, 4, 4, False),
    "cosmoflow": Workload("cosmoflow", "cosmoflow_v100", "tfrecord", 131072,
                          100000000, 1, 4, True),
}


def select_workload(parser: argparse.ArgumentParser):
    choices = parser.add_mutually_exclusive_group(required=True)
    for name in WORKLOADS:
        choices.add_argument(f"--{name}", action="store_const", const=name,
                             dest="workload")


def check_version(parser: argparse.ArgumentParser):
    if importlib.metadata.version("dlio-benchmark") != "2.0.0":
        parser.error("These commands require the installed DLIO 2.0.0")


def data_files(profile: Workload, pool: int):
    return sorted((DATA_ROOT / profile.folder / f"pool{pool}/dataset/train").glob(
        f"*.{profile.extension}"
    ))
