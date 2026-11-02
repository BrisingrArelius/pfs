#!/usr/bin/env python3
"""U-Net 3D: pilot or compare the existing datasets on BeeGFS pools 1/3/6."""

import argparse

from dlio_pool_profiles import WORKLOADS
from pool_common import run


if __name__ != "__main__":
    from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
    from dlio_benchmark.common.enumerations import Shuffle

    class SeededPyTorchCheckpointing(PyTorchCheckpointing):
        """Use stock torch.save; normalize DLIO 2.0.0's file-shuffle seed."""

        _instance = None

        @classmethod
        def get_instance(cls):
            if cls._instance is None:
                cls._instance = cls()
                if cls._instance.args.file_shuffle == "seed":
                    cls._instance.args.file_shuffle = Shuffle.SEED
            return cls._instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(WORKLOADS["unet"], "run_unet3d_pools.SeededPyTorchCheckpointing",
        model_bytes=WORKLOADS["unet"].model_bytes, pilot=args.pilot)


if __name__ == "__main__":
    main()
