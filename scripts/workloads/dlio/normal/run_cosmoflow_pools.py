#!/usr/bin/env python3
"""CosmoFlow: pilot or compare the existing datasets on BeeGFS pools 1/3/6."""

import argparse

from dlio_pool_profiles import WORKLOADS
from pool_common import run
from dlio_darshan import add_argument


if __name__ != "__main__":
    import tensorflow as tf
    from dlio_benchmark.checkpointing.tf_checkpointing import TFCheckpointing
    from dlio_benchmark.common.enumerations import Shuffle

    class CosmoFlowCheckpointing(TFCheckpointing):
        """Save trackable model state and honor seeded file order."""

        _instance = None

        @classmethod
        def get_instance(cls):
            if cls._instance is None:
                cls._instance = cls()
                if cls._instance.args.file_shuffle == "seed":
                    cls._instance.args.file_shuffle = Shuffle.SEED
            return cls._instance

        def get_tensor(self, size):
            return tf.Variable(super().get_tensor(size), trainable=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--repetitions",type=int,default=3,help="Full invocations per pool (default: 3; use 1 for capture plots)")
    add_argument(parser)
    args = parser.parse_args()
    if args.repetitions<1:
        parser.error("--repetitions must be positive")
    run(WORKLOADS["cosmoflow"], "run_cosmoflow_pools.CosmoFlowCheckpointing",
        model_bytes=WORKLOADS["cosmoflow"].model_bytes, pilot=args.pilot, darshan=args.darshan,
        repetitions=args.repetitions)


if __name__ == "__main__":
    main()
