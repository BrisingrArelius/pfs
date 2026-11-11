#!/usr/bin/env python3
"""ResNet-50: pilot or compare the existing datasets on BeeGFS pools 1/3/6."""

import argparse

from dlio_pool_profiles import WORKLOADS
from pool_common import run


# DLIO imports this module inside each MPI rank. Its stock TensorFlow saver
# checkpoints bare Tensors as empty files; Variables are trackable.
if __name__ != "__main__":
    import tensorflow as tf
    from dlio_benchmark.checkpointing.tf_checkpointing import TFCheckpointing
    from dlio_benchmark.common.enumerations import Shuffle

    class ResNetCheckpointing(TFCheckpointing):
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
    parser.add_argument("--model-size-bytes", type=int, required=True,
                        help="Synthetic model state per rank (previous pilot: 100000000)")
    args = parser.parse_args()
    if args.model_size_bytes <= 0:
        parser.error("--model-size-bytes must be positive")
    run(WORKLOADS["resnet"], "run_resnet50_pools.ResNetCheckpointing",
        model_bytes=args.model_size_bytes, pilot=args.pilot)


if __name__ == "__main__":
    main()
