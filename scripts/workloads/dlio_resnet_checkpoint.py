"""DLIO 2.0.0 TensorFlow checkpoint mechanism with trackable model state.

The stock saver serializes bare Tensor values as an empty checkpoint. DLIO's
checkpoint_mechanism_classname hook loads this class without editing DLIO.
"""

import tensorflow as tf
from dlio_benchmark.checkpointing.tf_checkpointing import TFCheckpointing


class ResNetCheckpointing(TFCheckpointing):
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_tensor(self, size):
        # tf.train.Checkpoint tracks Variables, not bare Tensors in a dict.
        return tf.Variable(super().get_tensor(size), trainable=False)
