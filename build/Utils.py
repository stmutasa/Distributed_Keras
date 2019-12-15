"""
This file will contain the utility files
"""

import tensorflow as tf
import numpy as np

# Define the flags class for variables
FLAGS = tf.compat.v1.flags


def dummy_inputs(local_batch_size):
    """
    Function to generate inputs and return a tf.data iterator
    :param local_batch_size: total batch size for this worker
    :return:
    """

    # Make a dummy mnist size batch
    dummy_inputs = np.random.uniform(-1.0, 1.0, [2000, 28, 28, 1]).astype((np.float32))
    dummy_labels = np.random.randint(0, 2, [2000, 1]).astype(np.float32)

    # Create the dataset object. Shuffle, then batch it, then make it repeat indefinitely
    dataset = tf.data.Dataset.from_tensor_slices((dummy_inputs, dummy_labels)).shuffle(200).repeat().batch(
        local_batch_size)

    # Return data as a dictionary
    return dataset
