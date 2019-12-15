"""
Distributed tensorflow 1.0 Keras example

Last ditch effort to get it to work before SIIM. We will use TF 2.0 (not bad) with Keras (Ugh)

"""

# This is the baseline process that every container will have a copy of
import tensorflow as tf
import build.Utils as utils
import os, json
import warnings

# Define command line arguments to define how this process will run. TF.app.flags is similar to sys.argv
FLAGS = tf.compat.v1.flags

# Define some of the default command line argumentstensorflow 2.0 code coverter

tf.compat.v1.flags.DEFINE_integer('task_index', 1,
                                  'Index of task within the job')

tf.compat.v1.flags.DEFINE_string('worker_hosts', 'localhost:2223,localhost:2224',
                                 'Comma separated list of hostname:port pairs')

# This represents the per worker batch size
batch_size = 128


def main(_):
    # Parse the command line arguments to get a lists of parameter servers and hosts
    try:
        ps_hosts = tf.compat.v1.flags.FLAGS.ps_hosts.split(",")
    except:
        warnings.warn('No Parameter Server Defined')
    worker_hosts = tf.compat.v1.flags.FLAGS.worker_hosts.split(",")

    # Prevent session from using all the gpu memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    except:
        warnings.warn('Cant set GPU growth, are we detecting all the GPU? %s' % gpus)

    # TF_CONFIG is used to specify the cluster configuration on each worker that is part of the cluster.
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': worker_hosts
        },
        'task': {'type': 'worker', 'index': tf.compat.v1.flags.FLAGS.task_index}
    })

    # Distribution strategy: MultiWorkerMirrored uses every gpu on multiple computers. Init function parses TF_CONFIG
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    # Som variables we will need
    NUM_WORKERS = len(worker_hosts)
    GLOBAL_BATCH_SIZE = batch_size * NUM_WORKERS

    # Everything run within the context of the distribution strategy. This will handle placing variables
    with strategy.scope():

        # Run the input custom function and bring back the data iterator object
        inputs = generate_inputs(batch_size)

        # Get the keras model
        model = define_model()

    # Now Fit the model
    model.fit(x=inputs, epochs=5)


def generate_inputs(local_batch_size):
    """
    Function to generate inputs and return a tf.data iterator
    Any input function can be placed here
    :param local_batch_size: total batch size for this worker
    :return:
    """

    return utils.dummy_inputs(local_batch_size)


def define_model():
    """
    Defines the model. Any custom model can be placed here
    :param inputs: tf.data inputs
    :return: The calculated logits
    """

    # Define the keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model (define loss, optimizer, and acc)
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Actually returns logits and L2 loss
    return model


if __name__ == "__main__":
    tf.compat.v1.app.run(main=main)
