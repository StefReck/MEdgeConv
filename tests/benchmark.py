"""
TF 2.3 (GTX 1080)
-----------------
Total time: 23.3371 s
Per batch: 235.7 ms
   median: 236.9 ms
Per sample: 3.6833 ms
   median: 3.7010 ms

TF 2.5 (GTX 1080)
-----------------
Total time: 23.7269 s
Per batch: 239.7 ms
   median: 238.4 ms
Per sample: 3.7448 ms
   median: 3.7243 ms

TF 2.5 & MEdgeconv 1.2 (ragged) (GTX 1080)
-----------------
Total time: 19.8413 s
Per batch: 200.4 ms
   median: 197.4 ms
Per sample: 3.1315 ms
   median: 3.0845 ms

"""

import time
import numpy as np
import tensorflow as tf
from medgeconv import DisjointEdgeConvBlock


NEXT_NEIGHBORS = 16
N_POINTS_MAX = 2000
N_FEATURES = 7
N_COORDS = 4
BATCHSIZE = 64
UNITS = (64, 64, 64)
N_NODES_OUTPUT = 2


def get_model():
    inps = (
        tf.keras.layers.Input((None, N_FEATURES), ragged=True),
        tf.keras.layers.Input((None, N_COORDS), ragged=True)
    )
    x = DisjointEdgeConvBlock(
        units=UNITS,
        next_neighbors=NEXT_NEIGHBORS,
        kernel_initializer="ones",
        pooling=True,
    )(inps)
    x = tf.keras.layers.Dense(N_NODES_OUTPUT)(x)
    model = tf.keras.Model(inps, x)
    model.compile("sgd", "mse")
    return model


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, length=1000, batches=10):
        self.length = length
        self.batches = batches
        self.x, self.y = get_data(self.batches)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        item = item % self.batches
        slc = slice(item*BATCHSIZE, (item+1)*BATCHSIZE)
        return [x_[slc] for x_ in self.x], self.y[slc]


def get_data(batches=10):
    """ Get all batches of data. Number of valid nodes is varied per graph. """
    # number of valid nodes, linearily decreasing
    n_valid_nodes = np.random.triangular(
        NEXT_NEIGHBORS+1, NEXT_NEIGHBORS+1, N_POINTS_MAX+1, batches*BATCHSIZE).astype(int)
    # exponential
    # np.minimum(N_POINTS_MAX+1, np.random.exponential(300, batches*BATCHSIZE)).astype(int) + NEXT_NEIGHBORS+1
    data = np.random.rand(n_valid_nodes.sum(), N_FEATURES).astype("float32")
    nodes = tf.RaggedTensor.from_row_lengths(data, n_valid_nodes)
    coords = nodes[:, :, :N_COORDS]

    x = nodes, coords
    y = np.zeros((batches*BATCHSIZE, N_NODES_OUTPUT))
    return x, y


def benchmark(warmup_batches=10, profile_batches=10, n_loops=100, tensorboarding=False):
    """
    Show average fit times of a simple model.

    For tensorboard profiling on GPU:
        (0.)  pip install -U tensorboard_plugin_profile
         1.   module load cuda/10.1
         2.   export LD_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64/

    To display:
        1. firefox &
        2. tensorboard --logdir benchmarks/
        3. URL:  http://localhost:6006/

    """
    model = get_model()
    model.summary()

    if tensorboarding:
        benchmark_dir = "benchmarks"
        print(f"TensorBoarding to '{benchmark_dir}'")
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=benchmark_dir,
            profile_batch=f"{warmup_batches+1}, {warmup_batches+profile_batches}",
            histogram_freq=1,
            write_images=True,
        )
        generator = DataGenerator(length=warmup_batches+profile_batches+1)
        val_data = generator[warmup_batches+profile_batches+1]
        model.fit(
            x=generator,
            validation_data=val_data,
            verbose=1,
            callbacks=[tensorboard],
            steps_per_epoch=warmup_batches+profile_batches,
        )
        print("tensorboarding complete")

    times = np.zeros(n_loops)
    generator = DataGenerator(length=n_loops)
    for i in range(n_loops):
        if i > 0 and i % 10 == 0:
            print(f"Batch {i}/{n_loops}...\t({times[i-10:i].mean():.4f} s per batch)")
        x, y = generator[i]
        start_time = time.time()
        model.fit(x, y, verbose=0)
        times[i] = time.time() - start_time
    print("\nDone! Times excluding first batch:")
    print(f"Total time: {times[1:].sum():.4f} s")
    print(f"Per batch: {1000*times[1:].mean():.1f} ms")
    print(f"   median: {1000*np.median(times[1:]):.1f} ms")
    print(f"Per sample: {1000*times[1:].mean()/BATCHSIZE:.4f} ms")
    print(f"   median: {1000*np.median(times[1:])/BATCHSIZE:.4f} ms")
    return times


if __name__ == '__main__':
    benchmark()

