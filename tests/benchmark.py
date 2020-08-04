import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from medgeconv.layers import EdgeConv


NEXT_NEIGHBORS = 16
N_POINTS_MAX = 2000
N_FEATURES = 7
N_COORDS = 4
BATCHSIZE = 64
UNITS = (64, 64, 64)


def get_model():
    inps = (
        tf.keras.layers.Input((N_POINTS_MAX, N_FEATURES), batch_size=BATCHSIZE),
        tf.keras.layers.Input((N_POINTS_MAX,), batch_size=BATCHSIZE),
        tf.keras.layers.Input((N_POINTS_MAX, N_COORDS), batch_size=BATCHSIZE)
    )
    edge_layer = EdgeConv(
        units=UNITS,
        next_neighbors=NEXT_NEIGHBORS,
        kernel_initializer="ones",
    )
    edge_out = edge_layer(inps)
    model = tf.keras.Model(inps, edge_out)
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

    def plot_n_nodes(self):
        plt.hist(self.x[1].sum(-1), bins=50)
        plt.xlabel("valid nodes")


def get_data(batches=10):
    points = np.random.rand(batches*BATCHSIZE, N_POINTS_MAX, N_FEATURES).astype("float32")
    is_valid = np.zeros((batches*BATCHSIZE, N_POINTS_MAX)).astype("float32")
    coords = points[:, :, :N_COORDS]
    # number of valid nodes, linearily decreasing
    n_valid_nodes = np.random.triangular(
        NEXT_NEIGHBORS+1, NEXT_NEIGHBORS+1, N_POINTS_MAX+1, batches*BATCHSIZE).astype(int)
    # exponential
    # np.minimum(N_POINTS_MAX+1, np.random.exponential(300, batches*BATCHSIZE)).astype(int) + NEXT_NEIGHBORS+1

    for i, l in enumerate(n_valid_nodes):
        is_valid[i, :l] = 1.

    x = points, is_valid, coords
    y = np.zeros((batches*BATCHSIZE, N_POINTS_MAX, UNITS[-1]))
    return x, y


def benchmark(warmup_batches=10, profile_batches=10, n_loops=100, tensorboarding=True, plotit=True):
    """
    For profiling on GPU:
        (0.)  pip install -U tensorboard_plugin_profile
         1.   module load cuda/10.1
         2.   export LD_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64/

    To display:
        1. firefox &
        2. tensorboard --logdir benchmarks/
        3. URL:  http://localhost:6006/

    """
    model = get_model()

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
        if i % 10 == 0:
            print(f"Batch {i}/{n_loops}...")
        x, y = generator[i]
        start_time = time.time()
        model.fit(x, y, verbose=0)
        times[i] = time.time() - start_time
    print("Done! Times excluding first batch:")
    print(f"Total time: {times[1:].sum():.4f}")
    print(f"Per batch: {times[1:].mean():.4f}")
    print(f"   median: {np.median(times[1:]):.4f}")
    if plotit:
        plt.plot(times[1:])
    return times


def _get_inp(batch_size=2):
    points = tf.keras.Input((5, 2), batch_size=batch_size)
    is_valid = tf.keras.Input((5, ), batch_size=batch_size)
    coords = tf.keras.Input((5, 2), batch_size=batch_size)
    return points, is_valid, coords


def _get_inp_numpy(batch_size=2):
    points = np.arange(0, batch_size*5*2, dtype="float32").reshape((batch_size, 5, 2))
    is_valid = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype="float32")
    coords = points
    return points, is_valid, coords


def _get_inp_const(batch_size=2):
    return [tf.constant(x) for x in _get_inp_numpy(batch_size)]


if __name__ == '__main__':
    benchmark(tensorboarding=False, plotit=False)

