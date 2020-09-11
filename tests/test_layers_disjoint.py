import os
import tempfile
import tensorflow as tf
import numpy as np
import medgeconv


class TestEdgy(tf.test.TestCase):
    def setUp(self):
        self.n_points, self.n_features, self.n_coords = 5, 4, 3
        self.batchsize = 2
        self.units = [5, 2]
        self.next_neighbors = 3

        inp_points = tf.keras.layers.Input((self.n_points, self.n_features), batch_size=self.batchsize)
        inp_valid = tf.keras.layers.Input((self.n_points, ), batch_size=self.batchsize)
        inp_coords = tf.keras.layers.Input((self.n_points, self.n_coords), batch_size=self.batchsize)
        self.inps = (inp_points, inp_valid, inp_coords)
        self.edge_layer = medgeconv.DisjointEdgeConvBlock(
            units=self.units,
            next_neighbors=self.next_neighbors,
            kernel_initializer="ones",
            to_disjoint=True
        )
        self.edge_out = self.edge_layer(self.inps)[0]
        self.model = tf.keras.Model(self.inps, self.edge_out)

        points = np.arange(
            self.batchsize * self.n_points * self.n_features).reshape(
            *self.model.input_shape[0]).astype("float32")
        is_valid = np.ones((self.batchsize, self.n_points))
        is_valid[:, -1] = 0
        coords = points[:, :, :self.n_coords]
        self.x = points, is_valid, coords
        self.y = np.zeros((int(is_valid.sum()), self.units[-1]))

    def test_output_shape(self):
        self.assertTupleEqual(
            self.model.output_shape, (None, self.units[-1]))

    def test_output(self):
        points = np.arange(
            self.batchsize * self.n_points * self.n_features).reshape(
            *self.model.input_shape[0]).astype("float32")
        is_valid = np.ones((self.batchsize, self.n_points))
        is_valid[:, -1] = 0
        coords = points[:, :, :self.n_coords]

        output = self.model.predict((points, is_valid, coords))
        target = np.array([
            [5.9970026,   5.9970026],
            [95.249084,  95.249084],
            [281.07126, 281.07126],
            [483.5435, 483.5435],

            [355.6873, 355.6873],
            [558.15955, 558.15955],
            [760.6317, 760.6317],
            [963.10394, 963.10394]
        ], dtype="float32")
        np.testing.assert_almost_equal(target, output, decimal=4)

    def test_train(self):
        self.model.compile("sgd", "mse")
        self.model.train_on_batch(x=self.x, y=self.y)

    def test_loading(self):
        self.model.compile("sgd", "mse")
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "temp.h5")
            tf.keras.models.save_model(self.model, path)
            loaded = tf.keras.models.load_model(
                path, custom_objects=medgeconv.custom_objects)
            loaded.train_on_batch(x=self.x, y=self.y)

    def test_with_pooling(self):
        edge_out = medgeconv.DisjointEdgeConvBlock(
            units=self.units,
            next_neighbors=self.next_neighbors,
            kernel_initializer="ones",
            to_disjoint=True,
            pooling=True,
        )(self.inps)
        model = tf.keras.Model(self.inps, edge_out)
        output = model.predict_on_batch(self.x)
        target = np.array([
            [216.46521, 216.46521],
            [659.3956, 659.3956]], dtype="float32")
        np.testing.assert_almost_equal(target, output)


class RepeatInEager(TestEdgy):
    """ Repeat the tests above in eager mode. Mostly to get the coverage right.
    """
    def setUp(self):
        tf.config.run_functions_eagerly(True)
        super().setUp()
    
    def tearDown(self):
        tf.config.run_functions_eagerly(False)
