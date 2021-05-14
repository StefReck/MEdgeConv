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
            to_disjoint=True,
            pooling=True,
        )
        self.edge_out = self.edge_layer(self.inps)
        self.model = tf.keras.Model(self.inps, self.edge_out)

        points = np.arange(
            self.batchsize * self.n_points * self.n_features).reshape(
            *self.model.input_shape[0]).astype("float32")
        is_valid = np.ones((self.batchsize, self.n_points), dtype="float32")
        is_valid[:, -1] = 0
        coords = points[:, :, :self.n_coords]
        self.x = points, is_valid, coords
        self.y = np.zeros((self.batchsize, self.units[-1]))

    def test_output_shape(self):
        self.assertTupleEqual(
            self.model.output_shape, (self.batchsize, self.units[-1]))

    def test_output(self):
        points = np.arange(
            self.batchsize * self.n_points * self.n_features).reshape(
            *self.model.input_shape[0]).astype("float32")
        is_valid = np.ones((self.batchsize, self.n_points))
        is_valid[:, -1] = 0
        coords = points[:, :, :self.n_coords]

        output = self.model.predict((points, is_valid, coords))
        target = np.array([
            [216.46521, 216.46521],
            [659.3956, 659.3956]], dtype="float32")
        np.testing.assert_almost_equal(target, output, decimal=4)

    def test_train(self):
        self.model.compile("sgd", "mse")
        train_loss = self.model.train_on_batch(x=self.x, y=self.y)
        np.testing.assert_almost_equal(train_loss, 1.4926, decimal=4)

    def test_loading(self):
        self.model.compile("sgd", "mse")
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "temp.h5")
            tf.keras.models.save_model(self.model, path)
            loaded = tf.keras.models.load_model(
                path, custom_objects=medgeconv.custom_objects)
            train_loss = loaded.train_on_batch(x=self.x, y=self.y)
            np.testing.assert_almost_equal(train_loss, 1.4926, decimal=4)


class RepeatInEager(TestEdgy):
    """ Repeat the tests above in eager mode. Mostly to get the coverage right.
    """
    def setUp(self):
        tf.config.run_functions_eagerly(True)
        super().setUp()
    
    def tearDown(self):
        tf.config.run_functions_eagerly(False)
