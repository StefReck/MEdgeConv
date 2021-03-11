import os
import tempfile
import tensorflow as tf
import numpy as np
import medgeconv
from medgeconv import layers


class TestLayers(tf.test.TestCase):
    def test_global_average_valid_pooling(self):
        n_points, n_features = 4, 3
        inp_points = tf.keras.layers.Input((n_points, n_features))
        inp_valid = tf.keras.layers.Input((n_points, ))

        x = layers.GlobalAvgValidPooling()((inp_points, inp_valid))
        self.assertListEqual([None, n_features], x.shape.as_list())

    def test_get_edge_features(self):
        n_points, n_features, n_coords = 4, 5, 3
        batchsize = 10
        k = 3
        inp_points = tf.keras.layers.Input((n_points, n_features), batch_size=batchsize)
        inp_valid = tf.keras.layers.Input((n_points, ), batch_size=batchsize)
        inp_coords = tf.keras.layers.Input((n_points, n_coords), batch_size=batchsize)

        x = layers.GetEdgeFeatures(next_neighbors=k)((inp_points, inp_valid, inp_coords))
        target_shape = [batchsize, n_points, k, n_features]
        self.assertListEqual(target_shape, x[0].shape.as_list())
        self.assertListEqual(target_shape, x[1].shape.as_list())

    def test_edge_conv(self):
        n_points, n_features, n_coords = 4, 5, 3
        batchsize = 10
        units = [5, 4]
        inp_points = tf.keras.layers.Input((n_points, n_features), batch_size=batchsize)
        inp_valid = tf.keras.layers.Input((n_points, ), batch_size=batchsize)
        inp_coords = tf.keras.layers.Input((n_points, n_coords), batch_size=batchsize)

        x = layers.EdgeConv(units=units, next_neighbors=3)(
            (inp_points, inp_valid, inp_coords))
        self.assertListEqual([batchsize, n_points, units[-1]], x.shape.as_list())


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
        self.edge_layer = layers.EdgeConv(
            units=self.units,
            next_neighbors=self.next_neighbors,
            kernel_initializer="ones",
        )
        self.edge_out = self.edge_layer(self.inps)
        self.model = tf.keras.Model(self.inps, self.edge_out)

        points = np.arange(
            self.batchsize * self.n_points * self.n_features).reshape(
            *self.model.input_shape[0]).astype("float32")
        is_valid = np.ones((self.batchsize, self.n_points))
        is_valid[:, -1] = 0
        coords = points[:, :, :self.n_coords]
        self.x = points, is_valid, coords
        self.y = np.zeros((self.batchsize, self.n_points, self.units[-1]))

    def test_output_shape(self):
        self.assertTupleEqual(
            self.model.output_shape,
            (self.batchsize, self.n_points, self.units[-1]))

    def test_output(self):
        points = np.arange(
            self.batchsize * self.n_points * self.n_features).reshape(
            *self.model.input_shape[0]).astype("float32")
        is_valid = np.ones((self.batchsize, self.n_points))
        is_valid[:, -1] = 0
        coords = points[:, :, :self.n_coords]

        output = self.model.predict((points, is_valid, coords))
        target = np.array([[
            [5.9970026,   5.9970026],
            [95.249084,  95.249084],
            [281.07126, 281.07126],
            [483.5435, 483.5435],
            [0.,   0.]],

            [[355.6873, 355.6873],
            [558.15955, 558.15955],
            [760.6317, 760.6317],
            [963.10394, 963.10394],
            [0.,   0.]]], dtype="float32")
        np.testing.assert_almost_equal(target, output, decimal=4)

    def test_train(self):
        self.model.compile("sgd", "mse")
        self.model.train_on_batch(x=self.x, y=self.y)

    def test_loading(self):
        """
        TODO This raises
         tensor_equals() missing 1 required positional argument: 'other'
         in the line 'load_model', maybe because tf.2.4 requires stuff to be inside
         keras layers. Maybe drop support?
        """
        self.model.compile("sgd", "mse")
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "temp.h5")
            tf.keras.models.save_model(self.model, path)
            loaded = tf.keras.models.load_model(
                path, custom_objects=medgeconv.custom_objects)
            loaded.train_on_batch(x=self.x, y=self.y)

    def test_with_pooling(self):
        x = layers.GlobalAvgValidPooling()((self.edge_out, self.inps[1]))
        model = tf.keras.Model(self.inps, x)
        output = model.predict_on_batch(self.x)
        target = np.array([
            [216.46521, 216.46521],
            [659.3956, 659.3956]], dtype="float32")
        np.testing.assert_almost_equal(target, output)
