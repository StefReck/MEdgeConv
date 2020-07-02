import os
import tempfile
import tensorflow as tf
import numpy as np
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


class TestModelFunctionality(tf.test.TestCase):
    def setUp(self):
        n_points, n_features, n_coords = 4, 5, 3
        batchsize = 10
        units = [5, 4]
        inp_points = tf.keras.layers.Input((n_points, n_features),
                                           batch_size=batchsize)
        inp_valid = tf.keras.layers.Input((n_points,), batch_size=batchsize)
        inp_coords = tf.keras.layers.Input((n_points, n_coords),
                                           batch_size=batchsize)
        inps = (inp_points, inp_valid, inp_coords)

        x = layers.EdgeConv(
            units=units,
            next_neighbors=3,
            pooling="GlobalAvgValidPooling",
            batchnorm_for_nodes=True)(inps)
        self.model = tf.keras.Model(inps, x)
        self.model.compile("sgd", "mse")
        self.x = [
            np.ones((batchsize, n_points, n_features)),  # points
            np.ones((batchsize, n_points)),  # is_valid
            np.ones((batchsize, n_points, n_coords)),  # coords
        ]
        self.y = np.zeros((batchsize, units[-1]))

    def test_train(self):
        self.model.train_on_batch(x=self.x, y=self.y)

    def test_loading(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "temp.h5")
            tf.keras.models.save_model(self.model, path)
            loaded = tf.keras.models.load_model(
                path, custom_objects=layers.custom_objects)
