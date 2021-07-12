import tensorflow as tf
from medgeconv import util


class TestTFFunctions(tf.test.TestCase):
    def test_get_knn_from_points(self):
        points = tf.constant([
            [0, 0], [0, 3], [0, 2],
            [1, 1], [4, 5], [5, 5], [1, 1.1],
        ], dtype="float32")
        n_nodes = tf.constant([3, 4], dtype="int32")
        points = tf.RaggedTensor.from_row_lengths(points, n_nodes)

        target = tf.constant([
            [2, 1], [2, 0], [1, 0],
            [6, 4], [5, 6], [4, 6], [3, 4],
        ], dtype="int32")
        result = util.get_knn(points, k=2)
        self.assertAllEqual(result, target)

    def test_get_knn_from_points_too_few_neighbors(self):
        points = tf.constant([
            [0, 0], [0, 3],
            [1, 1], [4, 5], [5, 5], [1, 1.1],
        ], dtype="float32")
        n_nodes = tf.constant([2, 4], dtype="int32")
        points = tf.RaggedTensor.from_row_lengths(points, n_nodes)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            util.get_knn(points, k=2)

    def test_get_knn_from_points_eager(self):
        """ map_fn turns the func into a tf.function apparently """
        tf.config.run_functions_eagerly(True)
        try:
            self.test_get_knn_from_points()
        finally:
            tf.config.run_functions_eagerly(False)
