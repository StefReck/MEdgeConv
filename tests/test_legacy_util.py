import tensorflow as tf
from medgeconv.legacy import util


class TestTFFunctions(tf.test.TestCase):
    def test_get_knn_from_points(self):
        points = tf.constant([
            [0, 0], [0, 3], [0, 2],
            [1, 1], [4, 5], [5, 5], [1, 1.1],
        ], dtype="float32")
        is_valid = tf.constant([
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ], dtype="int32")
        target = tf.constant([
            [2, 1], [2, 0], [1, 0],
            [6, 4], [5, 6], [4, 6], [3, 4],
        ], dtype="int32")
        result = util.get_knn_from_disjoint(points, k=2, is_valid=is_valid)
        self.assertAllEqual(result, target)

    def test_get_knn_from_points_too_few_neighbors(self):
        points = tf.constant([
            [0, 0], [0, 3], [0, 2],
            [1, 1], [4, 5], [5, 5], [1, 1.1],
        ], dtype="float32")
        is_valid = tf.constant([
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ], dtype="int32")
        with self.assertRaises(tf.errors.InvalidArgumentError):
            util.get_knn_from_disjoint(points, k=2, is_valid=is_valid)

    def test_get_knn_from_points_eager(self):
        """ map_fn turns the func into a tf.function apparently """
        tf.config.run_functions_eagerly(True)
        try:
            self.test_get_knn_from_points()
        finally:
            tf.config.run_functions_eagerly(False)

    def test_reduce_mean_valid(self):
        points = tf.constant([
            [1, 2], [3, 4],
            [9, 10], [11, 12], [13, 14], [15, 16],
        ], dtype="float32")
        is_valid = tf.constant([
            [1, 1, 0, 0],
            [1, 1, 1, 1]
        ], dtype="int32")
        target = tf.constant([
            [2, 3],
            [12, 13],
        ], dtype="float32")
        result = util.reduce_mean_valid_disjoint(
            points, is_valid=is_valid)
        self.assertAllClose(result, target)

    def test_get_graph_ids(self):
        is_valid = tf.constant([
            [1, 1, 0, 0],
            [1, 1, 1, 1]
        ], dtype="int32")
        graph_ids = util.get_graph_ids(is_valid)

        target = tf.constant([0, 0, 1, 1, 1, 1], dtype="int32")
        self.assertAllEqual(graph_ids, target)
