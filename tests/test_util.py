import tensorflow as tf
from medgeconv import util


class TestTFFunctions(tf.test.TestCase):
    def test_pdist(self):
        points = tf.constant([
            [[0, 0], [0, 1]],
            [[1, 1], [4, 5]],
            [[-1, -1], [-1, -1]],
        ], dtype="float32")

        target = tf.constant([
            [[0, 1], [1, 0]],
            [[0, 5], [5, 0]],
            [[0, 0], [0, 0]],
        ], dtype="float32")
        result = util.pdist(points)
        self.assertAllClose(result, target)

    def test_get_knn_from_points(self):
        points = tf.constant([
            [[0, 0], [0, 3], [0, 2], [0, 1]],
            [[1, 1], [4, 5], [5, 5], [1, 1.1]],
        ], dtype="float32")
        is_valid = tf.constant([
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ], dtype="float32")
        target = tf.constant([
            [[2, 1], [2, 0], [1, 0], [0, 0]],
            [[3, 1], [2, 3], [1, 3], [0, 1]],
        ], dtype="float32")
        result = util.get_knn_from_points(points, k=2, is_valid=is_valid)
        self.assertAllClose(result, target)

    def test_reduce_mean_valid(self):
        points = tf.constant([
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[9, 10], [11, 12], [13, 14], [15, 16]],
        ], dtype="float32")
        is_valid = tf.constant([
            [1, 1, 0, 0],
            [1, 1, 1, 1]
        ], dtype="float32")
        target = tf.constant([
            [2, 3],
            [12, 13],
        ], dtype="float32")
        result = util.reduce_mean_valid(
            points, is_valid=is_valid, divide_by_nodes=True)
        self.assertAllClose(result, target)

    def test_flatten_graphs(self):
        points = tf.constant([
            [[1, 2], [3, 4], [0, 0], [0, 0]],
            [[9, 10], [11, 12], [13, 14], [15, 16]],
        ], dtype="float32")
        is_valid = tf.constant([
            [1, 1, 0, 0],
            [1, 1, 1, 1]
        ], dtype="float32")
        flattened, rev = util.flatten_graphs(nodes=points, is_valid=is_valid)

        target_flattened = tf.constant([
            [1, 2], [3, 4], [9, 10], [11, 12], [13, 14], [15, 16]
        ])
        self.assertAllClose(flattened, target_flattened)
        self.assertAllClose(rev(flattened), points)
