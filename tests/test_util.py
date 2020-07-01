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
