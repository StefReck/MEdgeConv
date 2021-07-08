import tensorflow as tf
from medgeconv.tf_ops import knn_graph


def get_knn_from_disjoint(nodes, k, is_valid):
    """
    Get the k nearest neighbors depending on distance.

    Parameters
    ----------
    nodes : tf.Tensor
        shape (None, n_features)
    k : int
        Number of nearest neighbors (excluding self).
    is_valid : tf.Tensor
        int32 tensor, shape (bs, n_points).

    Returns
    -------
    shape (None, k)
        int32, for each point, the indices of the points that are the
        nearest neighbors.

    """
    n_valid_nodes = tf.reduce_sum(is_valid, axis=-1)
    assert_op = tf.debugging.assert_greater(
        n_valid_nodes,
        k,
        message="one or more graphs in the batch had too few nodes for "
                "k nearest neighbors calculation!"
    )
    with tf.control_dependencies([assert_op]):
        indices, dists = knn_graph(nodes, n_valid_nodes, k+1)
    return indices[:, 1:]


def reduce_mean_valid_disjoint(nodes, is_valid):
    """
    Average over valid nodes.

    Parameters
    ----------
    nodes : tf.Tensor
        shape (None, n_features)
    is_valid : tf.Tensor
        int32 tensor shape (bs, n_points).

    Returns
    -------
    shape (bs, n_features)
        For each feature, aggeregated over all valid nodes.

    """
    graph_ids = get_graph_ids(is_valid)
    pooled = tf.math.segment_mean(nodes, graph_ids)
    pooled.set_shape(is_valid.shape[:1] + pooled.shape[1:])
    return pooled


def get_graph_ids(is_valid):
    """ Shape (None,). To which graph each node belongs to."""
    return tf.gather_nd(
        is_valid * tf.expand_dims(tf.range(tf.shape(is_valid)[0]), -1),
        tf.where(is_valid == 1))
