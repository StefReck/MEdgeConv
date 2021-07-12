import tensorflow as tf
from medgeconv.tf_ops import knn_graph


def get_knn(nodes, k):
    """
    Get the k nearest neighbors depending on distance.

    Parameters
    ----------
    nodes : tf.Tensor
        shape (None, n_features)
    k : int
        Number of nearest neighbors (excluding self).

    Returns
    -------
    shape (None, k)
        int32, for each point, the indices of the points that are the
        nearest neighbors.

    """
    n_valid_nodes = tf.cast(nodes.nested_row_lengths()[0], "int32")
    nodes_disjoint = nodes.merge_dims(0, 1)
    assert_op = tf.debugging.assert_greater(
        n_valid_nodes,
        k,
        message="one or more graphs in the batch had too few nodes for "
                "k nearest neighbors calculation!"
    )
    with tf.control_dependencies([assert_op]):
        indices, dists = knn_graph(nodes_disjoint, n_valid_nodes, k+1)
    return indices[:, 1:]


def get_xixj(nodes_disjoint, knn, k):
    """
    Get the features of each edge in the graph.

    Paramters
    ---------
    nodes : tf.Tensor
        shape (None, n_features)
    knn : tf.Tensor
        shape (None, k)
        int32, for each point, the indices of the points that are the
        nearest neighbors.
    k : int
        Number of nearest neighbors (excluding self).

    Returns
    -------
    tuple
        Two Tensors with shape (None, k, n_features).
        --> [?, i, j] desribes the edge between points i and j.
        The first matrix is xi, i.e. for each edge the central point
        The 2nd matrix is xj, i.e. for each edge the other point.

    """
    nodes_central = tf.tile(
        tf.expand_dims(nodes_disjoint, axis=-2),
        [1, k, 1]
    )

    # TODO this produces a 'Converting sparse IndexedSlices to a dense Tensor
    #  of unknown shape.' warning. Thats because nodes has an unknown shape
    #  (None, n_features), along first axis is gathered.
    nodes_neighbors = tf.gather(nodes_disjoint, knn)

    return nodes_central, nodes_neighbors
