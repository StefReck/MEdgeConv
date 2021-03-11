import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_knn_op_module = load_library.load_op_library(
    resource_loader.get_path_to_datafile('./_knn_graph_ops.so'))
_knn_graph_op = _knn_op_module.knn_graph


def knn_graph(nodes, n_nodes_per_graph, k):
    """
    k-nearest neighbors on a batch of graphs.

    Parameters
    ----------
    nodes : tf.Tensor
        The disjoint node features. shape (n_nodes, n_dims), float32.
    n_nodes_per_graph : tf.Tensor
        The number of nodes per graph. Shape (batchsize, ), int32.
        E.g., if there are 3 graphs in the batch with 3, 5 and 4 nodes,
        n_nodes_per_graph is [3, 5, 4].
    k : int
        The number of nearest neighbors, including self connection.
        I.e. k=16 means self and 15 neighbors.

    Returns
    -------
    indices : tf.Tensor
        The nearest neighbor indices. Shape (n_nodes * k, ).
        Use tf.reshape(indices, (-1, k) to get into 2d shape.
    dists : tf.Tensor
        The distances to the neighbors. Shape (n_nodes+k, ).
        Use tf.reshape(indices, (-1, k) to get into 2d shape.

    """
    x_ptr = tf.concat([[0, ], tf.math.cumsum(n_nodes_per_graph)], axis=0)
    indices, dists = _knn_graph_op(nodes, x_ptr, k)
    return indices, dists
