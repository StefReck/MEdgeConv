import tensorflow as tf


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
    def func(args):
        graph = args
        euc_dist = pdist(graph, single_mode=True, take_sqrt=False)
        return tf.math.top_k(-euc_dist, k=k + 1)[1][:, 1:]

    n_valid_nodes = tf.reduce_sum(is_valid, axis=-1, keepdims=True)

    assert_op = tf.debugging.assert_greater(
        n_valid_nodes, 
        k, 
        message="one or more graphs in the batch had too few nodes for "
                "k nearest neighbors calculation!"
    )
    with tf.control_dependencies([assert_op]):
        # Shape (batchsize, None, n_features)
        nodes_ragged = tf.RaggedTensor.from_row_lengths(
            nodes, tf.squeeze(n_valid_nodes), validate=False)

    # shape (batchsize, None, k)
    knn_ragged = tf.map_fn(
        func,
        nodes_ragged,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, k),
            dtype="int32",
            ragged_rank=0,
            row_splits_dtype=tf.dtypes.int32),
    )

    # shape (None, k)
    knn = knn_ragged.merge_dims(0, 1)

    graph_indices = tf.gather_nd(
        is_valid * tf.cumsum(n_valid_nodes, exclusive=True),
        tf.where(is_valid == 1))

    return knn + tf.expand_dims(graph_indices, -1)


def pdist(points, take_sqrt=True, single_mode=False):
    """
    Computes pairwise distance between each pair of points

    Alternative 3d:
    tf.reduce_sum(
        tf.math.squared_difference(tf.expand_dims(points, 1), tf.expand_dims(points, 2)), -1)
    but see bug3...

    Parameters
    ----------
    points : tf.Tensor
        [bs, N, D] matrix representing a batch of N D-dimensional vectors each
    take_sqrt : bool
        Take the sqrt of the squared distance? Expensive...
    single_mode : bool
        If true, this is the operation for a single sample, not the batch

    Returns
    -------
    [bs, N, N] matrix of (squared) euclidean distances

    """
    if single_mode:
        transp_axes = None
    else:
        transp_axes = [0, 2, 1]

    x2 = tf.reduce_sum(tf.square(points), axis=-1, keepdims=True)

    cross = tf.matmul(
      points,
      tf.transpose(points, transp_axes)
    )

    output = x2 - 2 * cross + tf.transpose(x2, transp_axes)

    if take_sqrt:
        return tf.sqrt(output)
    else:
        return output


def get_xixj_disjoint(nodes, knn, k):
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
        tf.expand_dims(nodes, axis=-2),
        [1, k, 1]
    )

    # TODO this produces a 'Converting sparse IndexedSlices to a dense Tensor
    #  of unknown shape.' warning. Thats because nodes has an unknown shape
    #  (None, n_features), along first axis is gathered.
    nodes_neighbors = tf.gather(nodes, knn)

    return nodes_central, nodes_neighbors


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
    """ Shape (None,). """
    return tf.gather_nd(
        is_valid * tf.expand_dims(tf.range(is_valid.shape[0]), -1),
        tf.where(is_valid == 1))
