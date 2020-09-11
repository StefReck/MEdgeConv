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
        # euc_dist has to be done via mapping, see bug3
        euc_dist = pdist(graph, single_mode=True, take_sqrt=False)
        return tf.math.top_k(-euc_dist, k=k + 1)[1][:, 1:]

    n_valid_nodes = tf.reduce_sum(is_valid, axis=-1, keepdims=True)

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
    # knn_ragged.row_splits.set_shape(nodes_ragged.row_splits.shape)
    # doesn't work (see bug1):
    # tf.gather(nodes_ragged, knn_ragged, batch_dims=1)

    # shape (None, k)
    knn = knn_ragged.merge_dims(0, 1)

    graph_indices = tf.gather_nd(
        is_valid * tf.cumsum(n_valid_nodes, exclusive=True),
        tf.where(is_valid == 1))

    return knn + tf.expand_dims(graph_indices, -1)


def bug1():
    # gather with ragged tensors does not work in graph mode, only in eager
    params = tf.keras.Input((None, 1), batch_size=3, ragged=True)
    indices = tf.keras.Input((None, 2), batch_size=3, ragged=True, dtype="int32")
    tf.gather(params, indices, batch_dims=1)
    # OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed in Graph execution.
    # Use Eager execution or decorate this function with @tf.function.

    # this works though
    # params = tf.ragged.constant([[[0], [1]], [[2], [3]], [[4], [5], [6]]], ragged_rank=1, dtype="float32")
    # indices = tf.ragged.constant([[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]], ragged_rank=1)


def bug3():
    # broadcasting of two ragged dims in parallel does not work
    nodes_ragged = tf.ragged.constant(
        [[[0, 1], [1, 2]], [[2, 3], [3, 4]], [[4, 5], [5, 6], [6, 7]]],
        ragged_rank=1)  # shape (3, None, 2)
    squared_d = tf.math.squared_difference(
        tf.expand_dims(nodes_ragged, 1), tf.expand_dims(nodes_ragged, 2))


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
    # x2_ragged = tf.expand_dims(tf.reduce_sum(nodes_ragged**2, axis=-1), -1)

    # matmul doesnt work with ragged
    cross = tf.matmul(
      points,
      tf.transpose(points, transp_axes)
    )
    # transpose doesnt work with ragged
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
    # nodes_neighbors = gather_sparse(nodes, knn, k)

    return nodes_central, nodes_neighbors


def gather_sparse(nodes, knn, k):
    """
    Like tf.gather, but avoid actually gathering on an unknown dimension.
    Avoids the warning, but is still actually slower...

    dense: warning, 1:22
    sparse: no warning, 1:32

    """
    knn_64 = tf.cast(knn, "int64")
    knn_flat = tf.keras.backend.flatten(knn_64)

    base = tf.ones_like(knn_flat)
    base_cumu = tf.cumsum(base, exclusive=True)
    alternating = base_cumu % k
    node_index = base_cumu // k
    indices = tf.stack([node_index, alternating, knn_flat], axis=1)

    knn_sparse = tf.sparse.SparseTensor(
        indices=indices,
        values=tf.cast(base, "float32"),
        dense_shape=tf.cast(tf.concat([tf.shape(knn), tf.shape(knn)[:1]], axis=0), "int64")  # (None, k, None),
    )
    # Sparse matmul only works for rank 2 :-( (see bug2)
    # tf.sparse.sparse_dense_matmul(knn_sparse, nodes)
    # but i can reshape to 2d!:
    sparse_flat = tf.sparse.reshape(
        knn_sparse, tf.concat(([-1], tf.shape(knn_sparse)[-1:]), axis=0))
    out = tf.reshape(
        tf.sparse.sparse_dense_matmul(sparse_flat, nodes),
        tf.concat([tf.shape(knn_sparse)[:1], [knn_sparse.shape[1]], nodes.shape[-1:]], axis=0)
    )

    # out = tf.matmul(tf.sparse.to_dense(knn_sparse), nodes)
    return out


def bug2():
    # sparse matmul only works for rank 2 :-(
    sp_a = tf.keras.Input((5, None), sparse=True)
    b = tf.keras.Input((3, ))
    tf.sparse.sparse_dense_matmul(sp_a, b)


def temp():
    import numpy as np

    k = 3
    knn_numpy = np.array([[0, 2, 1], [0, 1, 3], [3, 1, 0], [1, 3, 0]], dtype="int64")
    nodes_numpy = np.array([[10, ], [11, ], [12, ], [13, ]], dtype="float32")

    knn = tf.constant(knn_numpy)
    nodes = tf.constant(nodes_numpy)

    knn = tf.keras.Input((3,))
    nodes = tf.keras.Input((1,))


def temp_2():
    import numpy as np

    k = 2
    nodes_numpy = np.array(
        [[10, ], [11, ], [12, ], [13, ], [14, ], [15, ], [16, ]], dtype="float32")
    is_valid_numpy = np.array(
        [[1, 1, 1, 1,], [1, 1, 1, 0]], dtype="int32"
    )

    is_valid = tf.constant(is_valid_numpy)
    nodes = tf.constant(nodes_numpy)

    is_valid = tf.keras.Input((4,), batch_size=2, dtype="int32")
    nodes = tf.keras.Input((1,))


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
