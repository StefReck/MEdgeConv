"""
Tensorflow functions for graph functionality.

"""
import tensorflow as tf


def get_knn_from_points(points, k, is_valid):
    """
    Get the k nearest neighbors depending on distance.

    Parameters
    ----------
    points : tf.Tensor
        shape (bs, n_points, n_features)
    k : int
        Number of nearest neighbors (excluding self).
    is_valid : tf.Tensor
        boolean or float tensor shape (bs, n_points).

    Returns
    -------
    shape (bs, n_points, k), int32
        The index of which points are the nearest neighbors.

    """
    def func(args):
        return _get_knn_map(args, k=k)
    return tf.map_fn(func, [points, is_valid], dtype="int32")


def get_knn_from_disjoint(points_flat, k, is_valid):
    """ Return shape (None, k). """
    def func(args):
        graph = args
        euc_dist = pdist(graph, single_mode=True)
        knn = tf.math.top_k(-euc_dist, k=k + 1)[1][:, 1:]
        return knn

    is_valid_int = tf.cast(is_valid, "int32")
    n_valid_nodes = tf.reduce_sum(is_valid_int, axis=-1, keepdims=True)

    # Shape (batchsize, None, n_features)
    points_ragged = tf.RaggedTensor.from_row_lengths(
        points_flat, tf.squeeze(n_valid_nodes), validate=False)
    # shape (batchsize, None, k)
    knn_ragged = tf.map_fn(
        func,
        points_ragged,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, k),
            dtype="int32",
            ragged_rank=0,
            row_splits_dtype=tf.dtypes.int32)
    )
    # shape (None, k)
    knn_flat = knn_ragged.merge_dims(0, 1)

    graph_indices = tf.gather_nd(
        is_valid_int * tf.cumsum(n_valid_nodes, exclusive=True),
        tf.where(is_valid == 1))

    return knn_flat + tf.expand_dims(graph_indices, -1)


def get_xixj_disjoint(points_flat, knn_flat, k):
    """
    Given points of shape (bs, n_points, n_dims), get
    two matrices with shape (bs, n_points, k, n_dims).

    --> [?, i, j] desribes the edge between points i and j.
    The first matrix is xi, i.e. for each edge the central point
    The 2nd matrix is xj, i.e. for each edge the other point.

    """
    point_central = tf.tile(
        tf.expand_dims(points_flat, axis=-2),
        [1, k, 1]
    )
    point_cloud_neighbors = tf.gather(points_flat, knn_flat)

    return point_central, point_cloud_neighbors




def get_edges_from_points_flat(points_flat, k, is_valid):
    """
    Calculate xi, xi - xj from the flattened points.

    Parameters
    ----------
    points_flat : tf.Tensor
        Shape (None, n_features).
    k : int
        Number of nearest neighbors (excluding self).
    is_valid : tf.Tensor
        boolean or float tensor shape (bs, n_points).

    Returns
    -------
    tf.Tensor
        Shape (None, k, n_features*2).

    """
    def func(args):
        graph = args
        euc_dist = pdist(graph, single_mode=True)
        knn = tf.math.top_k(-euc_dist, k=k + 1)[1][:, 1:]
        return knn
    # Shape (batchsize, None, n_features)
    points_ragged = tf.RaggedTensor.from_row_lengths(
        points_flat, tf.cast(tf.reduce_sum(is_valid, axis=-1), "int32"), validate=False)
    # shape (batchsize, None, k)
    knn_ragged = tf.map_fn(
        func,
        points_ragged,
        fn_output_signature=tf.RaggedTensorSpec(
            shape=(None, k),
            dtype="int32",
            ragged_rank=0,
            row_splits_dtype=tf.dtypes.int32)
    )
    # shape (None, k)
    knn_flat = knn_ragged.merge_dims(0, 1)

    point_central = tf.tile(
        tf.expand_dims(points_ragged, axis=-2),
        [1, 1, k, 1])
    point_neighbors = tf.gather(
        points_ragged, knn_ragged, batch_dims=1)
    # shape (batchsize, None, k, n_features*2)
    edge_features_ragged = tf.concat(
        [point_central, point_central - point_neighbors], axis=-1)

    edge_features_flat = edge_features_ragged.merge_dims(0, 1)
    # x.set_shape(tf.concat([is_valid.shape, points_flat.shape[1:]], -1))
    return edge_features_flat


def _temp_comp(points, k, is_valid):
    knn = get_knn_from_points(
        points,
        k,
        is_valid=is_valid,
    )
    xi_xj = get_xixj_knn(points, knn=knn, k=k)
    return xi_xj


def _get_knn_map(args, k):
    """ Get knn for one graph at a time. Only operates on real nodes. """
    graph, val = args
    # graph: (n_points, n_features), val (n_points, )

    # dummy tensor of fixed shape (n_points, k)
    knn_base = tf.zeros_like(val, dtype="int32")
    knn_base = tf.tile(tf.expand_dims(knn_base, -1), (1, k))

    # take only real nodes
    n_points_real = tf.cast(tf.reduce_sum(val), "int32")
    graph_prop = graph[:n_points_real]
    # get knn
    euc_dist = pdist(graph_prop, single_mode=True)  # (n_points_real, n_points_real)
    knn = tf.math.top_k(-euc_dist, k=k + 1)[1][:, 1:]  # (n_points_real, k)

    # insert into dummy tensor
    return tf.tensor_scatter_nd_update(
        knn_base, tf.expand_dims(tf.range(tf.shape(knn)[0]), -1), knn)


def get_xixj_knn(points, knn, k):
    """
    Given points of shape (bs, n_points, n_dims), get
    two matrices with shape (bs, n_points, k, n_dims).

    --> [?, i, j] desribes the edge between points i and j.
    The first matrix is xi, i.e. for each edge the central point
    The 2nd matrix is xj, i.e. for each edge the other point.

    """
    # TODO set invalid nodes to 0!
    point_central = tf.tile(
        tf.expand_dims(points, axis=-2),
        [1, 1, k, 1]
    )
    point_cloud_neighbors = tf.gather(points, knn, batch_dims=1)

    return point_central, point_cloud_neighbors


def pdist(points, take_sqrt=True, single_mode=False):
    """
    Computes pairwise distance between each pair of points

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


def reduce_mean_valid(points, is_valid, divide_by_nodes=True):
    """
    Sum up or average over valid nodes.

    Parameters
    ----------
    points : tf.Tensor
        shape (bs, n_points, n_features)
    is_valid : tf.Tensor
        boolean or float tensor shape (bs, n_points).
    divide_by_nodes : bool
        If true, divide by number of valid nodes (= average).

    Returns
    -------
    shape (bs, n_features)
        For each feature, aggeregated over all valid nodes.

    """
    is_valid = tf.cast(is_valid, points.dtype)
    # set node features of invalid nodes to 0
    valid_points = points * tf.expand_dims(is_valid, -1)
    if divide_by_nodes:
        # number of valid nodes in each batch
        n_valid_nodes = tf.reduce_sum(is_valid, axis=-1, keepdims=True)
        # sum over all nodes
        summed_nodes = tf.reduce_sum(valid_points, axis=-2)
        # divide the sum by the number of valid nodes per feature
        return summed_nodes / tf.maximum(n_valid_nodes, tf.keras.backend.epsilon())
    else:
        return tf.reduce_mean(valid_points, axis=1)


def reduce_mean_valid_disjoint(points_disjoint, is_valid):
    """
    Average over valid nodes.

    Parameters
    ----------
    points_disjoint : tf.Tensor
        shape (None, n_features)
    is_valid : tf.Tensor
        boolean or float tensor shape (bs, n_points).

    Returns
    -------
    shape (bs, n_features)
        For each feature, aggeregated over all valid nodes.

    """
    is_valid = tf.cast(is_valid, points_disjoint.dtype)

    placeholder = tf.tile(
        tf.expand_dims(tf.zeros_like(is_valid), -1),
        (1, 1, points_disjoint.shape[-1])
    )
    points_dense = tf.tensor_scatter_nd_update(
        placeholder, tf.where(is_valid == 1), points_disjoint)

    # number of valid nodes in each batch
    n_valid_nodes = tf.reduce_sum(is_valid, axis=-1, keepdims=True)
    # sum over all nodes
    summed_nodes = tf.reduce_sum(points_dense, axis=-2)
    # divide the sum by the number of valid nodes per feature
    return summed_nodes / tf.maximum(n_valid_nodes, tf.keras.backend.epsilon())



def flatten_graphs(nodes, is_valid):
    """
    Flatten node and batch dimension and remove non-existing nodes from
    graph. Also reverse this easily.

    Parameters
    ----------
    nodes : tf.Tensor
        The node features, shape (bs, n_nodes, ...)
    is_valid : tf.Tensor
        Which nodes are valid nodes? Shape (bs, n_nodes)

    Returns
    -------
    flattened : tf.Tensor
        The flattened nodes, shape (?, ...)
    rev : function
        Apply this on the flattened tensor to get the original shape back
        (with padded zeros), i.e. turn it back to (bs, n_nodes, ...).
        First axis (the batchsize/graph axis) must have
        not changed shape or this wont work.

    """
    valid_indices = tf.where(is_valid == 1)
    flattened = tf.gather_nd(nodes, valid_indices)

    def reverse(flat_nodes):
        empty = tf.tile(
            tf.expand_dims(tf.zeros_like(is_valid), -1),
            (1, 1, flat_nodes.shape[-1])
        )
        return tf.tensor_scatter_nd_update(
            empty, valid_indices, flat_nodes)

    return flattened, reverse


def dense_to_ragged(points, is_valid):
    """ (batchsize, n_points, n_features) --> (batchsize, None, n_features) """
    n_valid_nodes = tf.reduce_sum(tf.cast(is_valid, "int32"), -1)
    n_invalid_nodes = tf.ones_like(n_valid_nodes) * points.shape[1] - n_valid_nodes
    row_lengths = tf.reshape(tf.stack([n_valid_nodes, n_invalid_nodes], -1), [-1])
    points_flat = tf.reshape(points, [-1, points.shape[-1]])
    ragged = tf.RaggedTensor.from_row_lengths(points_flat, row_lengths, validate=False)
    return ragged[::2]


def dense_to_sparse(points, is_valid):
    return tf.sparse.from_dense(points * tf.expand_dims(is_valid, -1))
