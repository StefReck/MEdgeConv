"""
Keras layers for graph functionality.

"""
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
from . import util


class EdgeConv:
    """
    The EdgeConv layer like it is used in ParticleNet, implemented
    as a pseudo-block.
    Has a dense network as the kernel network.

    Input:
        (nodes, is_valid, coordinates)
    Output:
        (new_nodes, )

    Parameters
    ----------
    units : List or int
        How many dense layers are in the kernel network, and how many
        neurons does each of them have? E.g. [64, 64] means two dense
        layers with 64 neurons each.
    next_neighbors : int
        How many next neighbors to construct the edge features with for
        each node.
    shortcut : bool
        Add a shortcut connection between input and output?
    batchnorm_for_nodes : bool
        Add a batchnormalization to the nodes input? Useful if this is the
        first layer in the network.
    pooling : str or None
        Which pooling operation to use after the block, if any.
        Currently available: GlobalAvgValidPooling

    """
    def __init__(self, units,
                 next_neighbors=16,
                 shortcut=True,
                 batchnorm_for_nodes=False,
                 pooling=None):
        self.units = units
        self.next_neighbors = next_neighbors
        self.batchnorm_for_nodes = batchnorm_for_nodes
        self.shortcut = shortcut
        self.pooling = pooling

        self.pool_types = {
            "GlobalAvgValidPooling": GlobalAvgValidPooling,
        }

        self.kernel_network = lambda xi_xj: kernel_nn(xi_xj, units=self.units)

    def __call__(self, inputs):
        nodes, is_valid, coordinates = inputs

        if self.batchnorm_for_nodes:
            nodes = layers.BatchNormalization()(nodes)

        # get central and neighbour point features for each edge
        # between neighbours, defined by coordinates
        xi_xj = GetEdgeFeatures(
            next_neighbors=self.next_neighbors,
        )((nodes, is_valid, coordinates))

        x = self.get_edges(xi_xj)
        # remove padded nodes
        x, rev = util.flatten_graphs(x, is_valid)
        # apply MLP on points of each edge
        x = self.kernel_network(x)
        # aggregate over the nearest neighbours
        x = ks.backend.mean(x, axis=-2)

        if self.shortcut:
            sc, _ = util.flatten_graphs(nodes, is_valid)
            sc = ks.layers.Dense(self.units[-1], use_bias=False)(sc)
            sc = ks.layers.BatchNormalization()(sc)
            x = ks.layers.Activation("relu")(sc + x)

        x = rev(x)
        if self.pooling is not None:
            pool_op = self.pool_types[self.pooling]
            x = pool_op()((x, is_valid))

        return x

    def get_edges(self, xi_xj):
        """
        Get the edge features, derived from the central point and
        the neighboring points.

        Output shape: (bs, n_points, n_neighbors, 2*n_features)

        """
        xi, xj = xi_xj
        dif = layers.Subtract()([xi, xj])
        x = layers.Concatenate(axis=-1)([xi, dif])
        return x


def kernel_nn(x, units=(16, 16, 16)):
    """
    The kernel network used on each edge of the graph.

    """
    if isinstance(units, int):
        units = [units]
    for uts in units:
        x = layers.Dense(uts, use_bias=False, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


class GetEdgeFeatures(ks.layers.Layer):
    """
    Get the edge features of the graph.

    Input shape:
        [(bs, n_points, n_features),
         (bs, n_points),
         (bs, n_points, n_coordinates)]
        --> [nodes, is_valid, coordinates]
        - all the nodes of the graph, together with their features
        - Info about which nodes are valid nodes
        - Coordinates that will be used for calculateing the distance / knn

    Output shape:
        [
         (bs, n_points, next_neighbors, n_features),
         (bs, n_points, next_neighbors, n_features),
        ]
        --> [xi, xj]
        These matrices represent the edges of the graph:
        for [, i, j]:
        matrix 1 is xi (the central point),
        and matrix 2 is xj (the nearest neighbor j of central point i)

    """
    def __init__(self, next_neighbors, **kwargs):
        super().__init__(**kwargs)
        self.next_neighbors = next_neighbors

    def call(self, inputs):
        points, is_valid, coordinates = inputs
        # get the k nearest neighbours for each node (dont connect valid and invalid nodes)
        knn = util.get_knn_from_points(
            coordinates,
            self.next_neighbors,
            is_valid=is_valid,
        )
        # get central and neighbour point for each edge between neighbours
        xi_xj = util.get_xixj_knn(points, knn=knn, k=self.next_neighbors)
        return xi_xj

    def compute_output_shape(self, input_shape):
        return (input_shape[0, :2] + (self.next_neighbors,) + input_shape[0, -1:], ) * 2

    def get_config(self):
        config = super().get_config()
        config["next_neighbors"] = self.next_neighbors
        return config


class GlobalAvgValidPooling(ks.layers.Layer):
    """
    An average valid pooling layer.
    Pools a graph by averaging features over all nodes (or only all
    valid nodes).

    Input shape:    [(bs, n_points, n_features), (bs, n_points)]
    Output shape:   (bs, n_features)

    """
    def __init__(self, divide_by_nodes=True, **kwargs):
        super().__init__(**kwargs)
        self.divide_by_nodes = divide_by_nodes

    def call(self, inputs):
        points, is_valid = inputs
        return util.reduce_mean_valid(
            points, is_valid, divide_by_nodes=self.divide_by_nodes)

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]

    def get_config(self):
        config = super().get_config()
        config["divide_by_nodes"] = self.divide_by_nodes
        return config


custom_objects = {
    "GlobalAvgValidPooling": GlobalAvgValidPooling,
    "GetEdgeFeatures": GetEdgeFeatures,
}

