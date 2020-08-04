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

    Input: (nodes, is_valid, coordinates)

    nodes, shape (batchsize, n_nodes_max, n_features)
        Node features of the graph, padded to fixed size.
    is_valid, shape (batchsize, n_nodes_max)
        1 for actual node, 0 for padded node.
    coordinates, shape (batchsize, n_nodes_max, n_coords)
        Features of each node used for calculating nearest
        neighbors.

    Output, shape (batchsize, n_nodes_max, units[-1])

    Parameters
    ----------
    units : List or int
        How many dense layers are in the kernel network, and how many
        neurons does each of them have? E.g. [64, 64] means two dense
        layers with 64 neurons each.
    next_neighbors : int
        How many next neighbors to construct the edge features with for
        each node.
    kernel_initializer : str
    activation : str
    shortcut : bool
        Add a shortcut connection between input and output?

    """
    def __init__(self, units,
                 next_neighbors=16,
                 kernel_initializer="glorot_uniform",
                 activation="relu",
                 shortcut=True):
        self.units = units
        self.next_neighbors = next_neighbors
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.shortcut = shortcut

        self.kernel_network = lambda xi_xj: kernel_nn(
            xi_xj, units=self.units,
            kernel_initializer=self.kernel_initializer,
            activation=self.activation)

    def __call__(self, inputs):
        nodes, is_valid, coordinates = inputs

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
            sc = ks.layers.Dense(
                self.units[-1], use_bias=False,
                kernel_initializer=self.kernel_initializer)(sc)
            sc = ks.layers.BatchNormalization()(sc)
            x = ks.layers.Activation(self.activation)(sc + x)

        x = rev(x)
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


def kernel_nn(x, units, activation="relu", kernel_initializer='glorot_uniform'):
    """ The kernel network used on each edge of the graph. """
    if isinstance(units, int):
        units = [units]
    for uts in units:
        x = layers.Dense(uts, use_bias=False, activation=None,
                         kernel_initializer=kernel_initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
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
