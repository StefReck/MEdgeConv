import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as layers
from . import util


class GetEdgeFeaturesDisjoint(ks.layers.Layer):
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
        points_disjoint, is_valid, coordinates_disjoint = inputs
        # get the k nearest neighbours for each node (dont connect valid and invalid nodes)
        knn_disjoint = util.get_knn_from_disjoint(
            coordinates_disjoint,
            self.next_neighbors,
            is_valid=is_valid,
        )
        # get central and neighbour point for each edge between neighbours
        return util.get_xixj_disjoint(
            points_disjoint, knn_disjoint, k=self.next_neighbors)

    def compute_output_shape(self, input_shape):
        return (input_shape[0, :1] + (self.next_neighbors,) + input_shape[0, -1:], ) * 2

    def get_config(self):
        config = super().get_config()
        config["next_neighbors"] = self.next_neighbors
        return config


class DenseToDisjoint(ks.layers.Layer):
    """ (batchsize, n_nodes, n_features) --> (None, n_features) """
    def call(self, inputs):
        points, is_valid, coordinates = inputs
        points_disjoint, _ = util.flatten_graphs(points, is_valid)
        coordinates_disjoint, _ = util.flatten_graphs(coordinates, is_valid)
        return points_disjoint, coordinates_disjoint


class EdgeConvDisjoint:
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
        nodes_disjoint, is_valid, coordinates_disjoint = inputs

        # get central and neighbour point features for each edge
        # between neighbours, defined by coordinates
        xi_xj = GetEdgeFeaturesDisjoint(
            next_neighbors=self.next_neighbors,
        )((nodes_disjoint, is_valid, coordinates_disjoint))
        # fix warning:
        xi_xj = tf.stop_gradient(xi_xj)

        x = self.get_edges(xi_xj)
        # apply MLP on points of each edge
        x = self.kernel_network(x)
        # aggregate over the nearest neighbours
        x = ks.backend.mean(x, axis=-2)

        if self.shortcut:
            sc = ks.layers.Dense(
                self.units[-1], use_bias=False,
                kernel_initializer=self.kernel_initializer)(nodes_disjoint)
            sc = ks.layers.BatchNormalization()(sc)
            x = ks.layers.Activation(self.activation)(sc + x)

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


class GlobalAvgValidPoolingDisjoint(ks.layers.Layer):
    """
    An average valid pooling layer.
    Pools a graph by averaging features over all nodes (or only all
    valid nodes).

    Input shape:    [(None, n_features), (bs, n_points)]
    Output shape:   (bs, n_features)

    """
    def call(self, inputs):
        points_disjoint, is_valid = inputs
        return util.reduce_mean_valid_disjoint(
            points_disjoint, is_valid)

    def compute_output_shape(self, input_shape):
        return input_shape[1][:1] + input_shape[0][-1:]


custom_objects = {
    "GlobalAvgValidPooling": GlobalAvgValidPoolingDisjoint,
    "GetEdgeFeatures": GetEdgeFeaturesDisjoint,
    "DenseToDisjoint": DenseToDisjoint,
}