import tensorflow as tf
import tensorflow.keras as ks
import medgeconv.util as util
import medgeconv.layers as layers
import medgeconv.legacy.util as util_legacy


class DisjointEdgeConv:
    """
    The EdgeConv layer like it is used in ParticleNet, implemented
    as a pseudo-block and expecting disjoint input.
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
        For the kernel network.
    activation : str
        For the kernel network.
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

        self.kernel_network = lambda xi_xj: layers.kernel_nn(
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

    @staticmethod
    def get_edges(xi_xj):
        """
        Get the edge features, derived from the central point and
        the neighboring points.

        Output shape: (bs, n_nodes, n_neighbors, 2*n_features)

        """
        xi, xj = xi_xj
        dif = ks.layers.Subtract()([xi, xj])
        x = ks.layers.Concatenate(axis=-1)([xi, dif])
        return x


class GetEdgeFeaturesDisjoint(ks.layers.Layer):
    """
    Get the edge features of the graph.

    Input shape:
        [(None, n_features),
         (bs, n_nodes),
         (None, n_coordinates)]
        --> [nodes, is_valid, coordinates]
        - all the nodes of the graph, together with their features
        - Info about which nodes are valid nodes
        - Coordinates that will be used for calculateing the distance / knn

    Output shape:
        [
         (None, next_neighbors, n_features),
         (None, next_neighbors, n_features),
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
        nodes_disjoint, is_valid, coordinates_disjoint = inputs
        # get the k nearest neighbours for each node
        knn_disjoint = util_legacy.get_knn_from_disjoint(
            coordinates_disjoint,
            self.next_neighbors,
            is_valid=is_valid,
        )
        # get central and neighbour point for each edge between neighbours
        return util.get_xixj(
            nodes_disjoint, knn_disjoint, k=self.next_neighbors)

    def compute_output_shape(self, input_shape):
        return (input_shape[0, :1] + (self.next_neighbors,) + input_shape[0, -1:], ) * 2

    def get_config(self):
        config = super().get_config()
        config["next_neighbors"] = self.next_neighbors
        return config


class DenseToDisjoint(ks.layers.Layer):
    """ (batchsize, n_nodes, n_features) --> (None, n_features) """
    def call(self, inputs):
        nodes, is_valid, coordinates = inputs
        is_valid = tf.cast(is_valid, "int32")

        valid_indices = tf.where(is_valid == 1)
        nodes_disjoint = tf.gather_nd(nodes, valid_indices)
        coordinates_disjoint = tf.gather_nd(coordinates, valid_indices)

        return nodes_disjoint, is_valid, coordinates_disjoint


class GlobalAvgPoolingDisjoint(ks.layers.Layer):
    """
    An average valid pooling layer.
    Pools a graph by averaging features over all nodes (or only all
    valid nodes).

    Input shape:    [(None, n_features), (bs, n_nodes)]
    Output shape:   (bs, n_features)

    """
    def call(self, inputs):
        nodes_disjoint, is_valid = inputs
        return util_legacy.reduce_mean_valid_disjoint(
            nodes_disjoint, is_valid)

    def compute_output_shape(self, input_shape):
        return input_shape[1][:1] + input_shape[0][-1:]
