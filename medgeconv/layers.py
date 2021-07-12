import tensorflow as tf
import tensorflow.keras as ks
import medgeconv.util as util


class DisjointEdgeConv:
    """
    The EdgeConv layer like it is used in ParticleNet, implemented
    as a pseudo-block and expecting ragged inputs.
    Has a dense network as the kernel network.

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

        self.kernel_network = lambda xi_xj: kernel_nn(
            xi_xj, units=self.units,
            kernel_initializer=self.kernel_initializer,
            activation=self.activation)

    def __call__(self, inputs):
        """
        Parameters
        ----------
        inputs : tuple
            Length 2: (nodes, coordinates).
            nodes: tf.RaggedTensor
                shape (batchsize, None, n_features)
                Node features of the graph.
            coordinates: tf.RaggedTensor
                shape (batchsize, None, n_coords)
                Features of each node used for calculating nearest neighbors.

        Returns
        -------
        tf.RaggedTensor
            shape (batchsize, None, units[-1])

        """
        nodes, coordinates = inputs

        # get central and neighbour point features for each edge
        # between neighbours, defined by coordinates
        xi_xj = GetEdgeFeatures(
            next_neighbors=self.next_neighbors,
        )((nodes, coordinates))

        x = self.get_edges(xi_xj)
        # apply MLP on points of each edge
        x = self.kernel_network(x)
        # aggregate over the nearest neighbours
        x = ks.backend.mean(x, axis=-2)

        if self.shortcut:
            nodes_disjoint = nodes.merge_dims(0, 1)
            sc = ks.layers.Dense(
                self.units[-1], use_bias=False,
                kernel_initializer=self.kernel_initializer)(nodes_disjoint)
            sc = ks.layers.BatchNormalization()(sc)
            x = ks.layers.Activation(self.activation)(sc + x)
        x = tf.RaggedTensor.from_row_splits(x, nodes.row_splits)
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


class GetEdgeFeatures(ks.layers.Layer):
    """
    Get the edge features of the graph.

    Input shape:
        [(None, None, n_features),
         (None, None, n_coordinates)]
        --> [nodes, coordinates]
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
        nodes, coordinates = inputs
        # get the k nearest neighbours for each node
        knn_disjoint = util.get_knn(
            coordinates,
            self.next_neighbors,
        )
        nodes_disjoint = nodes.merge_dims(0, 1)
        # get central and neighbour point for each edge between neighbours
        return util.get_xixj(
            nodes_disjoint, knn_disjoint, k=self.next_neighbors)

    def compute_output_shape(self, input_shape):
        return (input_shape[0, :1] + (self.next_neighbors,) + input_shape[0, -1:], ) * 2

    def get_config(self):
        config = super().get_config()
        config["next_neighbors"] = self.next_neighbors
        return config


def kernel_nn(x, units, activation="relu", kernel_initializer='glorot_uniform'):
    """ The kernel network used on each edge of the graph. """
    if isinstance(units, int):
        units = [units]
    for uts in units:
        x = ks.layers.Dense(
            uts, use_bias=False, activation=None,
            kernel_initializer=kernel_initializer)(x)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.Activation(activation)(x)
    return x
