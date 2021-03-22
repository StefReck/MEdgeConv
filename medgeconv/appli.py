import tensorflow as tf
import medgeconv.layers as layers


class DisjointEdgeConvBlock:
    """
    EdgeConv with additional options as used in the ParticleNet
    paper.

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
    to_disjoint : bool
        Start by transforming dense input to disjoint.
        Has to be used if this is the first layer.
    batchnorm_for_nodes : bool
        Use batchnorm layers for node features?
        Recommended for first layer.
    pooling : bool
        Finish with average pooling?
        Has to be used if this is the last layer.

    """
    def __init__(self, units,
                 next_neighbors,
                 kernel_initializer="glorot_uniform",
                 activation="relu",
                 shortcut=True,
                 to_disjoint=False,
                 batchnorm_for_nodes=False,
                 pooling=False):
        self.to_disjoint = to_disjoint
        self.batchnorm_for_nodes = batchnorm_for_nodes
        self.pooling = pooling

        self.edgeconv = layers.DisjointEdgeConv(
            units=units,
            next_neighbors=next_neighbors,
            kernel_initializer=kernel_initializer,
            activation=activation,
            shortcut=shortcut,
        )

    def __call__(self, x):
        nodes, is_valid, coordinates = x

        if self.to_disjoint:
            nodes, is_valid, coordinates = layers.DenseToDisjoint()(
                (nodes, is_valid, coordinates))

        if self.batchnorm_for_nodes:
            nodes = tf.keras.layers.BatchNormalization()(nodes)

        nodes = self.edgeconv((nodes, is_valid, coordinates))

        if self.pooling:
            return layers.GlobalAvgPoolingDisjoint()((nodes, is_valid))
        else:
            return nodes, is_valid, nodes


custom_objects = {
    "GlobalAvgPoolingDisjoint": layers.GlobalAvgPoolingDisjoint,
    "GetEdgeFeaturesDisjoint": layers.GetEdgeFeaturesDisjoint,
    "DenseToDisjoint": layers.DenseToDisjoint,
}
