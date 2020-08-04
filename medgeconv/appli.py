import tensorflow as tf
from . import layers_disjoint


class DisjointEdgeConvBlock(layers_disjoint.EdgeConvDisjoint):
    """
    DisjointEdgeConv with additional options as used in the ParticleNet
    paper.

    Parameters
    ----------
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
    def __init__(self, *args,
                 to_disjoint=False,
                 batchnorm_for_nodes=False,
                 pooling=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.to_disjoint = to_disjoint
        self.batchnorm_for_nodes = batchnorm_for_nodes
        self.pooling = pooling

    def __call__(self, x):
        nodes, is_valid, coordinates = x

        if self.to_disjoint:
            nodes, is_valid, coordinates = layers_disjoint.DenseToDisjoint()(
                (nodes, is_valid, coordinates))

        if self.batchnorm_for_nodes:
            nodes = tf.keras.layers.BatchNormalization()(nodes)

        nodes = super()((nodes, is_valid, coordinates))

        if self.pooling:
            return layers_disjoint.GlobalAvgPoolingDisjoint()((nodes, is_valid))
        else:
            return nodes, is_valid, nodes


custom_objects = {
    "GlobalAvgPoolingDisjoint": layers_disjoint.GlobalAvgPoolingDisjoint,
    "GetEdgeFeaturesDisjoint": layers_disjoint.GetEdgeFeaturesDisjoint,
    "DenseToDisjoint": layers_disjoint.DenseToDisjoint,
}
