MEdgeConv
=========

.. image:: https://travis-ci.org/StefReck/MEdgeConv.svg?branch=master
    :target: https://travis-ci.org/StefReck/MEdgeConv

.. image:: https://codecov.io/gh/StefReck/MEdgeConv/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/StefReck/MEdgeConv

.. image:: https://badge.fury.io/py/medgeconv.svg
    :target: https://badge.fury.io/py/medgeconv

An efficient tensorflow 2 implementation of the edge-convolution layer
EdgeConv used in e.g. ParticleNet.

The structure of the layer is as described in 'ParticleNet: Jet Tagging
via Particle Clouds'
https://arxiv.org/abs/1902.08570. Graphs often have a varying number
of nodes. By making use of the disjoint unioon of graphs in a batch,
memory intensive operations in this implementation
are done only on the actual nodes. This is faster if the number of
nodes varies between graphs in the batch.


Install via::

    pip install medgeconv


Use e.g. like this:

.. code-block:: python

    import medgeconv

    nodes = medgeconv.DisjointEdgeConvBlock(
        units=[64, 64, 64],
        next_neighbors=16,
        to_disjoint=True,
        pooling=True)((nodes, is_valid, coordinates))


Inputs to EdgeConv are 3 dense tensors: nodes, is_valid and coordinates

- nodes, shape (batchsize, n_nodes_max, n_features)
    Node features of the graph, padded to fixed size.

- is_valid, shape (batchsize, n_nodes_max)
    1 for actual node, 0 for padded node.

- coordinates, shape (batchsize, n_nodes_max, n_coords)
    Features of each node used for calculating nearest
    neighbors.

Example for batchsize = 2, n_nodes_max = 4, n_features = 2:

.. code-block:: python

    nodes = np.array([
       [[2., 4.],
        [2., 6.],
        [0., 0.],  # <-- these nodes are padded, their
        [0., 0.]],  #           value doesn't matter

       [[0., 2.],
        [3., 7.],
        [4., 0.],
        [1., 2.]]])

    is_valid = np.array([
        [1, 1, 0, 0],  # <-- 0 defines these nodes as padded
        [1, 1, 1, 1]])

    coordinates = nodes


By using ``to_disjoint = True``, the dense tensors get transformed to
the disjoint union. The output is also disjoint, so this only needs to be
done once.
``pooling = True`` will attach a node-wise global
average pooling layer in the end, producing dense tensors again.


A full model could look like this:

.. code-block:: python

    import tensorflow as tf
    import medgeconv

    inp = (nodes, is_valid, coordinates)
    x = medgeconv.DisjointEdgeConvBlock(
        units=[64, 64, 64],
        to_disjoint=True,
        batchnorm_for_nodes=True)(inp)

    x = medgeconv.DisjointEdgeConvBlock(
        units=[128, 128, 128])(x)

    x = medgeconv.DisjointEdgeConvBlock(
        units=[256, 256, 256],
        pooling=True)(x)

    output = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inp, output)


To load models, use the custom_objects:

.. code-block:: python

    import medgeconv

    model = load_model(path, custom_objects=medgeconv.custom_objects)

Remarks:

- Batchsize has to be fixed (i.e. use Input(batch_size=bs, ...))
- in nodes array, valid nodes have to come first, then the padded nodes
