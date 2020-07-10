MEdgeConv
=========

.. image:: https://travis-ci.org/StefReck/MEdgeConv.svg?branch=master
    :target: https://travis-ci.org/StefReck/MEdgeConv

.. image:: https://codecov.io/gh/StefReck/MEdgeConv/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/StefReck/MEdgeConv

An efficient tensorflow 2 implementation of the edge-convolution layer
EdgeConv used in e.g. ParticleNet.

The structure of the layer is as described in 'ParticleNet: Jet Tagging
via Particle Clouds'
https://arxiv.org/abs/1902.08570. Graphs often have a varying number
of nodes. Memory intensive operations in MEdgeConv
are done only on the actual nodes, so this should be faster if the number of
nodes varies greatly between graphs in the batch.


Install e.g. via::

    pip install git+https://github.com/StefReck/MEdgeConv.git#egg=MEdgeConv


Use like this:

.. code-block:: python

    import medgeconv

    nodes = medgeconv.EdgeConv(units=[64, 64, 64], next_neighbors=16)((nodes, is_valid, coordinates))


Inputs to EdgeConv are 3 tensors: nodes, is_valid and coordinates

- nodes, shape (batchsize, n_nodes_max, n_features)
    Node features of the graph, padded to fixed size.

- is_valid, shape (batchsize, n_nodes_max)
    1 for actual node, 0 for padded node.

- coordinates, shape (batchsize, n_nodes_max, n_coords)
    Features of each node used for calculating nearest
    neighbors.

Output is a tensor with shape (batchsize, n_nodes_max, units[-1])

To pool in the end, use this:

.. code-block:: python

    import medgeconv

    x = medgeconv.GlobalAvgValidPooling()((nodes, is_valid))


To load models, use the custom_objects:

.. code-block:: python

    import medgeconv

    model = load_model(path, custom_objects=medgeconv.custom_objects)

Remarks:

- Batchsize has to be fixed (i.e. use Input(batch_size=bs, ...))
- in nodes array, valid nodes have to come first, then the padded nodes
