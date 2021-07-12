MEdgeConv
=========

.. image:: https://travis-ci.org/StefReck/MEdgeConv.svg?branch=master
    :target: https://github.com/StefReck/MEdgeConv/actions/workflows/cicd/badge.svg

.. image:: https://codecov.io/gh/StefReck/MEdgeConv/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/StefReck/MEdgeConv

.. image:: https://badge.fury.io/py/medgeconv.svg
    :target: https://badge.fury.io/py/medgeconv

An efficient tensorflow 2 implementation of the edge-convolution layer
EdgeConv used in e.g. ParticleNet.

The structure of the layer is as described in 'ParticleNet: Jet Tagging
via Particle Clouds' https://arxiv.org/abs/1902.08570.

Instructions
------------

Install via::

    pip install medgeconv


Use e.g. like this:

.. code-block:: python

    import medgeconv

    nodes = medgeconv.DisjointEdgeConvBlock(
        units=[64, 64, 64],
        next_neighbors=16,
    )((nodes, coordinates))


Inputs to EdgeConv are 2 ragged tensors: nodes and coordinates

- nodes, shape (batchsize, None, n_features)
    Node features of the graph. Secound dimension is the number of nodes,
    which can vary from graph to graph.

- coordinates, shape (batchsize, None, n_coords)
    Features of each node used for calculating nearest neighbors.


Example: Input for a graph with 2 features per node, and all node features
used as coordinates.

.. code-block:: python

    import tensorflow as tf

    nodes = tf.ragged.constant([
       # graph 1: 2 nodes
       [[2., 4.],
        [2., 6.]],
       # graph 2: 4 nodes
       [[0., 2.],
        [3., 7.],
        [4., 0.],
        [1., 2.]],
    ], ragged_rank=1)

    print(nodes.shape)  # output: (2, None, 2)

    # using all node features as coordinates
    coordinates = nodes

Example
-------

The full ParticleNet for n_features = n_coords = 2, and a dense layer
with 2 neurons as the output can be built like this:

.. code-block:: python

    import tensorflow as tf
    import medgeconv

    inp = (
        tf.keras.Input((None, 2), ragged=True),
        tf.keras.Input((None, 2), ragged=True),
    )
    x = medgeconv.DisjointEdgeConvBlock(
        units=[64, 64, 64],
        batchnorm_for_nodes=True,
        next_neighbors=16,
    )(inp)

    x = medgeconv.DisjointEdgeConvBlock(
        units=[128, 128, 128],
        next_neighbors=16,
    )(x)

    x = medgeconv.DisjointEdgeConvBlock(
        units=[256, 256, 256],
        next_neighbors=16,
        pooling=True,
    )(x)

    output = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inp, output)


The last EdgeConv layer has ``pooling = True``.
This will attach a node-wise global
average pooling layer in the end, producing normal not-ragged tensors again.

The model can then be used on ragged Tensors:

.. code-block:: python

    nodes = tf.RaggedTensor.from_tensor(tf.ones((3, 17, 2)))
    model.predict((nodes, nodes))


Loading models
--------------

To load models, use the custom_objects:

.. code-block:: python

    import medgeconv

    model = load_model(path, custom_objects=medgeconv.custom_objects)


knn_graph kernel
----------------

This package includes a cuda kernel for calculating the k nearest neighbors
on a batch of graphs. It comes with a precompiled kernel for the version of
tensorflow specified in requirements.txt.

To compile it locally, e.g. for a different version of
tensorflow, go to ``medgeconv/tf_ops`` and adjust the ``compile.sh`` bash script.
Running it will download the specified tf dev docker image and produce the
file ``medgeconv/tf_ops/python/ops/_knn_graph_ops.so``.
