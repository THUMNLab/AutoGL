.. _dataset:

AutoGL Dataset
==============

We provide various common datasets based on ``PyTorch-Geometric``, ``Deep Graph Library`` and ``OGB``.
Besides, users are able to leverage a unified abstraction provided in AutoGL, ``GeneralStaticGraph``, which is towards both static homogeneous graph and static heterogeneous graph.


A basic example to construct an instance of ``GeneralStaticGraph`` is shown as follows.

.. code-block:: python

    from autogl.data.graph import GeneralStaticGraph, GeneralStaticGraphGenerator

    ''' Construct a custom homogeneous graph '''
    custom_static_homogeneous_graph: GeneralStaticGraph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        {'x': torch.rand(2708, 3), 'y': torch.rand(2708, 1)}, torch.randint(0, 1024, (2, 10556))
    )

    ''' Construct a custom heterogemneous graph '''
    custom_static_heterogeneous_graph: GeneralStaticGraph = GeneralStaticGraphGenerator.create_heterogeneous_static_graph(
        {
            'author': {'x': torch.rand(1024, 3), 'y': torch.rand(1024, 1)},
            'paper': {'feat': torch.rand(2048, 10), 'z': torch.rand(2048, 13)}
        },
        {
            ('author', 'writing', 'paper'): (torch.randint(0, 1024, (2, 5120)), torch.rand(5120, 10)),
            ('author', 'reading', 'paper'): torch.randint(0, 1024, (2, 3840)),
        }
    )

Supporting datasets
-------------------
AutoGL now supports the following benchmarks for different tasks:

Semi-supervised node classification: Cora, Citeseer, Pubmed, Amazon Computers, Amazon Photo, Coauthor CS, Coauthor Physics, Reddit, etc.

+------------------+------------+-----------+--------------------------------+
|  Dataset         |  PyG       |  DGL      |  default train/val/test split  |
+==================+============+===========+================================+
| Cora             | ✓          | ✓         | ✓                              |
+------------------+------------+-----------+--------------------------------+
| Citeseer         | ✓          | ✓         | ✓                              |
+------------------+------------+-----------+--------------------------------+
| Pubmed           | ✓          | ✓         | ✓                              |
+------------------+------------+-----------+--------------------------------+
| Amazon Computers | ✓          | ✓         |                                |
+------------------+------------+-----------+--------------------------------+
| Amazon Photo     | ✓          | ✓         |                                |
+------------------+------------+-----------+--------------------------------+
| Coauthor CS      | ✓          | ✓         |                                |
+------------------+------------+-----------+--------------------------------+
| Coauthor Physics | ✓          | ✓         |                                |
+------------------+------------+-----------+--------------------------------+
| Reddit           | ✓          | ✓         | ✓                              |
+------------------+------------+-----------+--------------------------------+
| ogbn-products    | ✓          | ✓         | ✓                              |
+------------------+------------+-----------+--------------------------------+
| ogbn-proteins    | ✓          | ✓         | ✓                              |
+------------------+------------+-----------+--------------------------------+
| ogbn-arxiv       | ✓          | ✓         | ✓                              |
+------------------+------------+-----------+--------------------------------+
| ogbn-papers100M  | ✓          | ✓         | ✓                              |
+------------------+------------+-----------+--------------------------------+

Graph classification: MUTAG, IMDB-Binary, IMDB-Multi, PROTEINS, COLLAB, etc.

+-------------+------------+------------+--------------+------------+--------------------+
|  Dataset    |  PyG       |  DGL       | Node Feature | Label      |  Edge Features     |
+=============+============+============+==============+============+====================+
| MUTAG       | ✓          | ✓          |  ✓           | ✓          | ✓                  |
+-------------+------------+------------+--------------+------------+--------------------+
| IMDB-Binary | ✓          | ✓          |              | ✓          |                    |
+-------------+------------+------------+--------------+------------+--------------------+
| IMDB-Multi  | ✓          | ✓          |              | ✓          |                    |
+-------------+------------+------------+--------------+------------+--------------------+
| PROTEINS    | ✓          | ✓          |  ✓           | ✓          |                    |
+-------------+------------+------------+--------------+------------+--------------------+
| COLLAB      | ✓          | ✓          |              | ✓          |                    |
+-------------+------------+------------+--------------+------------+--------------------+
| ogbg-molhiv | ✓          | ✓          |  ✓           | ✓          | ✓                  |
+-------------+------------+------------+--------------+------------+--------------------+
| ogbg-molpcba| ✓          | ✓          |  ✓           | ✓          | ✓                  |
+-------------+------------+------------+--------------+------------+--------------------+
| ogbg-ppa    | ✓          | ✓          |              | ✓          | ✓                  |
+-------------+------------+------------+--------------+------------+--------------------+
| ogbg-code2  | ✓          | ✓          |  ✓           | ✓          | ✓                  |
+-------------+------------+------------+--------------+------------+--------------------+

Link Prediction: At present, AutoGL utilizes various homogeneous graphs towards node classification to conduct automatic link prediction.

Construct custom dataset by instances of GeneralStaticGraph
------------------------------------------------------------
The following example shows the way to compose a custom dataset by a sequence of instances of ``GeneralStaticGraph``.

.. code-block:: python
    from autogl.data import InMemoryDataset
    ''' Suppose the graphs is a sequence of instances of GeneralStaticGraph '''
    graphs = [ ... ]
    custom_dataset = InMemoryDataset(graphs)
