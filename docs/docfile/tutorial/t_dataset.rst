.. _dataset:

AutoGL Dataset
==============

We provide various common datasets of ``PyTorch-Geometric``, ``Deep Graph Library`` and ``OGB``.

Here is an example of using the dataset.

.. code-block:: python
    import os
    from autogl.datasets import build_dataset_from_name
    from autogl.solver import AutoNodeClassifier
    from autogl.module.train import NodeClassificationFullTrainer
    os.environ['AUTOGL_BACKEND'] = 'dgl'
    ''' Pyg dataset '''
    dataset = build_dataset_from_name('cora')
    solver = AutoNodeClassifier(
        graph_models=("gcn",),
        default_trainer=NodeClassificationFullTrainer(
            decoder=None,
            init=False,
            max_epoch=200,
            early_stopping_round=201,
            lr=0.01,
            weight_decay=0.0,
        ),
        hpo_module=None,
        device="auto"
    )
    solver.fit(dataset, evaluation_method=['acc'])
    result = solver.predict(core)
    print((result == cora[0].ndata['label'][cora[0].ndata['test_mask']].cpu().numpy()).astype('float').mean())

And you can custom your own dataset, here is an example using ``Deep Graph Library``.

.. code-block:: python
    import os
    import dgl
    import torch
    import urllib.request

    from autogl.datasets import build_dataset_from_name
    from autogl.solver import AutoNodeClassifier
    from autogl.module.train import NodeClassificationFullTrainer
    from dgl.data import DGLDataset

    class SynDataset(DGLDataset):
        def __init__(self):
            super().__init__(name="syn")

        def process(self):
            node_features = torch.rand((10, 32))
            node_labels = torch.randint(0, 2, size=(10,))
            edges_src = torch.randint(0, 10, (40,))
            edges_dst = torch.randint(0, 10, (40,))

            self.graph = dgl.graph(
                (edges_src, edges_dst), num_nodes=10
            )
            self.graph.ndata["feat"] = node_features
            self.graph.ndata["label"] = node_labels

            # If your dataset is a node classification dataset, you will need to assign
            # masks indicating whether a node belongs to training, validation, and test set.
            n_nodes = 10
            n_train = int(n_nodes * 0.6)
            n_val = int(n_nodes * 0.2)
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            train_mask[:n_train] = True
            val_mask[n_train : n_train + n_val] = True
            test_mask[n_train + n_val :] = True
            self.graph.ndata["train_mask"] = train_mask
            self.graph.ndata["val_mask"] = val_mask
            self.graph.ndata["test_mask"] = test_mask

        def __getitem__(self, i):
            return self.graph

        def __len__(self):
            return 1

    dataset = SynDataset()

    solver = AutoNodeClassifier(
        graph_models=("gcn",),
        default_trainer=NodeClassificationFullTrainer(
            decoder=None,
            init=False,
            max_epoch=200,
            early_stopping_round=201,
            lr=0.01,
            weight_decay=0.0,
        ),
        hpo_module=None,
        device="auto"
    )

    solver.fit(dataset, evaluation_method=["acc"])
    result = solver.predict(dataset)
    print((result == dataset[0].ndata['label'][dataset[0].ndata['test_mask']].cpu().numpy()).astype('float').mean())

For more details, you can consult the documentation of ``PyTorch-Geometric``, ``Deep Graph Library`` and ``OGB``.

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
