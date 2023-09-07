.. _dataset_cn:

AutoGL 数据集
==============

我们支持PyTorch-Geometric (PyG)，Deep Graph Learning (DGL)及Open Graph Benchmark (OGB)等图学习库提供的多种多样的常用数据集。

以下是一个在AutoGL中使用数据集的例子

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

同时你也可以自定义你自己的数据，下面是一个使用 ``Deep Graph Library`` 自定义数据集的例子

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

更多关于数据集的细节，你可以查询 ``PyTorch-Geometric``, ``Deep Graph Library`` 以及 ``OGB`` 的官方文档。

提供的常用数据集
----------------
AutoGL目前提供如下多种常用基准数据集：

半监督节点分类：

+------------------+------------+-----------+--------------------------------+
| 数据集           |  PyG       |  DGL      |  默认train/val/test划分        |
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


图分类任务： MUTAG, IMDB-Binary, IMDB-Multi, PROTEINS, COLLAB等

+-------------+------------+------------+--------------+------------+--------------------+
|  数据集     | PyG        | DGL        | 节点特征     | 标签       | 边特征             |
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


链接预测任务：目前AutoGL可以使用针对节点分类任务的多种图数据进行自动链接预测。
