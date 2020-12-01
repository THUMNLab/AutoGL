.. _dataset:

AutoGL Dataset
==============

We import the module of datasets from `CogDL` and `PyTorch Geometric` and add support for datasets from `OGB`. One can refer to the usage of creating and building datasets via the tutorial of `CogDL`_, `PyTorch Geometric`_, and `OGB`_.

.. _CogDL: https://cogdl.readthedocs.io/en/latest/tutorial.html
.. _PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
.. _OGB: https://ogb.stanford.edu/docs/dataset_overview/


Supporting datasets
-------------------
AutoGL now supports the following benchmarks for different tasks:

Semi-supervised node classification: Cora, Citeseer, Pubmed, Amazon Computers\*, Amazon Photo\*, Coauthor CS\*, Coauthor Physics\*, Reddit （\*: using `utils.random_splits_mask_class` for splitting dataset is recommended.).
For detailed information for supporting datasets, please kindly refer to `PyTorch Geometric Dataset`_.

.. _PyTorch Geometric Dataset: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html

+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+
|  Dataset         |  PyG       |  CogDL    | x          | y          | edge_index| edge_attr | train/val/test node | train/val/test mask |
+==================+============+===========+============+============+===========+============+====================+=====================+
| Cora             | ✓          |           |  ✓         | ✓          |  ✓        |  ✓         |                    | ✓                   |
+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+
| Citeseer         |  ✓         |           |         ✓  |      ✓     |     ✓     |         ✓  |                    |               ✓     |
+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+
| Pubmed           |        ✓   |           |         ✓  |          ✓ |        ✓  |         ✓  |                    |                   ✓ |
+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+
| Amazon Computers |         ✓  |           |  ✓         | ✓          |  ✓        |  ✓         |                    |                     |
+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+
| Amazon Photo     | ✓          |           |  ✓         | ✓          |  ✓        |  ✓         |                    |                     |
+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+
| Coauthor CS      | ✓          |           |  ✓         | ✓          |  ✓        |  ✓         |                    |                     |
+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+
| Coauthor Physics | ✓          |           |  ✓         | ✓          |  ✓        |  ✓         |                    |                     |
+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+
| Reddit           | ✓          |           |  ✓         | ✓          |  ✓        |  ✓         |                    | ✓                   |
+------------------+------------+-----------+------------+------------+-----------+------------+--------------------+---------------------+

Graph classification: MUTAG, IMDB-B, IMDB-M, PROTEINS, COLLAB

+-----------+------------+------------+-----------+------------+------------+-----------+
|  Dataset  |  PyG       |  CogDL     | x         | y          | edge_index | edge_attr |
+===========+============+============+===========+============+============+===========+
| MUTAG     | ✓          |            |  ✓        | ✓          |  ✓         |  ✓        |
+-----------+------------+------------+-----------+------------+------------+-----------+
| IMDB-B    | ✓          |            |           | ✓          | ✓          |           |
+-----------+------------+------------+-----------+------------+------------+-----------+
| IMDB-M    | ✓          |            |           | ✓          | ✓          |           |
+-----------+------------+------------+-----------+------------+------------+-----------+
| PROTEINS  | ✓          |            |  ✓        | ✓          | ✓          |           |
+-----------+------------+------------+-----------+------------+------------+-----------+
| COLLAB    | ✓          |            |           | ✓          | ✓          |           |
+-----------+------------+------------+-----------+------------+------------+-----------+

TODO: Supporting all datasets from `PyTorch Geometric`. 

OGB datasets
------------
AutoGL also supports the popular benchmark on `OGB` for node classification and graph classification tasks. For the summary of `OGB` datasets, please kindly refer to the their `docs`_.

.. _docs: https://ogb.stanford.edu/docs/nodeprop/

Since the loss and evaluation metric used for `OGB` datasets vary among different tasks, we also add `string` properties of datasets for identification:

+-----------------+----------------+-------------------+
|    Dataset      | dataset.metric |   datasets.loss   |
+=================+================+===================+
| ogbn-products   |    Accuracy    |    nll_loss       |
+-----------------+----------------+-------------------+
| ogbn-proteins   | ROC-AUC        | BCEWithLogitsLoss |
+-----------------+----------------+-------------------+
| ogbn-arxiv      |       Accuracy |          nll_loss |
+-----------------+----------------+-------------------+
| ogbn-papers100M |     Accuracy   |      nll_loss     |
+-----------------+----------------+-------------------+
|    ogbn-mag     |    Accuracy    |     nll_loss      |
+-----------------+----------------+-------------------+
|   ogbg-molhiv   |    ROC-AUC     | BCEWithLogitsLoss |
+-----------------+----------------+-------------------+
| ogbg-molpcba    |      AP        | BCEWithLogitsLoss |
+-----------------+----------------+-------------------+
|    ogbg-ppa     |     Accuracy   |  CrossEntropyLoss |
+-----------------+----------------+-------------------+
|    ogbg-code    |     F1 score   |  CrossEntropyLoss |
+-----------------+----------------+-------------------+


Create a dataset via URL
------------------------

If your dataset is the same as the 'ppi' dataset, which contains two matrices: 'network' and 'group', you can register your dataset directly use the above code. The default root for downloading dataset is `~/.cache-autogl`, you can also specify the root by passing the string to the `path` in `build_dataset(args, path)` or `build_dataset_from_name(dataset, path)`.

.. code-block:: python

    # following code-snippet is from autogl/datasets/matlab_matrix.py

    @register_dataset("ppi")
    class PPIDataset(MatlabMatrix):
        def __init__(self, path):
            dataset, filename = "ppi", "Homo_sapiens"
            url = "http://snap.stanford.edu/node2vec/"
            super(PPIDataset, self).__init__(path, filename, url)

You should declare the name of the dataset, the name of the file, and the URL, where our script can download the resource. Then you can use either `build_dataset(args, path)` or `build_dataset_from_name(dataset, path)` in your task to build a dataset with corresponding parameters.

Create a dataset locally
------------------------

If you want to test your local dataset, we recommend you to refer to the docs on `creating PyTorch Geometric dataset`_. 

.. _creating PyTorch Geometric dataset: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html


You can simply inherit from `torch_geometric.data.InMemoryDataset` to create an empty `dataset`, then create some `torch_geometric.data.Data` objects for your data and pass a regular python list holding them, then pass them to `torch_geometric.data.Dataset` or `torch_geometric.data.DataLoader`.
Let’s see this process in a simplified example:

.. code-block:: python

    from typing import Iterable
    from torch_geometric.data.data import Data
    from autogl.datasets import build_dataset_from_name
    from torch_geometric.data import InMemoryDataset

    class MyDataset(InMemoryDataset):
        def __init__(self, datalist) -> None:
            super().__init__()
            self.data, self.slices = self.collate(datalist)

    # Create your own Data objects

    # for example, if you have edge_index, features and labels
    # you can create a Data as follows
    # See pytorch geometric more info of Data
    data = Data()
    data.edge_index = edge_index
    data.x = features
    data.y = labels

    # create a list of Data object
    data_list = [data, Data(...), ..., Data(...)]

    # Initialize AutoGL Dataset with your own data
    myData = MyDataset(data_list)
