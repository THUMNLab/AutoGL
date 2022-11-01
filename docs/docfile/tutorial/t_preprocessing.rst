.. _preprocessing:

AutoGL Preprocessing
==========================

We provide a series of node and graph feature engineers for 
you to compose within a feature engineering pipeline. An automatic
feature engineering algorithm is also provided. We also provide several structural engineers. 

Quick Start
-----------
.. code-block :: python

    # 1. Choose a dataset.
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')

    # 2. Compose a preprocessing pipeline
    from autogl.module.preprocessing import DataPreprocessor
    from autogl.module.preprocessing.feature_engineering import AutoFeatureEngineer
    from autogl.module.preprocessing.feature_engineering._generators import OneHotFeatureGenerator
    from autogl.module.preprocessing.feature_engineering._selectors import GBDTFeatureSelector
    from autogl.module.preprocessing.feature_engineering._graph import NXLargeCliqueSize
    # you may compose preprocessing bases through operator & 
    fe = OneHotFeatureGenerator() & GBDTFeatureSelector(fixlen=100) & NXLargeCliqueSize()

    # 3. Fit and transform the data
    fe.fit(data)
    data1=fe.transform(data,inplace=False)

List of Feature Engineer Base Names
---------------------
Now three kinds of feature engineering bases are supported,namely ``generators``, ``selectors`` , ``graph``.You can import 
bases from according module as is mentioned in the ``Quick Start`` part. Or you may want to just list names of bases
in configurations or as arguments of the autogl solver. 

1. ``generators``

+---------------------------+-------------------------------------------------+
|           Base            |                   Description                   |
+===========================+=================================================+
| ``GraphletGenerator``              | concatenate local graphlet numbers as features. |
+---------------------------+-------------------------------------------------+
| ``EigenFeatureGenerator``                 | concatenate Eigen features.                     |
+---------------------------+-------------------------------------------------+
| ``PageRankFeatureGenerator``              | concatenate Pagerank scores.                    |
+---------------------------+-------------------------------------------------+
| ``LocalDegreeProfileGenerator`` | concatenate Local Degree Profile features.      |
+---------------------------+-------------------------------------------------+
| ``NormalizeFeatures``  | Normalize all node features                     |
+---------------------------+-------------------------------------------------+
| ``OneHotDegreeGenerator``       | concatenate degree one-hot encoding.            |
+---------------------------+-------------------------------------------------+
| ``OneHotFeatureGenerator``                | concatenate node id one-hot encoding.           |
+---------------------------+-------------------------------------------------+

2. ``selectors``

+----------------------+--------------------------------------------------------------------------------+
|         Base         |                                  Description                                   |
+======================+================================================================================+
| ``FilterConstant`` | delete all constant and one-hot encoding node features.                        |
+----------------------+--------------------------------------------------------------------------------+
| ``GBDTFeatureSelector``             | select top-k important node features ranked by Gradient Descent Decision Tree. |
+----------------------+--------------------------------------------------------------------------------+

3. ``graph``

``NetLSD`` is a graph feature generation method. please refer to the according document.

A set of graph feature extractors implemented in NetworkX are wrapped, please refer to NetworkX for details.  (``NxLargeCliqueSize``, ``NxAverageClusteringApproximate``, ``NxDegreeAssortativityCoefficient``, ``NxDegreePearsonCorrelationCoefficient``, ``NxHasBridge``
,``NxGraphCliqueNumber``, ``NxGraphNumberOfCliques``, ``NxTransitivity``, ``NxAverageClustering``, ``NxIsConnected``, ``NxNumberConnectedComponents``, 
``NxIsDistanceRegular``, ``NxLocalEfficiency``, ``NxGlobalEfficiency``, ``NxIsEulerian``)

The taxonomy of base types is based on the way of transforming features. ``generators`` concatenate the original features with ones newly generated
or just overwrite the original ones. Instead of generating new features , ``selectors`` try to select useful features and keep learned selecting methods
in the base itself. The former two types of bases can be exploited in node or edge level (modification upon each
node or edge feature) ,while ``graph`` focuses on feature engineering  in graph level (modification upon each graph feature). 
For your convenience in further development,you may want to design a new item by inheriting one of them. 
Of course, you can directly inherit the ``FeatureEngineer`` as well.

Create Your Own Feature Engineer
------------------
You can create your own feature engineering object by inheriting ``FeatureEngineer``and overwriting methods ``_fit`` and ``_transform``,
or simply inheriting one of feature engineering base types ,namely ``generators``, ``selectors`` , ``graph``.

.. code-block :: python

    # for example : create a node one-hot feature.
    import torch
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')
    
    from autogl.module.preprocessing.feature_engineering._generators._basic import BaseFeatureGenerator
    import numpy as np
    class GeOnehot(BaseFeatureGenerator):    
        def _extract_nodes_feature(self, data):
            num_nodes: int = (
                data.x.size(0)
                if data.x is not None and isinstance(data.x, torch.Tensor)
                else (data.edge_index.max().item() + 1)
            )
            return torch.eye(num_nodes)
    
    fe=GeOnehot()
    fe.fit(data)
    data1=fe.transform(data,inplace=False)

List of Structure Engineer Base Names
---------------------
+----------------------+--------------------------------------------------------------------------------+
|         Base         |                                  Description                                   |
+======================+================================================================================+
| ``GCNSVD`` | use Truncated SVD as preprocessing.                        |
+----------------------+--------------------------------------------------------------------------------+
| ``GCNJaccard``             | drop dissimilar edges. |
+----------------------+--------------------------------------------------------------------------------+

Create Your Own Structure Engineer
---------------------
You can create your own feature engineering object by inheriting ``StructureEngineer``, and overwriting methods ``_fit`` and ``_transform``

.. code-block :: python
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')
    
    from autogl.module.preprocessing.structure_engineering import *
    from autogl.module.preprocessing.structure_engineering._structure_engineer import * 
    from torch_geometric.utils import add_self_loops
    class AddSelfLoop(StructureEngineer):
        def _transform(self,data):
            adj = get_edges(data) # edge list
            modified_adj=add_self_loops(adj)
            set_edges(data,modified_adj)
            return data
        
    fe=AddSelfLoop()
    fe.fit(data)
    data1=fe.transform(data,inplace=False)