.. _fe:

AutoGL Feature Engineering
==========================

We provide a series of node and graph feature engineers for 
you to compose within a feature engineering pipeline. 

Quick Start
-----------
.. code-block :: python

    # 1. Choose a dataset.
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')

    # 2. Compose a feature engineering pipeline
    from autogl.module.feature._base_feature_engineer._base_feature_engineer import _ComposedFeatureEngineer
    from autogl.module.feature import EigenFeatureGenerator
    from autogl.module.feature import NetLSD

    # you may compose feature engineering bases through autogl.module.feature._base_feature_engineer
    fe = _ComposedFeatureEngineer([
        EigenFeatureGenerator(size=32),
        NetLSD()
    ])

    # 3. Fit and transform the data
    fe.fit(data)
    data1=fe.transform(data,inplace=False)

List of FE base names
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
| ``    LocalDegreeProfileGenerator `` | concatenate Local Degree Profile features.      |
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
Of course, you can directly inherit the ``BaseFeature`` as well.

Create Your Own FE
------------------
You can create your own feature engineering object by simply inheriting one of feature engineering base types ,namely ``generators``, ``selectors`` , ``graph``,
and overloading methods ``extract_xx_features``.

.. code-block :: python

    # for example : create a node one-hot feature.
    import autogl
    import torch
    from autogl.module.feature._generators._basic import BaseFeatureGenerator

    class OneHotFeatureGenerator(BaseFeatureGenerator):
        # if overrider_features==False , concat the features with original features; otherwise override.
        def __init__(self, override_features: bool = False): 
            super(BaseFeatureGenerator, self).__init__(override_features)

        def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
            num_nodes: int = (
                data.x.size(0)
                if data.x is not None and isinstance(data.x, torch.Tensor)
                else (data.edge_index.max().item() + 1)
            )
            return torch.eye(num_nodes)
