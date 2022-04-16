.. _fe_cn:

AutoGL 特征工程
==========================

我们提供了一系列的节点和图的特征工程方法。您可以挑选需要的特征工程方法，并在一个特性工程管道中编写。

快速开始
-----------
.. code-block :: python

    # 1. 选择一个数据集.
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')

    # 2. 选择特征工程方法
    from autogl.module.feature._base_feature_engineer._base_feature_engineer import _ComposedFeatureEngineer
    from autogl.module.feature import EigenFeatureGenerator
    from autogl.module.feature import NetLSD

    # 可以通过以下方式将多个特征工程方法组合起来
    fe = _ComposedFeatureEngineer([
        EigenFeatureGenerator(size=32),
        NetLSD()
    ])

    # 3.拟合变换数据
    fe.fit(data)
    data1=fe.transform(data,inplace=False)


特征工程方法
---------------------
现在支持3种类型的特征工程方法, 分别是 ``generators``, ``selectors`` , ``graph``. 你可以像在 ``快速开始`` 部分一样引入对应的模块，或者可以直接在Config或者Solver中传入需要的方法名称。

1. ``generators``

+---------------------------+-------------------------------------------------+
|           方法名            |                   描述                   |
+===========================+=================================================+
| ``GraphletGenerator``              | 生成local graphlet 数量作为节点特征 |
+---------------------------+-------------------------------------------------+
| ``EigenFeatureGenerator``                 | 生成特征向量作为节点特征                     |
+---------------------------+-------------------------------------------------+
| ``PageRankFeatureGenerator``              | 生成Pagerank 分数作为节点特征               |
+---------------------------+-------------------------------------------------+
| ``    LocalDegreeProfileGenerator `` | 生成Local Degree Profile作为节点特征      |
+---------------------------+-------------------------------------------------+
| ``NormalizeFeatures``  | 归一化所有节点特征                     |
+---------------------------+-------------------------------------------------+
| ``OneHotDegreeGenerator``       | 生成节点度的独热编码作为节点特征            |
+---------------------------+-------------------------------------------------+
| ``OneHotFeatureGenerator``                | 生成节点ID的独热编码作为节点特征           |
+---------------------------+-------------------------------------------------+

2. ``selectors``

+----------------------+--------------------------------------------------------------------------------+
|         方法名         |                                  描述                                   |
+======================+================================================================================+
| ``FilterConstant`` | 删除所有常量和独热编码节点特征                        |
+----------------------+--------------------------------------------------------------------------------+
| ``GBDTFeatureSelector``             | 通过梯度下降决策树对节点特征进行重要性排序，选择最重要的K个重要的节点特征 |
+----------------------+--------------------------------------------------------------------------------+

3. ``graph``

``NetLSD`` 是一种图特征生成方法。

一系列Networkx中的图特征生成方法被集成到库中, 若想了解详情，请查阅NetworkX的相关文档。  (``NxLargeCliqueSize``, ``NxAverageClusteringApproximate``, ``NxDegreeAssortativityCoefficient``, ``NxDegreePearsonCorrelationCoefficient``, ``NxHasBridge``
,``NxGraphCliqueNumber``, ``NxGraphNumberOfCliques``, ``NxTransitivity``, ``NxAverageClustering``, ``NxIsConnected``, ``NxNumberConnectedComponents``, 
``NxIsDistanceRegular``, ``NxLocalEfficiency``, ``NxGlobalEfficiency``, ``NxIsEulerian``)

特征工程类型根据变化特征的方法进行分类。 ``generators`` 生成新特征并拼接或覆盖原始的特征。 而 ``selectors`` 选择原始特征中有用的部分。 
前两种可以节点或者边的层级使用（更改节点或边的特征）, 而 ``graph`` 关注图级别的特征工程(在图特征上进行修改)。
如果您需要进一步开发使用，可以通过继承其中一种基础类进行修改；或者可以直接继承更加底层的``BaseFeature``类。

构建您自己的特征工程方法
------------------
您可以继承其中一种特征工程基础类 ``BaseFeatureGenerator``或 ``BaseFeatureSelector`` 进行修改， 重载方法 ``extract_xx_features``。对于图层级特征工程，可以参考 ``_NetworkXGraphFeatureEngineer`` 的实现。

.. code-block :: python

    # 例子：创建节点ID独热编码特征
    import autogl
    import torch
    from autogl.module.feature._generators._basic import BaseFeatureGenerator

    class OneHotFeatureGenerator(BaseFeatureGenerator):
        # 设置 overrider_features 为False , 则将原始特征拼接起来; 否则直接覆盖原始特征。
        def __init__(self, override_features: bool = False): 
            super(BaseFeatureGenerator, self).__init__(override_features)

        def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
            num_nodes: int = (
                data.x.size(0)
                if data.x is not None and isinstance(data.x, torch.Tensor)
                else (data.edge_index.max().item() + 1)
            )
            return torch.eye(num_nodes)
