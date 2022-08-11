.. _quickstart_cn:

快速开始
===========

本教程将帮助您快速了解智图（AutoGL）中重要类的概念和用法。在本教程中，您将在数据集`Cora`_上进行快速的自动图学习。

.. _Cora: https://graphsandnetworks.com/the-cora-dataset/

智图
---------------

基于自动机器学习（AutoML）的概念，自动图学习的目标是用图数据**自动地**解决任务。与传统的学习框架不同，自动图学习不需要人工参与实验循环。您只需要向智图求解器提供数据集和任务，本框架将自动为您找到合适的方法和超参数。

下图描述了智图框架的工作流程。

为了达到自动机器学习的目标，我们提出的自动图学习框架组织如下。我们用 ``数据集（dataset）``来维护由用户给出的图形数据集。为了指定目标任务，需要构建一个 ``求解器（solver）``对象。 ``求解器``内部，有五个子模型来帮助完成完成自动图学习任务，即 ``自动特征工程``， ``自动模型``， ``神经架构搜索``， ``超参数优化``和 ``自动集成学习``来根据您的需求自动预处理/增强您的数据，选择和优化深度模型并集成。
假设你想在数据集 ``Cora``上进行自动图学习。首先，您可以通过 ``dataset``模块得到 ``Cora``数据集：

.. code-block:: python

    from autogl.datasets import build_dataset_from_name
    cora_dataset = build_dataset_from_name('cora')

数据集将自动下载给您。请参考:ref:`dataset`或:ref:`dataset documentation`获取更多关于数据集构造、可用数据集、添加本地数据集等详细信息。

在导出数据集之后，您可以构建一个 ``节点分类求解器（node classification solver）``来处理自动训练过程：

.. code-block:: python

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from autogl.solver import AutoNodeClassifier
    solver = AutoNodeClassifier(
        feature_module='deepgl',
        graph_models=['gcn', 'gat'],
        hpo_module='anneal',
        ensemble_module='voting',
        device=device
    )

通过这种方式，我们构建了一个 ``节点分类求解器``，它使用 ``deepgl``进行特征工程，并使用 ``anneal``超参数优化器对给定的三个模型“ ``['gcn','gat']``进行优化。派生的模型将使用 ``voting``集成器进行集成。请参考相应的教程或文档，以了解可用子模块的定义和用法。

Then, you can fit the solver and then check the leaderboard:
接下来，你可以安装求解器，然后查看排行榜：

.. code-block:: python

    solver.fit(cora_dataset, time_limit=3600)
    solver.get_leaderboard().show()

``time_limit``设置为3600，这样整个自动绘图过程不会超过1小时。 ``solver.show()``将显示由 ``solver``维护的模型，以及它们在验证数据集上的性能。

然后，你可以使用提供的评估函数进行预测和结果评估：

.. code-block:: python

    from autogl.module.train import Acc
    predicted = solver.predict_proba()
    print('Test accuracy: ', Acc.evaluate(predicted, 
        cora_dataset.data.y[cora_dataset.data.test_mask].cpu().numpy()))

.. 注意:: 当预测时，你不需要再次传递``cora_dataset``，因为数据集被``求解器``**记住**，预测时如果没有传递数据集将被重用。然而，您也可以在预测时传递一个新的数据集，新的数据集将被使用，而不是被记住的数据集。详情请参考:ref:`solver`或:ref:`solver documentation`。