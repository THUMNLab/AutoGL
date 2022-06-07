.. _hetero_node_clf_cn:

.. Node Classification for Heterogeneous Graph
异质图上的节点分类
===========================================

.. This tutorial introduces how to use AutoGL to automate the learning of heterogeneous graphs in Deep Graph Library (DGL).
本教程指导如何利用AutoGL来对Deep Graph Library (DGL)中的异质图进行自动学习。

.. Creating a Heterogeneous Graph
创建一个异质图
------------------------------
.. AutoGL supports datasets created in DGL. We provide two datasets named "hetero-acm-han" and "hetero-acm-hgt" for HAN and HGT models, respectively [1].
AutoGL支持DGL内的数据集。我们分别对HAN和HGT这两个模型提供了两个数据集，分别叫做 ``hetero-acm-han`` 和 ``hetero-acm-hgt`` [1]。

.. The following code snippet is an example for loading a heterogeneous graph.
下面的代码片断提供了一个加载异质图的例子：

.. code-block:: python

    from autogl.datasets import build_dataset_from_name
    dataset = build_dataset_from_name("hetero-acm-han")

.. You can also access to data stored in the dataset object for more details:
你也可以通过访问存储在数据集对象中的数据来了解更多细节：

.. code-block:: python

    g = dataset[0]

    node_type = dataset.schema["target_node_type"]
    labels = g.nodes[node_type].data['label']
    num_classes = labels.max().item() + 1
    num_features=g.nodes[node_type].data['feat'].shape[1]

    train_mask = g.nodes[node_type].data['train_mask']
    val_mask = g.nodes[node_type].data['val_mask']
    test_mask = g.nodes[node_type].data['test_mask']

.. You can also build your own dataset and do feature engineering by adding files in the location AutoGL/autogl/datasets/_heterogeneous_datasets/_dgl_heterogeneous_datasets.py. We suggest users create a data object of type torch_geometric.data.HeteroData refering to the official documentation of DGL.
你也可以通过在 AutoGL/autogl/datasets/_heterogeneous_datasets/_dgl_heterogeneous_datasets.py 目录下添加文件来建立自己的数据集并进行特征工程。我们建议用户参考DGL的官方文档，创建一个类型为 ``torch_geometric.data.HeteroData`` 的数据对象。

.. Building Heterogeneous GNN Modules
构建异质图神经网络模块
----------------------------------
.. AutoGL integrates commonly used heterogeneous graph neural network models such as HeteroRGCN (Schlichtkrull et al., 2018) [2], HAN (Wang et al., 2019) [3] and HGT (Hu et al., 2029) [4].
AutoGL集成了常用的异质图神经网络模型，例如HeteroRGCN (Schlichtkrull et al., 2018) [2]，HAN (Wang et al., 2019) [3]和HGT (Hu et al., 2029) [4]：

.. code-block:: python

    from autogl.module.model.dgl import AutoHAN
    model = AutoHAN(
        dataset=dataset,
        num_features=num_features,
        num_classes=num_classes,
        device = args['device'],
        init=True
    ).model

.. Then you can train the model for 100 epochs.
然后你可以对模型进行100次的训练：

.. code-block:: python

    # Define the loss function.
    loss_fcn = torch.nn.CrossEntropyLoss()
    # Define the loss optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2,
                                 weight_decay=1e-2)

    # Training.
    for epoch in range(100):
        model.train()
        logits = model(g)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_loss, val_acc, _, _ = evaluate(model, g, labels, val_mask, loss_fcn)

.. Finally, evaluate the model.
最后，你可以评估该模型：

.. code-block:: python

    _, test_acc, _, _ = evaluate(model, g, labels, test_mask, loss_fcn)

.. You can also define your own heterogeneous graph neural network models by adding files in the location AutoGL/autogl/module/model/dgl/hetero.
你也可以通过在 AutoGL/autogl/module/model/dgl/hetero 目录下添加文件来定义自己的异质图神经网络模型。

.. Automatic Search for Node Classification Tasks
节点分类任务的自动搜索
----------------------------------------------
.. On top of the modules mentioned above, we provide a high-level API Solver to control the overall pipeline. We encapsulated the training process in the Building Heterogeneous GNN Modules part in the solver AutoHeteroNodeClassifier that supports automatic hyperparametric optimization as well as feature engineering and ensemble.
.. In this part, we will show you how to use AutoHeteroNodeClassifier to automatically predict the publishing conference of a paper using the ACM academic graph dataset.
在上述模块的基础上，我们提供了一个高级API求解器来控制整个流水线。我们将构建异质图神经网络模块部分的训练过程封装在求解器 ``AutoHeteroNodeClassifier`` 中，它支持自动超参数优化，特征工程及集成。
在这一部分，我们将使用ACM学术图数据集，来向你展示如何使用 ``AutoHeteroNodeClassifier`` 自动预测一篇论文发表在哪个会议上。

.. Firstly, we get the pre-defined model hyperparameter.
首先，我们得到预先定义的模型超参数：

.. code-block:: python

    from helper import get_encoder_decoder_hp
    model_hp, _ = get_encoder_decoder_hp(args.model)

.. You can also define your own model hyperparameters in a dict:
你也可以在一个字典（dict）中定义你自己的模型超参数：

.. code-block:: python

    model_hp = {
                "num_layers": 2,
                "hidden": [256],
                "heads": 4,
                "dropout": 0.2,
                "act": "leaky_relu",
            }

.. Secondly, use AutoHeteroNodeClassifier directly to bulid automatic heterogeneous GNN models in the following example:
然后，在下面的例子中，直接使用 ``AutoHeteroNodeClassifier`` 来构建自动异质图神经网络模型：

.. code-block:: python

    from autogl.solver import AutoHeteroNodeClassifier
    solver = AutoHeteroNodeClassifier(
                graph_models=["han"],
                hpo_module="random",
                ensemble_module=None,
                max_evals=1,
                device=args.device,
                trainer_hp_space=fixed(
                    max_epoch=100,
                    early_stopping_round=101,
                    lr=1e-3,
                    weight_decay=1e-2
                ),
                model_hp_spaces=[fixed(**model_hp)]
            )

.. Finally, fit and evlauate the model.
最后，对模型进行拟合和评估：

.. code-block:: python

    solver.fit(dataset)
    acc = solver.evaluate()

.. References:
参考文献：

[1] https://data.dgl.ai/dataset/ACM.mat

[2] Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018.

[3] Wang, Xiao, et al. "Heterogeneous graph attention network." The World Wide Web Conference. 2019.

[4] Yun, Seongjun, et al. "Graph transformer networks." Advances in Neural Information Processing Systems 32 (2019): 11983-11993.
