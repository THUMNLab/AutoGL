: _homo_cn:

==========================
图分类模型
==========================

构建图分类模块
=====================================
.. In AutoGL, we support two graph classification models, ``gin`` and  ``topk``.
在AutoGL中，我们支持两种图分类模型： ``gin`` and  ``topk`` 。

AutoGIN
>>>>>>>

.. The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper
图同构算子（Graph Isomorphism Operator）出自论文“How Powerful are Graph Neural Networks?”中，

图同构网络（Graph Isomorphism Network (GIN)）是一种图神经网络，出自论文 `“How Powerful are Graph Neural Networks” <https://arxiv.org/pdf/1810.00826.pdf>`_ 。

.. The layer is
层间更新方式为：

.. math::

    \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
    \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

或者：

.. math::

    \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
    (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

.. here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.
这里 :math:`h_{\mathbf{\Theta}}` 代表一个神经网络, 例如一个多层感知机（MLP）.

.. PARAMETERS:
参数包括：

.. - num_features: `int` - The dimension of features.
- num_features: `int` - 特征的维度.

.. - num_classes: `int` - The number of classes.
- num_classes: `int` - 类别的数量.

.. - device: `torch.device` or `str` - The device where model will be running on.
- device: `torch.device` or `str` - 用于运行模型的设备.

.. - init: `bool` - If True(False), the model will (not) be initialized.
- init: `bool` - 如果设为True（False），模型会（不会）被初始化.

.. code-block:: python

    class AutoGIN(BaseModel):
        r"""
        AutoGIN. The model used in this automodel is GIN, i.e., the graph isomorphism network from the `"How Powerful are
        Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper. The layer is

        .. math::
            \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
            \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

        or

        .. math::
            \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
            (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

        here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

        Parameters
        ----------
        num_features: `int`.
            The dimension of features.

        num_classes: `int`.
            The number of classes.

        device: `torch.device` or `str`
            The device where model will be running on.

        init: `bool`.
            If True(False), the model will (not) be initialized.
        """

        def __init__(
            self,
            num_features=None,
            num_classes=None,
            device=None,
            init=False,
            num_graph_features=None,
            **args
        ):

            super(AutoGIN, self).__init__()
            self.num_features = num_features if num_features is not None else 0
            self.num_classes = int(num_classes) if num_classes is not None else 0
            self.num_graph_features = (
                int(num_graph_features) if num_graph_features is not None else 0
            )
            self.device = device if device is not None else "cpu"

            self.params = {
                "features_num": self.num_features,
                "num_class": self.num_classes,
                "num_graph_features": self.num_graph_features,
            }
            self.space = [
                {
                    "parameterName": "num_layers",
                    "type": "DISCRETE",
                    "feasiblePoints": "4,5,6",
                },
                {
                    "parameterName": "hidden",
                    "type": "NUMERICAL_LIST",
                    "numericalType": "INTEGER",
                    "length": 5,
                    "minValue": [8, 8, 8, 8, 8],
                    "maxValue": [64, 64, 64, 64, 64],
                    "scalingType": "LOG",
                    "cutPara": ("num_layers",),
                    "cutFunc": lambda x: x[0] - 1,
                },
                {
                    "parameterName": "dropout",
                    "type": "DOUBLE",
                    "maxValue": 0.9,
                    "minValue": 0.1,
                    "scalingType": "LINEAR",
                },
                {
                    "parameterName": "act",
                    "type": "CATEGORICAL",
                    "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
                },
                {
                    "parameterName": "eps",
                    "type": "CATEGORICAL",
                    "feasiblePoints": ["True", "False"],
                },
                {
                    "parameterName": "mlp_layers",
                    "type": "DISCRETE",
                    "feasiblePoints": "2,3,4",
                },
                {
                    "parameterName": "neighbor_pooling_type",
                    "type": "CATEGORICAL",
                    "feasiblePoints": ["sum", "mean", "max"],
                },
                {
                    "parameterName": "graph_pooling_type",
                    "type": "CATEGORICAL",
                    "feasiblePoints": ["sum", "mean", "max"],
                },
            ]

            self.hyperparams = {
                "num_layers": 5,
                "hidden": [64,64,64,64],
                "dropout": 0.5,
                "act": "relu",
                "eps": "False",
                "mlp_layers": 2,
                "neighbor_pooling_type": "sum",
                "graph_pooling_type": "sum"
            }

            self.initialized = False
            if init is True:
                self.initialize()

.. Hyperparameters in GIN:
GIN中的超参数：

.. - num_layers: `int` - number of GIN layers.

.. - hidden: `List[int]` - hidden size for each hidden layer.

.. - dropout: `float` - dropout probability.

.. - act: `str` - type of activation function.

.. - eps: `str` - whether to train parameter :math:`epsilon` in the GIN layer.

.. - mlp_layers: `int` - number of MLP layers in the GIN layer.

.. - neighbor_pooling_type: `str` - pooling type in the  GIN layer.

.. - graph_pooling_type: `str` - graph pooling type following the last GIN layer.
- num_layers: `int` - GIN的层数。

- hidden: `List[int]` - 每个隐藏层的大小。

- dropout: `float` - 随机失活（Dropout）的概率。

- act: `str` - 激活函数的类型。

- eps: `str` - 是否在GIN层中训练参数 :math:`epsilon` 。

- mlp_layers: `int` - GIN中的多层感知机（MLP）层数。

- neighbor_pooling_type: `str` - GIN中的池化（pooling）层类型。

- graph_pooling_type: `str` - GIN最后一层之后的图池化（graph pooling）类型。


.. You could get define your own ``gin`` model by using ``from_hyper_parameter`` function and specify the hyperpameryers.
You could get define your own ``gin`` model by using ``from_hyper_parameter`` function and specify the hyperpameryers.
你可以通过使用 ``from_hyper_parameter`` 函数定义你自己的 ``gin`` 模型，并对其指定超参数。

.. code-block:: python

    # pyg version
    from autogl.module.model.pyg import AutoGIN
    # from autogl.module.model.dgl import AutoGIN  # dgl version
    model = AutoGIN(
                    num_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,
                    num_graph_features=0,
                    init=False
                ).from_hyper_parameter({
                    # hp from model
                    "num_layers": 5,
                    "hidden": [64,64,64,64],
                    "dropout": 0.5,
                    "act": "relu",
                    "eps": "False",
                    "mlp_layers": 2,
                    "neighbor_pooling_type": "sum",
                    "graph_pooling_type": "sum"
                }).model


.. Then you can train the model for 100 epochs.
然后你可以对模型进行100次的训练：

.. code-block:: python

    import torch.nn.functional as F

    # Define the loss optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    for epoch in range(100):
        model.train()
        for data in train_loader:
            data = data.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()

.. Finally, evaluate the trained model.
最后，你可以评估该模型：

.. code-block:: python

    def test(model, loader, args):
        model.eval()

        correct = 0
        for data in loader:
            data = data.to(args.device)
            output = model(data)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)

    acc = test(model, test_loader, args)


.. Automatic Search for Graph Classification Tasks
图分类任务的自动搜索
===============================================

.. In AutoGL, we also provide a high-level API Solver to control the overall pipeline. We encapsulated the training process in the Building GNN Modules part for graph classification tasks in the solver ``AutoGraphClassifier`` that supports automatic hyperparametric optimization as well as feature engineering and ensemble. In this part, we will show you how to use ``AutoGraphClassifier``.
.. In AutoGL, we also provide a high-level API Solver to control the overall pipeline. We encapsulated the training process in the Building GNN Modules part for graph classification tasks in the solver ``AutoGraphClassifier`` that supports automatic hyperparametric optimization as well as feature engineering and ensemble. In this part, we will show you how to use ``AutoGraphClassifier``.
在AutoGL中，我们还提供了一个高级的API求解器来控制整个流水线。我们将构建图神经网络模块部分的训练过程封装在求解器 ``AutoGraphClassifier`` 中以用于图分类任务，它支持自动超参数优化，特征工程及集成。
在这一部分，我们提供了一个例子来指导如何使用 ``AutoGraphClassifier`` ：

.. code-block:: python

    solver = AutoGraphClassifier(
                feature_module=None,
                graph_models=[args.model],
                hpo_module='random',
                ensemble_module=None,
                device=args.device, max_evals=1,
                trainer_hp_space = fixed(
                    **{
                        # hp from trainer
                        "max_epoch": args.epoch,
                        "batch_size": args.batch_size,
                        "early_stopping_round": args.epoch + 1,
                        "lr": args.lr,
                        "weight_decay": 0,
                    }
                ),
                model_hp_spaces=[
                    fixed(**{
                        # hp from model
                        "num_layers": 5,
                        "hidden": [64,64,64,64],
                        "dropout": 0.5,
                        "act": "relu",
                        "eps": "False",
                        "mlp_layers": 2,
                        "neighbor_pooling_type": "sum",
                        "graph_pooling_type": "sum"
                    }) if args.model == 'gin' else fixed(**{
                        "ratio": 0.8,
                        "dropout": 0.5,
                        "act": "relu"
                    }),
                ]
            )

    # fit auto model
    solver.fit(dataset, evaluation_method=['acc'])
    # prediction
    out = solver.predict(dataset, mask='test')
