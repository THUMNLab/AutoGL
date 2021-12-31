==========================
Graph Classification Model
==========================

Building Graph Classification Modules
=====================================

In AutoGL, we support two graph classification models, ``gin`` and  ``topk``.

AutoGIN
>>>>>>>

The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper

Graph Isomorphism Network (GIN) is one graph classification model from `"How Powerful are Graph Neural Networks" paper <https://arxiv.org/pdf/1810.00826.pdf>`_.

The layer is

.. math::

    \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
    \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

or

.. math::

    \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
    (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

PARAMETERS:
- num_features: `int` - The dimension of features.

- num_classes: `int` - The number of classes.

- device: `torch.device` or `str` - The device where model will be running on.

- init: `bool` - If True(False), the model will (not) be initialized.

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

Hyperparameters in GIN:

- num_layers: `int` - number of GIN layers.
  
- hidden: `List[int]` - hidden size for each hidden layer.

- dropout: `float` - dropout probability.

- act: `str` - type of activation function.

- eps: `str` - whether to train parameter :math:`epsilon` in the GIN layer.

- mlp_layers: `int` - number of MLP layers in the GIN layer.

- neighbor_pooling_type: `str` - pooling type in the  GIN layer.

- graph_pooling_type: `str` - graph pooling type following the last GIN layer.


You could get define your own ``gin`` model by using ``from_hyper_parameter`` function and specify the hyperpameryers.

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


Then you can train the model for 100 epochs.

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

Finally, evaluate the trained model.

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


Automatic Search for Graph Classification Tasks
===============================================

In AutoGL, we also provide a high-level API Solver to control the overall pipeline.
We encapsulated the training process in the Building GNN Modules part for graph classification tasks
in the solver ``AutoGraphClassifier`` that supports automatic hyperparametric optimization 
as well as feature engineering and ensemble. In this part, we will show you how to use 
``AutoGraphClassifier``.

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
