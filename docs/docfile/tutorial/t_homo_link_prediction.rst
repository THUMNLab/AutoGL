==========================
Link Prediction Model
==========================

Building Link Prediction Modules
=====================================

In AutoGL, we support three models for link prediction models, ``gcn``, ``gat`` and  ``sage``.

AutoLinkPredictor
>>>>>>>

Used to automatically solve the link prediction problems. For example, 


.. code-block:: python

    class AutoGCN(BaseAutoModel):
    r"""
    AutoGCN.
    The model used in this automodel is GCN, i.e., the graph convolutional network from the
    `"Semi-supervised Classification with Graph Convolutional
    Networks" <https://arxiv.org/abs/1609.02907>`_ paper. The layer is

    .. math::

        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Parameters
    ----------
    num_features: ``int``
        The dimension of features.

    num_classes: ``int``
        The number of classes.

    device: ``torch.device`` or ``str``
        The device where model will be running on.

    init: `bool`.
        If True(False), the model will (not) be initialized.
    """

    def __init__(
        self,
        num_features: int = ...,
        num_classes: int = ...,
        device: _typing.Union[str, torch.device] = ...,
        **kwargs
    ) -> None:
        super().__init__(num_features, num_classes, device, **kwargs)
        self.hyper_parameter_space = [
            {
                "parameterName": "add_self_loops",
                "type": "CATEGORICAL",
                "feasiblePoints": [1],
            },
            {
                "parameterName": "normalize",
                "type": "CATEGORICAL",
                "feasiblePoints": [1],
            },
            {
                "parameterName": "num_layers",
                "type": "DISCRETE",
                "feasiblePoints": "2,3,4",
            },
            {
                "parameterName": "hidden",
                "type": "NUMERICAL_LIST",
                "numericalType": "INTEGER",
                "length": 3,
                "minValue": [8, 8, 8],
                "maxValue": [128, 128, 128],
                "scalingType": "LOG",
                "cutPara": ("num_layers",),
                "cutFunc": lambda x: x[0] - 1,
            },
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.8,
                "minValue": 0.2,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
        ]

        self.hyper_parameters = {
            "num_layers": 3,
            "hidden": [128, 64],
            "dropout": 0,
            "act": "relu",
        }

    def _initialize(self):
        self._model = GCN(
            self.input_dimension,
            self.output_dimension,
            self.hyper_parameters.get("hidden"),
            self.hyper_parameters.get("act"),
            self.hyper_parameters.get("dropout", None),
            bool(self.hyper_parameters.get("add_self_loops", True)),
            bool(self.hyper_parameters.get("normalize", True)),
        ).to(self.device)

You could get define your own ``LinkPrediction`` model by using ``from_hyper_parameter`` function and specify the hyperpameryers.

.. code-block:: python

    # pyg version
    from autogl.module.model.pyg import AutoLinkPredictor
    # from autogl.module.model.dgl import AutoLinkPredictor  # dgl version
    model = AutoLinkPredictor(
            feature_module="NormalizeFeatures",
            graph_models=(args.model, ),
            hpo_module="random",
            ensemble_module=None,
            max_evals=1,
            trainer_hp_space=fixed(**{
                "max_epoch": 100,
                "early_stopping_round": 101,
                "lr": 1e-2,
                "weight_decay": 0.0,
            }),
            model_hp_spaces=[{"encoder": fixed(**model_hp), "decoder": fixed(**decoder_hp)}]
        ).model

Then you can train the model for 100 epochs.

.. code-block:: python

    import torch.nn.functional as F

    # Define the loss optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        z = model.lp_encode(splitted[0])
        link_logits = model.lp_decode(
            z, torch.stack(splitted[1].edges()), torch.stack(splitted[2].edges())
        )
        link_labels = get_link_labels(
            torch.stack(splitted[1].edges()), torch.stack(splitted[2].edges())
        )
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer.step()

        auc_val = evaluate(model, splitted, "val")

        if auc_val > best_auc:
            best_auc = auc_val
            best_parameters = pickle.dumps(model.state_dict())

Finally, evaluate the trained model.

.. code-block:: python

    model.load_state_dict(pickle.loads(best_parameters))
    evaluate(model, splitted, "test")


Automatic Search for Link Prediction Tasks
===============================================

In AutoGL, we also provide a high-level API Solver to control the overall pipeline.
We encapsulated the training process in the Building GNN Modules part for link prediction tasks
in the solver ``AutoLinkPredictor`` that supports automatic hyperparametric optimization 
as well as feature engineering and ensemble. In this part, we will show you how to use 
``AutoLinkPredictor``.

.. code-block:: python

    solver = AutoLinkPredictor(
            feature_module="NormalizeFeatures",
            graph_models=(args.model, ),
            hpo_module="random",
            ensemble_module=None,
            max_evals=1,
            trainer_hp_space=fixed(**{
                "max_epoch": 100,
                "early_stopping_round": 101,
                "lr": 1e-2,
                "weight_decay": 0.0,
            }),
            model_hp_spaces=[{"encoder": fixed(**model_hp), "decoder": fixed(**decoder_hp)}]
        )
    
    solver.fit(dataset, train_split=0.85, val_split=0.05, evaluation_method=["auc"], seed=seed)
    pre = solver.evaluate(metric="auc")
