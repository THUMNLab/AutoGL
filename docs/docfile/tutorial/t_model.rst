.. _model:

AutoGL Model
============

AutoGL project uses ``model`` to define the common graph nerual networks and ``automodel`` to denote the relative class that includes some auto functions. Currently, we support the following models and automodels:

* ``GCN`` and ``AutoGCN`` : graph convolutional network from https://arxiv.org/abs/1609.02907
* ``GAT`` and ``AutoGAT`` : graph attentional network from https://arxiv.org/abs/1710.10903
* ``GraphSAGE`` and ``AutoGraphSAGE`` : from the "Inductive Representation Learning on Large Graphs" https://arxiv.org/abs/1706.02216

And we also support the following models and automodels for graph classification tasks:
* ``GIN`` and ``AutoGIN`` : graph isomorphism network from https://arxiv.org/abs/1810.00826
* ``Topkpool`` and ``AutoTopkpool`` : graph U-Net from https://arxiv.org/abs/1905.05178, https://arxiv.org/abs/1905.02850

Define your own model and automodel
-----------------------------------

If you want to add your own model and automodel for some task, the only thing you should do is add a new model where the forward function should be fulfilled and a new automodel inherited from the basemodel.

For new models used in link prediction tasks, you should fulfill the lp_encode and lp_decode function. The difference between lp_encode and forward function is that there is not classification layer in lp_encode.


Firstly, you should define your model if it does not belong to the models above.

Secondly, you should define your corresponding automodel.

.. code-block:: python

    # 1. define your search space to self.space of your automodel instance
    [
        {'parameterName': 'num_layers', 'type': 'DISCRETE', 'feasiblePoints': '2,3,4'},
        {"parameterName": 'hidden', "type": "NUMERICAL_LIST", "numericalType": "INTEGER", "length": 3, "minValue": [8, 8, 8], "maxValue": [64, 64, 64], "scalingType": "LOG"},
        {'parameterName': 'dropout', 'type': 'DOUBLE', 'maxValue': 0.9, 'minValue': 0.1, 'scalingType': 'LINEAR'},
        {'parameterName': 'act', 'type': 'CATEGORICAL_LIST', "feasiblePoints": ['leaky_relu', 'relu', 'elu', 'tanh']},
    ]
    # 2. define the default point to self.hyperparams of your automodel instance
    {
        'num_layers': 2,
        'hidden': [16],
        'dropout': 0.2,
        'act': 'leaky_relu'
    }

Where ``self.space`` is a list of dictionary indicating the name, type, feasible point, min/max value and some properties of the parameter. ``self.hyperparams`` is a dictionary indicating the hyper-parameters used in this model.

Finally, you can use the defined model and automodel for the specific need.

.. code-block :: python

    # for example
    import torch
    from .base import BaseModel
    class YourGNN(torch.nn.Module):
        def forward(self, data):
            pass  # Your forward function

    class YourAutoGNN(BaseModel):
        def __init__(self, num_features=None, num_classes=None, device=None, init=True, **args):
            """
            num_features: the number of features
            num_classes: the number of classes
            device: your device to run code
            init: if True, the model will be initialize
            """
            self.space = XXX  # Define your search space
            self.hyperparams = XXX  # Define your hyper-parameters
            self.initialized = False
            if init is True:
                self.initialize()
