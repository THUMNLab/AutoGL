.. _trainer:

AutoGL Trainer
==============

AutoGL project use ``trainer`` to handle the auto-training of tasks. Currently, we support the following tasks:

* ``NodeClassificationTrainer`` for semi-supervised node classification
* ``GraphClassificationTrainer`` for supervised graph classification
* ``LinkPredictionTrainer`` for link prediction


Initialization
--------------

A trainer can either be initialized from its ``__init__()``. If you want to build a trainer by ``__init__()``, you need to pass the following parameters to it, namely as ``model``, ``num_features``, and ``num_classes`` and ``auto ensemble``. You can also define some parameters alternatively, including ``optimizer``, ``lr``, ``max_epoch``, ``early_stopping_round``, ``weight_decay`` and etc.

In the ``__init__()``, you need to define the space and hyperparameter of your trainer:  

.. code-block:: python

    # 1. define your search space of trainer
    self.space = [
        {'parameterName': 'max_epoch', 'type': 'INTEGER', 'maxValue': 300, 'minValue': 10, 'scalingType': 'LINEAR'},
        {'parameterName': 'early_stopping_round', 'type': 'INTEGER', 'maxValue': 30, 'minValue': 10,
             'scalingType': 'LINEAR'},
        {'parameterName': 'lr', 'type': 'DOUBLE', 'maxValue': 1e-3, 'minValue': 1e-4, 'scalingType': 'LOG'},
        {'parameterName': 'weight_decay', 'type': 'DOUBLE', 'maxValue': 5e-3, 'minValue': 5e-4,
             'scalingType': 'LOG'},
    ]

    # 2. define the initial point of hyperparameter search of your trainer
    self.hyperparams = {
        'max_epoch': self.max_epoch,
        'early_stopping_round': self.early_stopping_round,
        'lr': self.lr,
        'weight_decay': self.weight_decay
    }

Where ``self.space`` is a list of dictionary indicating the name, type, and some properties of the parameter. ``self.hyperparams`` is a dictionary indicating the hyper-parameters used in this trainer.

Train and Predict
-----------------
After initializing a trainer, you can train it on the given datasets.

We have given the training and testing functions for the tasks of node classification, graph classification, and link prediction up to now. You can also create your tasks following the similar patterns with ours. For training, you need to define ``train_only()`` and use it in ``train()``. For testing, you need to define ``predict_proba()`` and use it in ``predict()``.

The evaluation funtion is defined in ``evaluate()``, you can use your our evaluation metrics and methods.


