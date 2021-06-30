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

The evaluation function is defined in ``evaluate()``, you can use your our evaluation metrics and methods.

Node Classification with Sampling
------------------------------------
According to various present studies, training with spatial sampling has been demonstrated
as an efficient technique for representation learning on large-scale graph.
We provide implementations for various representative sampling mechanisms including
Neighbor Sampling, Layer Dependent Importance Sampling (LADIES), and GraphSAINT.
With the leverage of various efficient sampling mechanisms,
users can utilize this library on large-scale graph dataset, e.g. Reddit.

Specifically, as various sampling techniques generally require model to support
some layer-wise processing in forwarding, now only the provided GCN and GraphSAGE models are ready for
Node-wise Sampling (Neighbor Sampling) and Layer-wise Sampling (LADIES).
More models and more tasks are scheduled to support sampling in future version.

* Node-wise Sampling (GraphSAGE)
    Both ``GCN`` and ``GraphSAGE`` models are supported.

* Layer-wise Sampling (Layer Dependent Importance Sampling)
    Only the ``GCN`` model is supported in current version.

* Subgraph-wise Sampling (GraphSAINT)
    As The GraphSAINT sampling technique have no specific requirements for model to adopt,
    most of the available models are feasible for adopting GraphSAINT technique.
    However, the prediction process is a potential bottleneck or even obstacle
    when the GraphSAINT technique is actually applied on large-scale graph,
    thus the the model to adopt is better to support layer-wise prediction,
    and the provided ``GCN`` model already meet that enhanced requirement.
    According to empirical experiments,
    the implementation of GraphSAINT now has the leverage to support
    an integral graph smaller than the *Flickr* graph data.

The sampling techniques can be utilized by adopting corresponding trainer
``NodeClassificationGraphSAINTTrainer``,
``NodeClassificationLayerDependentImportanceSamplingTrainer``,
and ``NodeClassificationNeighborSamplingTrainer``.
You can either specify the corresponding name of trainer in YAML configuration file
or instantiate the solver ``AutoNodeClassifier``
with the instance of specific trainer as ``model`` argument.

A brief example is demonstrated as follows:

.. code-block:: python

    ladies_sampling_trainer = NodeClassificationLayerDependentImportanceSamplingTrainer(
        model='gcn', num_features=dataset.num_features, num_classes=dataset.num_classes,
        ...
    )
    AutoNodeClassifier(graph_models=(ladies_sampling_trainer,), ...)
