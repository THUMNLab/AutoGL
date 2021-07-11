.. _trainer:

AutoGL Trainer
==============

AutoGL project use ``trainer`` to handle the auto-training of tasks. Currently, we support the following tasks:

* ``NodeClassificationTrainer`` for semi-supervised node classification
* ``GraphClassificationTrainer`` for supervised graph classification
* ``LinkPredictionTrainer`` for link prediction


Lazy Initialization
-------------------
Similar reason to :ref:model, we also use lazy initialization for all trainers. Only (part of) the hyper-parameters will be set when ``__init__()`` is called. The ``trainer`` will have its core ``model`` only after ``initialize()`` is explicitly called, which will be done automatically in ``solver`` and ``duplicate_from_hyper_parameter()``, after all the hyper-parameters are set properly.


Train and Predict
-----------------
After initializing a trainer, you can train it on the given datasets.

We have given the training and testing functions for the tasks of node classification, graph classification, and link prediction up to now. You can also create your tasks following the similar patterns with ours. For training, you need to define ``train_only()`` and use it in ``train()``. For testing, you need to define ``predict_proba()`` and use it in ``predict()``.

The evaluation function is defined in ``evaluate()``, you can use your our evaluation metrics and methods.

Node Classification with Sampling
---------------------------------
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
with the instance of specific trainer. However, please make sure to manange some key
hyper-paramters properly inside the hyper-parameter space. Specifically:

For ``NodeClassificationLayerDependentImportanceSamplingTrainer``, you need to set the
hyper-parameter ``sampled_node_sizes`` properly. The space of ``sampled_node_sizes`` should
be a list of the same size with your **Sequential Model**. For example, if you have a
model with layer number 4, you need to pass the hyper-parameter space properly:

.. code-block:: python

    solver = AutoNodeClassifier(
        graph_models=(A_MODEL_WITH_4_LAYERS,),
        default_trainer='NodeClassificationLayerDependentImportanceSamplingTrainer',
        trainer_hp_space=[
            # (required) you need to set the trainer_hp_space properly.
            {
                'parameterName': 'sampled_node_sizes',
                'type': 'NUMERICAL_LIST', 
                "numericalType": "INTEGER",
                "length": 4,                    # same with the layer number of your model
                "minValue": [200,200,200,200],
                "maxValue": [1000,1000,1000,1000],
                "scalingType": "LOG"
            },
            ...
        ]
    )

If the layer number of your model is a searchable hyper-parameters, you can also set the ``cutPara``
and ``cutFunc`` properly, to make it connected with your layer number hyper-parameters of model.

.. code-block:: python

    '''
    Suppose the layer number of your model is of the following forms:
    {
        'parameterName': 'layer_number',
        'type': 'INTEGER',
        'minValue': 2,
        'maxValue': 4,
        'scalingType': 'LOG'
    }
    '''

    solver = AutoNodeClassifier(
        graph_models=(A_MODEL_WITH_DYNAMIC_LAYERS,),
        default_trainer='NodeClassificationLayerDependentImportanceSamplingTrainer',
        trainer_hp_space=[
            # (required) you need to set the trainer_hp_space properly.
            {
                'parameterName': 'sampled_node_sizes',
                'type': 'NUMERICAL_LIST', 
                "numericalType": "INTEGER",
                "length": 4,                    # max length
                "cutPara": ("layer_number", ),  # link with layer_number
                "cutFunc": lambda x:x[0],       # link with layer_number
                "minValue": [200,200,200,200],
                "maxValue": [1000,1000,1000,1000],
                "scalingType": "LOG"
            },
            ...
        ]
    )


Similarly, if you want to use ``NodeClassificationNeighborSamplingTrainer``, you need to
make sure setting the hyper-parameter ``sampling_sizes`` the same length as the layer number
of your model. For example:

.. code-block:: python

    '''
    Suppose the layer number of your model is of the following forms:
    {
        'parameterName': 'layer_number',
        'type': 'INTEGER',
        'minValue': 2,
        'maxValue': 4,
        'scalingType': 'LOG'
    }
    '''

    solver = AutoNodeClassifier(
        graph_models=(A_MODEL_WITH_DYNAMIC_LAYERS,),
        default_trainer='NodeClassificationNeighborSamplingTrainer',
        trainer_hp_space=[
            # (required) you need to set the trainer_hp_space properly.
            {
                'parameterName': 'sampling_sizes',
                'type': 'NUMERICAL_LIST', 
                "numericalType": "INTEGER",
                "length": 4,                    # max length
                "cutPara": ("layer_number", ),  # link with layer_number
                "cutFunc": lambda x:x[0],       # link with layer_number
                "minValue": [20,20,20,20],
                "maxValue": [100,100,100,100],
                "scalingType": "LOG"
            },
            ...
        ]
    )


You can also pass a trainer inside model list directly. A brief example is demonstrated as follows:

.. code-block:: python

    ladies_sampling_trainer = NodeClassificationLayerDependentImportanceSamplingTrainer(
        model='gcn', num_features=dataset.num_features, num_classes=dataset.num_classes, ...
    )

    ladies_sampling_trainer.hyper_parameter_space = [
        # (required) you need to set the trainer_hp_space properly.
        {
            'parameterName': 'sampled_node_sizes',
            'type': 'NUMERICAL_LIST', 
            "numericalType": "INTEGER",
            "length": 4,                    # max length
            "cutPara": ("num_layers", ),    # link with layer_number
            "cutFunc": lambda x:x[0],       # link with layer_number
            "minValue": [200,200,200,200],
            "maxValue": [1000,1000,1000,1000],
            "scalingType": "LOG"
        },
        ...
    ]

    AutoNodeClassifier(graph_models=(ladies_sampling_trainer,), ...)
