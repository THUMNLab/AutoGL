.. _solver:

AutoGL Solver
=============

AutoGL project use ``solver`` to handle the auto-solvation of tasks. Currently, we support the following tasks:

* ``AutoNodeClassifier`` for semi-supervised node classification
* ``AutoGraphClassifier`` for supervised graph classification
* ``AutoLinkPredictor`` for link prediction

Initialization
--------------

A solver can either be initialized from its ``__init__()`` or from a config dictionary or file.

Initialize from ``__init__()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to build a solver by ``__init__()``, you need to pass the key modules to it. You can either pass the keywords of corresponding modules or the initialized instances:

.. code-block:: python

    from autogl.solver import AutoNodeClassifier
    
    # 1. initialize from keywords
    solver = AutoNodeClassifier(
        feature_module='deepgl', 
        graph_models=['gat','gcn'], 
        hpo_module='anneal', 
        ensemble_module='voting',
        device='auto'
    )

    # 2. initialize using instances
    from autogl.module import AutoFeatureEngineer, AutoGCN, AutoGAT, AnnealAdvisorHPO, Voting
    solver = AutoNodeClassifier(
        feature_module=AutoFeatureEngineer(),
        graph_models=[AutoGCN(device='cuda'), AutoGAT(device='cuda')],
        hpo_module=AnnealAdvisorHPO(max_evals=10),
        ensemble_module=Voting(size=2),
        device='cuda'
    )

Where, the argument ``device`` means where to perform the training and searching, by setting to ``auto``, the ``cuda`` is used when it is available.

If you want to disable one module, you can set it to ``None``:

.. code-block:: python

    solver = AutoNodeClassifier(feature_module=None, hpo_module=None, ensemble_module=None)

You can also pass some important arguments of modules directly to solver, which will automatically be set for you:

.. code-block:: python

    solver = AutoNodeClassifier(hpo_module='anneal', max_evals=10)

Refer to :ref:`solver documentation` for more details of argument default value or important argument lists.

Initialize from config dictionary or file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also initialize a solver directly from a config dictionary or file. Currently, the AutoGL solver supports config file type of ``yaml`` or ``json``. You need to use ``from_config()`` when you want to initialize in this way:

.. code-block:: python

    # initialize from config file
    path_to_config = 'your/path/to/config'
    solver = AutoNodeClassifier.from_config(path_to_config)

    # initialize from a dictionary
    config = {
        'models':{'gcn': None, 'gat': None},
        'hpo': {'name': 'tpe', 'max_evals': 10},
        'ensemble': {'name': 'voting', 'size': 2}
    }
    solver = AutoNodeClassifier.from_config(config)

Refer to the config dictionary description :ref:`config` for more details.

Optimization
------------

After initializing a solver, you can optimize it on the given datasets (please refer to :ref:`dataset` and :ref:`dataset documentation` for creating datasets).

You can use ``fit()`` or ``fit_predict()`` to perform optimization, which shares similar argument lists:

.. code-block:: python

    # load your dataset here
    dataset = some_dataset()
    solver.fit(dataset, inplace=True)

The inplace argument is used for saving memory if set to ``True``. It will modify your dataset in an inplace manner during feature engineering.

You can also specify the ``train_split`` and ``val_split`` arguments to let solver auto-split the given dataset. If these arguments are given, the split dataset will be used instead of the default split specified by the dataset provided. All the models will be trained on ``train dataset``. Their hyperparameters will be optimized based on the performance of ``valid dataset``, as well as the final ensemble method. For example:

.. code-block:: python

    # split 0.2 of total nodes/graphs for train and 0.4 of nodes/graphs for validation, 
    # the rest 0.4 is left for test. 
    solver.fit(dataset, train_split=0.2, val_split=0.4)

    # split 600 nodes/graphs for train and 400 nodes/graphs for validation,
    # the rest nodes are left for test.
    solver.fit(dataset, train_split=600, val_split=400)

For the node classification problem, we also support balanced sampling of train and valid: force the number of sampled nodes in different classes to be the same. The balanced mode can be turned on by setting ``balanced=True`` in ``fit()``, which is by default set to ``True``.

.. note:: Solver will maintain the models with the best hyper-parameter of each model architecture you pass to solver (the ``graph_models`` argument when initialized). The maintained models will then be ensembled by ensemble module.

After ``fit()``, solver maintains the performances of every single model and the ensemble model in one leaderboard instance. You can output the performances on valid dataset by:

.. code-block:: python

    # get current leaderboard of the solver
    lb = solver.get_leaderboard()
    # show the leaderboard info
    lb.show()

You can refer to the leaderboard documentation in :ref:`solver documentation` for more usage.

Prediction
----------

After optimized on the given dataset, you can make predictions using the fitted ``solver``.

Prediction using ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the ensemble model constructed by solver to make the prediction, which is recommended and is the default choice:

.. code-block:: python

    solver.predict()

If you do not pass any dataset, the dataset during fitting will be used to give the prediction.

You can also pass the dataset when predicting, please make sure the ``inplaced`` argument is properly set.

.. code-block:: python

    solver.predict(dataset, inplaced=True, inplace=True)

The ``predict()`` function also has ``inplace`` argument, which is the same as the one in ``fit()``. As for the ``inplaced``, it means whether the passed dataset is already modified inplace or not (probably by ``fit()`` function). If you use ``fit()`` before, please make the ``inplaced`` of ``predict()`` stay the same with ``inplace`` in ``fit()``.

Prediction using one single model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also make the prediction using the best single model the solver maintains by:

.. code-block:: python

    solver.predict(use_ensemble=False, use_best=True)

Also, you can name the single model maintained by solver to make predictions.

.. code-block:: python

    solver.predict(use_ensemble=False, use_best=False, name=the_name_of_model)

The names of models can be derived by calling ``solver.trained_models.keys()``, which is the same as the names maintained by the leaderboard of solver.

.. note::

    By default, solver will only make predictions on the ``test`` split of given datasets. Please make sure the passed dataset has the ``test`` split when making predictions. You can also change the default prediction split by setting argument ``mask`` to ``train`` or ``valid``.

Appendix
--------

.. _config:

Config structure
~~~~~~~~~~~~~~~~
The structure of the config file or config is introduced here. The config should be a dict, with five optional keys, namely ``feature``, ``models``, ``trainer``, ``hpo`` and ``ensemble``. You can simply do not add one field if you want to use the default option. The default value of each module is the same as the one in ``__init__()``.

For key ``feature``, ``hpo`` and ``ensemble``, their corresponding values are all dictionaries, which contains one must key ``name`` and other arguments when initializing the corresponding modules. The value of key ``name`` specifies which algorithm should be used, where ``None`` can be passed if you do not want to enable the module. Other arguments are used to initialize the specified algorithm.

For key ``trainer``, you should specify the hyperparameter space of trainer. See :ref:`trainer` or :ref:`train documentation` for the detailed hyperparameter space of different trainers.

For key ``models``, the value is another dictionary with its keys being models that need optimized and the corresponding values being the hyperparameter space of that model. See :ref:`model` or :ref:`model documentation` for the detailed hyperparameter space of different models.

Below shows some examples of the config dictionary.

.. code-block:: python

    config_for_node_classification = {
        'feature': {
            'name': 'deepgl',       # name of auto feature module
            # following are the deepgl specified auto feature engineer arguments
            'fixlen': 100,
            'max_epoch': 5
        },
        'models': {
            'gcn': 
            # specify the hp space of gcn
            [
                {'parameterName': 'num_layers', 'type': 'DISCRETE', 'feasiblePoints': '2,3,4'}, 
                {'parameterName': 'hidden', 'type': 'NUMERICAL_LIST', 'numericalType': 'INTEGER', 'length': 3, 
                    'minValue': [8, 8, 8], 'maxValue': [64, 64, 64], 'scalingType': 'LOG'}, 
                {'parameterName': 'dropout', 'type': 'DOUBLE', 'maxValue': 0.9, 'minValue': 0.1, 'scalingType': 'LINEAR'}, 
                {'parameterName': 'act', 'type': 'CATEGORICAL', 'feasiblePoints': ['leaky_relu', 'relu', 'elu', 'tanh']}
            ],
            'gat': None,             # set to None to use default hp space
            'gin': None
        }
        'trainer': [
            # trainer hp space
            {'parameterName': 'max_epoch', 'type': 'INTEGER', 'maxValue': 300, 'minValue': 10, 'scalingType': 'LINEAR'}, 
            {'parameterName': 'early_stopping_round', 'type': 'INTEGER', 'maxValue': 30, 'minValue': 10, 'scalingType': 'LINEAR'}, 
            {'parameterName': 'lr', 'type': 'DOUBLE', 'maxValue': 0.001, 'minValue': 0.0001, 'scalingType': 'LOG'}, 
            {'parameterName': 'weight_decay', 'type': 'DOUBLE', 'maxValue': 0.005, 'minValue': 0.0005, 'scalingType': 'LOG'}
        ],
        'hpo': {
            'name': 'autone',       # name of hpo module
            # following are the autone specified auto hpo arguments
            'max_evals': 10,
            'subgraphs': 10,
            'sub_evals': 5
        }, 
        'ensemble': {
            'name': 'voting',       # name of ensemble module
            # following are the voting specified auto ensemble arguments
            'size': 2
        }
    }

    config_for_graph_classification = {
        'feature': None,            # set to None to disable this module
        # do not add field `model` to use default settings of solver
        'trainer': [
            # trainer hp space
            {'parameterName': 'max_epoch', 'type': 'INTEGER', 'maxValue': 300, 'minValue': 10, 'scalingType': 'LINEAR'},
            {'parameterName': 'batch_size', 'type': 'INTEGER', 'maxValue': 128, 'minValue': 32, 'scalingType': 'LOG'},
            {'parameterName': 'early_stopping_round', 'type': 'INTEGER', 'maxValue': 30, 'minValue': 10, 'scalingType': 'LINEAR'},
            {'parameterName': 'lr', 'type': 'DOUBLE', 'maxValue': 1e-3, 'minValue': 1e-4, 'scalingType': 'LOG'},
            {'parameterName': 'weight_decay', 'type': 'DOUBLE', 'maxValue': 5e-3, 'minValue': 5e-4, 'scalingType': 'LOG'},
        ],
        'hpo': {
            'name': 'random',       # name of hpo module
            # following are the random specified auto hpo arguments
            'max_evals': 10
        }, 
        'ensemble': None            # set to None to disable this module
    }