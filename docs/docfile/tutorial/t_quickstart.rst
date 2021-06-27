Quick Start
===========

This tutorial will help you quickly go through the concepts and usages of important classes in AutoGL. In this tutorial, you will conduct a quick auto graph learning on dataset `Cora`_.

.. _Cora: https://graphsandnetworks.com/the-cora-dataset/

AutoGL Learning
---------------

Based on the concept of autoML, auto graph learning aims at **automatically** solve tasks with data represented by graphs. Unlike conventional learning frameworks, auto graph learning, like autoML, does not need humans inside the experiment loop. You only need to provide the datasets and tasks to the AutoGL solver. This framework will automatically find suitable methods and hyperparameters for you.

The diagram below describes the workflow of AutoGL framework.

To reach the aim of autoML, our proposed auto graph learning framework is organized as follows. We have ``dataset`` to maintain the graph datasets given by users. A ``solver`` object needs to be built for specifying the target tasks. Inside ``solver``, there are five submodules to help complete the auto graph tasks, namely ``auto feature engineer``, ``auto model``, ``neural architecture search``, ``hyperparameter optimization`` and ``auto ensemble``, which will automatically preprocess/enhance your data, choose and optimize deep models and ensemble them in the best way for you.

Let's say you want to conduct an auto graph learning on dataset ``Cora``. First, you can easily get the ``Cora`` dataset using the ``dataset`` module:

.. code-block:: python

    from autogl.datasets import build_dataset_from_name
    cora_dataset = build_dataset_from_name('cora')

The dataset will be automatically downloaded for you. Please refer to :ref:`dataset` or :ref:`dataset documentation` for more details of dataset constructions, available datasets, add local datasets, etc.

After deriving the dataset, you can build a ``node classification solver`` to handle auto training process:

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

In this way, we build a ``node classification solver``, which will use ``deepgl`` as its feature engineer, and use ``anneal`` hyperparameter optimizer to optimize the given three models ``['gcn','gat']``. The derived models will then be ensembled using ``voting`` ensembler. Please refer to the corresponding tutorials or documentation to see the definition and usages of available submodules.

Then, you can fit the solver and then check the leaderboard:

.. code-block:: python

    solver.fit(cora_dataset, time_limit=3600)
    solver.get_leaderboard().show()

The ``time_limit`` is set to 3600 so that the whole auto graph process will not exceed 1 hour.  ``solver.show()`` will present the models maintained by ``solver``, with their performances on the validation dataset.

Then, you can make the predictions and evaluate the results using the evaluation functions provided:

.. code-block:: python

    from autogl.module.train import Acc
    predicted = solver.predict_proba()
    print('Test accuracy: ', Acc.evaluate(predicted, 
        cora_dataset.data.y[cora_dataset.data.test_mask].cpu().numpy()))

.. note:: You don't need to pass the ``cora_dataset`` again when predicting, since the dataset is **remembered** by the ``solver`` and will be reused when no dataset is passed at predicting. However, you can also pass a new dataset when predicting, and the new dataset will be used instead of the remembered one. Please refer to :ref:`solver` or :ref:`solver documentation` for more details.
