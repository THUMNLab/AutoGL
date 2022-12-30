Welcome to AutoGL's documentation!
==================================

AutoGL
------

*Actively under development by @THUMNLab*

AutoGL is developed for researchers and developers to quickly conduct autoML on the graph datasets & tasks.

The workflow below shows the overall framework of AutoGL.

.. image:: ../resources/workflow.svg
   :align: center

AutoGL uses ``AutoGL Dataset`` to maintain datasets for graph-based machine learning, which is based on the dataset in PyTorch Geometric or Deep Graph Library with some support added to corporate with the auto solver framework.

Different graph-based machine learning tasks are solved by different ``AutoGL Solvers`` , which make use of four main modules to automatically solve given tasks, namely ``Auto Feature Engineer``, ``Auto Model``, ``Neural Architecture Search``, ``HyperParameter Optimization``, and ``Auto Ensemble``. 

Installation
------------

Requirements
~~~~~~~~~~~~

Please make sure you meet the following requirements before installing AutoGL.

1. Python >= 3.6.0

2. PyTorch (>=1.6.0)

    see `PyTorch <https://pytorch.org/>`_ for installation.

3. Graph Library Backend

    You will need either PyTorch Geometric (PyG) or Deep Graph Library (DGL) as the backend.

3.1 PyTorch Geometric (>=1.7.0)

    see <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html> for installation.

3.2 Deep Graph Library (>=0.7.0)

    see <https://dgl.ai> for installation.

Installation
~~~~~~~~~~~~

Install from pip & conda
^^^^^^^^^^^^^^^^^^^^^^^^

Run the following command to install this package through pip.

.. code-block:: shell

   pip install autogl

Install from source
^^^^^^^^^^^^^^^^^^^

Run the following command to install this package from the source.

.. code-block:: shell

   git clone https://github.com/THUMNLab/AutoGL.git
   cd AutoGL
   python setup.py install

Install for development
^^^^^^^^^^^^^^^^^^^^^^^

If you are a developer of the AutoGL project, please use the following command to create a soft link, then you can modify the local package without installation again.

.. code-block:: shell

   pip install -e .


Modules
-------

In AutoGL, the tasks are solved by corresponding solvers, which in general do the following things:

1. Preprocess and feature engineer the given datasets. This is done by the module named **auto feature engineer**, which can automatically add/delete useful/useless attributes in the given datasets. Some topological features may also be extracted & combined to form stronger features for current tasks.

2. Find the best suitable model architectures through neural architecture search. This is done by modules named **nas**. AutoGL provides several search spaces, algorithms and estimators for finding the best architectures.

2. Automatically train and tune popular models specified by users. This is done by modules named **auto model** and **hyperparameter optimization**. In the auto model, several commonly used graph deep models are provided, together with their hyperparameter spaces. These kinds of models can be tuned using **hyperparameter optimization** module to find the best hyperparameter for the current task.

3. Find the best way to ensemble models found and trained in the last step. This is done by the module named **auto ensemble**. The suitable models available are ensembled here to form a more powerful learner.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   docfile/tutorial/t_quickstart
   docfile/tutorial/t_hetero_node_clf
   docfile/tutorial/t_homo_graph_classification_gin
   docfile/tutorial/t_backend
   docfile/tutorial/t_dataset
   docfile/tutorial/t_fe
   docfile/tutorial/t_model
   docfile/tutorial/t_trainer
   docfile/tutorial/t_ssl_trainer
   docfile/tutorial/t_hpo
   docfile/tutorial/t_nas
   docfile/tutorial/t_nas_bench_graph
   docfile/tutorial/t_robust
   docfile/tutorial/t_ensemble
   docfile/tutorial/t_solver

.. toctree::
   :maxdepth: 2
   :caption: 中文教程

   docfile/tutorial_cn/t_quickstart
   docfile/tutorial_cn/t_hetero_node_clf
   docfile/tutorial_cn/t_homo_graph_classification_gin
   docfile/tutorial_cn/t_backend
   docfile/tutorial_cn/t_dataset
   docfile/tutorial_cn/t_fe
   docfile/tutorial_cn/t_model
   docfile/tutorial_cn/t_trainer
   docfile/tutorial_cn/t_ssl_trainer
   docfile/tutorial_cn/t_hpo
   docfile/tutorial_cn/t_nas
   docfile/tutorial_cn/t_nas_bench_graph
   docfile/tutorial_cn/t_robust
   docfile/tutorial_cn/t_ensemble
   docfile/tutorial_cn/t_solver

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   docfile/documentation/data   
   docfile/documentation/dataset
   docfile/documentation/feature      
   docfile/documentation/model
   docfile/documentation/train
   docfile/documentation/hpo
   docfile/documentation/nas
   docfile/documentation/ensemble
   docfile/documentation/solver

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`