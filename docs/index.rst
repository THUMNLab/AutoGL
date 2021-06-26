Welcome to AutoGL's documentation!
==================================

AutoGL
------

*Actively under development by @THUMNLab*

AutoGL is developed for researchers and developers to quickly conduct autoML on the graph datasets & tasks. See our documentation for detailed information!

The workflow below shows the overall framework of AutoGL.

.. image:: ../resources/workflow.svg
   :align: center

AutoGL uses ``AutoGL Dataset`` to maintain datasets for graph-based machine learning, which is based on the dataset in PyTorch Geometric with some support added to corporate with the auto solver framework.

Different graph-based machine learning tasks are solved by different ``AutoGL Solvers`` , which make use of four main modules to automatically solve given tasks, namely ``Auto Feature Engineer``, ``Auto Model``, ``HyperParameter Optimization``, and ``Auto Ensemble``. 

Installation
------------

Requirements
~~~~~~~~~~~~

Please make sure you meet the following requirements before installing AutoGL.

1. Python >= 3.6.0

2. PyTorch (>=1.5.1)

    see `PyTorch <https://pytorch.org/>`_ for installation.

3. PyTorch Geometric

    see `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ for installation.

Installation
~~~~~~~~~~~~

Install from pip & conda
^^^^^^^^^^^^^^^^^^^^^^^^

Run the following command to install this package through pip.

.. code-block:: shell

   pip install auto-graph-learning

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

In AutoGL, the tasks are solved by corresponding learners, which in general do the following things:

1. Preprocess and feature engineer the given datasets. This is done by the module named **auto feature engineer**, which can automatically add/delete useful/useless attributes in the given datasets. Some topological features may also be extracted & combined to form stronger features for current tasks.

2. Automatically train and tune popular models specified by users. This is done by modules named **auto model** and **hyperparameter optimization**. In the auto model, several commonly used graph deep models are provided, together with their hyperparameter spaces. These kinds of models can be tuned using **hyperparameter optimization** module to find the best hyperparameter for the current task.

3. Find the best way to ensemble models found and trained in the last step. This is done by the module named **auto ensemble**. The suitable models available are ensembled here to form a more powerful learner.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   docfile/tutorial/t_quickstart
   docfile/tutorial/t_dataset
   docfile/tutorial/t_fe
   docfile/tutorial/t_model
   docfile/tutorial/t_trainer
   docfile/tutorial/t_hpo
   docfile/tutorial/t_nas
   docfile/tutorial/t_ensemble
   docfile/tutorial/t_solver

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   docfile/documentation/data
   docfile/documentation/dataset
   docfile/documentation/module
   docfile/documentation/solver

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`