.. _solver_cn:

AutoGL Solver
=============

AutoGL 项目用 ``solver`` 类来控制整个自动机器学习流程。目前，我们已经支持了一下任务：

* ``AutoNodeClassifier`` for semi-supervised node classification
* ``AutoGraphClassifier`` for supervised graph classification
* ``AutoLinkPredictor`` for link prediction

初始化
--------------

一个 solver 可以通过 ``__init__()`` 来进行初始化，也可以通过一个配置字典或文件初始化.

通过 ``__init__()`` 初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果你想通过 ``__init__()`` 初始化一个 solver, 你需要向其传入关键的模块。你可以选择传入相关模块的关键词，也可以传入初始化后的实例：

.. code-block:: python

    from autogl.solver import AutoNodeClassifier
    
    # 1. 通过关键词初始化
    solver = AutoNodeClassifier(
        feature_module='deepgl', 
        graph_models=['gat','gcn'], 
        hpo_module='anneal', 
        ensemble_module='voting',
        device='auto'
    )

    # 2. 使用实例初始化
    from autogl.module import AutoFeatureEngineer, AutoGCN, AutoGAT, AnnealAdvisorHPO, Voting
    solver = AutoNodeClassifier(
        feature_module=AutoFeatureEngineer(),
        graph_models=[AutoGCN(device='cuda'), AutoGAT(device='cuda')],
        hpo_module=AnnealAdvisorHPO(max_evals=10),
        ensemble_module=Voting(size=2),
        device='cuda'
    )

这里，参数 ``device`` 表示进行训练和搜索的平台。如果设置为 ``auto``，则会在 ``cuda`` 可以使用的时候使用 ``cuda`` 。

如果你想禁用某个模块，你可以将其设为 ``None``:

.. code-block:: python

    solver = AutoNodeClassifier(feature_module=None, hpo_module=None, ensemble_module=None)

你也可以直接向 solver 传入一些模块的重要参数，它们将会自动为你设置模块：

.. code-block:: python

    solver = AutoNodeClassifier(hpo_module='anneal', max_evals=10)

参考 :ref:`solver documentation` 以获得关于默认参数值和重要参数列表的更多细节。

通过配置字典或文件进行初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你也可以直接通过一个配置字典或文件进行初始化。目前，solver 支持 ``yaml`` 或者 ``json`` 类型的文件。如果你想通过这种方式初始化，你需要使用 ``from_config()``：  

.. code-block:: python

    # 通过配置文件初始化
    path_to_config = 'your/path/to/config'
    solver = AutoNodeClassifier.from_config(path_to_config)

    # 通过字典初始化
    config = {
        'models':{'gcn': None, 'gat': None},
        'hpo': {'name': 'tpe', 'max_evals': 10},
        'ensemble': {'name': 'voting', 'size': 2}
    }
    solver = AutoNodeClassifier.from_config(config)

参考配置字典的描述 :ref:`config` 以获取更多细节。 

优化
------------

初始化 solver 之后，你可以在给定的数据集上进行优化（请参考 :ref:`dataset_cn` 和 :ref:`dataset documentation` 以创建数据集）。

你可以使用 ``fit()`` 或 ``fit_predict()`` 来进行优化，它们有相似的参数列表：

.. code-block:: python

    # 加载数据集
    dataset = some_dataset()
    solver.fit(dataset, inplace=True)


如果设置 “inplace” 参数为真，它将会在特征工程步骤中把你的数据集设置为原地替换的模式以节省空间。

你也可以指定 ``train_split`` 和 ``val_split`` 参数来使 solver 自动分割给定的数据集。如果给定了这些参数，将会使用自动分割的数据集而不是数据集的默认分割。所有的模型都会在 ``train dataset`` 上进行训练。它们的超参数将会根据在 ``valid dataset`` 上的表现进行优化，包括最后的模型集成方式。例如

.. code-block:: python

    .. # 分割 20% 的节点/图用于训练，40% 的节点/图用于验证 
    # 剩余 40% 用于测试
    solver.fit(dataset, train_split=0.2, val_split=0.4)

    # 分割 600 个节点/图用于训练，400 个节点/图用于验证 
    # 剩余的用于测试
    solver.fit(dataset, train_split=600, val_split=400)

对于点分类问题，我们同样支持对训练和测试集的平衡采样：强制不同类别的节点数量相同。这种平衡模式可以通过在 ``fit()`` 使 ``balanced=True`` 来进行设置，而其默认值也是 ``True``。

.. note:: Solver 会维护每个你传入的模型（初始化时的 ``graph_models``）的最好的超参数。这些模型将会在集成模块中进行集成。

``fit()`` 操作之后，solver 在一个榜单中维护每个单独的模型以及集成模型的性能。你可以通过一下代码输出验证集的性能：

.. code-block:: python

    # 获取当前榜单
    lb = solver.get_leaderboard()
    # 展示榜单信息
    lb.show()

你可以参考榜单的文档 :ref:`solver documentation` 以获取更多使用细节。

预测
----------

在给定的数据集常优化之后，你可以通过 ``solver`` 来进行预测。

使用集成模型预测
~~~~~~~~~~~~~~~~~~~~~~~~~

你可以使用 slover 生成的集成模型来进行预测，这也是默认的选项，我们也推荐这样做：

.. code-block:: python

    solver.predict()

如果你没有传入任何数据集，那么用于拟合的数据集将会被用于预测。

你也可以在预测时传入数据集，请确认已经合理地设置了 ``inplaced`` 参数。

.. code-block:: python

    solver.predict(dataset, inplaced=True, inplace=True)

``predict()`` 函数也有 ``inplace`` 参数，这与在 ``fit()`` 中是一样的。至于 ``inplaced``，意味着无论传入的数据集是否被修改过（也许被 ``fit()`` 函数）。如果你之前使用过 ``fit()``，请确认 ``predict()`` 和 ``fit()`` 中的 ``inplaced`` 参数值是相同的。

使用单个模型预测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你也可以使用 solver 维护的最好的单个模型来进行预测：

.. code-block:: python

    solver.predict(use_ensemble=False, use_best=True)

同样的，你可以为 solver 维护的单个模型命名。

.. code-block:: python

    solver.predict(use_ensemble=False, use_best=False, name=the_name_of_model)

模型的名字可以通过 ``solver.trained_models.keys()`` 来进行调用，这与 solver 榜单中维护的名字是类似的。

.. note::
    默认地，solver 只会在数据集的 ``test`` 部分进行预测。请确认传入的数据集在预测时有 ``test`` 部分。你也可以通过将 ``mask`` 设置为 ``train`` 或者 ``valid`` 来改变默认预测对象。

附录
--------

.. _config:

配置文件格式
~~~~~~~~~~~~~~~~
这里介绍了配置文件的结构。配置是一个有五个键的字典，分别是 ``feature`` ，``models``，``trainer``，``hpo`` 和 ``ensemble``。如果使用默认配置，你可以你添加其中的某些键。模块的默认参数与 ``__init__()`` 中的值相同。

对于 ``feature``，``hpo`` 和 ``ensemble``，它们对应的值都是字典，里面至少有一个键是 ``name``， 其它的参数则用于初始化对应模型。``name`` 指定了所使用的的算法，如果你不想使用某个模块，你可以传入 ``None``。

对于 ``trainer``，你需要制指定它的超参数空间。请参考 :ref:`trainer_cn` 或者 :ref:`train documentation` 来获取不同 trainer 的详细超参数空间信息。

对于 ``models``，其值是另一个字典，它的键是需要优化的模型，值是对应模型的超参数空间。参考 :ref:`model_cn` 或者 :ref:`model documentation` 来获取不同模型的详细超参数信息。

下面展示了配置所需字典的一个例子。

.. code-block:: python

    config_for_node_classification = {
        'feature': {
            'name': 'deepgl',       # 自动特征工程模块的名字
            # 下面是 deepgl 特征工程模块的专有参数
            'fixlen': 100,
            'max_epoch': 5
        },
        'models': {
            'gcn': 
            # 指定 gcn 的超参数空间
            [
                {'parameterName': 'num_layers', 'type': 'DISCRETE', 'feasiblePoints': '2,3,4'}, 
                {'parameterName': 'hidden', 'type': 'NUMERICAL_LIST', 'numericalType': 'INTEGER', 'length': 3, 
                    'minValue': [8, 8, 8], 'maxValue': [64, 64, 64], 'scalingType': 'LOG'}, 
                {'parameterName': 'dropout', 'type': 'DOUBLE', 'maxValue': 0.9, 'minValue': 0.1, 'scalingType': 'LINEAR'}, 
                {'parameterName': 'act', 'type': 'CATEGORICAL', 'feasiblePoints': ['leaky_relu', 'relu', 'elu', 'tanh']}
            ],
            'gat': None,             # 设置为空则使用默认的超参数空间
            'gin': None
        }
        'trainer': [
            # trainer 超参数空间
            {'parameterName': 'max_epoch', 'type': 'INTEGER', 'maxValue': 300, 'minValue': 10, 'scalingType': 'LINEAR'}, 
            {'parameterName': 'early_stopping_round', 'type': 'INTEGER', 'maxValue': 30, 'minValue': 10, 'scalingType': 'LINEAR'}, 
            {'parameterName': 'lr', 'type': 'DOUBLE', 'maxValue': 0.001, 'minValue': 0.0001, 'scalingType': 'LOG'}, 
            {'parameterName': 'weight_decay', 'type': 'DOUBLE', 'maxValue': 0.005, 'minValue': 0.0005, 'scalingType': 'LOG'}
        ],
        'hpo': {
            'name': 'autone',       # 超参数优化模块的名字
            # 下面是 autone 超参数优化模块的专有参数
            'max_evals': 10,
            'subgraphs': 10,
            'sub_evals': 5
        }, 
        'ensemble': {
            'name': 'voting',       # 集成模块的名字
            # 下面是 voting 集成模块的专有参数
            'size': 2
        }
    }

    config_for_graph_classification = {
        'feature': None,            # 设置为空会禁用该模块
        # 不添加 `model` 域以使用默认设置
        'trainer': [
            # trainer 超参数空间
            {'parameterName': 'max_epoch', 'type': 'INTEGER', 'maxValue': 300, 'minValue': 10, 'scalingType': 'LINEAR'},
            {'parameterName': 'batch_size', 'type': 'INTEGER', 'maxValue': 128, 'minValue': 32, 'scalingType': 'LOG'},
            {'parameterName': 'early_stopping_round', 'type': 'INTEGER', 'maxValue': 30, 'minValue': 10, 'scalingType': 'LINEAR'},
            {'parameterName': 'lr', 'type': 'DOUBLE', 'maxValue': 1e-3, 'minValue': 1e-4, 'scalingType': 'LOG'},
            {'parameterName': 'weight_decay', 'type': 'DOUBLE', 'maxValue': 5e-3, 'minValue': 5e-4, 'scalingType': 'LOG'},
        ],
        'hpo': {
            'name': 'random',       # 超参数优化模块的名字
            # 下面是 random 超参数优化模块的专有参数
            'max_evals': 10
        }, 
        'ensemble': None            # 设置为空以禁用该模块
    }