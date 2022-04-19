.. _trainer_cn:

AutoGL训练器
==============

AutoGL项目使用``训练器``来处理任务的自动训练。目前，我们支持以下任务：

* ``NodeClassificationTrainer`` 用于半监督的节点分类
* ``GraphClassificationTrainer`` 用于有监督的图分类
* ``LinkPredictionTrainer`` 用于链接预测


简单初始化
-------------------
类似于模型，我们也对所有训练器使用简单的初始化。当调用`__init__()'时，只有（部分）超参数会被设置。只有在明确调用`initialize()`后，`训练器'才会有其核心`模型'，在所有的超参数被正确设置后，这将在`solver`和`duplicate_from_hyper_parameter()`中自动完成，


训练和预测
-----------------
初始化训练器后，你可以在给定的数据集上训练它。

到目前为止，我们已经给出了节点分类、图分类和链接预测等任务的训练和测试函数。你也可以按照与我们相似的模式创建你的任务。对于训练，你需要定义``train_only()``并在``train()``中使用。对于测试，你需要定义`predict_proba()`并在`predict()`中使用它。

评价函数在``evaluate()``中定义，你可以使用你的评价指标和方法。


基于采样而进行的节点分类
---------------------------------
根据目前的各种研究，用空间采样训练已被证明作为一种有效的技术在大规模图上进行表示学习。
我们提供了各种代表性的采样机制的实现，包括邻居采样、依赖层重要性采样（LADIES）和GraphSAINT。
利用各种有效的采样机制，用户可以在大规模的图数据集上利用这个库，例如Reddit。

具体来说，由于各种采样技术通常需要模型支持在转发中的一些分层处理，现在只有提供的GCN和GraphSAGE模型可以用于节点取样（邻居取样）和层级取样（LADIES）。
更多的模型和任务计划在未来的版本中支持采样。


* 节点采样（GraphSAGE)
    支持 ``GCN ``和 ``GraphSAGE``两种模型。

* 分层取样（依赖层的重要性取样）
    当前版本只支持``GCN``模型。

* 子图式采样（GraphSAINT）
    由于GraphSAINT采样技术对采用的模型没有具体要求。大多数可用的模型对于采用GraphSAINT技术是可行的。
    然而，在GraphSAINT技术实际应用时，预测过程是一个潜在的瓶颈，甚至是障碍。
    当GraphSAINT技术实际应用于大规模图时，预测过程是一个潜在的瓶颈甚至障碍。
    因此，要采用的模型最好是支持分层预测的。
    而我们提供的``GCN ``模型已经满足了这一强化要求。
    根据实验GraphSAINT的实现现在有了杠杆作用，可以支持小于*Flickr*规模的整体图数据。

采样技术可以通过采用相应的训练器来利用
``NodeClassificationGraphSAINTTrainer``,
``NodeClassificationLayerDependentImportanceSamplingTrainer``,
和 ``NodeClassificationNeighborSamplingTrainer``.
你可以在YAML配置文件中指定训练器的相应名称
或者用特定训练器的实例来实例化解算器`AutoNodeClassifier`。
的实例。然而，请确保正确管理一些关键的
超参数空间内的一些关键超参数。具体来说：


对于  ``NodeClassificationLayerDependentImportanceSamplingTrainer``, 你需要设置
超参数 ``sampled_node_sizes``。 ``sampled_node_sizes``的空间应该是一个和**Sequential Model**大小相同的列表. 例如，如果你有一个
层数为4的模型，你需要正确传递超参数空间：

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

如果你的模型的层数是一个可搜索的超参数，你也可以设置 ``cutPara``
和 ``cutFunc`` ， 使其与你的模型的层数超参数相联系。

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


类似地, 如果你想使用 ``NodeClassificationNeighborSamplingTrainer``, 你需要保证超参 ``sampling_sizes`` 和模型层数相同，例如:

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


你也可以直接在模型列表中传递一个训练器。一个简单的例子演示如下：

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