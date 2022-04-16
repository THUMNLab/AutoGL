.. _model_cn:

AutoGL 模型
============
在AutoGL中，我们使用 ``model`` 和 ``automodel`` 类定义图神经网络模型，并让它们和超参数优化(hyper parameter optimization, HPO)模块兼容。

当前版本下，我们支持节点分类、图分类和链接预测三种任务任务，支持的具体模型如下：


+----------------------+----------------------------+
|任务                   | 模型                       |
+======================+============================+
|节点分类               | ``gcn``, ``gat``, ``sage`` |
+----------------------+----------------------------+
|图分类                 | ``gin``, ``topk``          |
+----------------------+----------------------------+
|链接预测               | ``gcn``, ``gat``, ``sage`` |
+----------------------+----------------------------+


自定义模型和自动模型
------------------
我们强烈建议您同时定义 ``model`` 类和 ``automodel`` 类。
其中， ``model`` 类来管理参数的初始化与模型前向传播逻辑， ``automodel`` 类组织超参数相关的搜索。
``automodel`` 在 ``solver`` 和 ``trainer`` 模块会被调用。

示例
^^^^
以一个用于节点分类任务的多层感知机(MLP)为例。您可以使用AutoGL来帮您找到最合适的超参数。

首先，您可以定义一个MLP模型，并假设所有超参数已经给定。

.. code-block:: python

    import torch

    class MyMLP(torch.nn.Module):
        # 假定所有超参数可获得
        def __init__(self, args):
            super().__init__()
            in_channels, num_classes = args['in_channels'], args['num_classes']
            layer_num, dim = args['layer_num'], int(args['dim'])

            if layer_num == 1:
                ops = [torch.nn.Linear(in_channels, num_classes)]
            else:
                ops = [torch.nn.Linear(in_channels, dim)]
                for i in range(layer_num - 2):
                    ops.append(torch.nn.Linear(dim, dim))
                ops.append(torch.nn.Linear(dim, num_classes))
        
            self.core = torch.nn.Sequential(*ops)
        
        # 必须利用forward函数定义模型的前向传播逻辑
        def forward(self, data):
            assert hasattr(data, 'x'), 'MLP only support graph data with features'
            x = data.x
            return torch.nn.functional.log_softmax(self.core(x))


接下来，您可以定义自动模型 ``automodel`` 类以更好管理您的超参数。
对于来自于数据集的参数如输入维度与输出维度，可以直接传入 ``automodel`` 类中的初始化函数中 ``__init__()`` 。
而对于需要搜索的其他超参数，需要自定义搜索空间。

.. code-block:: python

    from autogl.module.model import BaseAutoModel
    
    # 定义自动模型类，需要从BaseAutoModel类继承
    class MyAutoMLP(BaseAutoModel):
        def __init__(self, num_features=None, num_classes=None, device=None, **args
        ):
            super().__init__(num_features, num_classes, device, **args)

            # (required) 需要定义搜索空间（包含超参数、超参数的类型以及搜索范围）
            self.space = [
                {'parameterName': 'layer_num', 'type': 'INTEGER', 'minValue': 1, 'maxValue': 5, 'scalingType': 'LINEAR'},
                {'parameterName': 'dim', 'type': 'INTEGER', 'minValue': 64, 'maxValue': 128, 'scalingType': 'LINEAR'}
            ]

            # 设置默认超参数
            self.hyper_parameters = {
                "layer_num": 2,
                "dim": 72,
            }


            # # (required) since we don't know the num_classes and num_features until we see the dataset,
            # # we cannot initialize the models when instantiated. the initialized will be set to False.
            # self.initialized = False


        # (required) instantiate the core MLP model using corresponding hyper-parameters
        def _initialize(self):
            # (required) you need to make sure the core model is named as `self.model`
            self.model = MyMLP({
                "in_channels": self.input_dimension,
                "num_classes": self.output_dimension,
                **self.hyper_parameters
            }
            ).to(self.device)

        

接着，只需要将定义好的自动图模型输入自动图分类任务的 ``solver`` 中，就可以利用它完成节点分类任务。
具体代码示例如下：
.. code-block :: python

    from autogl.solver import AutoNodeClassifier

    solver = AutoNodeClassifier(graph_models=(MyAutoMLP(num_features, num_classes,device=torch.device('cuda')),))



图分类任务的模型定义和整个流程和节点分类任务相似。详情参考图分类模型的tutorial。


用于链接预测任务的模型
^^^^^^^^^^^^^^^^^^^^

对于链接预测任务，模型的定义在 ``forward()`` 函数中略有不同。
为了更好地和链接预测训练器 ``LinkPredictionTrainer`` 与自动链接预测器 ``AutoLinkPredictor`` 交互，您需要定义编码函数 ``lp_encode(self, data)`` 与解码函数 ``lp_decode(self, x, pos_edge_index, neg_edge_index)`` 。

用同样的多层感知机作为示例，如果您想要将其用于链接预测任务，那么您不必再定义 ``forward()`` 函数，而是定义 ``lp_encode(self, data)`` 与 ``lp_decode(self, x, pos_edge_index, neg_edge_index)`` 两个函数。具体代码示例如下：

.. code-block:: python

    class MyMLPForLP(torch.nn.Module):
        def __init__(self, in_channels, layer_num, dim):
            super().__init__()
            ops = [torch.nn.Linear(in_channels, dim)]
            for i in range(layer_num - 1):
                ops.append(torch.nn.Linear(dim, dim))
        
            self.core = torch.nn.Sequential(*ops)

        # (required) 和trainer与solver模块交互
        def lp_encode(self, data):
            return self.core(data.x)

        # (required) 和trainer与solver模块交互
        def lp_decode(self, x, pos_edge_index, neg_edge_index):
            # 首先得到所有需要的正样本边与负样本边集合
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            # 利用点积计算logits，或者使用其他decode方法
            logits = (x[edge_index[0]] * x[edge_index[1]]).sum(dim=-1)
            return logits

    class MyAutoMLPForLP(MyAutoMLP):
        def initialize(self):
            self.model = MyMLPForLP(
                in_channels = self.num_features,
                layer_num = self.layer_num,
                dim = self.dim
            ).to(self.device)



支持采样的模型
^^^^^^^^^^^^^
为了高效地实现大规模图上表示学习，AutoGL目前支持使用节点级别(node-wise)的采样、层级别(layer-wise)的采样和子图级别(subgraph-wise)的采样等采样技术进行节点分类。
有关采样的更多信息，请参阅：:ref:`trainer_cn`。

根据图神经网络中的消息传递机制，一个节点的表达由它多跳邻居构成的子图决定。
但是，节点的邻居数量随着神经网络层数的增加呈现指数级增长，计算并储存所有节点的表达会占用许多的计算资源。
因此，在得到节点表达时，我们可以在每层神经网络输入不同的采样后的子图以达到高效计算的目的。
以torch_geometric的data为例，一个图包含节点特征x和边集合edge_index，在AutoGL的采样技巧中，我们会为data提供edge_indexes属性以表示不同的图卷积层采样出来的不同子图。

.. code-block:: python

    import autogl
    from autogl.module.model import ClassificationSupportedSequentialModel

    # 重新定义接收图作为输入的Linear类
    class Linear(torch.nn.Linear):
        def forward(self, data):
            return super().forward(data.x)

    class MyMLPSampling(ClassificationSupportedSequentialModel):
        def __init__(self, in_channels, num_classes, layer_num, dim):
            super().__init__()
            if layer_num == 1:
                ops = [Linear(in_channels, num_classes)]
            else:
                ops = [Linear(in_channels, dim)]
                for i in range(layer_num - 2):
                    ops.append(Linear(dim, dim))
                ops.append(Linear(dim, num_classes))

            self.core = torch.nn.ModuleList(ops)

        # (required) 覆盖序列编码层sequential_encoding_layers()，和sampling交互
        @property
        def sequential_encoding_layers(self) -> torch.nn.ModuleList:
            return self.core
        
        # (required) define the encode logic of classification for sampling
        def cls_encode(self, data):
            if hasattr(data, 'edge_indexes'):
                # edge_indexes是由edge_index组成的列表，每个edge_index代表每层图卷积所使用的边
                edge_indexes = data.edge_indexes
                edge_weights = [None] * len(self.core) if getattr(data, 'edge_weights', None) is None else data.edge_weights
            else:
                # 默认edge_index和edge_weight是相同的
                edge_indexes = [data.edge_index] * len(self.core)
                edge_weights = [getattr(data, 'edge_weight', None)] * len(self.core)

            x = data.x
            for i in range(len(self.core)):
                data = autogl.data.Data(x=x, edge_index=edge_indexes[i])
                data.edge_weight = edge_weights[i]
                x = self.sequential_encoding_layers[i](data)
            return x

        def cls_decode(self, x):
            return torch.nn.functional.log_softmax(x)

