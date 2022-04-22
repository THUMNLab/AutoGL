.. _nas_cn：

神经架构搜索
============================

我们在不同的搜索空间中支持不同的神经架构搜索算法。
神经架构搜索通常由三个模块构成：搜索空间、搜索策略和评估策略。

搜索空间描述了所有可能要搜索的架构。空间主要由两部分组成：操作（如GCNconv、GATconv）和输入输出关系。
大空间可能有更好的最优架构，但需要更多的算力来探索。
人类知识可以帮助设计合理的搜索空间，减少搜索策略的开销。

搜索策略则控制着如何来探索搜索空间。
它包含了经典的探索-利用困境。
一方面，人们希望快速找到性能良好的架构，
而另一方面，也需要避免过早收敛到次优架构区域。

评估策略在探索时给出架构的性能。
最简单的方法是对数据执行标准的架构训练和验证。
但由于在搜索过程中有很多架构需要评估，因此需要非常有效的评估策略来节省计算资源。

.. image:: ../../../resources/nas.svg
   :align: center

为了更加灵活易用，我们将神经架构搜索过程分解为三部分：算法、空间和评估器，分别对应于三个模块：搜索空间、搜索策略和评估策略。
不同部分的不同模型可以在满足一定条件的时候组合起来。
如果您想设计自己的神经架构搜索流程，您可以根据需要更改其中的任意部分。

用法
-----

您可以通过将算法、空间与评估器直接传递给求解器来在节点分类任务上启用架构搜索。
下面是一个例子：

.. code-block:: python

    # 在cora数据集上使用图神经架构搜索
    from autogl.datasets import build_dataset_from_name
    from autogl.solver import AutoNodeClassifier

    solver = AutoNodeClassifier(
        feature = 'PYGNormalizeFeatures',
        graph_models = (),
        hpo = 'tpe',
        ensemble = None,
        nas_algorithms=['rl'],
        nas_spaces='graphnasmacro',
        nas_estimators=['scratch']
    )

    cora = build_dataset_from_name('cora')
    solver.fit(cora)

上面的代码将先用 ``rl`` 搜索算法在空间 ``GraphnaSmaro`` 中找到最优架构。
然后通过超参数优化方法 ``tpe`` 进一步优化搜索到的架构。

.. note:: ``graph_models`` 参数与神经架构搜索模块不冲突。您可以将 ``graph_models`` 设置为
    神经架构搜索发现的模型外的，其他手工设计的模型。而神经架构搜索模块派生出来的架构
    的作用也与直接通过图模型模块传递的手工设计的模型相同。

搜索空间
------------

空间定义基于开源工具包NNI中使用的可变方式，定义为继承基础空间的模型。
定义搜索空间的方法主要有两种，一种支持单样本方法，而另一种则不支持。
目前，我们支持如下搜索空间：

+------------------------+-----------------------------------------------------------------+
| 空间                   | 描述                                                             |
+========================+=================================================================+
| ``singlepath`` [4]_    | 多个连续层构成的架构，其中每层                                     |
|                        | 只能有一个选择                                                   |
+------------------------+-----------------------------------------------------------------+
| ``graphnas``   [1]_    | 为监督节点分类问题的模型设计的                                     |
|                        | 图神经架构搜索微搜索空间                                          |
+------------------------+-----------------------------------------------------------------+
| ``graphnasmacro`` [1]_ | 为半监督节点分类问题的模型设计的                                   |
|                        | 图神经架构搜索宏搜索空间                                          |
+------------------------+-----------------------------------------------------------------+

您也可以定义自己的神经架构搜索空间。
如果需要支持单样本方式，可以使用函数 ``setLayerChoice`` 与 ``setInputChoice`` 来构建超网络。
下面是一个例子。

.. code-block:: python

    # 创建一个神经架构搜索空间的例子
    from autogl.module.nas.space.base import BaseSpace
    from autogl.module.nas.space.operation import gnn_map
    class YourOneShotSpace(BaseSpace):
        # 在初始化时获取基本参数
        def __init__(self, input_dim = None, output_dim = None):
            super().__init__()
            # 必须在空间中设定输入维度与输出维度，也可以在函数 `instantiate` 中初始化这两个参数`
            self.input_dim = input_dim
            self.output_dim = output_dim

        # 实例化超网络
        def instantiate(self, input_dim = None, output_dim = None):
            # 必须在函数中调用父类的实例化
            super().instantiate()
            self.input_dim = input_dim or self.input_dim
            self.output_dim = output_dim or self.output_dim
            # 按照顺序定义两层网络
            setattr(self, 'layer0', self.setLayerChoice(0, [gnn_map(op,self.input_dim,self.output_dim)for op in ['gcn', 'gat']], key = 'layer0')
            setattr(self, 'layer1', self.setLayerChoice(1, [gnn_map(op,self.input_dim,self.output_dim)for op in ['gcn', 'gat']], key = 'layer1')
            # 定义一个从两层的结果中选择的输入选项
            setattr(self, 'input_layer', self.setInputChoice(2, choose_from = ['layer0', 'layer1'], n_chosen = 1, returen_mask = False, key = 'input_layer'))
            self._initialized = True

        # 定义前向传播过程
        def forward(self, data):
            x, edges = data.x, data.edge_index
            x_0 = self.layer0(x, edges)
            x_1 = self.layer1(x, edges)
            y = self.input_layer([x_0, x_1])
            y = F.log_fostmax(y, dim = 1)
            return y

        # 对于单样本范式，您可以使用如 ``parse_model`` 函数中的方法
        def parse_model(self, selection, device) -> BaseModel:
            return self.wrap().fix(selection)

您也可以使用不支持单样本范式的方式。
这样的话，您可以直接复制模型，并进行少量更改。
但相应的，您也只能使用基于样本的搜索策略。

.. code-block:: python

    # 创建一个神经架构搜索空间的例子
    from autogl.module.nas.space.base import BaseSpace, map_nn
    from autogl.module.nas.space.operation import gnn_map
    # 在这里，我们以 `head` 作为参数，在三种图卷积上进行搜索
    # 在搜索 `heads` 时，我们在搜索图卷积
    from torch_geometric.nn import GATConv, FeaStConv, TransformerConv
    class YourNonOneShotSpace(BaseSpace):
        # 在初始化时获取基本参数
        def __init__(self, input_dim = None, output_dim = None):
            super().__init__()
            # 必须在空间中设定输入维度与输出维度，也可以在函数 `instantiate` 中初始化这两个参数`
            self.input_dim = input_dim
            self.output_dim = output_dim

        # 实例化超网络
        def instantiate(self, input_dim, output_dim):
            # 必须在函数中调用父类的实例化
            super().instantiate()
            self.input_dim = input_dim or self.input_dim
            self.output_dim = output_dim or self.output_dim
            # 设置你每一层的选择
            self.choice0 = self.setLayerChoice(0, map_nn(["gat", "feast", "transformer"]), key="conv")
            self.choice1 = self.setLayerChoice(1, map_nn([1, 2, 4, 8]), key="head")

        # 不要忘记在这里定义前向传播过程
        # 对于非单样本范式，您可以直接返回选择下的模型
        # ``YourModel`` 也就是您的模型必须继承基础空间
        def parse_model(self, selection, device) -> BaseModel:
            model = YourModel(selection, self.input_dim, self.output_dim).wrap()
            return model

.. code-block:: python

    # ``YourModel`` 也就是您的模型定义如下
    class YourModel(BaseSpace):
        def __init__(self, selection, input_dim, output_dim):
            self.input_dim = input_dim
            self.output_dim = output_dim
            if selection["conv"] == "gat":
                conv = GATConv
            elif selection["conv"] == "feast":
                conv = FeaStConv
            elif selection["conv"] == "transformer":
                conv = TransformerConv
            self.layer = conv(input_dim, output_dim, selection["head"])

        def forward(self, data):
            x, edges = data.x, data.edge_index
            y = self.layer(x, edges)
            return y

性能评估器
---------------------

性能评估器用于评估一个架构的优劣. 目前我们支持如下一些评估器:

+-------------------------+-------------------------------------------------------+
| 评估器                   | 描述                                                   |
+=========================+=======================================================+
| ``oneshot``             | 对于给定的子架构，无需训练地直接评估其效果                   |
+-------------------------+-------------------------------------------------------+
| ``scratch``             | 对于给定的子架构，从头开始训练直到收敛之后再评估其效果         |
+-------------------------+-------------------------------------------------------+

您也可以自己定义一个评估器. 下面是一个无需训练即可评估架构效果的评估器的例子 (通常应用于one-shot space).

.. code-block:: python

    # 例如，您也可以自己定义一个estimator
    from autogl.module.nas.estimator.base import BaseEstimator
    class YourOneShotEstimator(BaseEstimator):
        # 您所需要做的只是定义``infer``这个方法
        def infer(self, model: BaseSpace, dataset, mask="train"):
            device = next(model.parameters()).device
            dset = dataset[0].to(device)
            # 对架构直接进行前向传播
            pred = model(dset)[getattr(dset, f"{mask}_mask")]
            y = dset.y[getattr(dset, f'{mask}_mask')]
            # 使用默认的损失函数和评价指标来评估架构效果，当然，在这里您也可以选择其他的损失函数和评价指标
            loss = getattr(F, self.loss_f)(pred, y)
            probs = F.softmax(pred, dim = 1)
            metrics = [eva.evaluate(probs, y) for eva in self.evaluation]
            return metrics, loss

搜索策略
---------------

搜索策略定义了如何去搜索一个好的子架构. 目前我们支持如下一些搜索策略:

+-------------------------+-------------------------------------------------------+
| 策略                     | 描述                                                  |
+=========================+=======================================================+
| ``random``              | 通过均匀采样进行随机搜索                                  |
+-------------------------+-------------------------------------------------------+
| ``rl`` [1]_             | 通过强化学习方法来进行架构搜索                             |
+-------------------------+-------------------------------------------------------+
| ``enas`` [2]_           | 通过共享参数等方法，更高效地进行架构搜索                     |
+-------------------------+-------------------------------------------------------+
| ``darts`` [3]_          | 通过可微方法来进行架构搜索                                |
+-------------------------+-------------------------------------------------------+

基于采样的非共享权重的搜索策略在实现上更加简单
接下来，我们将向您展示如何自定义一个基于DFS的搜索策略来作为一个例子
如果您想要自定义更多复杂的搜索策略，您可以去参考NNI中Darts、Enas或者其他搜索策略的实现

.. code-block:: python

    from autogl.module.nas.algorithm.base import BaseNAS
    class RandomSearch(BaseNAS):
        # 接收需要采样的数量作为初始化
        def __init__(self, n_sample):
            super().__init__()
            self.n_sample = n_sample

        # NAS算法流程中的关键步骤，这个方法会根据给定的search space、dataset和estimator去搜索一个合理的架构
        def search(self, space: BaseSpace, dset, estimator):
            self.estimator=estimator
            self.dataset=dset
            self.space=space
                
            self.nas_modules = []
            k2o = get_module_order(self.space)
            # 寻找并存储search space中所有的mutables，这些mutables就是您在search space中定义的可搜索的部分
            replace_layer_choice(self.space, PathSamplingLayerChoice, self.nas_modules)
            replace_input_choice(self.space, PathSamplingInputChoice, self.nas_modules)
            # 根据给定的orders对mutables进行排序
            self.nas_modules = sort_replaced_module(k2o, self.nas_modules) 
            # 得到包含所有可能选择的一个字典
            selection_range={}
            for k,v in self.nas_modules:
                selection_range[k]=len(v)
            self.selection_dict=selection_range
                
            arch_perfs=[]
            # 定义DFS的流程
            self.selection = {}
            last_k = list(self.selection_dict.keys())[-1]
            def dfs():
                for k,v in self.selection_dict.items():
                    if not k in self.selection:
                        for i in range(v):
                            self.selection[k] = i
                            if k == last_k:
                                # 评估一个架构的效果
                                self.arch=space.parse_model(self.selection,self.device)
                                metric,loss=self._infer(mask='val')
                                arch_perfs.append([metric, self.selection.copy()])
                            else:
                                dfs()
                        del self.selection[k]
                        break
            dfs()

            # 得到在搜索过程中拥有最好效果的架构
            selection=arch_perfs[np.argmax([x[0] for x in arch_perfs])][1]
            arch=space.parse_model(selection,self.device)
            return arch 

不同的搜索策略需要与特定的搜索空间与评估器搭配使用
这与它们的实现相关，如非one-shot的搜索空间不能与one-shot的搜索策略搭配使用
下面的表格中给出了我们目前所支持的搭配组合

+----------------+-------------+-------------+------------------+
| Space          | single path | GraphNAS[1] | GraphNAS-macro[1]|
+================+=============+=============+==================+
| Random         |  ✓          |  ✓          |  ✓               | 
+----------------+-------------+-------------+------------------+
| RL             |  ✓          |  ✓          |  ✓               |
+----------------+-------------+-------------+------------------+
| GraphNAS [1]_  |  ✓          |  ✓          |  ✓               |
+----------------+-------------+-------------+------------------+
| ENAS [2]_      |  ✓          |             |                  |
+----------------+-------------+-------------+------------------+
| DARTS [3]_     |  ✓          |             |                  |
+----------------+-------------+-------------+------------------+

+----------------+-------------+-------------+
| Estimator      | one-shot    | Train       |
+================+=============+=============+
| Random         |             |  ✓          | 
+----------------+-------------+-------------+
| RL             |             |  ✓          |
+----------------+-------------+-------------+
| GraphNAS [1]_  |             |  ✓          |
+----------------+-------------+-------------+
| ENAS [2]_      |  ✓          |             |
+----------------+-------------+-------------+
| DARTS [3]_     |  ✓          |             |
+----------------+-------------+-------------+

.. [1] Gao, Yang, et al. "Graph neural architecture search." IJCAI. Vol. 20. 2020.
.. [2] Pham, Hieu, et al. "Efficient neural architecture search via parameters sharing." International Conference on Machine Learning. PMLR, 2018.
.. [3] Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "DARTS: Differentiable Architecture Search." International Conference on Learning Representations. 2018.
.. [4] Guo, Zichao, et al. “Single Path One-Shot Neural Architecture Search with Uniform Sampling.” European Conference on Computer Vision, 2019, pp. 544–560.