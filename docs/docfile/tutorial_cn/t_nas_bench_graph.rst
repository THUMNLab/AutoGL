.. _nas_bench_graph:

使用 NAS-Bench-Graph
============================

我们支持使用 NAS-Bench-Graph[1] 进行神经网络架构搜索。你可以从 NAS-Bench-Graph 中直接获得架构的性能评估，而不用去从头训练模型。
样例代码在 `nas_bench_graph_example.py` 中。

构建搜索空间
------------

要使用 NAS-Bench-Graph，你需要先按照论文中的方式定义搜索空间。
一般地，你可以无需任何改动，直接使用以下代码来定义搜索空间。

.. code-block:: python

    class StrModule(nn.Module):
        def __init__(self, lambd):
            super().__init__()
            self.name = lambd

        def forward(self, *args, **kwargs):
            return self.name

        def __repr__(self):
            return "{}({})".format(self.__class__.__name__, self.name)

    class BenchSpace(BaseSpace):
        def __init__(
            self,
            hidden_dim: _typ.Optional[int] = 64,
            layer_number: _typ.Optional[int] = 2,
            dropout: _typ.Optional[float] = 0.9,
            input_dim: _typ.Optional[int] = None,
            output_dim: _typ.Optional[int] = None,
            ops_type = 0
        ):
            super().__init__()
            self.layer_number = layer_number
            self.hidden_dim = hidden_dim
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.dropout = dropout
            self.ops_type=ops_type

        def instantiate(
            self,
            hidden_dim: _typ.Optional[int] = None,
            layer_number: _typ.Optional[int] = None,
            dropout: _typ.Optional[float] = None,
            input_dim: _typ.Optional[int] = None,
            output_dim: _typ.Optional[int] = None,
            ops_type=None
        ):
            super().instantiate()
            self.dropout = dropout or self.dropout
            self.hidden_dim = hidden_dim or self.hidden_dim
            self.layer_number = layer_number or self.layer_number
            self.input_dim = input_dim or self.input_dim
            self.output_dim = output_dim or self.output_dim
            self.ops_type = ops_type or self.ops_type
            self.ops = [gnn_list,gnn_list_proteins][self.ops_type]
            for layer in range(4):
                setattr(self,f"in{layer}",self.setInputChoice(layer,n_candidates=layer+1,n_chosen=1,return_mask=False,key=f"in{layer}"))
                setattr(self,f"op{layer}",self.setLayerChoice(layer,list(map(lambda x:StrModule(x),self.ops)),key=f"op{layer}"))
            self.dummy=nn.Linear(1,1)

        def forward(self, bench):
            lks = [getattr(self, "in" + str(i)).selected for i in range(4)]
            ops = [getattr(self, "op" + str(i)).name for i in range(4)]
            arch = Arch(lks, ops)
            h = arch.valid_hash()
            if h == "88888" or h==88888:
                return 0
            return bench[h]['perf']

        def parse_model(self, selection, device) -> BaseAutoModel:
            return self.wrap().fix(selection)

定义基准集评估器
--------------------

接下来你需要定义一个新的评估器来直接从 NAS-Bench-Graph 中获取架构在目标数据集上的性能。
你也可以直接使用以下代码来定义基准集评估器。

.. code-block:: python

    class BenchEstimator(BaseEstimator):
        def __init__(self, data_name, loss_f="nll_loss", evaluation=[Acc()]):
            super().__init__(loss_f, evaluation)
            self.evaluation = evaluation
            self.bench=light_read(data_name)

        def infer(self, model: BaseSpace, dataset, mask="train"):
            perf=model(self.bench)
            return [perf], 0

使用 NAS-Bench-Graph 进行架构搜索
--------------------

在运行阶段，我们首先初始化以上搜索空间和性能评估器。
然后我们选择一个神经网络架构搜索策略并进行初始化。
之后，进行搜索和推断过程。
实验结果写入到一个 `json` 文件中。

.. code-block:: python

    def run(data_name='cora',algo='graphnas',num_epochs=50,ctrl_steps_aggregate=20,log_dir='./logs/tmp'):
        print("Testing backend: {}".format("dgl" if DependentBackend.is_dgl() else "pyg"))
        if DependentBackend.is_dgl():
            from autogl.datasets.utils.conversion._to_dgl_dataset import to_dgl_dataset as convert_dataset
        else:
            from autogl.datasets.utils.conversion._to_pyg_dataset import to_pyg_dataset as convert_dataset

        # Only for initialization of the space class, no meaning 
        di=2
        do=2
        dataset=None

        ops_type=data_name=='proteins'

        # Initialization of the benchmark space and estimator
        space = BenchSpace().cuda()
        space.instantiate(input_dim=di, output_dim=do,ops_type=ops_type)
        esti = BenchEstimator(data_name)

        # Choosing a NAS search strategy in AutoGL
        if algo=='graphnas':
            algo = GraphNasRL(num_epochs=num_epochs,ctrl_steps_aggregate=ctrl_steps_aggregate)
        elif algo=='agnn':
            algo = AGNNRL(guide_type=1,num_epochs=num_epochs,ctrl_steps_aggregate=ctrl_steps_aggregate)
        else:
            assert False,f'Not implemented algo {algo}'

        # Searching with NAS-Bench-Graph
        model = algo.search(space, dataset, esti)
        result=esti.infer(model._model,None)[0][0]

        # Print and return the results
        import json
        archs=algo.allhist
        json.dump(archs,open(osp.join(log_dir,f'archs.json'),'w'))
        return result

.. [1] Qin, Yijian, et al. "NAS-Bench-Graph: Benchmarking Graph Neural Architecture Search." Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 
