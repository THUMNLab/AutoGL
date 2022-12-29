.. _nas_bench_graph:

Using with NAS-Bench-Graph
============================

We support running NAS methods with NAS-Bench-Graph[1]. You can directly get archtiecture performance estimation from NAS-Bench-Graph instead of training the architectures from scratch.
An example code is shown in `nas_bench_graph_example.py` in AutoGL.

Search Space Construction
------------

To use NAS-Bench-Graph, you should define the search space the same as the paper.
In general, you can copy the code below directly without any modification.

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

Benchmark Estimator Definition
--------------------

Then you need to define a new estimator which directly get performance of the given dataset from NAS-bench-graph instead of training the model.
You can also copy the code without modification.

.. code-block:: python

    class BenchEstimator(BaseEstimator):
        def __init__(self, data_name, loss_f="nll_loss", evaluation=[Acc()]):
            super().__init__(loss_f, evaluation)
            self.evaluation = evaluation
            self.bench=light_read(data_name)

        def infer(self, model: BaseSpace, dataset, mask="train"):
            perf=model(self.bench)
            return [perf], 0

Running NAS with NAS-Bench-Graph
--------------------

In the running part, we first initialize the above search space and performance estimator.
Then we choose a NAS search strategy and initialize it.
After that, run the searching and infering process.
The experimental results are written in a `json` file.

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
