"""
Test file for nas on node classification

AUTOGL_BACKEND=pyg python test/nas/node_classification.py
AUTOGL_BACKEND=dgl python test/nas/node_classification.py

TODO: make it a unit test file to test all the possible combinations
"""

import os
import logging

logging.basicConfig(level=logging.INFO)

from autogl.backend import DependentBackend

if DependentBackend.is_dgl():
    from autogl.module.model.dgl import BaseAutoModel
    from dgl.data import CoraGraphDataset
elif DependentBackend.is_pyg():
    from torch_geometric.datasets import Planetoid
    from autogl.module.model.pyg import BaseAutoModel
from autogl.datasets import build_dataset_from_name
import torch
from torch import nn
import torch.nn.functional as F
#from autogl.module.nas.algorithm.agnn_rl import AGNNRL
from autogl.module.nas.backend import bk_feat, bk_label
from autogl.module.nas.algorithm import Darts, RL, GraphNasRL, Enas, RandomSearch,Spos
from autogl.module.nas.estimator import BaseEstimator
from autogl.module.train.evaluation import Acc
import numpy as np
from autogl.solver.utils import set_seed
from autogl.module.nas.space import BaseSpace
import typing as _typ
from nas_bench_graph import light_read, gnn_list, gnn_list_proteins, Arch
import pandas as pd
import argparse
import os
import os.path as osp

# Define the search space in NAS-bench-graph
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

# Define a new estimator which directly get performance from NAS-bench-graph instead of training the model
class BenchEstimator(BaseEstimator):
    def __init__(self, data_name, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation
        self.bench=light_read(data_name)
    def infer(self, model: BaseSpace, dataset, mask="train"):
        perf=model(self.bench)
        return [perf],0

# Run NAS with NAS-bench-graph
def run(data_name='cora',algo='graphnas',num_epochs=50,ctrl_steps_aggregate=20,log_dir='./logs/tmp'):
    print("Testing backend: {}".format("dgl" if DependentBackend.is_dgl() else "pyg"))
    if DependentBackend.is_dgl():
        from autogl.datasets.utils.conversion._to_dgl_dataset import to_dgl_dataset as convert_dataset
    else:
        from autogl.datasets.utils.conversion._to_pyg_dataset import to_pyg_dataset as convert_dataset

    di=2
    do=2
    dataset=None

    ops_type=data_name=='proteins'

    space = BenchSpace().cuda()
    space.instantiate(input_dim=di, output_dim=do,ops_type=ops_type)
    esti = BenchEstimator(data_name)
    if algo=='graphnas':
        algo = GraphNasRL(num_epochs=num_epochs,ctrl_steps_aggregate=ctrl_steps_aggregate)
    elif algo=='agnn':
        algo = AGNNRL(guide_type=1,num_epochs=num_epochs,ctrl_steps_aggregate=ctrl_steps_aggregate)
    else:
        assert False,f'Not implemented algo {algo}'
    model = algo.search(space, dataset, esti)
    result=esti.infer(model._model,None)[0][0]

    os.makedirs(log_dir,exist_ok=True)
    with open(osp.join(log_dir,f'log.txt'),'w') as f:
        f.write(str(result))

    import json
    archs=algo.allhist
    json.dump(archs,open(osp.join(log_dir,f'archs.json'),'w'))

    arch_strs=[str(x[1]) for x in archs]
    print(f'number of archs: {len(arch_strs)} ; number of unique archs : {len(set(arch_strs))}')   

    scores=[-x[0] for x in archs] # accs
    idxs=np.argsort(scores) # increasing order
    with open(osp.join(log_dir,f'idx.txt'),'w') as f:
        f.write(str(idxs))
    return result

# Run NAS with NAS-bench-graph for all provided datasets
def run_all():
    data_names='arxiv citeseer computers cora cs photo physics proteins pubmed'.split()
    algos='graphnas agnn'.split()
    results=[]
    for data_name in data_names:
        for algo in algos:
            print(f'data {data_name} algo {algo}')
            # metric=run(data_name,algo,2,2)
            if data_name=='proteins':
                metric=run(data_name,algo,8,5)
            else:
                metric=run(data_name,algo,50,10)
            results.append([data_name,algo,metric])
    return results

if __name__ == "__main__":
    # results=run_all()
    # df=pd.DataFrame(results,columns='data algo v'.split()).pivot_table(values='v',index='algo',columns='data')
    # print(df.to_string())

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora', help='datasets')
    parser.add_argument('--algo', type=str, default='graphnas')
    parser.add_argument('--log_dir', type=str, default='./logs/')

    args = parser.parse_args()
    dname=args.data
    algo=args.algo
    log_dir= os.path.join(args.log_dir,f'{dname,algo}')
    if dname=='proteins':
        # 40 archs in total
        num_epochs=8
        ctrl_steps_aggregate=5
    else:
        # 500 archs in total
        num_epochs=50
        ctrl_steps_aggregate=10
    result=run(dname,algo,num_epochs,ctrl_steps_aggregate,log_dir)