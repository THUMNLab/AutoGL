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
import torch.nn.functional as F
from autogl.module.nas.space.single_path import SinglePathNodeClassificationSpace
from autogl.module.nas.space.graph_nas import GraphNasNodeClassificationSpace
from autogl.module.nas.space.graph_nas_macro import GraphNasMacroNodeClassificationSpace
from autogl.module.nas.estimator.one_shot import OneShotEstimator
from autogl.module.nas.estimator.train_scratch import TrainEstimator
from autogl.module.nas.algorithm.agnn_rl import AGNNRL
from autogl.module.nas.space.autoattend import AutoAttendNodeClassificationSpace
from autogl.module.nas.backend import bk_feat, bk_label
from autogl.module.nas.algorithm import Darts, RL, GraphNasRL, Enas, RandomSearch,Spos
import numpy as np
from autogl.solver.utils import set_seed

set_seed(202106)
from autogl.module.nas.space import BaseSpace
import typing as _typ
from torch import nn
class StrModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.name = lambd

    def forward(self, *args, **kwargs):
        return self.name

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)


gnn_list_proteins = [
    "gcn",  # GCN
    "cheb",  # chebnet
    "arma",
    "fc",  # skip connection
    "skip"  # skip connection
]

gnn_list = [
    "gat",  # GAT with 2 heads
    "gcn",  # GCN
    "gin",  # GIN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "graph",  
    "fc",  # skip connection
    "skip"  # skip connection
]


class Arch:
    def __init__(self, lk=None, op=None):
        self.link = lk
        self.ops = op

    # def random_arch(self):
    #     self.ops = []
    #     self.link = random.choice(link_list)
    #     for i in self.link:
    #         self.ops.append(random.choice(gnn_list))

    def hash_arch(self, use_proteins = False):
        lk = self.link
        op = self.ops
        if use_proteins:
            gnn_g = {name: i for i, name in enumerate(gnn_list_proteins)}
            b = len(gnn_list_proteins) + 1
        else:
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            b = len(gnn_list) + 1
        if lk == [0,0,0,0]:
            lk_hash = 0
        elif lk == [0,0,0,1]:
            lk_hash = 1
        elif lk == [0,0,1,1]:
            lk_hash = 2
        elif lk == [0,0,1,2]:
            lk_hash = 3
        elif lk == [0,0,1,3]:
            lk_hash = 4
        elif lk == [0,1,1,1]:
            lk_hash = 5
        elif lk == [0,1,1,2]:
            lk_hash = 6
        elif lk == [0,1,2,2]:
            lk_hash = 7
        elif lk == [0,1,2,3]:
            lk_hash = 8

        for i in op:
            lk_hash = lk_hash * b + gnn_g[i]
        return lk_hash

    def regularize(self):
        lk = self.link[:]
        ops = self.ops[:]
        if lk == [0,0,0,2]:
            lk = [0,0,0,1]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0,0,0,3]:
            lk = [0,0,0,1]
            ops = [ops[2], ops[0], ops[1], ops[3]]
        elif lk == [0,0,1,0]:
            lk = [0,0,0,1]
            ops = [ops[0], ops[1], ops[3], ops[2]]
        elif lk == [0,0,2,0]:
            lk = [0,0,0,1]
            ops = [ops[1], ops[0], ops[3], ops[2]]
        elif lk == [0,0,2,1]:
            lk = [0,0,1,2]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0,0,2,2]:
            lk = [0,0,1,1]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0,0,2,3]:
            lk = [0,0,1,3]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0,1,0,0]:
            lk = [0,0,0,1]
            ops = [ops[0], ops[2], ops[3], ops[1]]
        elif lk == [0,1,0,1]:
            lk = [0,0,1,1]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0,1,0,2]:
            lk = [0,0,1,3]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0,1,0,3]:
            lk = [0,0,1,2]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0,1,1,0]:
            lk = [0,0,1,1]
            ops = [ops[0], ops[3], ops[1], ops[2]]
        elif lk == [0,1,1,3]:
            lk = [0,1,1,2]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0,1,2,0]:
            lk = [0,0,1,3]
            ops = [ops[0], ops[3], ops[1], ops[2]]
        elif lk == [0,1,2,1]:
            lk = [0,1,1,2]
            ops = [ops[0], ops[1], ops[3], ops[2]]
        return Arch(lk, ops)

    def equalpart_sort(self):
        lk = self.link
        op = self.ops
        ops = op[:]
        def part_sort(ids, ops):
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            opli = [gnn_g[ops[i]] for i in ids]
            opli.sort()
            for posid, opid in zip(ids, opli):
                ops[posid] = gnn_list[opid]
            return ops

        def sort0012(ops):
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            if gnn_g[op[0]] > gnn_g[op[1]] or op[0] == op[1] and gnn_g[op[2]] > gnn_g[op[3]]:
                ops = [ops[1], ops[0], ops[3], ops[2]]
            return ops

        if lk == [0,0,0,0]:
            ids = [0,1,2,3]
        elif lk == [0,0,0,1]:
            ids = [1,2] 
        elif lk == [0,0,1,1]:
            ids = [2,3] 
        elif lk == [0,0,1,2]:
            ids = None
            ops = sort0012(ops)
        elif lk == [0,1,1,1]:
            ids = [1,2,3] 
        elif lk == [0,1,2,2]:
            ids = [2,3] 
        else:
            ids = None

        if ids:
            ops = part_sort(ids, ops)

        self.ops = ops

    def move_skip_op(self):
        link = self.link[:]
        ops = self.ops[:]
        def move_one(k, link, ops):
            ops = [ops[k]] + ops[:k] + ops[k + 1:]
            for i, father in enumerate(link):
                if father == k + 1:
                    link[i] = link[k]
                if father <= k:
                    link[i] = link[i] + 1
            link = [0] + link[:k] + link[k + 1:]
            return link, ops

        def check_dim(k, link, ops):
            # check if a dimension is original dimension
            while k > -1:
                if ops[k] != 'skip':
                    return False
                k = link[k] - 1
            return True

        for i in range(len(link)):
            if ops[i] != 'skip':
                continue
            son = False
            brother = False
            for j, fa in enumerate(link):
                if fa == i + 1:
                    son = True
                elif j != i and fa == link[i]:
                    brother = True
            if son or not brother or check_dim(i, link, ops) and not son:
                link, ops = move_one(i, link, ops)

        if link == [0,1,2,1]:
            link = [0,1,1,2]
            ops = ops[:2] + [ops[3], ops[2]]
        elif link == [0,1,1,3]:
            link = [0,1,1,2]
            ops = [ops[0], ops[2], ops[1], ops[3]]

        #if link not in link_list:
        #    print(lk, link)
            
        self.link = link
        self.ops = ops

    def valid_hash(self):
        b = self.regularize()
        b.move_skip_op()
        b.equalpart_sort()
        return b.hash_arch()

    def check_isomorph(self):
        link, ops = self.link, self.ops
        linkm = link[:]
        opsm = ops[:]
        self.move_skip_op()
        self.equalpart_sort()
        #print(self.link, self.ops)
        return linkm == self.link and opsm == self.ops

import nni
def map_value(l, label):
    return nni.retiarii.nn.pytorch.ValueChoice(l, label = label)
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

import os.path as osp
bench_path='/DATA/DATANAS1/zzy/bench/light'
import pickle

def light_read(dname):
    f = open(osp.join(bench_path,f"{dname}.bench"), "rb")
    bench = pickle.load(f)
    f.close()
    return bench

from autogl.module.nas.estimator import BaseEstimator
from autogl.module.train.evaluation import Acc
class BenchEstimator(BaseEstimator):
    def __init__(self, data_name, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation
        self.bench=light_read(data_name)
    def infer(self, model: BaseSpace, dataset, mask="train"):
        perf=model(self.bench)
        return [perf],0

def run(data_name='cora',algo='graphnas',num_epochs=50,ctrl_steps_aggregate=20,log_dir='./logs/tmp'):
    print("Testing backend: {}".format("dgl" if DependentBackend.is_dgl() else "pyg"))
    if DependentBackend.is_dgl():
        from autogl.datasets.utils.conversion._to_dgl_dataset import to_dgl_dataset as convert_dataset
    else:
        from autogl.datasets.utils.conversion._to_pyg_dataset import to_pyg_dataset as convert_dataset

    # dataset = build_dataset_from_name('cora')
    # dataset = convert_dataset(dataset)
    # data = dataset[0]

    # di = bk_feat(data).shape[1]
    # do = len(np.unique(bk_label(data)))
    
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
import pandas as pd
import argparse
import torch
import os



if __name__ == "__main__":
    # results=run_all()
    # df=pd.DataFrame(results,columns='data algo v'.split()).pivot_table(values='v',index='algo',columns='data')
    # print(df.to_string())

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora', help='datasets')
    parser.add_argument('--algo', type=str, default='agnn')
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
    

    