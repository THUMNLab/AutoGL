import os
import sys
import logging
logging.basicConfig(level=logging.INFO)

sys.path.append("../../")
print(os.getcwd())
os.environ["AUTOGL_BACKEND"] = "dgl"
from autogl.backend import DependentBackend
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import torch
import torch.nn.functional as F
from autogl.module.nas.space.single_path import SinglePathNodeClassificationSpace
from autogl.module.nas.space.graph_nas import GraphNasNodeClassificationSpace
from autogl.module.nas.estimator.one_shot import OneShotEstimator
from autogl.module.nas.estimator.train_scratch import TrainEstimator
from autogl.module.nas.backend import bk_feat, bk_label
from autogl.module.nas.algorithm import Darts,RL,GraphNasRL,Enas,RandomSearch
from pdb import set_trace
import numpy as np
from autogl.datasets import build_dataset_from_name
from autogl.solver.utils import set_seed
set_seed(202106)

if __name__ == '__main__':
    isdgl=DependentBackend.is_dgl()
    print('Using backend: %s'%('dgl' if isdgl else 'pyg'))
    data = CoraGraphDataset()
    di=bk_feat(data[0]).shape[1]
    do=len(np.unique(bk_label(data[0])))

    print("Random search + graphnas ")
    space=GraphNasNodeClassificationSpace().cuda()
    space.instantiate(input_dim=di,output_dim=do)
    esti=OneShotEstimator()
    algo=RandomSearch(num_epochs=10)
    algo.search(space,data,esti)
    
    print("Random search + singlepath ")
    space=SinglePathNodeClassificationSpace().cuda()
    space.instantiate(input_dim=di,output_dim=do)
    esti=OneShotEstimator()
    algo=RandomSearch(num_epochs=10)
    algo.search(space,data,esti)

    print("rl + graphnas ")
    space=GraphNasNodeClassificationSpace().cuda()
    space.instantiate(input_dim=di,output_dim=do)
    esti=OneShotEstimator()
    algo=RL(num_epochs=10)
    algo.search(space,data,esti)

    print("graphnasrl + graphnas ")
    space=GraphNasNodeClassificationSpace().cuda()
    space.instantiate(input_dim=di,output_dim=do)
    esti=OneShotEstimator()
    algo=GraphNasRL(num_epochs=10)
    algo.search(space,data,esti)

    print("enas + graphnas ")
    space=GraphNasNodeClassificationSpace().cuda()
    space.instantiate(input_dim=di,output_dim=do)
    esti=OneShotEstimator()
    algo=Enas(num_epochs=10)
    algo.search(space,data,esti)

    # Darts can not run
    # print("darts + graphnas ")
    # space=GraphNasNodeClassificationSpace().cuda()
    # space.instantiate(input_dim=di,output_dim=do)
    # esti=OneShotEstimator()
    # algo=Darts(num_epochs=10)
    # algo.search(space,data,esti)

