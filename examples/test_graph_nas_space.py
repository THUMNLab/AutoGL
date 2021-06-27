import sys
from nni.nas.pytorch.mutables import Mutable
sys.path.append('../')
from torch_geometric.nn import GCNConv
import torch
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module.train import NodeClassificationFullTrainer
from autogl.module.nas import Darts, OneShotEstimator
from autogl.module.nas.space.graph_nas import *
from autogl.module.train import Acc
from autogl.module.nas.algorithm.enas import Enas
from autogl.module.nas.algorithm.rl import *
from autogl.module.nas.estimator.one_shot import TrainEstimator
import logging
import numpy as np
from tqdm import  tqdm
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    dataset = build_dataset_from_name('cora')
    space=GraphNasNodeClassificationSpace(hidden_dim=16,search_act_con=True,layer_number=2)
    space.instantiate(input_dim=dataset[0].x.shape[1],
                output_dim=dataset.num_classes,)
    estim=TrainEstimator()
    # solver.fit(dataset)
    # solver.get_leaderboard().show()
    # out = solver.predict_proba()
    
    # print('acc on cora', Acc.evaluate(out, dataset[0].y[dataset[0].test_mask].detach().numpy()))
    class Tmp:
        def __init__(self,space):
            self.model = space
            self.nas_modules = []
            k2o = get_module_order(self.model)
            replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
            replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
            self.nas_modules = sort_replaced_module(k2o, self.nas_modules)
    
    t=Tmp(space)
    print(t.nas_modules)
    nm=t.nas_modules
    selection_range={}
    for k,v in nm:
        selection_range[k]=len(v)
    ks=list(selection_range.keys())
    selections=[]
    def dfs(selection,d):
        if d>=len(ks):
            selections.append(selection.copy())
            return 
        k=ks[d]
        r=selection_range[k]
        for i in range(r):
            selection[k]=i
            dfs(selection,d+1)
    dfs({},0)
    print(f'#selections {len(selections)}')
    device=torch.device('cuda:0')
    accs=[]
    from datetime import datetime
    timestamp=datetime.now().strftime('%m%d-%H-%M-%S')
    log=open(f'acclog{timestamp}.txt','w')
    with tqdm(selections) as bar:
        for selection in bar:
            arch=space.export(selection,device)
            m,l=estim.infer(arch,dataset,'test')
            bar.set_postfix(m=m,l=l.item())
            log.write(f'{arch}\n{selection}\n{m},{l}\n')
            log.flush()
            accs.append(m)

    np.save(f'space_acc{timestamp}',np.array(accs))
    print(f'max acc {np.max(accs)}')