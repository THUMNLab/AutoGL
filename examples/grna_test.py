import os
os.environ["AUTOGL_BACKEND"] = "pyg"
import sys
sys.path.append('../')
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module.train import Acc
from autogl.solver.utils import set_seed
import argparse

import torch
from tqdm import tqdm
import numpy as np
import time

def test_from_data(trainer, dataset, args):
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        trainer.train(dataset)
        acc = trainer.evaluate(dataset, mask='test')

    return acc

if __name__ == '__main__':
    time0 = time.time()
    set_seed(202106)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/nodeclf_nas_grna.yml')
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed'], default='citeseer', type=str)
    parser.add_argument('--repeat', type=int, default=1)

    args = parser.parse_args()
    device = 'cuda'

    import yaml
    path_or_dict = yaml.load(open(args.config, "r").read(), Loader=yaml.FullLoader)

    from torch_geometric.datasets import Planetoid
    dataset_pyg = Planetoid('./data', args.dataset)
    print('finish loading dataset pyg')
    dataset = build_dataset_from_name(args.dataset, path='./')
    dataset[0] = dataset_pyg[0]
    
    solver = AutoNodeClassifier.from_config(args.config)
    solver.fit(dataset)
    solver.get_leaderboard().show()
    orig_acc = solver.evaluate(metric="acc")
    
    trainer = solver.graph_model_list[0]
    trainer.device = device


    ## test searched model on perturbed data
    # load perturb data with attacker=mettack
    modified_adj = torch.from_numpy(np.load(os.path.join('./perturb_data','Mettack', \
                args.dataset.lower()+'_A_hat_'+str(5)+'.npy')))
    new_edge_index = torch.nonzero(modified_adj).T.to(device)
    perturbed_data = dataset[0].clone()
    perturbed_data.edge_index = new_edge_index
    dataset[0] = perturbed_data
    
    ptb_acc = test_from_data(trainer, dataset, args)
    # print('acc after poisoning attack:', ptb_acc)




