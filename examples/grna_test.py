import os
os.environ["AUTOGL_BACKEND"] = "pyg"
import sys
sys.path.append('../')
from autogl.datasets import build_dataset_from_name
from autogl.datasets.utils.conversion._to_pyg_dataset import to_pyg_dataset
from autogl.solver import AutoNodeClassifier
from autogl.module.train import Acc
from autogl.solver.utils import set_seed
import argparse

import torch
from tqdm import tqdm
import numpy as np
import time

from torch_geometric.utils import to_scipy_sparse_matrix
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack


def metattack(data):
    print('Meta-attack...')
    adj, features, labels = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes), data.x.numpy(), data.y.numpy()
    idx = np.arange(data.num_nodes)
    idx_train, idx_val, idx_test = idx[data.train_mask], idx[data.val_mask], idx[data.test_mask]
    idx_unlabeled = np.union1d(idx_val, idx_test)
    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    # Setup Attack Model
    model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
    # Attack
    n_perturbations = int(data.edge_index.size(1)/2 * 0.05)
    model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=n_perturbations, ll_constraint=False)
    perturbed_adj = model.modified_adj
    perturbed_data = data.clone()
    perturbed_data.edge_index = torch.LongTensor(perturbed_adj.nonzero().T)

    return perturbed_data

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
    dataset = build_dataset_from_name(args.dataset)
    
    print('architecture search')
    solver = AutoNodeClassifier.from_config(args.config)
    solver.fit(dataset)
    solver.get_leaderboard().show()
    orig_acc = solver.evaluate(metric="acc")
    trainer = solver.graph_model_list[0]
    trainer.device = device

    ## test searched model on clean data
    dataset = to_pyg_dataset(dataset)
    acc = test_from_data(trainer, dataset, args)

    ## test searched model on perturbed data
    data = dataset[0].cpu()
    dataset[0] = metattack(data).to(device)
    ptb_acc = test_from_data(trainer, dataset, args)
    