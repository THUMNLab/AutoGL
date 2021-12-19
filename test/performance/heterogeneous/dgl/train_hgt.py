"""
Performance check of AutoGL trainer + DGL dataset
"""
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import urllib
import scipy
import dgl
import os.path as osp
os.environ["AUTOGL_BACKEND"] = "dgl"
import sys
import pickle
import random
from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from autogl.module.train import NodeClassificationHetTrainer
from autogl.solver.utils import set_seed
import logging
import argparse
from scipy import sparse
from scipy import io as sio 
from pprint import pprint
logging.basicConfig(level=logging.ERROR)

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def main(args):
    if args.dataset == 'Acm':
        # data_url = 'https://data.dgl.ai/dataset/ACM.mat'
        # data_file_path = '/tmp/ACM.mat'
        # urllib.request.urlretrieve(data_url, data_file_path)
        data_file_path = '/home/jcai/code/AutoGL/test/performance/heterogeneous/dgl/ACM.mat'
        data = scipy.io.loadmat(data_file_path)
    
    G = dgl.heterograph({
            ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
            ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
            ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
            ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
        })

    # generate labels
    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    labels = pvc.indices
    labels = torch.tensor(labels).long()
    
    # generate train/val/test split
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:900]).long()
    test_idx = torch.tensor(shuffle[900:]).long()

    edge_dict = {}
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
    for etype in G.etypes:
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * len(edge_dict)

    #Random initialize input feature
    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 256), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        G.nodes[ntype].data['feat'] = emb

    G = G.to(args.device)

    num_features = 256
    num_classes = labels.max().item()+1

    data["labels"] = labels
    data["val_mask"] = get_binary_mask(len(pid), val_idx)

    accs = []

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        if args.model == 'hgt':
            model_hp = {
                # hp from model
                "num_layers": 2,
                "hidden": [256,256,256],
                "heads": 4,
                "dropout": 0.2,
                "act": "gelu",
                "use_norm": True,
            }
        elif args.model=='HeteroRGCN':
            model_hp = {
                # hp from model
                "num_layers": 2,
                "hidden": [256],
                "heads": 4,
                "dropout": 0.2,
                "act": "leaky_relu",
            }
        
        trainer = NodeClassificationHetTrainer(
            model=args.model,
            G = G,
            num_features=num_features,
            num_classes=num_classes,
            device=args.device,
            init=False,
            feval=['acc'],
            loss="nll_loss",
        ).duplicate_from_hyper_parameter({
            "max_epoch": args.epoch,
            "early_stopping_round": args.epoch + 1,
            "lr": args.max_lr,
            "weight_decay": args.weight_decay,
            **model_hp
        })

        trainer.train(data, G, False, train_idx)
        output = trainer.predict(G, test_idx).to("cpu")
        acc = (output == labels[test_idx]).float().mean().item()
        print(acc)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('dgl trainer_hgt')
 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['gat', 'gcn', 'sage'], default='gat')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--max_lr',  type=float, default=1e-3) 
    parser.add_argument('--n_inp',   type=int, default=256)
    parser.add_argument('--clip',    type=int, default=1.0) 
    args = parser.parse_args()

    args.dataset = 'Acm'
    args.model = 'HeteroRGCN'
    torch.manual_seed(0)
    print(args)

    main(args)
