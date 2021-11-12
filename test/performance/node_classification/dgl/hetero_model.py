"""
Performance check of AutoGL model + DGL (trainer + dataset)
"""
import os
import numpy as np
from tqdm import tqdm
import dgl
os.environ["AUTOGL_BACKEND"] = "dgl"
import sys
sys.path.append("../../../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from autogl.module.model.dgl import AutoHGT
from autogl.solver.utils import set_seed
import logging


import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import argparse


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(model, G, args):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits = model(G, 'paper')
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(args.device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        if epoch % 5 == 0:
            model.eval()
            logits = model(G, 'paper')
            pred   = logits.argmax(1).cpu()
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
            val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
            test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print('Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                epoch,
                optimizer.param_groups[0]['lr'], 
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
            ))
    return best_test_acc


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='hetero dgl model')

    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--n_inp',   type=int, default=256)
    parser.add_argument('--clip',    type=int, default=1.0) 
    parser.add_argument('--max_lr',  type=float, default=1e-3) 
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
        
    torch.manual_seed(0)
    data_url = 'https://data.dgl.ai/dataset/ACM.mat'
    data_file_path = '/tmp/ACM.mat'

    urllib.request.urlretrieve(data_url, data_file_path)
    data = scipy.io.loadmat(data_file_path)

    G = dgl.heterograph({
            ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
            ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
            ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
            ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
        })
    print(G)

    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    # generate labels
    labels = pvc.indices
    labels = torch.tensor(labels).long()


    # generate train/val/test split
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:900]).long()
    test_idx = torch.tensor(shuffle[900:]).long()

    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

    #     Random initialize input feature
    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 256), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        G.nodes[ntype].data['inp'] = emb

    G = G.to(args.device)

    num_features = 256
    num_classes = labels.max().item()+1
    accs = []

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        model = AutoHGT(G=G,
            node_dict=node_dict, 
            edge_dict=edge_dict,
            num_features=num_features,
            num_classes=num_classes,
            device=args.device,
            init=False
        ).from_hyper_parameter({
            # hp from model
            "num_layers": 2,
            "hidden": [256],
            "heads": 4,
            "dropout": 0.2,
            "act": "gelu",
            "use_norm": True,
        }).model

        model.to(args.device)

        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
        print('Training MLP with #param: %d' % (get_n_params(model)))
        best_test_acc = train(model, G, args)
        accs.append(best_test_acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))

