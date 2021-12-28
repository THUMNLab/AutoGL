"""
Performance check of AutoGL model + PYG (trainer + dataset)
"""
import os
import random
import numpy as np
from tqdm import tqdm

os.environ["AUTOGL_BACKEND"] = "pyg"

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from autogl.module.model.pyg import AutoGIN, AutoTopkpool
from autogl.datasets import utils
from autogl.solver.utils import set_seed
import logging

logging.basicConfig(level=logging.ERROR)

def test(model, loader, args):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(args.device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def train(model, train_loader, val_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    parameters = model.state_dict()
    best_acc = 0.
    for epoch in range(args.epoch):
        model.train()
        for data in train_loader:
            data = data.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()

        val_acc = test(model, val_loader, args)
        if val_acc > best_acc:
            best_acc = val_acc
            parameters = model.state_dict()
    
    model.load_state_dict(parameters)
    return model

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('pyg trainer')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'NCI1', 'PROTEINS', 'PTC', 'REDDITBINARY', 'REDDITMULTI5K'], default='MUTAG')
    parser.add_argument('--dataset_seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['gin', 'topkpool'], default='gin')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)

    args = parser.parse_args()

    # seed = 100
    dataset = TUDataset(os.path.expanduser('~/.pyg'), args.dataset)
    
    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(args.dataset_seed)
    random.shuffle(dataids)
    
    fold = int(len(dataset) * 0.1)
    train_index = dataids[:fold * 8]
    val_index = dataids[fold * 8: fold * 9]
    test_index = dataids[fold * 9: ]
    dataset.train_index = train_index
    dataset.val_index = val_index
    dataset.test_index = test_index
    dataset.train_split = dataset[dataset.train_index]
    dataset.val_split = dataset[dataset.val_index]
    dataset.test_split = dataset[dataset.test_index]

    labels = np.array([data.y.item() for data in dataset.test_split])

    train_loader = DataLoader(dataset.train_split, batch_size=args.batch_size)
    val_loader = DataLoader(dataset.val_split, batch_size=args.batch_size)
    test_loader = DataLoader(dataset.test_split, batch_size=args.batch_size)
    
    accs = []

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

        if args.model == 'gin':
            model = AutoGIN(
                num_features=dataset.num_node_features,
                num_classes=dataset.num_classes,
                num_graph_features=0,
                init=False
            ).from_hyper_parameter({
                # hp from model
                "num_layers": 5,
                "hidden": [64,64,64,64],
                "dropout": 0.5,
                "act": "relu",
                "eps": "False",
                "mlp_layers": 2,
                "neighbor_pooling_type": "sum",
                "graph_pooling_type": "sum"
            }).model
        elif args.model == 'topkpool':
            model = AutoTopkpool(
                num_features=dataset.num_node_features,
                num_classes=dataset.num_classes,
                num_graph_features=0,
                init=False
            ).from_hyper_parameter({
                "ratio": 0.8,
                "dropout": 0.5,
                "act": "relu"
            }).model
        
        model.to(args.device)

        train(model, train_loader, val_loader, args)
        acc = test(model, test_loader, args)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
