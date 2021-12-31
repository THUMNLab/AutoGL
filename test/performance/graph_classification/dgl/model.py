"""
Performance check of AutoGL model + DGL (dataset + trainer)
"""

import os
import pickle
os.environ["AUTOGL_BACKEND"] = "dgl"

# from dgl.dataloading.pytorch.dataloader import GraphDataLoader
from dgl.dataloading import GraphDataLoader
import numpy as np
from tqdm import tqdm

import random

import torch
import torch.nn as nn
import torch.optim as optim

from dgl.data import GINDataset

import torch
import torch.nn as nn
from autogl.module.model.dgl.gin import AutoGIN
from autogl.module.model.dgl.topkpool import AutoTopkpool
from autogl.solver.utils import set_seed
import argparse

class DatasetAbstraction():
    def __init__(self, graphs, labels):
        for g in graphs:
            g.ndata['feat'] = g.ndata['attr']
        self.graphs, self.labels = [], []
        for g, l in zip(graphs, labels):
            self.graphs.append(g)
            self.labels.append(l)
        self.gclasses = max(self.labels).item() + 1
        self.graph = self.graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif isinstance(idx, torch.BoolTensor):
            idx = [i for i in range(len(idx)) if idx[i]]
        elif isinstance(idx, torch.Tensor) and idx.unique()[0].sum().item() == 1:
            idx = [i for i in range(len(idx)) if idx[i]]
        return DatasetAbstraction([self.graphs[i] for i in idx], [self.labels[i] for i in idx])

def train(net, trainloader, validloader, optimizer, criterion, epoch, device):
    best_model = pickle.dumps(net.state_dict())
    
    best_acc = 0.
    for e in range(epoch):
        net.train()
        for graphs, labels in trainloader:

            labels = labels.to(device)
            graphs = graphs.to(device)
            # outputs = net((graphs, labels))
            # feat = graphs.ndata.pop('attr')
            # outputs = net(graphs, feat)
            outputs = net(graphs)

            loss = criterion(outputs, labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        gt = []
        pr = []
        net.eval()
        for graphs, labels in validloader:
            labels = labels.to(device)
            graphs = graphs.to(device)
            gt.append(labels)
            # feat = graphs.ndata.pop('attr')
            # outputs = net(graphs, feat)
            # outputs = net((graphs, labels))
            outputs = net(graphs)
            pr.append(outputs.argmax(1))
        gt = torch.cat(gt, dim=0)
        pr = torch.cat(pr, dim=0)
        acc = (gt == pr).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_model = pickle.dumps(net.state_dict())
    
    net.load_state_dict(pickle.loads(best_model))

    return net

def eval_net(net, dataloader, device):
    net.eval()

    total = 0
    total_correct = 0

    for data in dataloader:
        graphs, labels = data
        graphs = graphs.to(device)
        labels = labels.to(device)
        # feat = graphs.ndata.pop('attr')
        total += len(labels)
        # outputs = net(graphs, feat)
        # outputs = net((graphs, labels))
        outputs = net(graphs)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()

    acc = 1.0 * total_correct / total

    net.train()

    return acc


def main(args):

    device = torch.device(args.device)
    dataset_ = GINDataset(args.dataset, False)
    dataset = DatasetAbstraction([g[0] for g in dataset_], [g[1] for g in dataset_])
    
    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(args.dataset_seed)
    random.shuffle(dataids)
    
    fold = int(len(dataset) * 0.1)
    train_dataset = dataset[dataids[:fold * 8]]
    val_dataset = dataset[dataids[fold * 8: fold * 9]]
    test_dataset = dataset[dataids[fold * 9: ]]

    trainloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valloader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    testloader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    accs = []
    for seed in tqdm(range(args.repeat)):
        # set up seeds, args.seed supported
        set_seed(seed)

        if args.model == 'gin':
            model = AutoGIN(
                num_features=dataset_.dim_nfeats,
                num_classes=dataset_.gclasses,
                device=device,
            ).from_hyper_parameter({
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
                num_features=dataset_.dim_nfeats,
                num_classes=dataset_.gclasses,
                device=device,
            ).from_hyper_parameter({
                "num_layers": 5,
                "hidden": [64,64,64,64],
                "dropout": 0.5
            }).model

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()  # defaul reduce is true
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model = train(model, trainloader, valloader, optimizer, criterion, args.epoch, device)
        acc = eval_net(model, testloader, device)
        accs.append(acc)

    print('{:.2f} ~ {:.2f}'.format(np.mean(accs) * 100, np.std(accs) * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('model parser')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'NCI1', 'PROTEINS', 'PTC', 'REDDITBINARY', 'REDDITMULTI5K'], default='MUTAG')
    parser.add_argument('--dataset_seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['gin', 'topkpool'], default='gin')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)

    args = parser.parse_args()

    main(args)
