import sys
sys.path.append('../../')

from dgl.dataloading.pytorch.dataloader import GraphDataLoader
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
    best_model = net.state_dict()
    
    best_acc = 0.
    for e in range(epoch):
        for graphs, labels in trainloader:
            net.train()

            labels = labels.to(device)
            graphs = graphs.to(device)
            outputs = net((graphs, labels))
            # feat = graphs.ndata.pop('attr')
            # outputs = net(graphs, feat)

            loss = criterion(outputs, labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        gt = []
        pr = []
        for graphs, labels in validloader:
            labels = labels.to(device)
            graphs = graphs.to(device)
            gt.append(labels)
            # feat = graphs.ndata.pop('attr')
            # outputs = net(graphs, feat)
            outputs = net((graphs, labels))
            pr.append(outputs.argmax(1))
        gt = torch.cat(gt, dim=0)
        pr = torch.cat(pr, dim=0)
        acc = (gt == pr).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_model = net.state_dict()
    
    net.load_state_dict(best_model)

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
        outputs = net((graphs, labels))
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()

    acc = 1.0 * total_correct / total

    net.train()

    return acc


def main():

    device = torch.device('cuda:1')
    dataset_ = GINDataset('MUTAG', False)
    dataset = DatasetAbstraction([g[0] for g in dataset_], [g[1] for g in dataset_])
    
    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(2021)
    random.shuffle(dataids)
    
    fold = int(len(dataset) * 0.1)
    train_dataset = dataset[dataids[:fold * 8]]
    val_dataset = dataset[dataids[fold * 8: fold * 9]]
    test_dataset = dataset[dataids[fold * 9: ]]

    trainloader = GraphDataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = GraphDataLoader(val_dataset, batch_size=32, shuffle=False)
    testloader = GraphDataLoader(test_dataset, batch_size=32, shuffle=False)

    accs = []
    for seed in tqdm(range(50)):
        # set up seeds, args.seed supported
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)

        model = AutoGIN(
            num_features=dataset_.dim_nfeats,
            num_classes=dataset_.gclasses,
            device=device,
        ).from_hyper_parameter(
            {
                # hp from model
                "num_layers": 5,
                "hidden": [64,64,64,64],
                "dropout": 0.5,
                "act": "relu",
                "eps": "False",
                "mlp_layers": 2,
                "neighbor_pooling_type": "sum",
                "graph_pooling_type": "sum"
            }
        ).model

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()  # defaul reduce is true
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        model = train(model, trainloader, valloader, optimizer, criterion, 100, device)
        acc = eval_net(model, testloader, device)
        accs.append(acc)

    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))

if __name__ == '__main__':
    main()
