"""
Performance check of PYG (model + trainer + dataset)
"""
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch_geometric
from torch_geometric.datasets import TUDataset
if int(torch_geometric.__version__.split(".")[0]) >= 2:
    from torch_geometric.loader import DataLoader
else:
    from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import logging

torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True)

logging.basicConfig(level=logging.ERROR)

class GIN(torch.nn.Module):
    def __init__(self):
        super(GIN, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class TopKPool(torch.nn.Module):
    def __init__(self):
        super(TopKPool, self).__init__()

        self.conv1 = GraphConv(dataset.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

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
    torch.manual_seed(args.dataset_seed)
    np.random.seed(args.dataset_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.dataset_seed)
    
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

    def seed_worker(worker_id):
        #seed =  torch.initial_seed()
        torch.manual_seed(args.dataset_seed)
        np.random.seed(args.dataset_seed)
        random.seed(args.dataset_seed)
    g = torch.Generator()
    g.manual_seed(args.dataset_seed)

    train_loader = DataLoader(dataset.train_split, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(dataset.val_split, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(dataset.test_split, batch_size=args.batch_size, worker_init_fn=seed_worker, generator=g)

    #train_loader = DataLoader(dataset.train_split, batch_size=args.batch_size, shuffle=False)
    #val_loader = DataLoader(dataset.val_split, batch_size=args.batch_size, shuffle=False)
    #test_loader = DataLoader(dataset.test_split, batch_size=args.batch_size, shuffle=False)

    accs = []

    for seed in tqdm(range(args.repeat)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        #random.seed(seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(seed)

        if args.model == 'gin':
            model = GIN()
        elif args.model == 'topkpool':
            model = TopKPool()
        
        model.to(args.device)

        train(model, train_loader, val_loader, args)
        acc = test(model, test_loader, args)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
