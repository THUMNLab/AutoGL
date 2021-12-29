"""
Performance check of DGL model + trainer + dataset
"""
import numpy as np
from tqdm import tqdm
import pickle

import torch
import torch.nn.functional as F

from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import logging

logging.basicConfig(level=logging.ERROR)

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(num_features, 16)
        self.conv2 = GraphConv(16, num_classes)

    def forward(self, graph):
        features = graph.ndata['feat']
        features = F.relu(self.conv1(graph, features))
        features = F.dropout(features, training=self.training)
        features = self.conv2(graph, features)
        return F.log_softmax(features, dim=-1)

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, 8, feat_drop=.6, attn_drop=.6, activation=F.relu)
        self.conv2 = GATConv(8 * 8, num_classes, 1, feat_drop=.6, attn_drop=.6)

    def forward(self, graph):
        features = graph.ndata['feat']
        features = self.conv1(graph, features).flatten(1)
        features = self.conv2(graph, features).mean(1)
        return F.log_softmax(features, dim=-1)

class SAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers, num_classes):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            inc = outc = hidden_channels
            if i == 0:
                inc = num_features
            if i == num_layers - 1:
                outc = num_classes
            self.convs.append(SAGEConv(inc, outc, "gcn"))
        self.dropout = torch.nn.Dropout()

    def forward(self, graph):
        h = graph.ndata['feat']
        h = self.dropout(h)
        for i, conv in enumerate(self.convs):
            h = conv(graph, h)
            if i != self.num_layers - 1:
                h = h.relu()
                h = self.dropout(h)
        return F.log_softmax(h, dim=-1)

def test(model, graph, mask, label):
    model.eval()

    pred = model(graph)[mask].max(1)[1]
    acc = pred.eq(label[mask]).sum().item() / mask.sum().item()
    return acc

def train(model, graph, args, label, train_mask, val_mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    parameters = model.state_dict()
    best_acc = 0.
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        output = model(graph)
        loss = F.nll_loss(output[train_mask], label[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = test(model, graph, val_mask, label)
        if val_acc > best_acc:
            best_acc = val_acc
            parameters = pickle.dumps(model.state_dict())
            
    model.load_state_dict(pickle.loads(parameters))
    return model

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('dgl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['gat', 'gcn', 'sage'], default='gat')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epoch', type=int, default=200)

    args = parser.parse_args()

    # seed = 100
    if args.dataset == 'Cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'CiteSeer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'PubMed':
        dataset = PubmedGraphDataset()
    graph = dataset[0].to(args.device)
    label = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    accs = []

    for seed in tqdm(range(args.repeat)):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if args.model == 'gat':
            model = GAT(graph.ndata['feat'].size(1), dataset.num_classes)
        elif args.model == 'gcn':
            model = GCN(graph.ndata['feat'].size(1), dataset.num_classes)
        elif args.model == 'sage':
            model = SAGE(graph.ndata['feat'].size(1), 64, 2, dataset.num_classes)
        
        model.to(args.device)

        train(model, graph, args, label, train_mask, val_mask)
        acc = test(model, graph, test_mask, label)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
