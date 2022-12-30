import os
import tqdm
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import Evaluator
from autogl.datasets import build_dataset_from_name
from autogl import backend

if backend.DependentBackend.is_dgl():
    feat = 'feat'
    label = 'label'
else:
    feat = 'x'
    label = 'y'

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, x, y, edge_index, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)[train_idx]
    loss = F.nll_loss(out, y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, x, y, edge_index, split_idx, evaluator):
    model.eval()
    out = model(x, edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']].view(-1, 1),
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']].view(-1, 1),
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']].view(-1, 1),
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

class Node:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __le__(self, other):
        return self.a <= other.a
    
    def __lt__(self, other):
        if self.a < other.a:
            return True
        elif self.a == other.a:
            return self.b < other.b
        else:
            return False

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # print(oedge_index)
    

    dataset = build_dataset_from_name('ogbn_arxiv')

    data = dataset[0]
    x = data.nodes.data[feat].to(device)
    y = data.nodes.data[label].to(device)
    edge_index = data.edges.connections.to(device)
    # edge_index = data_transfer(edge_index, row, col)
    print(edge_index)
    # print(edge_index.shape)

    train_mask = data.nodes.data['train_mask']
    val_mask = data.nodes.data['val_mask']
    test_mask = data.nodes.data['test_mask']
    split_idx = {
        'train': train_mask,
        'valid': val_mask,
        'test': test_mask
    }

    # split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    labels = dataset[0].nodes.data[label]
    num_classes = len(np.unique(labels.numpy()))

    if args.use_sage:
        model = SAGE(dataset[0].nodes.data[feat].size(1), args.hidden_channels,
                     num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(dataset[0].nodes.data[feat].size(1), args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')

    best_accs = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_valid = 0.0
        best_test = 0.0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y, edge_index, train_idx, optimizer)
            result = test(model, x, y, edge_index, split_idx, evaluator)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_test = test_acc
        best_accs.append(best_test)
    print(best_accs)
    print(np.mean(best_accs))
    print(np.std(best_accs))

if __name__ == "__main__":
    main()