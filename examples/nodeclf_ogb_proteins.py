import argparse
import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from autogl import backend
from autogl.datasets import build_dataset_from_name

if backend.DependentBackend.is_dgl():
    ylabel = 'label'
else:
    ylabel = 'y'

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


def train(model, x, y, edge_index, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(x, edge_index)[train_idx]
    loss = criterion(out, y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y, edge_index, split_idx, evaluator):
    model.eval()

    y_pred = model(x, edge_index)

    train_rocauc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    autogl_dataset = build_dataset_from_name('ogbn-proteins')
    data = autogl_dataset[0]
    y = data.nodes.data[ylabel].to(device)
    num_nodes = data.nodes.data['species'].shape[0]
    edge_index = data.edges.connections
    row = edge_index[0].type(torch.long).to(device)
    col = edge_index[1].type(torch.long).to(device)
    edge_feat = data.edges.data['edge_feat'].to(device)
    edge_index = SparseTensor(row=row, col=col, value=edge_feat, sparse_sizes=(num_nodes, num_nodes))
    x = edge_index.mean(dim=1).to(device)
    edge_index.set_value_(None)
    
    train_mask = data.nodes.data['train_mask']
    val_mask = data.nodes.data['val_mask']
    test_mask = data.nodes.data['test_mask']
    split_idx = {
        'train': train_mask,
        'valid': val_mask,
        'test': test_mask
    }
    labels = data.nodes.data[ylabel]
    num_classes = len(np.unique(labels.numpy()))
    train_idx = split_idx['train']

    if args.use_sage:
        model = SAGE(x.size(1), args.hidden_channels, 112,
                     args.num_layers, args.dropout).to(device)
    else:
        model = GCN(x.size(1), args.hidden_channels, 112,
                    args.num_layers, args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = edge_index.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        edge_index = adj_t

    evaluator = Evaluator(name='ogbn-proteins')

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y, edge_index, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, x, y, edge_index, split_idx, evaluator)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')


if __name__ == "__main__":
    main()