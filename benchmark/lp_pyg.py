import sys
sys.path.append('../')
import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import train_test_split_edges

import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['gcn', 'sage', 'gat'], type=str, default='gcn', help='model to train')
parser.add_argument('--dataset', choices=['cora', 'citseer', 'pubmed'], type=str, default='cora', help='dataset to evaluate')
parser.add_argument('--times', type=int, default=10, help='time to rerun')

args = parser.parse_args()

DIM = 64
dataset = pickle.load(open(f'/DATA/DATANAS1/guancy/github/AutoGL/env/cache/{args.dataset}-edge.data', 'rb'))
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

def _decode(z, pos_edge_index, neg_edge_index):
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x, edge_index):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)

class GCN(GNN):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, DIM)

class GAT(GNN):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 16, 8)
        self.conv2 = GATConv(128, DIM // 8, 8)

class SAGE(GNN):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, DIM)

MODEL = {
    'gcn': GCN,
    'gat': GAT,
    'sage': SAGE
}

scores = []

for t in range(args.times):

    model = MODEL[args.model](dataset.num_features).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    def get_link_labels(pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def train(data):
        model.train()

        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))

        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)
        link_logits = _decode(z, data.train_pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer.step()

        return loss


    @torch.no_grad()
    def test(data):
        model.eval()

        z = model.encode(data.x, data.train_pos_edge_index)

        results = []
        for prefix in ['val', 'test']:
            pos_edge_index = data[f'{prefix}_pos_edge_index']
            neg_edge_index = data[f'{prefix}_neg_edge_index']
            link_logits = _decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = get_link_labels(pos_edge_index, neg_edge_index)
            results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
        return results


    best_val_auc = test_auc = 0
    for epoch in range(1, 101):
        loss = train(data)
        val_auc, tmp_test_auc = test(data)
        if val_auc > best_val_auc:
            best_val = val_auc
            test_auc = tmp_test_auc
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
        #     f'Test: {test_auc:.4f}')

    scores.append(test_auc)
    print('time', t, test_auc)
print('mean', np.mean(scores), 'std', np.std(scores))
open('lp_pyg.log', 'a').write('\t'.join([args.dataset, args.model, str(np.mean(scores)), str(np.std(scores)), '\n']))
