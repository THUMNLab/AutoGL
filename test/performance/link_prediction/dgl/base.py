"""
Baseline that use early stopping
"""

import pickle
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import random
from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
import dgl.data
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

from dgl.nn import SAGEConv
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn import GATConv

from sklearn.metrics import roc_auc_score

parser = ArgumentParser(
    "auto link prediction", formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument("--dataset", default="Cora", type=str, help="dataset to use", choices=["Cora", "CiteSeer", "PubMed"],)
parser.add_argument("--model", default="sage", type=str,help="model to use", choices=["gcn","gat","sage"],)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument('--repeat', type=int, default=10)
parser.add_argument("--device", default=0, type=int, help="GPU device")

args = parser.parse_args()

if args.device >= 0:
    device = torch.device(f"cuda:{args.device}")
else:
    device = torch.device("cpu")

if args.dataset == 'Cora':
    dataset = CoraGraphDataset()
elif args.dataset == 'CiteSeer':
    dataset = CiteseerGraphDataset()
elif args.dataset == 'PubMed':
    dataset = PubmedGraphDataset()
else:
    assert False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, data):
        g = data
        in_feat = data.ndata['feat']
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, data):
        g = data
        in_feat = data.ndata['feat']
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats // 8, 8)
        self.conv2 = GATConv(h_feats, h_feats// 8, 8)

    def forward(self, data):
        g = data
        in_feat = data.ndata['feat']
        h = self.conv1(g, in_feat).flatten(1)
        h = F.relu(h)
        h = self.conv2(g, h).flatten(1)
        return h


def split_train_valid_test(g):
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)

    valid_size = int(len(eids) * 0.1)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size - valid_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    valid_pos_u, valid_pos_v =  u[eids[test_size:test_size+valid_size]], v[eids[test_size:test_size+valid_size]]
    train_pos_u, train_pos_v = u[eids[test_size+valid_size:]], v[eids[test_size+valid_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    valid_neg_u, valid_neg_v = neg_u[neg_eids[test_size:test_size+valid_size]], neg_v[neg_eids[test_size:test_size+valid_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size+valid_size:]], neg_v[neg_eids[test_size+valid_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size+valid_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    valid_pos_g = dgl.graph((valid_pos_u, valid_pos_v), num_nodes=g.number_of_nodes())
    valid_neg_g = dgl.graph((valid_neg_u, valid_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[: pos_edge_index.size(1)] = 1.0
    return link_labels

def lp_decode(z, pos_edge_index, neg_edge_index):
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    return logits

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    if mask == "val": offset = 3
    else: offset = 5
    z = model(data[0])
    link_logits = lp_decode(
        z, torch.stack(data[offset].edges()), torch.stack(data[offset + 1].edges())
    )
    link_probs = link_logits.sigmoid()
    link_labels = get_link_labels(
        torch.stack(data[offset].edges()), torch.stack(data[offset + 1].edges())
    )

    result = roc_auc_score(link_labels.cpu().numpy(),  link_probs.cpu().numpy())
    return result

res = []
for seed in tqdm(range(1234, 1234+args.repeat)):
    setup_seed(seed)
    g = dataset[0]
    splitted = list(split_train_valid_test(g))
    
    if args.model == 'gcn' or args.model == 'gat':
        splitted[0] = dgl.add_self_loop(splitted[0])

    splitted = [g.to(device) for g in splitted]

    if args.model == 'gcn':
        model = GCN(splitted[0].ndata['feat'].shape[1], 16).to(device)
    elif args.model == 'gat':
        model = GAT(splitted[0].ndata['feat'].shape[1], 64).to(device)
    elif args.model == 'sage':
        model = GraphSAGE(splitted[0].ndata['feat'].shape[1], 16).to(device)
    else:
        assert False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_auc = 0.
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        z = model(splitted[0])
        link_logits = lp_decode(
            z, torch.stack(splitted[1].edges()), torch.stack(splitted[2].edges())
        )
        link_labels = get_link_labels(
            torch.stack(splitted[1].edges()), torch.stack(splitted[2].edges())
        )
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer.step()

        auc_val = evaluate(model, splitted, "val")

        if auc_val > best_auc:
            best_auc = auc_val
            best_parameters = pickle.dumps(model.state_dict())
    
    model.load_state_dict(pickle.loads(best_parameters))
    res.append(evaluate(model, splitted, "test"))

print("{:.2f} ~ {:.2f}".format(np.mean(res) * 100, np.std(res) * 100))
