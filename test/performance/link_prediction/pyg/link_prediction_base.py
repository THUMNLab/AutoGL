import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import dgl.function as fn
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import negative_sampling
import os.path as osp
import sys
sys.path.insert(0, "../")
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import urllib
from sklearn.metrics import roc_auc_score

# Reference: https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial12/Tutorial12%20GAE%20for%20link%20prediction.ipynb#scrollTo=7-dfpy3sULEZ

def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features//2)

    def encode(self, data):
        x = self.conv1(data.x, data.train_pos_edge_index)
        return self.conv2(x, data.train_pos_edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z): 
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_features):
        super(GAT, self).__init__()

        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, hidden_features // 2, heads=1, concat=False,
                             dropout=0.6)
    def encode(self, data):
        x, edge_index = data.x, data.train_pos_edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z): 
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_features):
        super(GraphSAGE, self).__init__()
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            inc = outc = hidden_features
            if i == 0:
                inc = num_features
            if i == self.num_layers - 1:
                outc = hidden_features // 2
            self.convs.append(SAGEConv(inc, outc))

    def encode(self, data):
        x, edge_index = data.x, data.train_pos_edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z): 
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

parser = ArgumentParser(
    "auto link prediction", formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument("--dataset", default="Cora", type=str, help="dataset to use", choices=["Cora", "CiteSeer", "PubMed"],)
parser.add_argument("--model", default="sage", type=str,help="model to use", choices=["gcn","gat","sage"],)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument('--repeat', type=int, default=10)
parser.add_argument("--device", default=0, type=int, help="GPU device")
args = parser.parse_args()

args.device = torch.device('cuda:0')
device = torch.device('cuda:0')

dataset = Planetoid(osp.expanduser('~/.cache-autogl'), args.dataset, transform=T.NormalizeFeatures())

def train():
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, #positive edges
        num_nodes=data.num_nodes, # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1)).to(device) # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()
    z = model.encode(data) #encode

    # print(data)
    # print("trainen_shape",data.x.shape, data.train_pos_edge_index.shape)
    # print("trainde_shape",z.shape, data.train_pos_edge_index.shape,neg_edge_index.shape)
    # trainen_shape torch.Size([2708, 1433]) torch.Size([2, 8976])
    # trainde_shape torch.Size([2708, 64]) torch.Size([2, 8976]) torch.Size([2, 8976])   
 
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index) # decode
    
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        # print(prefix)
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        z = model.encode(data) # encode train
        # print("testen_shape",data.x.shape, data.train_pos_edge_index.shape)
        # print("testde_shape",z.shape, data.train_pos_edge_index.shape,neg_edge_index.shape)
        # val
        # testen_shape torch.Size([2708, 1433]) torch.Size([2, 8976])
        # testde_shape torch.Size([2708, 64]) torch.Size([2, 8976]) torch.Size([2, 263])
        # test
        # testen_shape torch.Size([2708, 1433]) torch.Size([2, 8976])
        # testde_shape torch.Size([2708, 64]) torch.Size([2, 8976]) torch.Size([2, 527])
        link_logits = model.decode(z, pos_edge_index, neg_edge_index) # decode test or val
        link_probs = link_logits.sigmoid() # apply sigmoid
        
        link_labels = get_link_labels(pos_edge_index, neg_edge_index) # get link
        
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu())) #compute roc_auc score
    return perfs

res = []
for seed in tqdm(range(1234, 1234+args.repeat)):
    setup_seed(seed)
    data = dataset[0].to(device)
    # use train_test_split_edges to create neg and positive edges
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data).to(device)
 
    if args.model == 'gcn':
        model = GCN(dataset.num_features, 128).to(device)
    elif args.model == 'gat':
        model = GAT(dataset.num_features, 128).to(device)
    elif args.model == 'sage':
        model = GraphSAGE(dataset.num_features, 128).to(device)
    else:
        assert False
        
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_val_perf = test_perf = 0
    for epoch in range(100):
        train_loss = train()
        val_perf, tmp_test_perf = test()
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # if epoch % 10 == 0:
        #     print(log.format(epoch, train_loss, best_val_perf, test_perf))
    res.append(test_perf)

print(np.mean(res), np.std(res))












