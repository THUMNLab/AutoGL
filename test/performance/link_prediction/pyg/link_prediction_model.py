import os
os.environ["AUTOGL_BACKEND"] = "pyg"

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import os.path as osp
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from autogl.module.model.pyg import AutoGCN, AutoGAT, AutoSAGE
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import negative_sampling
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

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

# args.dataset = 'Cora'
# args.model = 'gat'
print(args.dataset)
print(args.model)
# load the dataset

# path = osp.join('.', 'data', args.dataset)
path = osp.join('data', args.dataset)
if args.dataset == 'Cora':
    dataset = Planetoid(path, name='Cora',transform=T.NormalizeFeatures())
elif args.dataset == 'CiteSeer':
    dataset = Planetoid(path, name='CiteSeer',transform=T.NormalizeFeatures())
elif args.dataset == 'PubMed':
    dataset = Planetoid(path, name='PubMed',transform=T.NormalizeFeatures())
else:
    assert False

def train():
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, #positive edges
        num_nodes=data.num_nodes, # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1)).to(device) # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()
    train_edge_index=data.train_pos_edge_index
    train_data = Data(x=data.x,edge_index=train_edge_index)
    # print("trainen_shape",train_data.x.shape, train_data.edge_index.shape)
    # torch.Size([2708, 1433]) torch.Size([2, 17952]) #
    z = model.lp_encode(train_data) #encode
    # print("trainde_shape",z.shape, data.train_pos_edge_index.shape,neg_edge_index.shape)
    # torch.Size([2708, 64]) torch.Size([2, 8976]) torch.Size([2, 8976])
    link_logits = model.lp_decode(z, data.train_pos_edge_index, neg_edge_index) # decode
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss, train_data

@torch.no_grad()
def test(train_data):
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:

        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        z = model.lp_encode(train_data) # encode train
        # print("testen_shape",train_data.x.shape, train_data.edge_index.shape)
        # print("testde_shape",z.shape, data.train_pos_edge_index.shape,neg_edge_index.shape)
        # val
        # testen_shape torch.Size([2708, 1433]) torch.Size([2, 17952])
        # testde_shape torch.Size([2708, 64]) torch.Size([2, 8976]) torch.Size([2, 263])
        # test
        # testen_shape torch.Size([2708, 1433]) torch.Size([2, 17952])
        # testde_shape torch.Size([2708, 64]) torch.Size([2, 8976]) torch.Size([2, 527])
        # print(prefix)
        link_logits = model.lp_decode(z, pos_edge_index, neg_edge_index) # decode test or val
        link_probs = link_logits.sigmoid() # apply sigmoid
        
        link_labels = get_link_labels(pos_edge_index, neg_edge_index) # get link
        
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu())) #compute roc_auc score
    return perfs

res = []
for seed in tqdm(range(1234, 1234+args.repeat)):
    setup_seed(seed)
    g = dataset[0].to(device)
    data = dataset[0].to(device)
    # use train_test_split_edges to create neg and positive edges
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data).to(device)
    if args.model == 'gcn':
        model = AutoGCN(dataset=dataset,
                num_features=dataset.num_features,
                num_classes=2, # num_class对linkpre任务似乎没有用？
                device=args.device,
                init=False
            ).from_hyper_parameter({
                'num_layers': 3,
                'hidden': [128,64],
                'dropout': 0.0,
                'act': 'relu', # 对linkpre任务似乎没有用？
                'agg': 'mean',
                'add_self_loops': 'False',
                'normalize': 'False',
            }).model
    elif args.model == 'gat':
        model = AutoGAT(dataset=dataset,
                num_features=dataset.num_features,
                num_classes=2,
                device=args.device,
                init=False
            ).from_hyper_parameter({
                'num_layers': 3,
                'hidden': [128,64],
                'dropout': 0.0,
                'act': 'relu',
                'agg': 'mean',
                'add_self_loops': 'False',
                'normalize': 'False',
            }).model
    elif args.model == 'sage':
        model = AutoSAGE(dataset=dataset,
                num_features=dataset.num_features,
                num_classes=2,
                device=args.device,
                init=False
            ).from_hyper_parameter({
                'num_layers': 3,
                'hidden': [128,64],
                'dropout': 0.0,
                'act': 'relu',
                'agg': 'mean',
                'add_self_loops': 'False',
                'normalize': 'False',
            }).model
    else:
        assert False

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_val_perf = test_perf = 0
    for epoch in range(100):
        train_loss, train_data = train()
        val_perf, tmp_test_perf = test(train_data)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # if epoch % 10 == 0:
        #     print(log.format(epoch, train_loss, best_val_perf, test_perf))
    res.append(test_perf)

print(np.mean(res), np.std(res))









