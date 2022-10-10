import os
import torch
# import sys
# sys.path.insert(0, '/n/scratch2/xz204/Dr37/lib/python3.7/site-packages')
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
# from deeprobust.graph.defense import * # GCN, GAT, GIN, JK, GCN_attack,accuracy_1
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset, PrePtbDataset, Dpr2Pyg, AmazonPyg
from tqdm import tqdm
import scipy
import numpy as np
from sklearn.preprocessing import normalize
import pickle

os.environ["AUTOGL_BACKEND"] = "pyg"

from autogl.module.model.pyg import AutoGNNGuard
from autogl.solver.utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=14, help='Random seed.')
# cora and citeseer are binary, pubmed has not binary features
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--DPlabel', type=int, default=9,  help='0-10')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=1029):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
seed_torch(args.seed)

args.dataset = "cora"
args.modelname = "GCN"

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)
"""set the number of training/val/testing nodes"""
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
"""add undirected edges, orgn-arxiv is directed graph, we transfer it to undirected closely following 
https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
"""
adj = adj + adj.T
adj[adj>1] = 1

pyg_data = Dpr2Pyg(data).process().to(args.device)
pyg_data.num_classes = len(set(labels))


def main(flag):

    # Setup Surrogate model
    surrogate = GCN_attack(nfeat=features.shape[1], nclass=labels.max().item()+1, n_edge=adj.nonzero()[0].shape[0], nhid=16, dropout=0, with_relu=False, with_bias=False, device=args.device, )
    surrogate = surrogate.to(args.device)
    surrogate.fit(features, adj, labels, idx_train, train_iters=201)  # change this train_iters to 201: train_iters=201

    # Setup Attack Model
    target_node = 859

    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=args.device)
    model = model.to(args.device)

    degrees = adj.sum(0).A1
    # How many perturbations to perform. Default: Degree of the node
    n_perturbations = int(degrees[target_node])

    # # indirect attack/ influencer attack
    model.attack(features, adj, labels, target_node, n_perturbations, direct=True)
    modified_adj = model.modified_adj
    modified_features = model.modified_features

    print('=== testing GNN on original(clean) graph ===')
    test(adj, features, target_node,  attention=flag)

    print('=== testing GCN on perturbed graph ===')
    test(modified_adj, modified_features, target_node,attention=flag)

def test(adj, features, target_node, attention=False):
    ''
    """test on GCN """
    """model_name could be 'GCN', 'GAT', 'GIN','JK'  """
    # for orgn-arxiv: nhid =256, layers =3, epoch =500

    gcn = globals()[args.modelname](nfeat=features.shape[1], nhid=256,  nclass=labels.max().item() + 1, dropout=0.5,
              device=args.device)
    gcn = gcn.to(args.device)
    gcn.fit(features, adj, labels, idx_train, idx_val=idx_val,
            idx_test=idx_test,
            attention=attention, verbose=True, train_iters=81)
    gcn.eval()
    _, output = gcn.test(idx_test=idx_test)

    probs = torch.exp(output[[target_node]])[0]
    print('probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

def main_autogl(flag): 
    # Setup Surrogate model
    surrogate = AutoGNNGuard(
                num_features=pyg_data.num_node_features,
                num_classes=pyg_data.num_classes,
                device=args.device,
                init=False
            ).from_hyper_parameter(model_hp).model
    surrogate = surrogate.to(args.device)
    surrogate.fit(features, adj, labels, idx_train, train_iters=201)  # change this train_iters to 201: train_iters=201

    # Setup Attack Model
    target_node = 859

    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=args.device)
    model = model.to(args.device)

    degrees = adj.sum(0).A1
    # How many perturbations to perform. Default: Degree of the node
    n_perturbations = int(degrees[target_node])

    # # indirect attack/ influencer attack
    model.attack(features, adj, labels, target_node, n_perturbations, direct=True)
    modified_adj = model.modified_adj
    modified_features = model.modified_features

    print('=== testing GNN on original(clean) graph (AutoGL) ===')
    test_autogl(adj, features, target_node,  attention=flag)

    print('=== testing GCN on perturbed graph (AutoGL) ===')
    test_autogl(modified_adj, modified_features, target_node,attention=flag)


def test_autogl(adj, features, target_node, attention=False):
    ''
    """test on GCN """
    """model_name could be 'GCN', 'GAT', 'GIN','JK'  """
    # for orgn-arxiv: nhid =256, layers =3, epoch =500

    gcn = AutoGNNGuard(
                num_features=pyg_data.num_node_features,
                num_classes=pyg_data.num_classes,
                device=args.device,
                init=False
            ).from_hyper_parameter(model_hp).model
    gcn = gcn.to(args.device)
    gcn.fit(features, adj, labels, idx_train, idx_val=idx_val,
            idx_test=idx_test,
            attention=attention, verbose=True, train_iters=81)
    gcn.eval()
    _, output = gcn.test(idx_test=idx_test)

    probs = torch.exp(output[[target_node]])[0]
    print('probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

model_hp = {
        "num_layers": 2,
        "hidden": [16],
        "dropout": 0.0,
        "act": "relu"
    }

if __name__ == '__main__':
    # Check the performance of GCN under directed attack without defense
    main(flag=False) 
    # Use GNNGuard for defense
    main(flag=True)
    # Check the performance of GCN under directed attack without defense（AutoGL）
    main_autogl(flag=False)
    # Use GNNGuard for defense（AutoGL）
    main_autogl(flag=True)
