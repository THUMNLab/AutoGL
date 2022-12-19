import os
import torch
# import sys
# sys.path.insert(0, '/n/scratch2/xz204/Dr37/lib/python3.7/site-packages')
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import Metattack
import argparse
# from deeprobust.graph.defense import * # GCN, GAT, GIN, JK, GCN_attack,accuracy_1
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset, PrePtbDataset, Dpr2Pyg, AmazonPyg
from scipy.sparse import csr_matrix
from tqdm import tqdm
import scipy
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import normalize
import pickle

os.environ["AUTOGL_BACKEND"] = "pyg"

from autogl.module.model.pyg import AutoGNNGuard, AutoGNNGuard_attack
from autogl.solver.utils import set_seed

def seed_torch(seed=1029):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
seed_torch(2048)

def main(dataset, adj, features, device):
    # from deeprobust.graph.data import PrePtbDataset
    # perturbed_data = PrePtbDataset(root='/tmp/', name=dataset, attack_method='meta', ptb_rate=0.2)
    # modified_adj = perturbed_data.adj

    # Setup Surrogate model
    surrogate = GCN_attack(nfeat=features.shape[1], nclass=labels.max().item()+1, n_edge=adj.nonzero()[0].shape[0], nhid=16, dropout=0, with_relu=False, with_bias=False, device=args.device, )

    surrogate = surrogate.to(args.device)
    surrogate.fit(features, adj, labels, idx_train, train_iters=201)  # change this train_iters to 201: train_iters=201

    # Setup Attack Model
    # model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=args.device, lambda_=0.5) # lambda_ is used to weight the two objectives in Eq. (10) in the paper.
    # model = model.to(args.device)

    # """save the mettacked adj"""
    # model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    # modified_adj = sp.csr_matrix(model.modified_adj.cpu())

    # from deeprobust.graph.data import PrePtbDataset
    perturbed_data = PrePtbDataset(root='/tmp/', name=dataset, attack_method='meta', ptb_rate=0.2)
    modified_adj = perturbed_data.adj

    # Check the performance of GCN under directed attack without defense
    # flag = False
    # print('=== testing GNN on original(clean) graph ===')
    # print("acc_test:",test(adj, features, device, attention=flag))
    # print('=== testing GCN on perturbed graph ===')
    # print("acc_test:",test(modified_adj, features, device, attention=flag))

    # Use GNNGuard for defense
    flag = True
    print('=== testing GNN on original(clean) graph + GNNGuard ===')
    print("acc_test:",test(adj, features, device, attention=flag))
    print('=== testing GCN on perturbed graph + GNNGuard ===')
    print("acc_test:",test(modified_adj, features, device, attention=flag))

def test(adj, features, device, attention):
    accs = []
    for seed in tqdm(range(5)):

        classifier = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)

        classifier = classifier.to(device)

        classifier.fit(features, adj, labels, idx_train, train_iters=201,
                    idx_val=idx_val,
                    idx_test=idx_test,
                    verbose=True, attention=attention) # idx_val=idx_val, idx_test=idx_test , model_name=model_name
        classifier.eval()

        # classifier.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
        acc_test, output = classifier.test(idx_test)
        accs.append(acc_test.item())
    mean = np.mean(accs)
    std = np.std(accs)
    return {"mean": mean, "std": std}

def main_autogl(dataset, model_hp, adj, features, device):
    
    # Setup Surrogate model
    # surrogate = AutoGNNGuard_attack(
    #             num_features=pyg_data.num_node_features,
    #             num_classes=pyg_data.num_classes,
    #             device=args.device,
    #             init=False
    #         ).from_hyper_parameter(model_hp).model
    # surrogate = surrogate.to(args.device)
    # surrogate.fit(features, adj, labels, idx_train, train_iters=201)  # change this train_iters to 201: train_iters=201

    # Setup Attack Model
    # model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=args.device, lambda_=0.5) # lambda_ is used to weight the two objectives in Eq. (10) in the paper.
    # model = model.to(args.device)

    # """save the mettacked adj"""
    # model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    # modified_adj = sp.csr_matrix(model.modified_adj.cpu())

    # from deeprobust.graph.data import PrePtbDataset
    perturbed_data = PrePtbDataset(root='/tmp/', name=dataset, attack_method='meta', ptb_rate=0.2)
    modified_adj = perturbed_data.adj

    # Check the performance of GCN under directed attack without defense（AutoGL）
    # flag = False
    # print('=== testing GNN on original(clean) graph (AutoGL) ===')
    # print("acc_test:",test_autogl(adj, features, device, attention=flag))
    # print('=== testing GCN on perturbed graph (AutoGL) ===')
    # print("acc_test:",test_autogl(modified_adj, features, device, attention=flag))

    # Use GNNGuard for defense（AutoGL）
    flag = True
    print('=== testing GNN on original(clean) graph (AutoGL) + GNNGuard ===')
    print("acc_test:",test_autogl(adj, features, device, attention=flag))
    print('=== testing GCN on perturbed graph (AutoGL) + GNNGuard ===')
    print("acc_test:",test_autogl(modified_adj, features, device, attention=flag))

def test_autogl(adj, features, device, attention):
    ''
    """test on GCN """
    """model_name could be 'GCN', 'GAT', 'GIN','JK'  """
    accs = []
    for seed in tqdm(range(5)):
        # defense model
        gcn = AutoGNNGuard(
                    num_features=pyg_data.num_node_features,
                    num_classes=pyg_data.num_classes,
                    device=args.device,
                    init=False
                ).from_hyper_parameter(model_hp).model
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, idx_train, idx_val=idx_val,
                idx_test=idx_test,
                attention=attention, verbose=True, train_iters=81)
        gcn.eval()
        acc_test, output = gcn.test(idx_test=idx_test)
        accs.append(acc_test.item())
    mean = np.mean(accs)
    std = np.std(accs)
    return {"mean": mean, "std": std}

if __name__ == '__main__':

    model_hp = {
        "num_layers": 2,
        "hidden": [16],
        "dropout": 0.5,
        "act": "relu"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=14, help='Random seed.')
    # cora and citeseer are binary, pubmed has not binary features
    parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
    parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
    parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
    parser.add_argument('--DPlabel', type=int, default=9,  help='0-10')

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('cuda: %s' % args.device)

    args.dataset = "cora"
    args.modelname = "GCN"

    data = Dataset(root='/tmp/', name=args.dataset)
    pyg_data = Dpr2Pyg(data).process().to(args.device)
    pyg_data.num_classes = len(set(data.labels))

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    if scipy.sparse.issparse(features)==False:
        features = scipy.sparse.csr_matrix(features)

    perturbations = int(args.ptb_rate * (adj.sum()//2)) ###
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    # to CSR sparse
    adj, features = csr_matrix(adj), csr_matrix(features)

    """add undirected edges, orgn-arxiv is directed graph, we transfer it to undirected closely following 
    https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
    """
    adj = adj + adj.T
    adj[adj>1] = 1

    # main(args.dataset, adj, features, device=args.device)
    main_autogl(args.dataset, model_hp, adj, features, device=args.device)
