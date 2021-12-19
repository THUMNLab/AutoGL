"""
Performance check of AutoGL trainer + DGL dataset
"""
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import urllib
import scipy
import dgl
import os.path as osp
os.environ["AUTOGL_BACKEND"] = "dgl"
import sys
import pickle
import random
from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from autogl.module.train import NodeClassificationHetTrainer
from autogl.solver.utils import set_seed
import logging
from scipy import sparse
from scipy import io as sio 
from pprint import pprint
logging.basicConfig(level=logging.ERROR)

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_acm(remove_self_loop):
    url = 'dataset/ACM3025.pkl'
    data_path = get_download_dir() + '/ACM3025.pkl'
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    labels, features = torch.from_numpy(data['label'].todense()).long(), \
                       torch.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g, subject_g]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    filename = 'ACM.mat'
    url = 'dataset/' + filename
    data_path = get_download_dir() + '/' + filename
    if osp.exists(data_path):
        print(f'Using existing file {filename}', file=sys.stderr)
    else:
        download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    hg.nodes['paper'].data['feat'] = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask

def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'ACMRaw':
        return load_acm_raw(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    #g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    G, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])
    data = {"G":G, "labels":labels, "num_classes":num_classes, "train_idx":train_idx,
             "val_idx":val_idx, "test_idx":test_idx, "train_mask":train_mask, 
             "val_mask":val_mask, "test_mask":test_mask}
    G = G.to(args['device'])

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    meta_paths=[['pa', 'ap'], ['pf', 'fp']]

    num_features = G.nodes['paper'].data['feat'].shape[1]
    num_classes = data["num_classes"]
    accs = []

    for seed in tqdm(range(args["repeat"])):
        set_seed(seed)
        if args["model"] == 'han':
            model_hp = {
                # hp from model
                "num_layers": 2,
                "hidden": [256], ##
                "heads": [8], ##
                "dropout": 0.2,
                "act": "gelu",
            }
        trainer = NodeClassificationHetTrainer(
            model=args["model"],
            G = G,
            meta_paths = meta_paths, #
            num_features=num_features,
            num_classes=num_classes,
            device=args["device"],
            init=False,
            feval=['acc'],
            loss="nll_loss",
        ).duplicate_from_hyper_parameter({
            "max_epoch": args["num_epochs"],
            "early_stopping_round": args["num_epochs"] + 1,
            "lr": args["lr"],
            "weight_decay": args["weight_decay"],
            **model_hp
        })

        trainer.train(data, G, False, train_mask)
        output = trainer.predict(G, test_idx)
        acc = (output == labels[test_idx]).float().mean().item()
        print(acc)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('dgl trainer_han')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    parser.add_argument('--repeat', type=int, default=10) # 50
    parser.add_argument('--model', type=str, choices=['gat', 'gcn', 'sage'], default='gat')
    parser.add_argument('--n_inp',   type=int, default=256)
    parser.add_argument('--clip',    type=int, default=1.0) 

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')

    args = parser.parse_args().__dict__
    args['dataset'] = 'ACMRaw' if not args['hetero'] else 'ACM'
    args['device'] = 'cuda:6' if torch.cuda.is_available() else 'cpu'
    args['num_epochs'] = 200
    args['model'] = "han"
    set_random_seed(args['seed'])
    print(args)
    
    main(args)
