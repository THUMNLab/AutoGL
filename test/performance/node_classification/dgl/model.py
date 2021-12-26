"""
Performance check of AutoGL model + DGL (trainer + dataset)
"""
import numpy as np
from tqdm import tqdm
import pickle

import torch
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from autogl.module.model.dgl import AutoGCN, AutoGAT, AutoSAGE
from autogl.solver.utils import set_seed
import logging

logging.basicConfig(level=logging.ERROR)

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
    parser = argparse.ArgumentParser('dgl model')
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

    # graph = dgl.remove_self_loop(graph)
    # graph = dgl.add_self_loop(graph)

    label = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    num_features = graph.ndata['feat'].size(1)
    num_classes = dataset.num_classes
    accs = []

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

        if args.model == 'gat':
            model = AutoGAT(
                input_dimension=num_features,
                output_dimension=num_classes,
                device=args.device
            ).from_hyper_parameter({
                # hp from model
                "num_layers": 2,
                "hidden": [8],
                "heads": 8,
                "feat_drop": 0.6,
                "dropout": 0.6,
                "act": "relu",
            }).model
        elif args.model == 'gcn':
            model = AutoGCN(
                input_dimension=num_features,
                output_dimension=num_classes,
                device=args.device
            ).from_hyper_parameter({
                "num_layers": 2,
                "hidden": [16],
                "dropout": 0.5,
                "act": "relu"
            }).model
        elif args.model == 'sage':
            model = AutoSAGE(
                input_dimension=num_features,
                output_dimension=num_classes,
                device=args.device
            ).from_hyper_parameter({
                "num_layers": 2,
                "hidden": [64],
                "dropout": 0.5,
                "act": "relu",
                "agg": "gcn",
            }).model
        
        model.to(args.device)

        train(model, graph, args, label, train_mask, val_mask)
        acc = test(model, graph, test_mask, label)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
