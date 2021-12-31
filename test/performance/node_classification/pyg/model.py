"""
Performance check of AutoGL model + PYG (trainer + dataset)
"""
import os
import pickle
import numpy as np
from tqdm import tqdm

os.environ["AUTOGL_BACKEND"] = "pyg"

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from autogl.module.model.pyg import AutoGCN, AutoGAT, AutoSAGE
from autogl.datasets import utils
from autogl.solver.utils import set_seed
from helper import get_encoder_decoder_hp
import logging

logging.basicConfig(level=logging.ERROR)

def test(model, data, mask):
    model.eval()

    if hasattr(model, 'cls_forward'):
        out = model.cls_forward(data)[mask]
    else:
        out = model(data)[mask]
    pred = out.max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc

def train(model, data, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    parameters = model.state_dict()
    best_acc = 0.
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        if hasattr(model, 'cls_forward'):
            output = model.cls_forward(data)
        else:
            output = model(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc = test(model, data, data.val_mask)
        if val_acc > best_acc:
            best_acc = val_acc
            parameters = pickle.dumps(model.state_dict())
            
    model.load_state_dict(pickle.loads(parameters))
    return model

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('pyg model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['gat', 'gcn', 'sage'], default='gat')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epoch', type=int, default=200)

    args = parser.parse_args()

    # seed = 100
    dataset = Planetoid(os.path.expanduser('~/.cache-autogl'), args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(args.device)

    accs = []

    model_hp, _ = get_encoder_decoder_hp(args.model)

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

        if args.model == 'gat':
            model = AutoGAT(
                num_features=dataset.num_node_features,
                num_classes=dataset.num_classes,
                device=args.device,
                init=False
            ).from_hyper_parameter(model_hp).model
        elif args.model == 'gcn':
            model = AutoGCN(
                num_features=dataset.num_node_features,
                num_classes=dataset.num_classes,
                device=args.device,
                init=False
            ).from_hyper_parameter(model_hp).model
        elif args.model == 'sage':
            model = AutoSAGE(
                num_features=dataset.num_node_features,
                num_classes=dataset.num_classes,
                device=args.device,
                init=False
            ).from_hyper_parameter(model_hp).model
        
        model.to(args.device)

        train(model, data, args)
        acc = test(model, data, data.test_mask)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
