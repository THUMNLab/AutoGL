"""
Performance check of AutoGL trainer + PYG dataset
"""
import os
import numpy as np
from tqdm import tqdm

os.environ["AUTOGL_BACKEND"] = "pyg"

import torch
import torch.nn.functional as F
from autogl.module.feature import NormalizeFeatures
from autogl.module.train import NodeClassificationFullTrainer
from autogl.datasets import utils, build_dataset_from_name
from autogl.solver.utils import set_seed
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
            parameters = model.state_dict()
    
    model.load_state_dict(parameters)
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
    dataset = build_dataset_from_name('cora')
    dataset = NormalizeFeatures().fit_transform(dataset)
    dataset = utils.conversion.general_static_graphs_to_pyg_dataset(dataset)
    data = dataset[0].to(args.device)
    num_features = data.x.size(1)
    num_classes = max([label.item() for label in data.y]) + 1

    accs = []

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

        if args.model == 'gat':
            model_hp = {
                # hp from model
                "num_layers": 2,
                "hidden": [8],
                "heads": 8,
                "dropout": 0.6,
                "act": "elu",
            }
        elif args.model == 'gcn':
            model_hp = {
                "num_layers": 2,
                "hidden": [16],
                "dropout": 0.5,
                "act": "relu"
            }
        elif args.model == 'sage':
            model_hp = {
                "num_layers": 2,
                "hidden": [64],
                "dropout": 0.5,
                "act": "relu",
                "agg": "mean",
            }

        trainer = NodeClassificationFullTrainer(
            model=args.model,
            num_features=num_features,
            num_classes=num_classes,
            device=args.device,
            init=False,
            feval=['acc'],
            loss="nll_loss",
        ).duplicate_from_hyper_parameter({
            "max_epoch": args.epoch,
            "early_stopping_round": args.epoch + 1,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            **model_hp
        })

        trainer.train(dataset, False)
        output = trainer.predict(dataset, 'test')
        acc = (output == data.y[data.test_mask]).float().mean().item()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
