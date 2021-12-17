"""
Performance check of AutoGL trainer + DGL dataset
"""
import os
import numpy as np
from tqdm import tqdm

os.environ["AUTOGL_BACKEND"] = "dgl"

from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from autogl.module.train import NodeClassificationFullTrainer
from autogl.solver.utils import set_seed
import logging

logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('dgl trainer')
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
                "agg": "gcn",
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
            "trainer": {
                "max_epoch": args.epoch,
                "early_stopping_round": args.epoch + 1,
                "lr": args.lr,
                "weight_decay": args.weight_decay
            },
            "encoder": model_hp
        })

        trainer.train(dataset, False)
        output = trainer.predict(dataset, 'test')
        acc = (output == label[test_mask]).float().mean().item()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
