"""
Performance check of AutoGL trainer + PYG dataset
"""
import os
import torch
os.environ["AUTOGL_BACKEND"] = "pyg"

import numpy as np
from tqdm import tqdm

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from autogl.module.train import NodeClassificationFullTrainer
from autogl.datasets import utils
from autogl.solver.utils import set_seed
from helper import get_encoder_decoder_hp
from autogl.datasets import build_dataset_from_name
from autogl.datasets.utils import random_splits_mask
from autogl.datasets.utils.conversion import to_pyg_dataset as convert_dataset
import logging

logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('pyg model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed', 'amazon_photo', 'amazon_photo'], default='cora')
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--model', type=str, choices=['gat', 'gcn', 'sage', 'gin'], default='gcn')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=300)

    args = parser.parse_args()

    transform = T.RandomNodeSplit(
        split='test_rest',
        num_train_per_class=20,
        num_val=240
    )
    dataset = build_dataset_from_name(args.dataset, split='public')
    data = dataset[0]
    num_features = dataset[0].x.shape[1]
    num_classes = len(np.unique(dataset[0].y.cpu().numpy()))
    # dataset = [data]

    accs = []

    model_hp, decoder_hp = get_encoder_decoder_hp(args.model, decoupled=True)
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

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
                "weight_decay": args.weight_decay,
            },
            "encoder": model_hp,
            "decoder": decoder_hp
        })

        trainer.train(dataset, False)
        output = trainer.predict(dataset, 'test')
        output = output.cpu().detach()
        acc = (output == dataset[0].y[dataset[0].test_mask]).float().mean().item()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))

    dataset = build_dataset_from_name('amazon_photo')
    data = dataset[0].to(args.device)
    data = transform(data)

    num_features = dataset[0].x.shape[1]
    num_classes = len(np.unique(dataset[0].y.cpu().numpy()))

    dataset = [data]

    accs = []

    model_hp, decoder_hp = get_encoder_decoder_hp(args.model, decoupled=True)
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

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
                "weight_decay": args.weight_decay,
            },
            "encoder": model_hp,
            "decoder": decoder_hp
        })

        trainer.train(dataset, False)
        output = trainer.predict(dataset, 'test')
        acc = (output == data.y[data.test_mask]).float().mean().item()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
