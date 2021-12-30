"""
Performance check of AutoGL (trainer + model) + DGL dataset
"""

import os
os.environ["AUTOGL_BACKEND"] = "dgl"

import torch
import random
import numpy as np
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader

from autogl.datasets import utils
from autogl.module.train import GraphClassificationFullTrainer
from autogl.solver.utils import set_seed
from helper import get_encoder_decoder_hp
import logging

logging.basicConfig(level=logging.ERROR)

class DatasetAbstraction():
    def __init__(self, graphs, labels):
        for g in graphs:
            g.ndata['feat'] = g.ndata['attr']
        self.graphs, self.labels = [], []
        for g, l in zip(graphs, labels):
            self.graphs.append(g)
            self.labels.append(l)
        self.gclasses = max(self.labels).item() + 1
        self.graph = self.graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif isinstance(idx, torch.BoolTensor):
            idx = [i for i in range(len(idx)) if idx[i]]
        elif isinstance(idx, torch.Tensor) and idx.unique()[0].sum().item() == 1:
            idx = [i for i in range(len(idx)) if idx[i]]
        return DatasetAbstraction([self.graphs[i] for i in idx], [self.labels[i] for i in idx])

    @classmethod
    def build_from_train_val(cls, train, val, test=None):
        dataset = cls(train.graphs + val.graphs, train.labels + val.labels)
        dataset.train_index = list(range(len(train)))
        dataset.val_index = list(range(len(train), len(train) + len(val)))
        if test is not None:
            dataset.test_index = list(range(len(train) + len(val), len(train) + len(val) + len(test)))
        dataset.train_split = train
        dataset.val_split = val
        if test is not None:
            dataset.test_split = test
        return dataset


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('dgl trainer')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'NCI1', 'PROTEINS', 'PTC', 'REDDITBINARY', 'REDDITMULTI5K'], default='MUTAG')
    parser.add_argument('--dataset_seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['gin', 'gat', 'gcn', 'sage', 'topk'], default='gin')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)

    args = parser.parse_args()

    # seed = 100
    # dataset = build_dataset_from_name('mutag')
    dataset_ = GINDataset(args.dataset, False)
    dataset = DatasetAbstraction([g[0] for g in dataset_], [g[1] for g in dataset_])

    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(args.dataset_seed)
    random.shuffle(dataids)
    
    fold = int(len(dataset) * 0.1)
    train_dataset = dataset[dataids[:fold * 8]]
    val_dataset = dataset[dataids[fold * 8: fold * 9]]
    test_dataset = dataset[dataids[fold * 9: ]]

    dataset = DatasetAbstraction.build_from_train_val(train_dataset, val_dataset, test_dataset)

    labels = np.array([x.item() for x in test_dataset.labels])

    accs = []

    if args.model == "gin":
        decoder = "JKSumPoolMLP"
    else:
        decoder = "sumpoolmlp"

    model_hp, decoder_hp = get_encoder_decoder_hp(args.model, decoder=decoder)

    from tqdm import tqdm
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

        trainer = GraphClassificationFullTrainer(
            model=(args.model, decoder),
            device=args.device,
            init=False,
            num_features=dataset.graph[0].ndata['feat'].size(1),
            num_classes=dataset.gclasses,
            loss='cross_entropy',
            feval = ('acc',)
        ).duplicate_from_hyper_parameter({
            "trainer": {
                # hp from trainer
                "max_epoch": args.epoch,
                "batch_size": args.batch_size, 
                "early_stopping_round": args.epoch + 1, 
                "lr": args.lr, 
                "weight_decay": 0
                },
            "encoder": model_hp,
            "decoder": decoder_hp
        })

        trainer.train(dataset, False)
        out = trainer.predict(dataset, 'test').detach().cpu().numpy()
        acc = (out == labels).astype('float').mean()
        accs.append(acc)
    print('{:.2f} ~ {:.2f}'.format(np.mean(accs) * 100, np.std(accs) * 100))
