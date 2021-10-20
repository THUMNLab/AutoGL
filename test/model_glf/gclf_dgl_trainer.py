import sys

sys.path.append('../../')

import torch
import random
import numpy as np
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from gin_helper import Parser, GINDataLoader

from autogl.solver import AutoGraphClassifier
from autogl.datasets import utils, build_dataset_from_name
from autogl.module.train import GraphClassificationFullTrainer
from autogl.module.model.dgl.gin import AutoGIN
from autogl.solver.utils import set_seed
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

def graph_get_split(
    dataset, mask="train", is_loader=True, batch_size=128, num_workers=0
):
    assert hasattr(
        dataset, "%s_split" % (mask)
    ), "Given dataset do not have %s split" % (mask)
    if is_loader:
        return GraphDataLoader(
            getattr(dataset, "%s_split" % (mask)),
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        return getattr(dataset, "%s_split" % (mask))


utils.graph_get_split = graph_get_split

def fixed(**kwargs):
    return [{
        'parameterName': k,
        "type": "FIXED",
        "value": v
    } for k, v in kwargs.items()]

if __name__ == '__main__':

    # seed = 100
    # dataset = build_dataset_from_name('mutag')
    dataset_ = GINDataset('MUTAG', False)
    dataset = DatasetAbstraction([g[0] for g in dataset_], [g[1] for g in dataset_])

    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(2021)
    random.shuffle(dataids)
    
    fold = int(len(dataset) * 0.1)
    train_dataset = dataset[dataids[:fold * 8]]
    val_dataset = dataset[dataids[fold * 8: fold * 9]]
    test_dataset = dataset[dataids[fold * 9: ]]

    dataset = DatasetAbstraction.build_from_train_val(train_dataset, val_dataset, test_dataset)

    labels = np.array([x.item() for x in test_dataset.labels])

    accs = []
    from tqdm import tqdm
    for seed in tqdm(range(10)):
        set_seed(seed)

        trainer = GraphClassificationFullTrainer(
            model='gin',
            device='cuda:1',
            init=False,
            num_features=dataset.graph[0].ndata['feat'].size(1),
            num_classes=dataset.gclasses,
            loss='cross_entropy'
        ).duplicate_from_hyper_parameter(
            {
                # hp from trainer
                "max_epoch": 100,
                "batch_size": 32, 
                "early_stopping_round": 101, 
                "lr": 0.0001, 
                "weight_decay": 0,

                # hp from model
                "num_layers": 5,
                "hidden": [64],
                "dropout": 0.5,
                "act": "relu",
                "eps": "False",
                "mlp_layers": 2,
                "neighbor_pooling_type": "sum",
                "graph_pooling_type": "sum"
            }
        )

        trainer.train(dataset, False)
        out = trainer.predict(dataset, 'test').detach().cpu().numpy()
        acc = (out == labels).astype('float').mean()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
