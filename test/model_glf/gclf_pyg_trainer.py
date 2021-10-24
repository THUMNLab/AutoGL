import sys
import os

os.environ["AUTOGL_BACKEND"] = "pyg"

sys.path.append('../../')

import random
import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from autogl.datasets import utils
from autogl.module.train import GraphClassificationFullTrainer
from autogl.solver.utils import set_seed
import logging

logging.basicConfig(level=logging.ERROR)

def fixed(**kwargs):
    return [{
        'parameterName': k,
        "type": "FIXED",
        "value": v
    } for k, v in kwargs.items()]

def graph_get_split(dataset, mask, is_loader=True, batch_size=128, num_workers=0):
    out = getattr(dataset, f'{mask}_split')
    if is_loader:
        out = DataLoader(out, batch_size, num_workers=num_workers)
    return out

utils.graph_get_split = graph_get_split

if __name__ == '__main__':

    # seed = 100
    dataset = TUDataset(os.path.expanduser('~/.pyg'), 'MUTAG')
    
    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(2021)
    random.shuffle(dataids)
    
    fold = int(len(dataset) * 0.1)
    train_index = dataids[:fold * 8]
    val_index = dataids[fold * 8: fold * 9]
    test_index = dataids[fold * 9: ]
    dataset.train_index = train_index
    dataset.val_index = val_index
    dataset.test_index = test_index
    dataset.train_split = dataset[dataset.train_index]
    dataset.val_split = dataset[dataset.val_index]
    dataset.test_split = dataset[dataset.test_index]

    labels = np.array([data.y.item() for data in dataset.test_split])

    accs = []
    from tqdm import tqdm
    for seed in tqdm(range(10)):
        set_seed(seed)

        trainer = GraphClassificationFullTrainer(
            model='gin',
            device='cuda:2',
            init=False,
            num_features=dataset[0].x.size(1),
            num_classes=max([data.y.item() for data in dataset]) + 1,
            loss='cross_entropy',
            feval=('acc')
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
                "hidden": [64,64,64,64],
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
