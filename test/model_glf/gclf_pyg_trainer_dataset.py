import sys

sys.path.append('../../')

import random
import numpy as np

from autogl.datasets import build_dataset_from_name, utils
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

if __name__ == '__main__':

    # seed = 100
    dataset = build_dataset_from_name('mutag')
    
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

    labels = np.array([data.data['y'].item() for data in dataset.test_split])

    dataset = utils.conversion.general_static_graphs_to_pyg_dataset(dataset)

    accs = []
    from tqdm import tqdm
    for seed in tqdm(range(50)):
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
