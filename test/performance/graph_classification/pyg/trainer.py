"""
Performance check of AutoGL trainer + PYG dataset
"""

import os

os.environ["AUTOGL_BACKEND"] = "pyg"

import random
import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from autogl.datasets import utils
from autogl.module.train import GraphClassificationFullTrainer
from autogl.solver.utils import set_seed
from helper import get_encoder_decoder_hp
import logging

logging.basicConfig(level=logging.ERROR)

def fixed(**kwargs):
    return [{
        'parameterName': k,
        "type": "FIXED",
        "value": v
    } for k, v in kwargs.items()]

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('pyg trainer')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'NCI1', 'PROTEINS', 'PTC', 'REDDITBINARY', 'REDDITMULTI5K'], default='MUTAG')
    parser.add_argument('--dataset_seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['gin', 'gat', 'gcn', 'sage'], default='gin')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)

    args = parser.parse_args()

    # seed = 100
    dataset = TUDataset(os.path.expanduser('~/.pyg'), args.dataset)
    
    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(args.dataset_seed)
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

    model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

    from tqdm import tqdm
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

        trainer = GraphClassificationFullTrainer(
            model=args.model,
            device=args.device,
            init=False,
            num_features=dataset[0].x.size(1),
            num_classes=max([data.y.item() for data in dataset]) + 1,
            num_graph_features=0,
            loss='nll_loss',
            feval=('acc')
        ).duplicate_from_hyper_parameter(
            {
                "trainer": {
                    # hp from trainer
                    "max_epoch": args.epoch,
                    "batch_size": args.batch_size, 
                    "early_stopping_round": args.epoch + 1, 
                    "lr": args.lr, 
                    "weight_decay": 0,
                },
                "encoder": model_hp,
                "decoder": decoder_hp
            }
        )

        trainer.train(dataset, False)
        out = trainer.predict(dataset, 'test').detach().cpu().numpy()
        acc = (out == labels).astype('float').mean()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
