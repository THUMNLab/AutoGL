"""
Performance check of AutoGL Solver
"""

import os
os.environ["AUTOGL_BACKEND"] = "pyg"

import random
import numpy as np
from tqdm import tqdm

from autogl.solver import AutoGraphClassifier
from autogl.datasets import build_dataset_from_name, utils
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
    parser = argparse.ArgumentParser('pyg solver')
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
    dataset = build_dataset_from_name(args.dataset.lower())
    
    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(args.dataset_seed)
    random.shuffle(dataids)
    
    fold = int(len(dataset) * 0.1)
    dataset.train_index = dataids[:fold * 8]
    dataset.val_index = dataids[fold * 8: fold * 9]
    dataset.test_index = dataids[fold * 9: ]
    dataset.loss = 'nll_loss'

    labels = np.array([x.data['y'].item() for x in dataset.test_split])

    model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

    accs = []
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        solver = AutoGraphClassifier(
            feature_module=None,
            graph_models=[args.model],
            hpo_module='random',
            ensemble_module=None,
            device=args.device, max_evals=1,
            trainer_hp_space = fixed(
                **{
                    # hp from trainer
                    "max_epoch": args.epoch,
                    "batch_size": args.batch_size, 
                    "early_stopping_round": args.epoch + 1, 
                    "lr": args.lr, 
                    "weight_decay": 0,
                }
            ),
            model_hp_spaces=[{"encoder": fixed(**model_hp), "decoder": fixed(**decoder_hp)}]
        )
        solver.fit(dataset, evaluation_method=['acc'])
        out = solver.predict(dataset, mask='test')
        acc = (out == labels).astype('float').mean()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
