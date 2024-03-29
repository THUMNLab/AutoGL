"""
Performance check of AutoGL (trainer + model + dataset)
"""

import os
os.environ["AUTOGL_BACKEND"] = "dgl"

import random
import numpy as np

from autogl.datasets import build_dataset_from_name, utils
from autogl.module.train import GraphClassificationFullTrainer
from autogl.solver.utils import set_seed
from helper import get_encoder_decoder_hp
import logging

logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('dgl dataset')
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
    dataset = build_dataset_from_name(args.dataset.lower())
    
    # 1. split dataset [fix split]
    dataids = list(range(len(dataset)))
    random.seed(args.dataset_seed)
    random.shuffle(dataids)

    utils.graph_random_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=23)

    labels = np.array([x[1].item() for x in dataset.test_split])

    accs = []

    if args.model == "gin":
        decoder = "JKSumPoolMLP"
    else:
        decoder = "sumpoolmlp"

    model_hp, decoder_hp = get_encoder_decoder_hp(args.model, decoder)

    from tqdm import tqdm
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

        trainer = GraphClassificationFullTrainer(
            model=(args.model, decoder),
            device=args.device,
            init=False,
            num_features=dataset[0][0].ndata['attr'].size(1),
            num_classes=max([graph[1].item() for graph in dataset]) + 1,
            loss='cross_entropy',
            feval=('acc'),
        ).duplicate_from_hyper_parameter(
            {
                "trainer": {
                    # hp from trainer
                    "max_epoch": args.epoch,
                    "batch_size": args.batch_size, 
                    "early_stopping_round": args.epoch + 1, 
                    "lr": args.lr, 
                    "weight_decay": 0
                },
                "encoder": model_hp,
                "decoder": decoder_hp,
            }
        )

        trainer.train(dataset, False)
        out = trainer.predict(dataset, 'test').detach().cpu().numpy()
        acc = (out == labels).astype('float').mean()
        accs.append(acc)
    print('{:.2f} ~ {:.2f}'.format(np.mean(accs) * 100, np.std(accs) * 100))
