import os
import random
import torch
import torch.nn as nn
import numpy as np

from autogl.module.train.ssl import GraphCLSemisupervisedTrainer
from autogl.datasets import build_dataset_from_name, utils
from autogl.datasets.utils.conversion import to_pyg_dataset as convert_dataset
from autogl.module.model.encoders.base_encoder import AutoHomogeneousEncoderMaintainer
from autogl.module.model.decoders import BaseDecoderMaintainer
from autogl.solver.utils import set_seed

def fixed(**kwargs):
    return [{
        'parameterName': k,
        'type': "FIXED",
        'value': v
    } for k, v in kwargs.items()]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('ssl pyg trainer')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'NCI1', 'PROTEINS', 'PTC_MR'], default='PROTEINS')
    parser.add_argument('--dataset_seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=50)
    # parser.add_argument('--model', type=str, choices=['gin', 'gat', 'gcn', 'sage'], default='gin')
    parser.add_argument('--encoder', type=str, choices=['gin', 'gcn'], default='gcn')
    parser.add_argument('--p_lr', type=float, default=0.0001)
    parser.add_argument('--p_weight_decay', type=float, default=0)
    parser.add_argument('--p_epoch', type=int, default=100)
    parser.add_argument('--f_lr', type=float, default=0.001)
    parser.add_argument('--f_weight_decay', type=float, default=0)
    parser.add_argument('--f_epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=100)

    args=parser.parse_args()

    # split dataset 
    dataset = build_dataset_from_name(args.dataset)
    dataset = convert_dataset(dataset)
    utils.graph_random_splits(dataset, train_ratio=0.1, val_ratio=0.1, seed=2022)

    accs = [[],[],[]]

    encoder_hp = {
        "num_layers": 5,
        "hidden": [32, 64, 64, 64],
        "dropout": 0.5,
        "act": "elu",
        "eps": "true"
    }
    decoder_hp = {
        "hidden": 32,
        "act": "tanh",
        "dropout": 0.35
    }
    prediction_head = {
        "hidden": 128,
        "act": "relu",
        "dropout": 0.4
    }
    from tqdm import tqdm
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        trainer = GraphCLSemisupervisedTrainer(
            model=(args.encoder, 'sumpoolmlp'),
            prediction_head='sumpoolmlp',
            views_fn=['random2', 'random2'],
            device=args.device,
            num_features=dataset[0].x.size(1),
            num_classes=max([data.y.item() for data in dataset]) + 1,
            batch_size=args.batch_size,
            # p_lr=args.p_lr,
            # p_weight_decay=args.p_weight_decay,
            # p_epoch=args.p_epoch,
            # f_lr=args.f_lr,
            # f_weight_decay=args.f_weight_decay,
            # f_epoch=args.f_epoch,
            z_dim=128,
            init=False
        )
        trainer.initialize()
        trainer = trainer.duplicate_from_hyper_parameter(
            {
                'trainer': {
                    'batch_size': args.batch_size,
                    'p_lr': args.p_lr,
                    'p_weight_decay': args.p_weight_decay,
                    'p_epoch': args.p_epoch,
                    'p_early_stopping_round': args.p_epoch + 1,
                    'f_lr': args.f_lr,
                    'f_weight_decay': args.f_weight_decay,
                    'f_epoch': args.f_epoch,
                    'f_early_stopping_round': args.f_epoch + 1,
                },
                "encoder": encoder_hp,
                "decoder": decoder_hp,
                "prediction_head": prediction_head
            }
        )
        trainer.train(dataset, False)
        out = trainer.predict(dataset, 'test').detach().cpu().numpy()
        train_result = trainer.evaluate(dataset, 'train')
        valid_result = trainer.evaluate(dataset, 'val')
        test_result = trainer.evaluate(dataset, 'test')
        print(f"{train_result[0]} - {valid_result[0]} - {test_result[0]}")
        accs[0].append(train_result[0])
        accs[1].append(valid_result[0])
        accs[2].append(test_result[0])
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs[0]), np.std(accs[0])))
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs[1]), np.std(accs[1])))
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs[2]), np.std(accs[2])))