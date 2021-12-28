"""
Performance check of AutoGL trainer + PYG dataset
"""
import os
import numpy as np
from tqdm import tqdm

os.environ["AUTOGL_BACKEND"] = "pyg"

from autogl.module.feature import NormalizeFeatures
from autogl.solver import AutoNodeClassifier
from autogl.datasets import utils, build_dataset_from_name
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
    parser = argparse.ArgumentParser('pyg model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['gat', 'gcn', 'sage', 'gin'], default='gat')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epoch', type=int, default=200)

    args = parser.parse_args()

    # seed = 100
    dataset = build_dataset_from_name(args.dataset.lower())
    label = dataset[0].nodes.data['y'][dataset[0].nodes.data['test_mask']].numpy()
    accs = []

    model_hp, decoder_hp = get_encoder_decoder_hp(args.model, decoupled=True)

    for seed in tqdm(range(args.repeat)):

        solver = AutoNodeClassifier(
            feature_module='NormalizeFeatures',
            graph_models=(args.model,),
            ensemble_module=None,
            max_evals=1,
            hpo_module='random',
            trainer_hp_space=fixed(**{
                "max_epoch": args.epoch,
                "early_stopping_round": args.epoch + 1,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }),
            model_hp_spaces=[{"encoder": fixed(**model_hp), "decoder": fixed(**decoder_hp)}]
        )

        solver.fit(dataset, seed=seed)
        output = solver.predict(dataset)
        acc = (output == label).astype('float').mean()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
