"""
Performance check of AutoGL trainer + PYG dataset
"""
import os
import numpy as np
from tqdm import tqdm

os.environ["AUTOGL_BACKEND"] = "pyg"

from autogl.module.feature import NormalizeFeatures
from autogl.module.train import NodeClassificationFullTrainer
from autogl.datasets import utils, build_dataset_from_name
from autogl.solver.utils import set_seed
from helper import get_encoder_decoder_hp
import logging

logging.basicConfig(level=logging.ERROR)

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
    dataset = NormalizeFeatures().fit_transform(dataset)
    dataset = utils.conversion.to_pyg_dataset(dataset)
    data = dataset[0].to(args.device)
    num_features = data.x.size(1)
    num_classes = max([label.item() for label in data.y]) + 1

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
