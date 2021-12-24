"""
Performance check of AutoGL trainer + dataset
"""
import os
import numpy as np
from tqdm import tqdm

os.environ["AUTOGL_BACKEND"] = "dgl"

from autogl.datasets import build_dataset_from_name
from autogl.datasets.utils.conversion import to_dgl_dataset
from autogl.module.train import NodeClassificationFullTrainer
from autogl.solver.utils import set_seed
import logging
from helper import get_encoder_decoder_hp

logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('dgl trainer dataset')
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
    dataset = to_dgl_dataset(dataset)
    data = dataset[0].to(args.device)
    num_features = data.ndata['feat'].size(1)
    num_classes = data.ndata['label'].max().item() + 1
    label = data.ndata['label']
    test_mask = data.ndata['test_mask']

    accs = []

    model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

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
        acc = (output == label[test_mask]).float().mean().item()
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
