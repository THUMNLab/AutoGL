"""
Performance check of AutoGL trainer + DGL dataset
"""
import os
import numpy as np
from tqdm import tqdm
import torch
os.environ["AUTOGL_BACKEND"] = "dgl"

from autogl.module.train import NodeClassificationHetTrainer
from autogl.solver.utils import set_seed
import logging
import argparse
from autogl.datasets import build_dataset_from_name

logging.basicConfig(level=logging.ERROR)

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def main(args):
    
    dataset = build_dataset_from_name("hetero-acm-hgt")
    G = dataset[0].to(args.device)
    field = dataset.schema["target_node_type"]
    print(G)

    num_features = 256
    labels = G.nodes[field].data['label'].to(args.device)
    num_classes = labels.max().item()+1
    test_mask = G.nodes[field].data['test_mask'].to(args.device)
    accs = []

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        if args.model == 'hgt':
            model_hp = {
                # hp from model
                "num_layers": 2,
                "hidden": [256,256,256],
                "heads": 4,
                "dropout": 0.2,
                "act": "gelu",
                "use_norm": True,
            }
        elif args.model=='HeteroRGCN':
            model_hp = {
                # hp from model
                "num_layers": 2,
                "hidden": [256],
                "heads": 4,
                "dropout": 0.2,
                "act": "leaky_relu",
            }
        
        trainer = NodeClassificationHetTrainer(
            model=args.model,
            dataset = dataset,
            num_features=num_features,
            num_classes=num_classes,
            device=args.device,
            init=False,
            feval=['acc'],
            loss="cross_entropy",
            optimizer=torch.optim.AdamW
        ).duplicate_from_hyper_parameter({
            "trainer": {
                "max_epoch": args.epoch,
                "early_stopping_round": args.epoch + 1,
                "lr": args.max_lr,
                "weight_decay": args.weight_decay,
            },
            "encoder": model_hp
        })

        trainer.train(dataset, False)
        acc = trainer.evaluate(dataset, test_mask, "acc")
        print(acc)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('dgl trainer_hgt')
 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['hgt', 'HeteroRGCN'], default='hgt')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--max_lr',  type=float, default=1e-3) 
    parser.add_argument('--n_inp',   type=int, default=256)
    parser.add_argument('--clip',    type=int, default=1.0) 
    args = parser.parse_args()

    torch.manual_seed(0)
    print(args)

    main(args)
