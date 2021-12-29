"""
Performance check of AutoGL trainer + DGL dataset
"""
import os
import numpy as np
from tqdm import tqdm
import torch
os.environ["AUTOGL_BACKEND"] = "dgl"
import random
from autogl.module.train import NodeClassificationHetTrainer
from autogl.solver.utils import set_seed
import logging
from autogl.datasets import build_dataset_from_name

logging.basicConfig(level=logging.ERROR)

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main(args):
    dataset = build_dataset_from_name("hetero-acm-han")
    field = dataset.schema["target_node_type"]
    g = dataset[0].to(args['device'])

    labels = g.nodes[field].data['label']
    num_classes = labels.max().item()+1

    #features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = g.nodes[field].data['train_mask'].to(args['device'])
    val_mask = g.nodes[field].data['val_mask'].to(args['device'])
    test_mask = g.nodes[field].data['test_mask'].to(args['device'])

    num_features = g.nodes[field].data['feat'].shape[1]
    num_classes = labels.max().item()+1
    accs = []

    for seed in tqdm(range(args["repeat"])):
        set_seed(seed)
        if args["model"] == 'han':
            model_hp = {
                "num_layers": 2,
                "hidden": [256], ##
                "heads": [8], ##
                "dropout": 0.2,
                "act": "gelu",
            }
        trainer = NodeClassificationHetTrainer(
            model=args["model"],
            dataset = dataset,
            num_features=num_features,
            num_classes=num_classes,
            device=args["device"],
            init=False,
            feval=['acc'],
            loss="cross_entropy",
            optimizer=torch.optim.AdamW,
        ).duplicate_from_hyper_parameter({
            "trainer": {
                "max_epoch": args["num_epochs"],
                "early_stopping_round": args["num_epochs"] + 1,
                "lr": args["lr"],
                "weight_decay": args["weight_decay"],
            },
            "encoder": model_hp
        })

        trainer.train(dataset, False)
        acc = trainer.evaluate(dataset, "test", "acc")
        print(acc)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('dgl trainer_han')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--repeat', type=int, default=10) # 50
    parser.add_argument('--model', type=str, choices=['han'], default='han')
    parser.add_argument('--n_inp',   type=int, default=256)
    parser.add_argument('--clip',    type=int, default=1.0) 

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')

    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    
    main(args)
