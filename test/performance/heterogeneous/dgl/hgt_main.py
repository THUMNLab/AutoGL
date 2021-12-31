"""
Performance check of AutoGL model + DGL (trainer + dataset)
"""
import os
os.environ["AUTOGL_BACKEND"] = "dgl"
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from autogl.module.model.dgl import AutoHGT, AutoHeteroRGCN
from autogl.solver.utils import set_seed
import numpy as np
import argparse
from autogl.datasets import build_dataset_from_name

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def test(model, graph, idx, labels):
    model.eval()
    pred = model(graph)[idx].max(1)[1]
    acc = (pred == labels[idx]).float().mean()
    return acc.item()

def train(model, G, args, train_mask, labels):
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits = model(G)
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask].to(args.device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='hetero dgl model')

    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--n_inp',   type=int, default=256)
    parser.add_argument('--clip',    type=int, default=1.0) 
    parser.add_argument('--max_lr',  type=float, default=1e-3) 
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, choices=['hgt', 'HeteroRGCN'], default='hgt')

    args = parser.parse_args()

    torch.manual_seed(0)

    dataset = build_dataset_from_name("hetero-acm-hgt")
    G = dataset[0].to(args.device)
    print(G)

    target_field = dataset.schema["target_node_type"]
    labels = G.nodes[target_field].data["label"].to(args.device)

    train_mask = G.nodes[target_field].data["train_mask"].nonzero().flatten()
    val_mask = G.nodes[target_field].data["val_mask"].nonzero().flatten()
    test_mask = G.nodes[target_field].data["test_mask"].nonzero().flatten()

    num_features = G.nodes[target_field].data["feat"].size(1)
    num_classes = labels.max().item() + 1
    accs = []

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        if args.model=='hgt':
            model = AutoHGT(dataset=dataset,
                num_features=num_features,
                num_classes=num_classes,
                device=args.device,
                init=False
            ).from_hyper_parameter({
                "num_layers": 2,
                "hidden": [256,256],
                "heads": 4,
                "dropout": 0.2,
                "act": "gelu",
                "use_norm": True,
            }).model
        elif args.model=='HeteroRGCN':
            model = AutoHeteroRGCN(dataset=dataset,
                num_features=num_features,
                num_classes=num_classes,
                device=args.device,
                init=False
            ).from_hyper_parameter({
                "num_layers": 2,
                "hidden": [256],
                "act": "leaky_relu",
            }).model

        model.to(args.device)

        train(model, G, args, train_mask, labels)
        accs.append(test(model, G, test_mask, labels))
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
