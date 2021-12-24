"""
Performance check of AutoGL model + DGL (trainer + dataset)
"""
import os
os.environ["AUTOGL_BACKEND"] = "dgl"

import numpy as np
from tqdm import tqdm

import torch
from autogl.module.model.dgl import AutoHAN
from helper import EarlyStopping
import argparse

import random

from sklearn.metrics import f1_score
from autogl.datasets import build_dataset_from_name

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

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def main(args):
    dataset = build_dataset_from_name("hetero-acm-han")
    node_type = dataset.schema["target_node_type"]
    g = dataset[0].to(args['device'])

    labels = g.nodes[node_type].data['label']
    num_classes = labels.max().item() + 1

    labels = labels.to(args['device'])
    train_mask = g.nodes[node_type].data['train_mask'].to(args['device'])
    val_mask = g.nodes[node_type].data['val_mask'].to(args['device'])
    test_mask = g.nodes[node_type].data['test_mask'].to(args['device'])

    model = AutoHAN(
        dataset=dataset,
        num_features=g.nodes[node_type].data['feat'].shape[1],
        num_classes=num_classes,
        device = args['device'],
        init=True
    ).model
    g = g.to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_loss, val_acc, _, _ = evaluate(model, g, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)
        
        if early_stop:
            break

    stopper.load_checkpoint(model)
    _, test_acc, _, _ = evaluate(model, g, labels, test_mask, loss_fcn)
    
    return test_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HAN dgl model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    #parser.add_argument('-ld', '--log-dir', type=str, default='results',
    #                    help='Dir for saving training results')
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
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args().__dict__
    args['device'] = 'cuda:{}'.format(args['device']) if args['device'] > -1 else 'cpu'
    args['num_epochs'] = 10
    set_random_seed(args['seed'])
    print(args)
    accs = []

    for seed in tqdm(range(50)):
        np.random.seed(seed)
        torch.manual_seed(seed)
        acc = main(args)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))

