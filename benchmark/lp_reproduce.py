"""
Used to reproduce the statistics from pyg
"""

import sys
sys.path.append('../')
import pickle
import torch
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from autogl.module.train import LinkPredictionTrainer
# Fix data split

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['gcn', 'sage', 'gat'], type=str, default='gcn', help='model to train')
parser.add_argument('--dataset', choices=['cora', 'citseer', 'pubmed'], type=str, default='cora', help='dataset to evaluate')
parser.add_argument('--times', type=int, default=10, help='time to rerun')

args = parser.parse_args()

DIM = 64
dataset = pickle.load(open(f'/DATA/DATANAS1/guancy/github/AutoGL/env/cache/{args.dataset}-edge.data', 'rb'))
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

HP = {
    'gcn' : {
        'num_layers': 3,
        'hidden': [128, DIM],
        'dropout': 0.0,
        'act': 'relu'
    },
    'gat' : {
        'num_layers': 3,
        'hidden': [16, DIM // 8],
        'dropout': 0.0,
        'act': 'relu',
        'heads': 8
    },
    'sage': {
        'num_layers': 3,
        'hidden': [128, DIM],
        'dropout': 0.0,
        'act': 'relu',
        'aggr': 'mean'
    }
}

scores = []

for t in range(args.times):

    trainer = LinkPredictionTrainer(
        args.model,
        num_features=dataset.num_features,
        lr=0.01,
        max_epoch=100,
        early_stopping_round=101,
        weight_decay=0,
        device='cuda',
        init=False,
        feval='auc',
        loss="binary_cross_entropy_with_logits",
    )

    trainer = trainer.duplicate_from_hyper_parameter(HP[args.model], restricted=False)
    trainer.train([data], keep_valid_result=True)
    y = trainer.predict([data], 'test')
    y_ = y.cpu().numpy()
    
    pos_edge_index = data[f'test_pos_edge_index']
    neg_edge_index = data[f'test_neg_edge_index']
    link_labels = trainer.get_link_labels(pos_edge_index, neg_edge_index)
    label = link_labels.cpu().numpy()
    test_auc = roc_auc_score(label, y_)
    scores.append(test_auc)
    print('time', t, test_auc)
print('mean', np.mean(scores), 'std', np.std(scores))
open('lp_reproduce.log', 'a').write('\t'.join([args.dataset, args.model, str(np.mean(scores)), str(np.std(scores)), '\n']))
