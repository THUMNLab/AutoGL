import os.path as osp
import sys
sys.path.insert(0, '../')
import torch
from autogl.datasets import build_dataset_from_name
from autogl.module.train import LinkPredictionTrainer
import numpy as np
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score

dataset = build_dataset_from_name('cora')

print('len', len(dataset))
print('num_class', dataset.num_classes)
print('num_node_features', dataset.num_node_features)

a = []
for _ in range(10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0]

    data = data.to(device)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)

    clf = LinkPredictionTrainer(
        'gcn',
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        max_epoch=100,
        early_stopping_round=101,
        feval=['auc'],
        lr=0.01,
        weight_decay=0,
        lr_scheduler_type=None,
    )
    clf.train([data], keep_valid_result=True)
    print(clf.valid_score, end=',')
    y = clf.predict([data], 'test')
    y_ = y.cpu().numpy()
    # acc_ = y.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    # print(acc_, end=',')

    pos_edge_index = data[f'test_pos_edge_index']
    neg_edge_index = data[f'test_neg_edge_index']
    link_labels = clf.get_link_labels(pos_edge_index, neg_edge_index)
    label = link_labels.cpu().numpy()
    ret = roc_auc_score(label, y_)
    print(ret)
    a.append(ret)
print(np.mean(a), np.std(a))


