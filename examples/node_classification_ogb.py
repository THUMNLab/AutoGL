import os
os.environ["AUTOGL_BACKEND"] = "pyg"
import autogl
from autogl.datasets import build_dataset_from_name
ogbn_dataset = build_dataset_from_name('ogbn-arXiv')
import torch
# temporal fix to add masks, only for one graph, not recommended
split_index = ogbn_dataset.get_idx_split()
num_nodes = ogbn_dataset[0].num_nodes
ogbn_dataset.data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
ogbn_dataset.data.train_mask[split_index['train']] = True
ogbn_dataset.data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
ogbn_dataset.data.val_mask[split_index['valid']] = True
ogbn_dataset.data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
ogbn_dataset.data.test_mask[split_index['test']] = True
ogbn_dataset.data.y = ogbn_dataset.data.y.squeeze(-1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from autogl.solver import AutoNodeClassifier
solver = AutoNodeClassifier(
    feature_module='deepgl',
    graph_models=['gcn', 'gat'],
    hpo_module='anneal',
    ensemble_module='voting',
    device=device
)

solver.fit(ogbn_dataset, time_limit=30)
solver.get_leaderboard().show()

from autogl.module.train import Acc
from autogl.solver.utils import get_graph_labels, get_graph_masks

predicted = solver.predict_proba()
label = get_graph_labels(ogbn_dataset[0])[get_graph_masks(ogbn_dataset[0], 'test')].cpu().numpy()
print('Test accuracy: ', Acc.evaluate(predicted, label))
