import pkg_resources
import autogl
from autogl.datasets import build_dataset_from_name
cora_dataset = build_dataset_from_name('cora')

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from autogl.solver import AutoNodeClassifier
solver = AutoNodeClassifier(
    feature_module='deepgl',
    graph_models=['gcn', 'gat'],
    hpo_module='anneal',
    ensemble_module='voting',
    device=device
)

solver.fit(cora_dataset, time_limit=30)
solver.get_leaderboard().show()

from autogl.module.train import Acc
from autogl.solver.utils import get_graph_labels, get_graph_masks

predicted = solver.predict_proba()
label = get_graph_labels(cora_dataset[0])[get_graph_masks(cora_dataset[0], 'test')].cpu().numpy()
print('Test accuracy: ', Acc.evaluate(predicted, label))
