
import autogl
from autogl.datasets import build_dataset_from_name
cora_dataset = build_dataset_from_name('cora', path = '/home/qinyj/AGL/')

import torch
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from autogl.solver import AutoNodeClassifier
solver = AutoNodeClassifier(
    feature_module='deepgl',
    graph_models=['gcn', 'gat'],
    hpo_module='anneal',
    ensemble_module='voting',
    device=device
)

solver.fit(cora_dataset, time_limit=3600)
solver.get_leaderboard().show()

from autogl.module.train import Acc
predicted = solver.predict_proba()
print('Test accuracy: ', Acc.evaluate(predicted, 
    cora_dataset.data.y[cora_dataset.data.test_mask].cpu().numpy()))
