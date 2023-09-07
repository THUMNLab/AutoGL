import os
os.environ["AUTOGL_BACKEND"] = "dgl"

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module.train import NodeClassificationFullTrainer
from autogl.backend import DependentBackend

key = "y" if DependentBackend.is_pyg() else "label"

cora = build_dataset_from_name("cora")
# print(cora.__class__)

solver = AutoNodeClassifier(
    graph_models=("gcn",),
    default_trainer=NodeClassificationFullTrainer(
        decoder=None,
        init=False,
        max_epoch=200,
        early_stopping_round=201,
        lr=0.01,
        weight_decay=0.0,
    ),
    hpo_module=None,
    device="auto"
)

solver.fit(cora, evaluation_method=["acc"])
result = solver.predict(cora)
if DependentBackend.is_pyg():
    print((result == cora[0].y[cora[0].test_mask].cpu().numpy()).astype('float').mean())
else:
    print((result == cora[0].ndata['label'][cora[0].ndata['test_mask']].cpu().numpy()).astype('float').mean())
