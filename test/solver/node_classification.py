from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module.train import NodeClassificationFullTrainer

cora = build_dataset_from_name("cora")

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
print((result == cora[0].nodes.data["y"][cora[0].nodes.data["test_mask"]].cpu().numpy()).astype('float').mean())
