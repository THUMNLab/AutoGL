from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoGraphClassifier
from autogl.datasets import utils
from autogl.backend import DependentBackend

mutag = build_dataset_from_name("mutag")
utils.graph_random_splits(mutag, 0.8, 0.1)

solver = AutoGraphClassifier(
    graph_models=("gin",),
    hpo_module=None,
    device="auto"
)

solver.fit(mutag, evaluation_method=["acc"])
result = solver.predict(mutag)
if DependentBackend.is_dgl():
    print("Acc:", sum([d[1].item() == r for d, r in zip(mutag.test_split, result)]) / len(result))
else:
    print("Acc:", sum([d.y.item() == r for d, r in zip(mutag.test_split, result)]) / len(result))
