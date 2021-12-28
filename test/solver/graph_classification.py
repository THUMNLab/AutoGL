from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoGraphClassifier
from autogl.datasets import utils

mutag = build_dataset_from_name("mutag")
utils.graph_random_splits(mutag, 0.8, 0.1)

solver = AutoGraphClassifier(
    graph_models=("gin",),
    hpo_module=None,
    device="auto"
)

solver.fit(mutag, evaluation_method=["acc"])
result = solver.predict(mutag)
print("Acc:", sum([d.data["y"].item() == r for d, r in zip(mutag.test_split, result)]) / len(result))
