from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoLinkPredictor
from autogl.datasets.utils.conversion._to_pyg_dataset import general_static_graphs_to_pyg_dataset
from torch_geometric.utils import train_test_split_edges

cora = build_dataset_from_name("cora")
cora = general_static_graphs_to_pyg_dataset(cora)
cora[0] = train_test_split_edges(cora[0])

solver = AutoLinkPredictor(
    graph_models=("gin",),
    hpo_module=None,
    device="auto"
)

solver.fit(cora, evaluation_method=["acc"])
result = solver.predict(cora)

print(result)
