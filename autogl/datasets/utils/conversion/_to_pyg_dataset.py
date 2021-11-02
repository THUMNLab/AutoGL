import typing as _typing
from autogl.data import Data, Dataset, InMemoryDataset
from autogl.data.graph import GeneralStaticGraph
from autogl.data.graph.utils import conversion


def general_static_graphs_to_pyg_dataset(
        graphs: _typing.Iterable[GeneralStaticGraph]
) -> Dataset[Data]:
    if isinstance(graphs, Dataset):
        return InMemoryDataset(
            [conversion.static_graph_to_pyg_data(g) for g in graphs],
            graphs.train_index, graphs.val_index, graphs.test_index
        )
    else:
        return InMemoryDataset(
            [conversion.static_graph_to_pyg_data(g) for g in graphs]
        )
