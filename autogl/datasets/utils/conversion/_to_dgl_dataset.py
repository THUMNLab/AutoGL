import dgl
import torch
import typing as _typing
from autogl.data import Dataset, InMemoryDataset
from autogl.data.graph import GeneralStaticGraph
from autogl.data.graph.utils import conversion


def general_static_graphs_to_dgl_dataset(
        general_static_graphs: _typing.Iterable[GeneralStaticGraph]
) -> Dataset[_typing.Union[dgl.DGLGraph, _typing.Tuple[dgl.DGLGraph, torch.Tensor]]]:
    def _transform(
            general_static_graph: GeneralStaticGraph
    ) -> _typing.Union[dgl.DGLGraph, _typing.Tuple[dgl.DGLGraph, torch.Tensor]]:
        if not isinstance(general_static_graph, GeneralStaticGraph):
            raise TypeError
        if 'label' in general_static_graph.data:
            label: _typing.Optional[torch.Tensor] = general_static_graph.data['label']
        elif 'y' in general_static_graph.data:
            label: _typing.Optional[torch.Tensor] = general_static_graph.data['y']
        else:
            label: _typing.Optional[torch.Tensor] = None
        if label is not None and isinstance(label, torch.Tensor) and torch.is_tensor(label):
            return conversion.general_static_graph_to_dgl_graph(general_static_graph), label
        else:
            return conversion.general_static_graph_to_dgl_graph(general_static_graph)

    if isinstance(general_static_graphs, Dataset):
        return InMemoryDataset(
            [_transform(g) for g in general_static_graphs],
            general_static_graphs.train_index,
            general_static_graphs.val_index,
            general_static_graphs.test_index
        )
    else:
        return InMemoryDataset([_transform(g) for g in general_static_graphs])
