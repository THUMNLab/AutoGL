import dgl
import torch
import typing as _typing
from autogl.data import Dataset, InMemoryDataset
from autogl.data.graph import GeneralStaticGraph
from autogl.data.graph.utils import conversion


def to_dgl_dataset(
        dataset: _typing.Union[Dataset, _typing.Iterable[GeneralStaticGraph]]
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

    transformed_datalist: _typing.MutableSequence[
        _typing.Union[dgl.DGLGraph, _typing.Tuple[dgl.DGLGraph, torch.Tensor]]
    ] = []
    for item in dataset:
        if isinstance(item, GeneralStaticGraph):
            transformed_datalist.append(_transform(item))
        elif isinstance(item, dgl.DGLGraph):
            transformed_datalist.append(item)
        elif (
                isinstance(item, _typing.Sequence) and len(item) == 2 and
                isinstance(item[0], dgl.DGLGraph) and isinstance(item[1], torch.Tensor)
        ):
            transformed_datalist.append(tuple(item))
        else:
            raise ValueError(f"Illegal data item as {item}")

    return (
        InMemoryDataset(transformed_datalist, dataset.train_index, dataset.val_index, dataset.test_index, dataset.schema)
        if isinstance(dataset, InMemoryDataset)
        else InMemoryDataset(transformed_datalist)
    )
