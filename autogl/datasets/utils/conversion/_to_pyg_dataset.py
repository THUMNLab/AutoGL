import typing as _typing
import torch
import torch_geometric
from autogl.data import Dataset, InMemoryDataset
from autogl.data.graph import GeneralStaticGraph
from autogl.data.graph.utils import conversion


def to_pyg_dataset(
        dataset: _typing.Union[Dataset, _typing.Iterable[GeneralStaticGraph]]
) -> Dataset[torch_geometric.data.Data]:
    transformed_datalist: _typing.MutableSequence[torch_geometric.data.Data] = []
    for item in dataset:
        if isinstance(item, torch_geometric.data.Data):
            transformed_datalist.append(item)
        elif isinstance(item, GeneralStaticGraph):
            transformed_datalist.append(conversion.static_graph_to_pyg_data(item))
        elif (
                isinstance(item, _typing.Mapping) and
                all([
                    (isinstance(k, str) and isinstance(v, torch.Tensor))
                    for k, v in item.items()
                ])
        ):
            transformed_datalist.append(torch_geometric.data.Data(**item))
        else:
            raise NotImplementedError(
                f"Unsupported data item {type(item)}<{item}> to convert as "
                f"{torch_geometric.data.Data}"
            )
    return (
        InMemoryDataset(transformed_datalist, dataset.train_index, dataset.val_index, dataset.test_index, dataset.schema)
        if isinstance(dataset, InMemoryDataset)
        else InMemoryDataset(transformed_datalist)
    )
