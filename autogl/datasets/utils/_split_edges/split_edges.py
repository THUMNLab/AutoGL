import typing as _typing
from autogl.data import InMemoryDataset, Dataset
import autogl

if autogl.backend.DependentBackend.is_dgl():
    from ._dgl_compatible import split_edges_for_data
elif autogl.backend.DependentBackend.is_pyg():
    from ._pyg_compatible import split_edges_for_data
else:
    raise NotImplementedError


def split_edges(
        dataset: _typing.Iterable, train_ratio: float, val_ratio: _typing.Optional[float] = ...
) -> Dataset:
    if isinstance(val_ratio, float) and not 0 < train_ratio + val_ratio < 1:
        raise ValueError
    elif not 0 < train_ratio < 1:
        raise ValueError
    if (
            autogl.backend.DependentBackend.is_pyg() and
            not (isinstance(val_ratio, float) and 0 < val_ratio < 1)
    ):
        raise ValueError(
            "For PyG as backend, val_ratio MUST be specific float between 0 and 1, "
            "i.e. 0 < val_ratio < 1"
        )
    return (
        InMemoryDataset(
            [split_edges_for_data(item, train_ratio, val_ratio) for item in dataset],
            dataset.train_index, dataset.val_index, dataset.test_index, dataset.schema
        )
        if isinstance(dataset, Dataset)
        else
        InMemoryDataset(
            [split_edges_for_data(item, train_ratio, val_ratio) for item in dataset]
        )
    )
