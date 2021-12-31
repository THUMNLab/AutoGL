import typing as _typing
from ._dataset import InMemoryDataset
from ..graph import GeneralStaticGraph


class InMemoryStaticGraphSet(InMemoryDataset[GeneralStaticGraph]):
    def __init__(
            self, graphs: _typing.Iterable[GeneralStaticGraph],
            train_index: _typing.Optional[_typing.Iterable[int]] = ...,
            val_index: _typing.Optional[_typing.Iterable[int]] = ...,
            test_index: _typing.Optional[_typing.Iterable[int]] = ...
    ):
        super(InMemoryStaticGraphSet, self).__init__(
            graphs, train_index, val_index, test_index
        )

    def __iter__(self) -> _typing.Iterator[GeneralStaticGraph]:
        return super(InMemoryStaticGraphSet, self).__iter__()

    def __getitem__(self, index: int) -> GeneralStaticGraph:
        return super(InMemoryStaticGraphSet, self).__getitem__(index)

    def __setitem__(self, index: int, data: GeneralStaticGraph):
        super(InMemoryStaticGraphSet, self).__setitem__(index, data)
