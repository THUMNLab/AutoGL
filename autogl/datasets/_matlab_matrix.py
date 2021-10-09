import itertools
import os

import scipy.io
import torch
import typing as _typing

from autogl.data import Data, download_url, InMemoryStaticGraphSet
from autogl.data.graph import GeneralStaticGraphGenerator
from ._dataset_registry import DatasetUniversalRegistry
from ._data_source import OnlineDataSource
from .. import backend as _backend


class _MATLABMatrix(OnlineDataSource):
    @property
    def _raw_filenames(self) -> _typing.Iterable[str]:
        splits = [self.__name]
        files = ["mat"]
        return [
            "{}.{}".format(s, f) for s, f
            in itertools.product(splits, files)
        ]

    @property
    def _processed_filenames(self) -> _typing.Iterable[str]:
        return ["data.pt"]

    def _fetch(self):
        for name in self._raw_filenames:
            download_url(self.__url + name, self._raw_directory)

    def _process(self):
        path = os.path.join(self._raw_directory, f"{self.__name}.mat")
        mat = scipy.io.loadmat(path)
        adj_matrix, group = mat["network"], mat["group"]

        y = torch.from_numpy(group.todense()).to(torch.float)

        row_ind, col_ind = adj_matrix.nonzero()
        edge_index = torch.stack([torch.tensor(row_ind), torch.tensor(col_ind)], dim=0)
        edge_attr = torch.tensor(adj_matrix[row_ind, col_ind])
        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=None, y=y)
        torch.save(data, list(self._processed_file_paths)[0])

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.__data

    def __init__(self, path: str, name: str, url: str):
        self.__name: str = name
        self.__url: str = url
        super(_MATLABMatrix, self).__init__(path)
        self.__data = torch.load(
            list(self._processed_file_paths)[0]
        )


@DatasetUniversalRegistry.register_dataset("BlogCatalog".lower())
class BlogCatalogDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        filename: str = "BlogCatalog".lower()
        url: str = "http://leitang.net/code/social-dimension/data/"
        data = _MATLABMatrix(path, filename, url)[0]
        if _backend.DependentBackend.is_dgl():
            super(BlogCatalogDataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {'label': data.y}, data.edge_index,
                        {'edge_attr': data.edge_attr}
                    )
                ]
            )
        elif _backend.DependentBackend.is_pyg():
            super(BlogCatalogDataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {'y': data.y}, data.edge_index,
                        {'edge_attr': data.edge_attr}
                    )
                ]
            )


@DatasetUniversalRegistry.register_dataset("WikiPEDIA".lower())
class WIKIPEDIADataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        filename: str = "POS"
        url = "http://snap.stanford.edu/node2vec/"
        data = _MATLABMatrix(path, filename, url)[0]
        if _backend.DependentBackend.is_dgl():
            super(WIKIPEDIADataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {'label': data.y}, data.edge_index,
                        {'attr': data.edge_attr}
                    )
                ]
            )
        elif _backend.DependentBackend.is_pyg():
            super(WIKIPEDIADataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {'y': data.y}, data.edge_index,
                        {'attr': data.edge_attr}
                    )
                ]
            )
