import os
import os.path as osp
import shutil
import pickle
import numpy as np
import torch
import typing as _typing

from autogl.data import Data, download_url, InMemoryStaticGraphSet
from autogl.data.graph import GeneralStaticGraphGenerator
from ._dataset_registry import DatasetUniversalRegistry
from ._data_source import OnlineDataSource
from .. import backend as _backend


def _untar(path, fname, delete_tar=True):
    """
    Unpacks the given archive file to the same directory, then (by default)
    deletes the archive file.
    """
    print("unpacking " + fname)
    full_path = os.path.join(path, fname)
    shutil.unpack_archive(full_path, path)
    if delete_tar:
        os.remove(full_path)


class _GTNDataSource(OnlineDataSource):
    def __init__(self, path: str, name: str):
        self.__name: str = name
        self.__url: str = (
            f"https://github.com/cenyk1230/gtn-data/blob/master/{name}.zip?raw=true"
        )
        super(_GTNDataSource, self).__init__(path)
        self.__data = torch.load(list(self._processed_file_paths)[0])

    @property
    def _raw_filenames(self) -> _typing.Iterable[str]:
        return ["edges.pkl", "labels.pkl", "node_features.pkl"]

    @property
    def _processed_filenames(self) -> _typing.Iterable[str]:
        return ["data.pt"]

    def __read_gtn_data(self, directory):
        edges = pickle.load(open(osp.join(directory, "edges.pkl"), "rb"))
        labels = pickle.load(open(osp.join(directory, "labels.pkl"), "rb"))
        node_features = pickle.load(open(osp.join(directory, "node_features.pkl"), "rb"))

        data = Data()
        data.x = torch.from_numpy(node_features).float()

        num_nodes = edges[0].shape[0]

        node_type = np.zeros(num_nodes, dtype=int)
        assert len(edges) == 4
        assert len(edges[0].nonzero()) == 2

        node_type[edges[0].nonzero()[0]] = 0
        node_type[edges[0].nonzero()[1]] = 1
        node_type[edges[1].nonzero()[0]] = 1
        node_type[edges[1].nonzero()[1]] = 0
        node_type[edges[2].nonzero()[0]] = 0
        node_type[edges[2].nonzero()[1]] = 2
        node_type[edges[3].nonzero()[0]] = 2
        node_type[edges[3].nonzero()[1]] = 0

        print(node_type)
        data.pos = torch.from_numpy(node_type)

        edge_list = []
        for i, edge in enumerate(edges):
            edge_tmp = torch.from_numpy(
                np.vstack((edge.nonzero()[0], edge.nonzero()[1]))
            ).long()
            edge_list.append(edge_tmp)
        data.edge_index = torch.cat(edge_list, 1)

        A = []
        for i, edge in enumerate(edges):
            edge_tmp = torch.from_numpy(
                np.vstack((edge.nonzero()[0], edge.nonzero()[1]))
            ).long()
            value_tmp = torch.ones(edge_tmp.shape[1]).float()
            A.append((edge_tmp, value_tmp))
        edge_tmp = torch.stack(
            (torch.arange(0, num_nodes), torch.arange(0, num_nodes))
        ).long()
        value_tmp = torch.ones(num_nodes).float()
        A.append((edge_tmp, value_tmp))
        data.adj = A

        data.train_node = torch.from_numpy(np.array(labels[0])[:, 0]).long()
        data.train_target = torch.from_numpy(np.array(labels[0])[:, 1]).long()
        data.valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).long()
        data.valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).long()
        data.test_node = torch.from_numpy(np.array(labels[2])[:, 0]).long()
        data.test_target = torch.from_numpy(np.array(labels[2])[:, 1]).long()

        y = np.zeros(num_nodes, dtype=int)
        x_index = torch.cat((data.train_node, data.valid_node, data.test_node))
        y_index = torch.cat((data.train_target, data.valid_target, data.test_target))
        y[x_index.numpy()] = y_index.numpy()
        data.y = torch.from_numpy(y)
        self.__data = data

    def __transform_gtn_data(self):
        self.__data.train_mask = torch.zeros(self.__data.x.size(0), dtype=torch.bool)
        self.__data.val_mask = torch.zeros(self.__data.x.size(0), dtype=torch.bool)
        self.__data.test_mask = torch.zeros(self.__data.x.size(0), dtype=torch.bool)
        self.__data.train_mask[getattr(self.__data, "train_node")] = True
        self.__data.val_mask[getattr(self.__data, "valid_node")] = True
        self.__data.test_mask[getattr(self.__data, "test_node")] = True

    def _fetch(self):
        download_url(self.__url, self._raw_directory, name=f"{self.__name}.zip")
        _untar(self._raw_directory, f"{self.__name}.zip")

    def _process(self):
        self.__read_gtn_data(self._raw_directory)
        self.__transform_gtn_data()
        torch.save(self.__data, list(self._processed_file_paths)[0])

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError
        return self.__data


@DatasetUniversalRegistry.register_dataset("gtn-acm")
class GTNACMDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        data = _GTNDataSource(path, "gtn-acm")[0]
        if _backend.DependentBackend.is_dgl():
            super(GTNACMDataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {
                            'feat': getattr(data, 'x'),
                            'label': getattr(data, 'y'),
                            'pos': getattr(data, 'pos'),
                            'train_mask': getattr(data, 'train_mask'),
                            'val_mask': getattr(data, 'val_mask'),
                            'test_mask': getattr(data, 'test_mask')
                        },
                        getattr(data, 'edge_index')
                    )
                ]
            )
        elif _backend.DependentBackend.is_pyg():
            super(GTNACMDataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {
                            'x': getattr(data, 'x'),
                            'y': getattr(data, 'y'),
                            'pos': getattr(data, 'pos'),
                            'train_mask': getattr(data, 'train_mask'),
                            'val_mask': getattr(data, 'val_mask'),
                            'test_mask': getattr(data, 'test_mask')
                        },
                        getattr(data, 'edge_index')
                    )
                ]
            )


@DatasetUniversalRegistry.register_dataset("gtn-dblp")
class GTNDBLPDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        data = _GTNDataSource(path, "gtn-dblp")[0]
        if _backend.DependentBackend.is_dgl():
            super(GTNDBLPDataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {
                            'feat': getattr(data, 'x'),
                            'label': getattr(data, 'y'),
                            'pos': getattr(data, 'pos'),
                            'train_mask': getattr(data, 'train_mask'),
                            'val_mask': getattr(data, 'val_mask'),
                            'test_mask': getattr(data, 'test_mask')
                        },
                        getattr(data, 'edge_index')
                    )
                ]
            )
        elif _backend.DependentBackend.is_pyg():
            super(GTNDBLPDataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {
                            'x': getattr(data, 'x'),
                            'y': getattr(data, 'y'),
                            'pos': getattr(data, 'pos'),
                            'train_mask': getattr(data, 'train_mask'),
                            'val_mask': getattr(data, 'val_mask'),
                            'test_mask': getattr(data, 'test_mask')
                        },
                        getattr(data, 'edge_index')
                    )
                ]
            )


@DatasetUniversalRegistry.register_dataset("gtn-imdb")
class GTNIMDBDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        data = _GTNDataSource(path, "gtn-imdb")[0]
        if _backend.DependentBackend.is_dgl():
            super(GTNIMDBDataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {
                            'feat': getattr(data, 'x'),
                            'label': getattr(data, 'y'),
                            'pos': getattr(data, 'pos'),
                            'train_mask': getattr(data, 'train_mask'),
                            'val_mask': getattr(data, 'val_mask'),
                            'test_mask': getattr(data, 'test_mask')
                        },
                        getattr(data, 'edge_index')
                    )
                ]
            )
        elif _backend.DependentBackend.is_pyg():
            super(GTNIMDBDataset, self).__init__(
                [
                    GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                        {
                            'x': getattr(data, 'x'),
                            'y': getattr(data, 'y'),
                            'pos': getattr(data, 'pos'),
                            'train_mask': getattr(data, 'train_mask'),
                            'val_mask': getattr(data, 'val_mask'),
                            'test_mask': getattr(data, 'test_mask')
                        },
                        getattr(data, 'edge_index')
                    )
                ]
            )
