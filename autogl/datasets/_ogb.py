import numpy as np
import torch
import typing as _typing
from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset
from ogb.graphproppred import GraphPropPredDataset

from torch_sparse import SparseTensor

from autogl import backend as _backend
from autogl.data import InMemoryStaticGraphSet
from autogl.data.graph import (
    GeneralStaticGraph, GeneralStaticGraphGenerator
)
from ._dataset_registry import DatasetUniversalRegistry
from .utils import index_to_mask


class _OGBDatasetUtil:
    ...


class _OGBNDatasetUtil(_OGBDatasetUtil):
    @classmethod
    def ogbn_data_to_general_static_graph(
            cls, ogbn_data: _typing.Mapping[str, _typing.Union[np.ndarray, int]],
            nodes_label: np.ndarray = ..., nodes_label_key: str = ...,
            train_index: _typing.Optional[np.ndarray] = ...,
            val_index: _typing.Optional[np.ndarray] = ...,
            test_index: _typing.Optional[np.ndarray] = ...,
            nodes_data_key_mapping: _typing.Optional[_typing.Mapping[str, str]] = ...,
            edges_data_key_mapping: _typing.Optional[_typing.Mapping[str, str]] = ...,
            graph_data_key_mapping: _typing.Optional[_typing.Mapping[str, str]] = ...
    ) -> GeneralStaticGraph:
        # TODO
        edge_index = ogbn_data['edge_index']
        num_nodes = ogbn_data['num_nodes']
        edge_feat = ogbn_data['edge_feat']
        if edge_feat is not None:
            edge_feat = torch.tensor(edge_feat)
        edge_index = SparseTensor(row=torch.tensor(edge_index[0]), col=torch.tensor(edge_index[1]), value=edge_feat, sparse_sizes=(num_nodes, num_nodes))
        _, _, value = edge_index.coo()
        if value is not None:
            ogbn_data['edge_feat'] = value.cpu().detach().numpy()
        else:
            ogbn_data['edge_feat'] = edge_feat
        edge_index = edge_index.to_symmetric()
        row, col, _ = edge_index.coo()
        edge_index = np.array([row.cpu().detach().numpy(), col.cpu().detach().numpy()])
        homogeneous_static_graph: GeneralStaticGraph = (
            GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                dict([
                    (target_key, torch.from_numpy(ogbn_data[source_key]))
                    for source_key, target_key in nodes_data_key_mapping.items()
                ]),
                torch.tensor(edge_index),
                dict([
                    (target_key, torch.from_numpy(ogbn_data[source_key]))
                    for source_key, target_key in edges_data_key_mapping.items()
                ]) if isinstance(edges_data_key_mapping, _typing.Mapping) else ...,
                dict([
                    (target_key, torch.from_numpy(ogbn_data[source_key]))
                    for source_key, target_key in graph_data_key_mapping.items()
                ]) if isinstance(graph_data_key_mapping, _typing.Mapping) else ...
            )
        )
        if isinstance(nodes_label, np.ndarray) and isinstance(nodes_label_key, str):
            if ' ' in nodes_label_key:
                raise ValueError("Illegal nodes label key")
            homogeneous_static_graph.nodes.data[nodes_label_key] = (
                torch.from_numpy(nodes_label.squeeze()).squeeze()
            )
        if isinstance(train_index, np.ndarray):
            homogeneous_static_graph.nodes.data['train_mask'] = index_to_mask(
                torch.from_numpy(train_index), ogbn_data['num_nodes']
            )
        if isinstance(val_index, np.ndarray):
            homogeneous_static_graph.nodes.data['val_mask'] = index_to_mask(
                torch.from_numpy(val_index), ogbn_data['num_nodes']
            )
        if isinstance(test_index, np.ndarray):
            homogeneous_static_graph.nodes.data['test_mask'] = index_to_mask(
                torch.from_numpy(test_index), ogbn_data['num_nodes']
            )
        return homogeneous_static_graph

    @classmethod
    def ogbn_dataset_to_general_static_graph(
            cls, ogbn_dataset: NodePropPredDataset,
            nodes_label_key: str,
            nodes_data_key_mapping: _typing.Optional[_typing.Mapping[str, str]] = ...,
            edges_data_key_mapping: _typing.Optional[_typing.Mapping[str, str]] = ...,
            graph_data_key_mapping: _typing.Optional[_typing.Mapping[str, str]] = ...
    ) -> GeneralStaticGraph:
        split_idx = ogbn_dataset.get_idx_split()
        return cls.ogbn_data_to_general_static_graph(
            ogbn_dataset[0][0],
            ogbn_dataset[0][1],
            nodes_label_key,
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
            nodes_data_key_mapping,
            edges_data_key_mapping,
            graph_data_key_mapping
        )


@DatasetUniversalRegistry.register_dataset("ogbn-products")
class OGBNProductsDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbn_dataset = NodePropPredDataset("ogbn-products", path)
        if _backend.DependentBackend.is_dgl():
            super(OGBNProductsDataset, self).__init__([
                _OGBNDatasetUtil.ogbn_dataset_to_general_static_graph(
                    ogbn_dataset, "label",
                    {"node_feat": "feat"},
                    {"edge_feat": "edge_feat"}
                )
            ])
        elif _backend.DependentBackend.is_pyg():
            super(OGBNProductsDataset, self).__init__([
                _OGBNDatasetUtil.ogbn_dataset_to_general_static_graph(
                    ogbn_dataset, "y",
                    {"node_feat": "x"}
                )
            ])


@DatasetUniversalRegistry.register_dataset("ogbn-proteins")
class OGBNProteinsDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbn_dataset = NodePropPredDataset("ogbn-proteins", path)
        if _backend.DependentBackend.is_dgl():
            super(OGBNProteinsDataset, self).__init__([
                _OGBNDatasetUtil.ogbn_dataset_to_general_static_graph(
                    ogbn_dataset, "label",
                    {"node_species": "species"},
                    {"edge_feat": "edge_feat"}
                )
            ])
        elif _backend.DependentBackend.is_pyg():
            super(OGBNProteinsDataset, self).__init__([
                _OGBNDatasetUtil.ogbn_dataset_to_general_static_graph(
                    ogbn_dataset, "y",
                    {"node_species": "species"},
                    {"edge_feat": "edge_feat"}
                )
            ])


@DatasetUniversalRegistry.register_dataset("ogbn-arxiv")
class OGBNArxivDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbn_dataset = NodePropPredDataset("ogbn-arxiv", path)
        if _backend.DependentBackend.is_dgl():
            super(OGBNArxivDataset, self).__init__([
                _OGBNDatasetUtil.ogbn_dataset_to_general_static_graph(
                    ogbn_dataset, "label",
                    {
                        "node_feat": "feat",
                        "node_year": "year"
                    }
                )
            ])
        elif _backend.DependentBackend.is_pyg():
            super(OGBNArxivDataset, self).__init__([
                _OGBNDatasetUtil.ogbn_dataset_to_general_static_graph(
                    ogbn_dataset, "y",
                    {
                        "node_feat": "x",
                        "node_year": "year"
                    }
                )
            ])


@DatasetUniversalRegistry.register_dataset("ogbn-papers100M")
class OGBNPapers100MDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbn_dataset = NodePropPredDataset("ogbn-papers100M", path)
        if _backend.DependentBackend.is_dgl():
            super(OGBNPapers100MDataset, self).__init__([
                _OGBNDatasetUtil.ogbn_dataset_to_general_static_graph(
                    ogbn_dataset, "label",
                    {
                        "node_feat": "feat",
                        "node_year": "year"
                    }
                )
            ])
        elif _backend.DependentBackend.is_pyg():
            super(OGBNPapers100MDataset, self).__init__([
                _OGBNDatasetUtil.ogbn_dataset_to_general_static_graph(
                    ogbn_dataset, "y",
                    {
                        "node_feat": "x",
                        "node_year": "year"
                    }
                )
            ])


# todo: currently homogeneous dataset `ogbn-mag` NOT supported


class _OGBLDatasetUtil(_OGBDatasetUtil):
    @classmethod
    def ogbl_data_to_general_static_graph(
            cls, ogbl_data: _typing.Mapping[str, _typing.Union[np.ndarray, int]],
            heterogeneous_edges: _typing.Mapping[
                _typing.Tuple[str, str, str],
                _typing.Union[
                    torch.Tensor,
                    _typing.Tuple[torch.Tensor, _typing.Optional[_typing.Mapping[str, torch.Tensor]]]
                ]
            ] = ...,
            nodes_data_key_mapping: _typing.Optional[_typing.Mapping[str, str]] = ...,
            graph_data_key_mapping: _typing.Optional[_typing.Mapping[str, str]] = ...
    ) -> GeneralStaticGraph:
        return GeneralStaticGraphGenerator.create_heterogeneous_static_graph(
            {
                '': dict([
                    (target_data_key, torch.from_numpy(ogbl_data[source_data_key]).squeeze())
                    for source_data_key, target_data_key in nodes_data_key_mapping.items()
                ])
            },
            heterogeneous_edges,
            dict([
                (target_data_key, torch.from_numpy(ogbl_data[source_data_key]).squeeze())
                for source_data_key, target_data_key in graph_data_key_mapping.items()
            ]) if isinstance(graph_data_key_mapping, _typing.Mapping) else ...
        )


@DatasetUniversalRegistry.register_dataset("ogbl-ppa")
class OGBLPPADataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbl_dataset = LinkPropPredDataset("ogbl-ppa", path)
        edge_split = ogbl_dataset.get_edge_split()
        super(OGBLPPADataset, self).__init__([
            _OGBLDatasetUtil.ogbl_data_to_general_static_graph(
                ogbl_dataset[0], {
                    ('', '', ''): torch.from_numpy(ogbl_dataset[0]['edge_index']),
                    ('', 'train_pos_edge', ''): torch.from_numpy(edge_split['train']['edge']),
                    ('', 'val_pos_edge', ''): torch.from_numpy(edge_split['valid']['edge']),
                    ('', 'val_neg_edge', ''): torch.from_numpy(edge_split['valid']['edge_neg']),
                    ('', 'test_pos_edge', ''): torch.from_numpy(edge_split['test']['edge']),
                    ('', 'test_neg_edge', ''): torch.from_numpy(edge_split['test']['edge_neg'])
                },
                {'node_feat': 'feat'} if _backend.DependentBackend.is_dgl() else {'node_feat': 'x'}
            )
        ])


@DatasetUniversalRegistry.register_dataset("ogbl-collab")
class OGBLCOLLABDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbl_dataset = LinkPropPredDataset("ogbl-collab", path)
        edge_split = ogbl_dataset.get_edge_split()
        super(OGBLCOLLABDataset, self).__init__([
            _OGBLDatasetUtil.ogbl_data_to_general_static_graph(
                ogbl_dataset[0], {
                    ('', '', ''): torch.from_numpy(ogbl_dataset[0]['edge_index']),
                    ('', 'train_pos_edge', ''): (
                        torch.from_numpy(edge_split['train']['edge']),
                        {
                            'weight': torch.from_numpy(edge_split['train']['weight']),
                            'year': torch.from_numpy(edge_split['train']['year'])
                        }
                    ),
                    ('', 'val_pos_edge', ''): (
                        torch.from_numpy(edge_split['valid']['edge']),
                        {
                            'weight': torch.from_numpy(edge_split['valid']['weight']),
                            'year': torch.from_numpy(edge_split['valid']['year'])
                        }
                    ),
                    ('', 'val_neg_edge', ''): torch.from_numpy(edge_split['valid']['edge_neg']),
                    ('', 'test_pos_edge', ''): (
                        torch.from_numpy(edge_split['test']['edge']),
                        {
                            'weight': torch.from_numpy(edge_split['test']['weight']),
                            'year': torch.from_numpy(edge_split['test']['year'])
                        }
                    ),
                    ('', 'test_neg_edge', ''): torch.from_numpy(edge_split['test']['edge_neg'])
                },
                {'node_feat': 'feat'} if _backend.DependentBackend.is_dgl() else {'node_feat': 'x'}
            )
        ])


@DatasetUniversalRegistry.register_dataset("ogbl-ddi")
class OGBLDDIDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbl_dataset = LinkPropPredDataset("ogbl-ddi", path)
        edge_split = ogbl_dataset.get_edge_split()
        super(OGBLDDIDataset, self).__init__([
            GeneralStaticGraphGenerator.create_heterogeneous_static_graph(
                {'': {'_NID': torch.arange(ogbl_dataset[0]['num_nodes'])}},
                {
                    ('', '', ''): torch.from_numpy(ogbl_dataset[0]['edge_index']),
                    ('', 'train_pos_edge', ''): torch.from_numpy(edge_split['train']['edge']),
                    ('', 'val_pos_edge', ''): torch.from_numpy(edge_split['valid']['edge']),
                    ('', 'val_neg_edge', ''): torch.from_numpy(edge_split['valid']['edge_neg']),
                    ('', 'test_pos_edge', ''): torch.from_numpy(edge_split['test']['edge']),
                    ('', 'test_neg_edge', ''): torch.from_numpy(edge_split['test']['edge_neg'])
                }
            )
        ])


@DatasetUniversalRegistry.register_dataset("ogbl-citation")
@DatasetUniversalRegistry.register_dataset("ogbl-citation2")
class OGBLCitation2Dataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbl_dataset = LinkPropPredDataset("ogbl-citation2", path)
        edge_split = ogbl_dataset.get_edge_split()
        super(OGBLCitation2Dataset, self).__init__([
            _OGBLDatasetUtil.ogbl_data_to_general_static_graph(
                ogbl_dataset[0],
                {
                    ('', '', ''): torch.from_numpy(ogbl_dataset[0]['edge_index']),
                    ('', 'train_pos_edge', ''): torch.from_numpy(edge_split['train']['edge']),
                    ('', 'val_pos_edge', ''): torch.from_numpy(edge_split['valid']['edge']),
                    ('', 'val_neg_edge', ''): torch.from_numpy(edge_split['valid']['edge_neg']),
                    ('', 'test_pos_edge', ''): torch.from_numpy(edge_split['test']['edge']),
                    ('', 'test_neg_edge', ''): torch.from_numpy(edge_split['test']['edge_neg'])
                },
                (
                    {'node_feat': 'feat', 'node_year': 'year'}
                    if _backend.DependentBackend.is_dgl()
                    else {'node_feat': 'x', 'node_year': 'year'}
                )
            )
        ])


# todo: currently homogeneous dataset `ogbl-wikikg2` and `ogbl-biokg` NOT supported


class _OGBGDatasetUtil:
    ...


@DatasetUniversalRegistry.register_dataset("ogbg-molhiv")
class OGBGMOLHIVDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbl_dataset = GraphPropPredDataset("ogbg-molhiv", path)
        idx_split: _typing.Mapping[str, np.ndarray] = ogbl_dataset.get_idx_split()
        train_index: _typing.Any = idx_split['train'].tolist()
        test_index: _typing.Any = idx_split['test'].tolist()
        val_index: _typing.Any = idx_split['valid'].tolist()
        super(OGBGMOLHIVDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    (
                        {"feat": torch.from_numpy(data['node_feat'])}
                        if _backend.DependentBackend.is_dgl()
                        else {"x": torch.from_numpy(data['node_feat'])}
                    ),
                    torch.from_numpy(data['edge_index']),
                    {'edge_feat': torch.from_numpy(data['edge_feat'])},
                    (
                        {'label': torch.from_numpy(label)}
                        if _backend.DependentBackend.is_dgl()
                        else {'y': torch.from_numpy(label)}
                    )
                ) for data, label in ogbl_dataset
            ],
            train_index, val_index, test_index
        )


@DatasetUniversalRegistry.register_dataset("ogbg-molpcba")
class OGBGMOLPCBADataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbl_dataset = GraphPropPredDataset("ogbg-molhiv", path)
        idx_split: _typing.Mapping[str, np.ndarray] = ogbl_dataset.get_idx_split()
        train_index: _typing.Any = idx_split['train'].tolist()
        test_index: _typing.Any = idx_split['test'].tolist()
        val_index: _typing.Any = idx_split['valid'].tolist()
        super(OGBGMOLPCBADataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    (
                        {"feat": torch.from_numpy(data['node_feat'])}
                        if _backend.DependentBackend.is_dgl()
                        else {"x": torch.from_numpy(data['node_feat'])}
                    ),
                    torch.from_numpy(data['edge_index']),
                    {'edge_feat': torch.from_numpy(data['edge_feat'])},
                    (
                        {'label': torch.from_numpy(label)}
                        if _backend.DependentBackend.is_dgl()
                        else {'y': torch.from_numpy(label)}
                    )
                ) for data, label in ogbl_dataset
            ],
            train_index, val_index, test_index
        )


@DatasetUniversalRegistry.register_dataset("ogbg-ppa")
class OGBGPPADataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbl_dataset = GraphPropPredDataset("ogbg-molhiv", path)
        idx_split: _typing.Mapping[str, np.ndarray] = ogbl_dataset.get_idx_split()
        train_index: _typing.Any = idx_split['train'].tolist()
        test_index: _typing.Any = idx_split['test'].tolist()
        val_index: _typing.Any = idx_split['valid'].tolist()
        super(OGBGPPADataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'_NID': torch.arange(data['num_nodes'])},
                    torch.from_numpy(data['edge_index']),
                    {'edge_feat': torch.from_numpy(data['edge_feat'])},
                    (
                        {'label': torch.from_numpy(label)}
                        if _backend.DependentBackend.is_dgl()
                        else {'y': torch.from_numpy(label)}
                    )
                ) for data, label in ogbl_dataset
            ],
            train_index, val_index, test_index
        )


@DatasetUniversalRegistry.register_dataset("ogbg-code")
@DatasetUniversalRegistry.register_dataset("ogbg-code2")
class OGBGCode2Dataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        ogbl_dataset = GraphPropPredDataset("ogbg-molhiv", path)
        idx_split: _typing.Mapping[str, np.ndarray] = ogbl_dataset.get_idx_split()
        train_index: _typing.Any = idx_split['train'].tolist()
        test_index: _typing.Any = idx_split['test'].tolist()
        val_index: _typing.Any = idx_split['valid'].tolist()
        super(OGBGCode2Dataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    (
                        {
                            "feat": torch.from_numpy(data['node_feat']),
                            "node_is_attributed": torch.from_numpy(data["node_is_attributed"]),
                            "node_dfs_order": torch.from_numpy(data["node_dfs_order"]),
                            "node_depth": torch.from_numpy(data["node_depth"])
                        }
                        if _backend.DependentBackend.is_dgl()
                        else
                        {
                            "x": torch.from_numpy(data['node_feat']),
                            "node_is_attributed": torch.from_numpy(data["node_is_attributed"]),
                            "node_dfs_order": torch.from_numpy(data["node_dfs_order"]),
                            "node_depth": torch.from_numpy(data["node_depth"])
                        }
                    ),
                    torch.from_numpy(data['edge_index'])
                ) for data, label in ogbl_dataset
            ],
            train_index, val_index, test_index
        )
