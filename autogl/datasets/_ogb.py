import numpy as np
import torch
import typing as _typing

from autogl import backend as _backend

if _backend.DependentBackend.is_pyg():
    from ogb.nodeproppred import PygNodePropPredDataset as NodePropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset as LinkPropPredDataset
    from ogb.graphproppred import PygGraphPropPredDataset as GraphPropPredDataset
elif _backend.DependentBackend.is_dgl():
    from ogb.nodeproppred import DglNodePropPredDataset as NodePropPredDataset
    from ogb.linkproppred import DglLinkPropPredDataset as LinkPropPredDataset
    from ogb.graphproppred import DglGraphPropPredDataset as GraphPropPredDataset

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
def get_ogbn_products_dataset(path, *args, **kwargs):
    return NodePropPredDataset('ogbn-products', path, *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ogbn-proteins")
def get_ogbn_proteins_dataset(path, *args, **kwargs):
    return NodePropPredDataset('ogbn-proteins', path, *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ogbn-arxiv")
def get_ogbn_arxiv_dataset(path, *args, **kwargs):
    return NodePropPredDataset("ogbn-arxiv", path, *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ogbn-papers100M")
def get_ogbn_papers_hm(path, *args, **kwargs):
    return NodePropPredDataset("ogbn-papers100M", path, *args, **kwargs)


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
def get_ogbl_ppa_dataset(path, *args, **kwargs):
    return LinkPropPredDataset("ogbl-ppa", path, *args, **kwargs)



@DatasetUniversalRegistry.register_dataset("ogbl-collab")
def get_ogbl_collab_dataset(path, *args, **kwargs):
    return LinkPropPredDataset("ogbl-collab", path, *args, **kwargs)




@DatasetUniversalRegistry.register_dataset("ogbl-ddi")
def get_ogbl_ddi_dataset(path, *args, **kwargs):
    return LinkPropPredDataset("ogbl-ddi", path, *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ogbl-citation")
@DatasetUniversalRegistry.register_dataset("ogbl-citation2")
def get_ogbl_citation_dataset(path, *args, **kwargs):
    return LinkPropPredDataset("ogbl-citation2", path, *args, **kwargs)

# todo: currently homogeneous dataset `ogbl-wikikg2` and `ogbl-biokg` NOT supported


class _OGBGDatasetUtil:
    ...


@DatasetUniversalRegistry.register_dataset("ogbg-molhiv")
def get_ogbg_molhiv_dataset(path, *args, **kwargs):
    return GraphPropPredDataset("ogbg-molhiv", path, *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ogbg-molpcba")
def get_ogbg_molpcba_dataset(path, *args, **kwargs):
    return GraphPropPredDataset("ogbg-molpcba", path, *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ogbg-ppa")
def get_ogbg_ppa_dataset(path, *args, **kwargs):
    return GraphPropPredDataset("ogbg-ppa", path, *args, **kwargs)


@DatasetUniversalRegistry.register_dataset("ogbg-code")
@DatasetUniversalRegistry.register_dataset("ogbg-code2")
def get_ogbg_code_dataset(path, *args, **kwargs):
    return GraphPropPredDataset("ogbg-code", path, *args, **kwargs)