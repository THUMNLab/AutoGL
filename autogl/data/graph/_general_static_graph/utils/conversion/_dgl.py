import dgl
import torch
import typing as _typing
from ..._general_static_graph import GeneralStaticGraph
from ... import (
    _general_static_graph_generator, _general_static_graph_dgl_implementation
)


class GeneralStaticGraphToDGLGraph:
    def __init__(self, *__args, **__kwargs):
        pass

    def __call__(self, static_graph: GeneralStaticGraph) -> dgl.DGLGraph:
        dgl_graph: dgl.DGLGraph = dgl.heterograph(
            dict([
                (
                    (
                        canonical_edge_type.source_node_type,
                        canonical_edge_type.relation_type,
                        canonical_edge_type.target_node_type
                    ),
                    (
                        static_graph.edges[canonical_edge_type].connections[0],
                        static_graph.edges[canonical_edge_type].connections[1]
                    )
                )
                for canonical_edge_type in static_graph.edges
            ])
        )
        for node_type in static_graph.nodes:
            for data_key in static_graph.nodes[node_type].data:
                dgl_graph.nodes[node_type].data[data_key] = (
                    static_graph.nodes[node_type].data[data_key]
                )
        for canonical_edge_type in static_graph.edges:
            for data_key in static_graph.edges[canonical_edge_type].data:
                dgl_graph.edges[
                    (
                        canonical_edge_type.source_node_type,
                        canonical_edge_type.relation_type,
                        canonical_edge_type.target_node_type
                    )
                ].data[data_key] = (
                    static_graph.edges[canonical_edge_type].data[data_key]
                )
        # Set graph level data by `setattr`
        if len(static_graph.data) > 0:
            setattr(dgl_graph, "graph_data", dict(static_graph.data))
            if "gf" in static_graph.data:
                setattr(dgl_graph, "gf", static_graph.data["gf"].detach().clone())
        return dgl_graph


class DGLGraphToGeneralStaticGraph:
    def __init__(
            self, as_universal_storage_format: bool = False,
            *__args, **__kwargs
    ):
        if not isinstance(as_universal_storage_format, bool):
            raise TypeError
        else:
            self._as_universal_storage_format: bool = as_universal_storage_format

    def __call__(
            self, dgl_graph: dgl.DGLGraph,
            as_universal_storage_format: _typing.Optional[bool] = ...,
            *__args, **__kwargs
    ) -> GeneralStaticGraph:
        if not (
                as_universal_storage_format in (Ellipsis, None) or
                isinstance(as_universal_storage_format, bool)
        ):
            raise TypeError
        _as_universal_storage_format: bool = (
            as_universal_storage_format
            if isinstance(as_universal_storage_format, bool)
            else self._as_universal_storage_format
        )

        if not _as_universal_storage_format:
            general_static_graph: GeneralStaticGraph = (
                _general_static_graph_dgl_implementation.GeneralStaticGraphDGLImplementation(dgl_graph)
            )

        else:
            general_static_graph: GeneralStaticGraph = (
                _general_static_graph_generator.GeneralStaticGraphGenerator.create_heterogeneous_static_graph(
                    dict([(node_type, dgl_graph.nodes[node_type].data) for node_type in dgl_graph.ntypes]),
                    dict([
                        (
                            canonical_edge_type,
                            (
                                torch.vstack(dgl_graph.edges(etype=canonical_edge_type)),
                                dgl_graph.edges[canonical_edge_type].data
                            )
                        )
                        for canonical_edge_type in dgl_graph.canonical_etypes]
                    )
                )
            )
        if (
                hasattr(dgl_graph, "graph_data") and
                isinstance(getattr(dgl_graph, "graph_data"), _typing.Mapping)
        ):
            graph_data: _typing.Mapping[str, torch.Tensor] = getattr(dgl_graph, "graph_data")
            for k, v in graph_data.items():
                if (
                        isinstance(k, str) and ' ' not in k and
                        isinstance(v, torch.Tensor)
                ):
                    general_static_graph.data[k] = v
        for k in ("gf",):
            if (
                    hasattr(dgl_graph, k) and
                    isinstance(getattr(dgl_graph, k), torch.Tensor)
            ):
                general_static_graph.data[k] = getattr(dgl_graph, k)
        return general_static_graph


def general_static_graph_to_dgl_graph(
        general_static_graph: GeneralStaticGraph, *__args, **__kwargs
) -> dgl.DGLGraph:
    return GeneralStaticGraphToDGLGraph(*__args, **__kwargs).__call__(
        general_static_graph
    )


def dgl_graph_to_general_static_graph(
        dgl_graph: dgl.DGLGraph, as_universal_storage_format: bool = False,
        *__args, **__kwargs
) -> GeneralStaticGraph:
    return DGLGraphToGeneralStaticGraph(as_universal_storage_format).__call__(
        dgl_graph, as_universal_storage_format
    )
