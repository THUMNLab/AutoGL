import torch
import typing as _typing
from . import _general_static_graph
from ._general_static_graph_default_implementation import (
    HeterogeneousNodesContainer, HeterogeneousNodesContainerImplementation,
    HomogeneousEdgesContainerImplementation,
    HeterogeneousEdgesAggregation, HeterogeneousEdgesAggregationImplementation,
    StaticGraphDataAggregation, GeneralStaticGraphImplementation
)


class GeneralStaticGraphGenerator:
    @classmethod
    def create_heterogeneous_static_graph(
            cls, heterogeneous_nodes_data: _typing.Mapping[str, _typing.Mapping[str, torch.Tensor]],
            heterogeneous_edges: _typing.Mapping[
                _typing.Tuple[str, str, str],
                _typing.Union[
                    torch.Tensor,
                    _typing.Tuple[
                        torch.Tensor,
                        _typing.Optional[_typing.Mapping[str, torch.Tensor]]
                    ]
                ]
            ],
            graph_data: _typing.Optional[_typing.Mapping[str, torch.Tensor]] = ...
    ) -> _general_static_graph.GeneralStaticGraph:
        _heterogeneous_nodes_container: HeterogeneousNodesContainer = (
            HeterogeneousNodesContainerImplementation(heterogeneous_nodes_data)
        )
        _heterogeneous_edges_aggregation: HeterogeneousEdgesAggregation = (
            HeterogeneousEdgesAggregationImplementation()
        )
        for canonical_edge_type, specific_typed_edges in heterogeneous_edges.items():
            if isinstance(specific_typed_edges, torch.Tensor):
                connections = specific_typed_edges
                data = None
            elif (
                    isinstance(specific_typed_edges, _typing.Sequence) and
                    len(specific_typed_edges) == 2 and
                    isinstance(specific_typed_edges[0], torch.Tensor) and
                    (
                            specific_typed_edges[1] is None or
                            isinstance(specific_typed_edges[1], _typing.Mapping)
                    )
            ):
                connections = specific_typed_edges[0]
                data = specific_typed_edges[1]
            else:
                raise TypeError
            _heterogeneous_edges_aggregation[canonical_edge_type] = (
                HomogeneousEdgesContainerImplementation(connections, data)
            )
        return GeneralStaticGraphImplementation(
            _heterogeneous_nodes_container,
            _heterogeneous_edges_aggregation,
            StaticGraphDataAggregation(graph_data)
        )

    @classmethod
    def create_homogeneous_static_graph(
            cls, nodes_data: _typing.Mapping[str, torch.Tensor],
            edges_connections: torch.Tensor,
            edges_data: _typing.Optional[_typing.Mapping[str, torch.Tensor]] = ...,
            graph_data: _typing.Optional[_typing.Mapping[str, torch.Tensor]] = ...
    ) -> _general_static_graph.GeneralStaticGraph:
        _heterogeneous_nodes_container: HeterogeneousNodesContainer = (
            HeterogeneousNodesContainerImplementation({'': nodes_data})
        )
        _heterogeneous_edges_aggregation: HeterogeneousEdgesAggregation = (
            HeterogeneousEdgesAggregationImplementation()
        )
        _heterogeneous_edges_aggregation[('', '', '')] = (
            HomogeneousEdgesContainerImplementation(edges_connections, edges_data)
        )
        return GeneralStaticGraphImplementation(
            _heterogeneous_nodes_container,
            _heterogeneous_edges_aggregation,
            StaticGraphDataAggregation(graph_data)
        )
