import torch
import typing as _typing
import autogl
from ... import GeneralStaticGraph


class StaticGraphToGeneralData:
    def __init__(self, *__args, **__kwargs):
        pass

    def __call__(
            self, static_graph: GeneralStaticGraph,
            *__args, **__kwargs
    ):
        if not isinstance(static_graph, GeneralStaticGraph):
            raise TypeError
        elif not static_graph.nodes.is_homogeneous:
            raise ValueError("Provided static graph MUST consist of homogeneous nodes")
        homogeneous_node_type: _typing.Optional[str] = (
            list(static_graph.nodes)[0]
            if len(list(static_graph.nodes)) > 0 else None
        )
        data: _typing.Dict[str, torch.Tensor] = dict()
        if isinstance(homogeneous_node_type, str):
            node_and_edge_data_keys_intersection: _typing.Set[str] = (
                    set(static_graph.nodes.data) & set(static_graph.data)
            )
            if len(node_and_edge_data_keys_intersection) > 0:
                raise ValueError(
                    f"Provided static graph contains duplicate data "
                    f"with same keys {node_and_edge_data_keys_intersection}"
                    f"for homogeneous nodes data and graph-level data, "
                    f"please refer to doc for more details."
                )
            data.update(static_graph.nodes.data)
            data.update(static_graph.data)
        else:
            data.update(static_graph.data)

        if len(list(static_graph.edges)) == 1:
            data["edge_index"] = static_graph.edges.connections
            if len(set(data.keys()) & set(static_graph.edges.data.keys())) > 0:
                raise ValueError(
                    "Provided static graph contains duplicate data with same key, "
                    "please refer to doc for more details."
                )
            data.update(static_graph.edges.data)
        elif len(list(static_graph.edges)) > 1:
            for canonical_edge_type in static_graph.edges:
                if homogeneous_node_type is not None and isinstance(homogeneous_node_type, str) and (
                        canonical_edge_type.source_node_type != homogeneous_node_type or
                        canonical_edge_type.target_node_type != homogeneous_node_type
                ):
                    continue
                if len(canonical_edge_type.relation_type) < 4 or canonical_edge_type[-4:] != 'edge':
                    continue
                data[f"{canonical_edge_type.relation_type}_index"] = (
                    static_graph.edges[canonical_edge_type].connections
                )

                edge_type_prefix: str = canonical_edge_type.relation_type[:-4]
                for data_key in static_graph.edges[canonical_edge_type].data:
                    if len(data_key) >= 4 and data_key[:4] == 'edge':
                        data[f"{edge_type_prefix}{data_key}"] = (
                            static_graph.edges[canonical_edge_type].data[data_key].detach()
                        )
                    else:
                        data[f"{canonical_edge_type.relation_type}_{data_key}"] = (
                            static_graph.edges[canonical_edge_type].data[data_key].detach()
                        )

        general_data = autogl.data.Data()
        for key, value in data.items():
            setattr(general_data, key, value)
        return general_data


def static_graph_to_general_data(static_graph: GeneralStaticGraph) -> autogl.data.Data:
    return StaticGraphToGeneralData().__call__(static_graph)
