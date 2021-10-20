import torch
import typing as _typing
import autogl
from ... import GeneralStaticGraph


class StaticGraphToPyGData:
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
        pyg_data: autogl.data.Data = autogl.data.Data()
        for data_key in static_graph.nodes.data:
            setattr(pyg_data, data_key, static_graph.nodes.data[data_key].detach())
        homogeneous_node_type: _typing.Optional[str] = (
            list(static_graph.nodes)[0]
            if len(list(static_graph.nodes)) > 0 else None
        )
        if len(list(static_graph.edges)) == 1:
            pyg_data.edge_index = static_graph.edges.connections
            for data_key in static_graph.edges.data:
                if (
                        hasattr(pyg_data, data_key) and
                        getattr(pyg_data, data_key) is not None and
                        isinstance(getattr(pyg_data, data_key), torch.Tensor)
                ):
                    raise ValueError(
                        "Provided static graph contains duplicate data with same key, "
                        "please refer to doc for more details."
                    )
                else:
                    setattr(pyg_data, data_key, static_graph.edges.data[data_key].detach())
        elif len(list(static_graph.edges)) > 1:
            for canonical_edge_type in static_graph.edges:
                if homogeneous_node_type is not None and isinstance(homogeneous_node_type, str) and (
                        canonical_edge_type.source_node_type != homogeneous_node_type or
                        canonical_edge_type.target_node_type != homogeneous_node_type
                ):
                    continue
                if len(canonical_edge_type.relation_type) < 4 or canonical_edge_type[-4:] != 'edge':
                    continue
                edge_type_prefix: str = canonical_edge_type.relation_type[:-4]
                for data_key in static_graph.edges[canonical_edge_type].data:
                    if len(data_key) >= 4 and data_key[:4] == 'edge':
                        setattr(
                            pyg_data, edge_type_prefix + data_key,
                            static_graph.edges[canonical_edge_type].data[data_key].detach()
                        )
                    else:
                        setattr(
                            pyg_data, f"{canonical_edge_type.relation_type}_{data_key}",
                            static_graph.edges[canonical_edge_type].data[data_key].detach()
                        )
        for data_key in static_graph.data:
            if (
                    hasattr(pyg_data, data_key) and
                    getattr(pyg_data, data_key) is not None and
                    isinstance(getattr(pyg_data, data_key), torch.Tensor)
            ):
                raise ValueError(
                    "Provided static graph contains duplicate data with same key, "
                    "please refer to doc for more details."
                )
            else:
                setattr(pyg_data, data_key, static_graph.data[data_key].detach())
        return pyg_data


def static_graph_to_pyg_data(static_graph: autogl.data.graph.GeneralStaticGraph):
    return StaticGraphToPyGData().__call__(static_graph)
