import typing as _typing
import networkx as nx
from autogl.data.graph._general_static_graph import GeneralStaticGraph


class HomogeneousStaticGraphToNetworkX:
    def __init__(
            self, remove_self_loops: bool = False, to_undirected: bool = False,
            *__args, **__kwargs
    ):
        if not isinstance(remove_self_loops, bool):
            raise TypeError
        if not isinstance(to_undirected, bool):
            raise TypeError
        self.__remove_self_loops: bool = remove_self_loops
        self.__to_undirected: bool = to_undirected

    def __call__(
            self, homogeneous_static_graph: GeneralStaticGraph,
            remove_self_loops: _typing.Optional[bool] = ...,
            to_undirected: _typing.Optional[bool] = ...,
            *args, **kwargs
    ):
        if not isinstance(homogeneous_static_graph, GeneralStaticGraph):
            raise TypeError
        elif not (
                homogeneous_static_graph.nodes.is_homogeneous and
                homogeneous_static_graph.edges.is_homogeneous
        ):
            raise ValueError("Only homogeneous static graph can be converted to NetworkX")

        if not (remove_self_loops in (Ellipsis, None) or isinstance(remove_self_loops, bool)):
            raise TypeError
        else:
            __remove_self_loops: bool = (
                remove_self_loops if isinstance(remove_self_loops, bool)
                else self.__remove_self_loops
            )
        if not (to_undirected in (Ellipsis, None) or isinstance(to_undirected, bool)):
            raise TypeError
        else:
            __to_undirected: bool = (
                to_undirected if isinstance(to_undirected, bool)
                else self.__to_undirected
            )

        num_nodes: int = homogeneous_static_graph.edges.connections.max().item() + 1
        # todo: Note that this is an assumption

        g: nx.Graph = nx.Graph() if __to_undirected else nx.DiGraph()
        g.add_nodes_from(range(num_nodes))

        nodes_data: _typing.MutableMapping[str, list] = {}
        for data_key in homogeneous_static_graph.nodes.data:
            nodes_data[data_key] = (
                homogeneous_static_graph.nodes.data[data_key].squeeze().tolist()
            )
        edges_data: _typing.MutableMapping[str, list] = {}
        for data_key in homogeneous_static_graph.edges.data:
            edges_data[data_key] = (
                homogeneous_static_graph.edges.data[data_key].squeeze().tolist()
            )
        for i, (u, v) in enumerate(homogeneous_static_graph.edges.connections.t().tolist()):
            if __remove_self_loops and v == u:
                continue
            g.add_edge(u, v)
            for data_key in edges_data:
                g[u][v][data_key] = edges_data[data_key][i]
        for data_key in nodes_data:
            for i, feature_dict in g.nodes(data=True):
                feature_dict.update(
                    {data_key: nodes_data[data_key][i]}
                )
        return g
