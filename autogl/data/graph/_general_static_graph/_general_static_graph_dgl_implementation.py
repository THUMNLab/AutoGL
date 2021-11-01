import dgl
import torch
import typing as _typing
from . import (
    _abstract_views,
    _canonical_edge_type,
    _general_static_graph
)


class _DGLGraphHolder:
    def __init__(self, dgl_graph: dgl.DGLGraph):
        if not isinstance(dgl_graph, dgl.DGLGraph):
            raise TypeError
        self.__graph: dgl.DGLGraph = dgl_graph

    @property
    def graph(self) -> dgl.DGLGraph:
        return self.__graph

    @graph.setter
    def graph(self, dgl_graph: dgl.DGLGraph):
        if not isinstance(dgl_graph, dgl.DGLGraph):
            raise TypeError
        else:
            self.__graph = dgl_graph


class _SpecificTypedNodeDataView(_abstract_views.SpecificTypedNodeDataView):
    def __init__(
            self, dgl_graph_holder: _DGLGraphHolder,
            node_type: _typing.Optional[str] = ...
    ):
        if not isinstance(dgl_graph_holder, _DGLGraphHolder):
            raise TypeError
        if not (node_type in (Ellipsis, None) or isinstance(node_type, str)):
            raise TypeError
        elif isinstance(node_type, str) and ' ' in node_type:
            raise ValueError("Illegal node type")
        self.__dgl_graph_holder: _DGLGraphHolder = dgl_graph_holder
        self.__optional_node_type: _typing.Optional[str] = (
            node_type if isinstance(node_type, str) else None
        )

    def __getitem__(self, data_key: str) -> torch.Tensor:
        if not isinstance(data_key, str):
            raise TypeError
        elif ' ' in data_key:
            raise ValueError("Illegal data key")
        if isinstance(self.__optional_node_type, str):
            node_type: str = self.__optional_node_type
        else:
            if len(self.__dgl_graph_holder.graph.ntypes) == 0:
                raise ValueError("the graph is empty")
            elif len(self.__dgl_graph_holder.graph.ntypes) > 1:
                raise ValueError(
                    "Unable to automatically determine node type, "
                    "the graph consists of heterogeneous node types"
                )
            else:
                node_type: str = self.__dgl_graph_holder.graph.ntypes[0]
        if data_key in self.__dgl_graph_holder.graph.nodes[node_type].data:
            return self.__dgl_graph_holder.graph.nodes[node_type].data[data_key]
        else:
            raise KeyError  # todo: Complete message

    def __setitem__(self, data_key: str, value: torch.Tensor):
        if not isinstance(data_key, str):
            raise TypeError
        elif ' ' in data_key:
            raise ValueError("Illegal data key")
        if not isinstance(value, torch.Tensor):
            raise TypeError
        elif value.dim() == 0:
            raise ValueError
        if isinstance(self.__optional_node_type, str):
            node_type: str = self.__optional_node_type
        else:
            if len(self.__dgl_graph_holder.graph.ntypes) == 0:
                raise ValueError("the graph is empty")
            elif len(self.__dgl_graph_holder.graph.ntypes) > 1:
                raise ValueError(
                    "Unable to automatically determine node type, "
                    "the graph consists of heterogeneous node types"
                )
            else:
                node_type: str = self.__dgl_graph_holder.graph.ntypes[0]
        if value.size(0) != self.__dgl_graph_holder.graph.num_nodes(node_type):
            raise ValueError  # todo: Complete error message
        else:
            # todo: 现在这个方法没有处理node_type不存在的情况
            self.__dgl_graph_holder.graph.nodes[node_type].data[data_key] = value

    def __delitem__(self, data_key: str) -> None:
        if not isinstance(data_key, str):
            raise TypeError
        elif ' ' in data_key:
            raise ValueError("Illegal data key")
        if isinstance(self.__optional_node_type, str):
            node_type: str = self.__optional_node_type
        else:
            if len(self.__dgl_graph_holder.graph.ntypes) == 0:
                raise ValueError("the graph is empty")
            elif len(self.__dgl_graph_holder.graph.ntypes) > 1:
                raise ValueError(
                    "Unable to automatically determine node type, "
                    "the graph consists of heterogeneous node types"
                )
            else:
                node_type: str = self.__dgl_graph_holder.graph.ntypes[0]
        if data_key in self.__dgl_graph_holder.graph.nodes[node_type].data:
            try:
                del self.__dgl_graph_holder.graph.nodes[node_type].data[data_key]
            except KeyError:
                pass  # todo: Use logger to warn

    def __len__(self) -> int:
        if isinstance(self.__optional_node_type, str):
            node_type: str = self.__optional_node_type
        else:
            if len(self.__dgl_graph_holder.graph.ntypes) == 0:
                raise ValueError("the graph is empty")
            elif len(self.__dgl_graph_holder.graph.ntypes) > 1:
                raise ValueError(
                    "Unable to automatically determine node type, "
                    "the graph consists of heterogeneous node types"
                )
            else:
                node_type: str = self.__dgl_graph_holder.graph.ntypes[0]
        return len(self.__dgl_graph_holder.graph.nodes[node_type].data)

    def __iter__(self) -> _typing.Iterator[str]:
        if isinstance(self.__optional_node_type, str):
            node_type: str = self.__optional_node_type
        else:
            if len(self.__dgl_graph_holder.graph.ntypes) == 0:
                raise ValueError("the graph is empty")
            elif len(self.__dgl_graph_holder.graph.ntypes) > 1:
                raise ValueError(
                    "Unable to automatically determine node type, "
                    "the graph consists of heterogeneous node types"
                )
            else:
                node_type: str = self.__dgl_graph_holder.graph.ntypes[0]
        return iter(self.__dgl_graph_holder.graph.nodes[node_type].data)


class _SpecificTypedNodeView(_abstract_views.SpecificTypedNodeView):
    def __init__(
            self, dgl_graph_holder: _DGLGraphHolder,
            node_type: _typing.Optional[str] = ...
    ):
        if not isinstance(dgl_graph_holder, _DGLGraphHolder):
            raise TypeError
        if not (node_type in (Ellipsis, None) or isinstance(node_type, str)):
            raise TypeError
        elif isinstance(node_type, str) and ' ' in node_type:
            raise ValueError("Illegal node type")
        self.__dgl_graph_holder: _DGLGraphHolder = dgl_graph_holder
        self.__optional_node_type: _typing.Optional[str] = (
            node_type if isinstance(node_type, str) else None
        )

    @property
    def data(self) -> _SpecificTypedNodeDataView:
        return _SpecificTypedNodeDataView(
            self.__dgl_graph_holder, self.__optional_node_type
        )

    @data.setter
    def data(self, nodes_data: _typing.Mapping[str, torch.Tensor]):
        raise NotImplementedError  # todo: Currently, DGL not support this operation


class _HeterogeneousNodeView(_abstract_views.HeterogeneousNodeView):
    def __init__(self, dgl_graph_holder: _DGLGraphHolder):
        if not isinstance(dgl_graph_holder, _DGLGraphHolder):
            raise TypeError
        self.__dgl_graph_holder: _DGLGraphHolder = dgl_graph_holder

    @property
    def data(self) -> _SpecificTypedNodeDataView:
        return _SpecificTypedNodeDataView(self.__dgl_graph_holder, ...)

    @data.setter
    def data(self, nodes_data: _typing.Mapping[str, torch.Tensor]):
        if not isinstance(nodes_data, _typing.Mapping):
            raise TypeError
        _SpecificTypedNodeView(self.__dgl_graph_holder, ...).data = nodes_data

    def __getitem__(self, node_type: _typing.Optional[str]) -> _SpecificTypedNodeView:
        if not (node_type in (Ellipsis, None) or isinstance(node_type, str)):
            raise TypeError
        elif isinstance(node_type, str) and ' ' in node_type:
            raise ValueError("Illegal edge type")
        return _SpecificTypedNodeView(self.__dgl_graph_holder, node_type)

    def __setitem__(
            self, node_type: _typing.Optional[str],
            nodes_data: _typing.Mapping[str, torch.Tensor]
    ):
        if not (node_type in (Ellipsis, None) or isinstance(node_type, str)):
            raise TypeError
        elif isinstance(node_type, str) and ' ' in node_type:
            raise ValueError("Illegal edge type")
        if not isinstance(nodes_data, _typing.Mapping):
            raise TypeError
        _SpecificTypedNodeView(
            self.__dgl_graph_holder, node_type if isinstance(node_type, str) else None
        ).data = nodes_data

    def __delitem__(self, node_t: _typing.Optional[str]):
        raise NotImplementedError  # todo: Currently, DGL not support this operation

    def __iter__(self) -> _typing.Iterator[str]:
        return iter(self.__dgl_graph_holder.graph.ntypes)

    @property
    def is_homogeneous(self) -> bool:
        return len(self.__dgl_graph_holder.graph.ntypes) <= 1


class _HomogeneousEdgesDataView(_abstract_views.HomogeneousEdgesDataView):
    def __init__(
            self, dgl_graph_holder: _DGLGraphHolder,
            edge_type: _typing.Union[
                None, str, _typing.Tuple[str, str, str],
                _canonical_edge_type.CanonicalEdgeType
            ] = ...
    ):
        if not isinstance(dgl_graph_holder, _DGLGraphHolder):
            raise TypeError
        self.__dgl_graph_holder: _DGLGraphHolder = dgl_graph_holder
        if edge_type in (Ellipsis, None):
            self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = None
        elif isinstance(edge_type, str):
            if ' ' in edge_type:
                raise ValueError("Illegal edge type")
            self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = edge_type
        elif isinstance(edge_type, _typing.Sequence) and not isinstance(edge_type, str):
            if not (
                    len(edge_type) == 3 and
                    isinstance(edge_type[0], str) and ' ' not in edge_type[0] and
                    isinstance(edge_type[1], str) and ' ' not in edge_type[1] and
                    isinstance(edge_type[2], str) and ' ' not in edge_type[2]
            ):
                raise ValueError("Illegal edge type")
            self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = tuple(edge_type)
        elif isinstance(edge_type, _canonical_edge_type.CanonicalEdgeType):
            self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = (
                edge_type.source_node_type, edge_type.relation_type, edge_type.target_node_type
            )
        else:
            raise TypeError

    def __get_canonical_edge_type(self) -> _typing.Tuple[str, str, str]:
        if self.__optional_edge_type in (Ellipsis, None):
            if len(self.__dgl_graph_holder.graph.canonical_etypes) == 0:
                raise ValueError("The graph is empty")
            elif len(self.__dgl_graph_holder.graph.canonical_etypes) > 1:
                raise ValueError(
                    "Unable to automatically determine edge type, "
                    "the graph consists of heterogeneous edge types."
                )
            else:
                return self.__dgl_graph_holder.graph.canonical_etypes[0]
        elif isinstance(self.__optional_edge_type, str):
            try:
                canonical_edge_type = self.__dgl_graph_holder.graph.to_canonical_etype(
                    self.__optional_edge_type
                )
            except dgl.DGLError as e:
                raise e
            else:
                return canonical_edge_type
        else:
            return self.__optional_edge_type

    def __getitem__(self, data_key: str) -> torch.Tensor:
        if not isinstance(data_key, str):
            raise TypeError
        elif ' ' in data_key:
            raise ValueError("Illegal data key")
        edge_type: _typing.Tuple[str, str, str] = self.__get_canonical_edge_type()

        found = False
        for et in self.__dgl_graph_holder.graph.canonical_etypes:
            if all([a == b for a, b in zip(et, edge_type)]):
                found = True
                break
        if not found:
            raise ValueError("edge type not exist")

        if data_key in self.__dgl_graph_holder.graph.edges[edge_type].data:
            return self.__dgl_graph_holder.graph.edges[edge_type].data[data_key]
        else:
            raise KeyError  # todo: Complete error message

    def __setitem__(self, data_key: str, value: torch.Tensor):
        if not isinstance(data_key, str):
            raise TypeError
        elif ' ' in data_key:
            raise ValueError("Illegal data key")
        if not isinstance(value, torch.Tensor):
            raise TypeError
        if value.dim() == 0:
            raise ValueError
        edge_type: _typing.Tuple[str, str, str] = self.__get_canonical_edge_type()

        found = False
        for et in self.__dgl_graph_holder.graph.canonical_etypes:
            if all([a == b for a, b in zip(et, edge_type)]):
                found = True
                break
        if not found:
            raise ValueError("edge type not exist")

        self.__dgl_graph_holder.graph.edges[edge_type].data[data_key] = value

    def __delitem__(self, data_key: str):
        if not isinstance(data_key, str):
            raise TypeError
        elif ' ' in data_key:
            raise ValueError("Illegal data key")
        edge_type: _typing.Tuple[str, str, str] = self.__get_canonical_edge_type()

        found = False
        for et in self.__dgl_graph_holder.graph.canonical_etypes:
            if all([a == b for a, b in zip(et, edge_type)]):
                found = True
                break
        if not found:
            raise ValueError("edge type not exist")

        if data_key in self.__dgl_graph_holder.graph.edges[edge_type].data:
            del self.__dgl_graph_holder.graph.edges[edge_type].data[data_key]
        else:
            raise KeyError  # todo: Complete error message

    def __len__(self) -> int:
        edge_type: _typing.Tuple[str, str, str] = self.__get_canonical_edge_type()

        found = False
        for et in self.__dgl_graph_holder.graph.canonical_etypes:
            if all([a == b for a, b in zip(et, edge_type)]):
                found = True
                break
        if not found:
            raise ValueError("edge type not exist")

        return len(self.__dgl_graph_holder.graph.edges[edge_type].data)

    def __iter__(self) -> _typing.Iterator[str]:
        edge_type: _typing.Tuple[str, str, str] = self.__get_canonical_edge_type()

        found = False
        for et in self.__dgl_graph_holder.graph.canonical_etypes:
            if all([a == b for a, b in zip(et, edge_type)]):
                found = True
                break
        if not found:
            raise ValueError("edge type not exist")

        return iter(self.__dgl_graph_holder.graph.edges[edge_type].data)


class _HomogeneousEdgesView(_abstract_views.HomogeneousEdgesView):
    def __init__(
            self, dgl_graph_holder: _DGLGraphHolder,
            edge_type: _typing.Union[
                None, str, _typing.Tuple[str, str, str],
                _canonical_edge_type.CanonicalEdgeType
            ] = ...
    ):
        if not isinstance(dgl_graph_holder, _DGLGraphHolder):
            raise TypeError
        self.__dgl_graph_holder: _DGLGraphHolder = dgl_graph_holder
        if edge_type in (Ellipsis, None):
            self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = None
        elif isinstance(edge_type, str):
            if ' ' in edge_type:
                raise ValueError("Illegal edge type")
            self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = edge_type
        elif isinstance(edge_type, _typing.Sequence) and not isinstance(edge_type, str):
            if not (
                    len(edge_type) == 3 and
                    isinstance(edge_type[0], str) and ' ' not in edge_type[0] and
                    isinstance(edge_type[1], str) and ' ' not in edge_type[1] and
                    isinstance(edge_type[2], str) and ' ' not in edge_type[2]
            ):
                raise ValueError("Illegal edge type")
            self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = tuple(edge_type)
        elif isinstance(edge_type, _canonical_edge_type.CanonicalEdgeType):
            self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = (
                edge_type.source_node_type, edge_type.relation_type, edge_type.target_node_type
            )
        else:
            raise TypeError

    def __get_canonical_edge_type(self) -> _typing.Tuple[str, str, str]:
        if self.__optional_edge_type in (Ellipsis, None):
            if len(self.__dgl_graph_holder.graph.canonical_etypes) == 0:
                raise ValueError("The graph is empty")
            elif len(self.__dgl_graph_holder.graph.canonical_etypes) > 1:
                raise ValueError(
                    "Unable to automatically determine edge type, "
                    "the graph consists of heterogeneous edge types."
                )
            else:
                return self.__dgl_graph_holder.graph.canonical_etypes[0]
        elif isinstance(self.__optional_edge_type, str):
            try:
                canonical_edge_type = self.__dgl_graph_holder.graph.to_canonical_etype(
                    self.__optional_edge_type
                )
            except dgl.DGLError as e:
                raise e
            else:
                return canonical_edge_type
        else:
            return self.__optional_edge_type

    @property
    def connections(self) -> torch.Tensor:
        return torch.vstack(
            self.__dgl_graph_holder.graph.edges(etype=self.__get_canonical_edge_type())
        )

    @property
    def data(self) -> _HomogeneousEdgesDataView:
        return _HomogeneousEdgesDataView(self.__dgl_graph_holder, self.__optional_edge_type)


class _HeterogeneousEdgesView(_abstract_views.HeterogeneousEdgesView):
    def __init__(self, dgl_graph_holder: _DGLGraphHolder):
        if not isinstance(dgl_graph_holder, _DGLGraphHolder):
            raise TypeError
        self.__dgl_graph_holder: _DGLGraphHolder = dgl_graph_holder
        self.__optional_edge_type: _typing.Union[None, str, _typing.Tuple[str, str, str]] = None

    def __get_canonical_edge_type(self) -> _typing.Tuple[str, str, str]:
        if self.__optional_edge_type in (Ellipsis, None):
            if len(self.__dgl_graph_holder.graph.canonical_etypes) == 0:
                raise ValueError("The graph is empty")
            elif len(self.__dgl_graph_holder.graph.canonical_etypes) > 1:
                raise ValueError(
                    "Unable to automatically determine edge type, "
                    "the graph consists of heterogeneous edge types."
                )
            else:
                return self.__dgl_graph_holder.graph.canonical_etypes[0]
        elif isinstance(self.__optional_edge_type, str):
            try:
                canonical_edge_type = self.__dgl_graph_holder.graph.to_canonical_etype(
                    self.__optional_edge_type
                )
            except dgl.DGLError as e:
                raise e
            else:
                return canonical_edge_type
        else:
            return self.__optional_edge_type

    @property
    def connections(self) -> torch.Tensor:
        return _HomogeneousEdgesView(self.__dgl_graph_holder, ...).connections

    @property
    def data(self) -> _HomogeneousEdgesDataView:
        return _HomogeneousEdgesView(self.__dgl_graph_holder, ...).data

    @property
    def is_homogeneous(self) -> bool:
        return len(self.__dgl_graph_holder.graph.canonical_etypes) <= 1

    def set(
            self, edge_t: _typing.Union[None, str, _typing.Tuple[str, str, str]],
            connections: torch.LongTensor,
            data: _typing.Optional[_typing.Mapping[str, torch.Tensor]] = ...
    ):
        raise NotImplementedError  # todo: Complete this function or this error message

    def __getitem__(
            self,
            edge_t: _typing.Union[
                None, str, _typing.Tuple[str, str, str], _canonical_edge_type.CanonicalEdgeType
            ]
    ) -> _HomogeneousEdgesView:
        return _HomogeneousEdgesView(self.__dgl_graph_holder, edge_t)

    def __setitem__(
            self,
            edge_t: _typing.Union[
                None, str, _typing.Tuple[str, str, str], _canonical_edge_type.CanonicalEdgeType
            ],
            edges: _typing.Union[torch.LongTensor]
    ):
        raise NotImplementedError  # todo: Complete this function or this error message

    def __delitem__(
            self,
            edge_t: _typing.Union[
                None, str, _typing.Tuple[str, str, str], _canonical_edge_type.CanonicalEdgeType
            ]
    ):
        raise NotImplementedError  # todo: Complete this function or this error message

    def __len__(self) -> int:
        return len(self.__dgl_graph_holder.graph.canonical_etypes)

    def __iter__(self) -> _typing.Iterator[_canonical_edge_type.CanonicalEdgeType]:
        return iter([
            _canonical_edge_type.CanonicalEdgeType(et[0], et[1], et[2])
            for et in self.__dgl_graph_holder.graph.canonical_etypes
        ])

    def __contains__(
            self,
            edge_type: _typing.Union[
                str, _typing.Tuple[str, str, str], _canonical_edge_type.CanonicalEdgeType
            ]
    ) -> bool:
        # raise NotImplementedError
        if isinstance(edge_type, str):
            if ' ' in edge_type:
                raise ValueError("Illegal edge type")
            else:
                return edge_type in self.__dgl_graph_holder.graph.etypes
        elif isinstance(edge_type, _typing.Sequence) and not isinstance(edge_type, str):
            if not (
                    len(edge_type) == 3 and
                    isinstance(edge_type[0], str) and ' ' not in edge_type[0] and
                    isinstance(edge_type[1], str) and ' ' not in edge_type[1] and
                    isinstance(edge_type[2], str) and ' ' not in edge_type[2]
            ):
                raise ValueError("Illegal edge type")
            found = False
            for et in self.__dgl_graph_holder.graph.canonical_etypes:
                if all([a == b for a, b in zip(et, edge_type)]):
                    found = True
                    break
            return found
        elif isinstance(edge_type, _canonical_edge_type.CanonicalEdgeType):
            found = False
            for et in self.__dgl_graph_holder.graph.canonical_etypes:
                if (
                        et[0] == edge_type.source_node_type and
                        et[1] == edge_type.relation_type and
                        et[2] == edge_type.target_node_type
                ):
                    found = True
                    break
            return found
        else:
            raise TypeError


class _StaticGraphDataContainer(_typing.MutableMapping[str, torch.Tensor]):
    def __setitem__(self, data_key: str, data: torch.Tensor) -> None:
        raise NotImplementedError

    def __delitem__(self, data_key: str) -> None:
        raise NotImplementedError

    def __getitem__(self, data_key: str) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> _typing.Iterator[str]:
        raise NotImplementedError


class StaticGraphDataAggregation(_StaticGraphDataContainer):
    def __init__(
            self, graph_data: _typing.Optional[_typing.Mapping[str, torch.Tensor]] = ...
    ):
        self.__data: _typing.MutableMapping[str, torch.Tensor] = (
            dict(graph_data) if isinstance(graph_data, _typing.Mapping)
            else {}
        )

    def __setitem__(self, data_key: str, data: torch.Tensor) -> None:
        self.__data[data_key] = data

    def __delitem__(self, data_key: str) -> None:
        del self.__data[data_key]

    def __getitem__(self, data_key: str) -> torch.Tensor:
        return self.__data[data_key]

    def __len__(self) -> int:
        return len(self.__data)

    def __iter__(self) -> _typing.Iterator[str]:
        return iter(self.__data)


class _StaticGraphDataView(_abstract_views.GraphDataView):
    def __init__(self, graph_data_container: _StaticGraphDataContainer):
        self.__graph_data_container: _StaticGraphDataContainer = (
            graph_data_container
        )

    def __setitem__(self, data_key: str, data: torch.Tensor) -> None:
        self.__graph_data_container[data_key] = data

    def __delitem__(self, data_key: str) -> None:
        del self.__graph_data_container[data_key]

    def __getitem__(self, data_key: str) -> torch.Tensor:
        return self.__graph_data_container[data_key]

    def __len__(self) -> int:
        return len(self.__graph_data_container)

    def __iter__(self) -> _typing.Iterator[str]:
        return iter(self.__graph_data_container)


class GeneralStaticGraphDGLImplementation(
    _general_static_graph.GeneralStaticGraph
):
    def __init__(
            self, dgl_graph: dgl.DGLGraph,
            graph_data: _typing.Optional[_typing.Mapping[str, torch.Tensor]] = ...
    ):
        if not isinstance(dgl_graph, dgl.DGLGraph) and (
                graph_data in (Ellipsis, None) or
                isinstance(graph_data, _typing.Mapping)
        ):
            raise TypeError
        self.__dgl_graph_holder: _DGLGraphHolder = _DGLGraphHolder(dgl_graph)
        self.__graph_data_container: _StaticGraphDataContainer = (
            StaticGraphDataAggregation(
                graph_data if isinstance(graph_data, _typing.Mapping) else None
            )
        )

    @property
    def nodes(self) -> _abstract_views.HeterogeneousNodeView:
        return _HeterogeneousNodeView(self.__dgl_graph_holder)

    @property
    def edges(self) -> _abstract_views.HeterogeneousEdgesView:
        return _HeterogeneousEdgesView(self.__dgl_graph_holder)

    @property
    def data(self) -> _abstract_views.GraphDataView:
        return _StaticGraphDataView(self.__graph_data_container)
