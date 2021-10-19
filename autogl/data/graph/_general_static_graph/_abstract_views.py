import torch
import typing as _typing
from . import _canonical_edge_type


class SpecificTypedNodeDataView(_typing.MutableMapping[str, torch.Tensor]):
    def __getitem__(self, data_key: str) -> torch.Tensor:
        raise NotImplementedError

    def __setitem__(self, data_key: str, value: torch.Tensor):
        raise NotImplementedError

    def __delitem__(self, data_key: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> _typing.Iterator[str]:
        raise NotImplementedError


class SpecificTypedNodeView:
    @property
    def data(self) -> SpecificTypedNodeDataView:
        raise NotImplementedError

    @data.setter
    def data(self, nodes_data: _typing.Mapping[str, torch.Tensor]):
        raise NotImplementedError


class HeterogeneousNodeView(_typing.Iterable[str]):
    @property
    def data(self) -> SpecificTypedNodeDataView:
        raise NotImplementedError

    @data.setter
    def data(self, nodes_data: _typing.Mapping[str, torch.Tensor]):
        raise NotImplementedError

    def __getitem__(self, node_type: _typing.Optional[str]) -> SpecificTypedNodeView:
        raise NotImplementedError

    def __setitem__(
            self, node_t: _typing.Optional[str],
            nodes_data: _typing.Mapping[str, torch.Tensor]
    ):
        raise NotImplementedError

    def __delitem__(self, node_t: _typing.Optional[str]):
        raise NotImplementedError

    def __iter__(self) -> _typing.Iterator[str]:
        raise NotImplementedError

    @property
    def is_homogeneous(self) -> bool:
        raise NotImplementedError


class HomogeneousEdgesDataView(_typing.MutableMapping[str, torch.Tensor]):
    def __getitem__(self, data_key: str) -> torch.Tensor:
        raise NotImplementedError

    def __setitem__(self, data_key: str, value: torch.Tensor):
        raise NotImplementedError

    def __delitem__(self, data_key: str):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> _typing.Iterator[str]:
        raise NotImplementedError


class HomogeneousEdgesView:
    @property
    def connections(self) -> torch.LongTensor:
        raise NotImplementedError

    @property
    def data(self) -> HomogeneousEdgesDataView:
        raise NotImplementedError


class HeterogeneousEdgesView(_typing.Collection[_canonical_edge_type.CanonicalEdgeType]):
    @property
    def connections(self) -> torch.LongTensor:
        raise NotImplementedError

    @property
    def data(self) -> HomogeneousEdgesDataView:
        raise NotImplementedError

    @property
    def is_homogeneous(self) -> bool:
        raise NotImplementedError

    def set(
            self, edge_t: _typing.Union[None, str, _typing.Tuple[str, str, str]],
            connections: torch.LongTensor, data: _typing.Optional[_typing.Mapping[str, torch.Tensor]] = ...
    ):
        raise NotImplementedError

    def __getitem__(
            self,
            edge_t: _typing.Union[
                None, str, _typing.Tuple[str, str, str], _canonical_edge_type.CanonicalEdgeType
            ]
    ) -> HomogeneousEdgesView:
        raise NotImplementedError

    def __setitem__(
            self,
            edge_t: _typing.Union[
                None, str, _typing.Tuple[str, str, str], _canonical_edge_type.CanonicalEdgeType
            ],
            edges: _typing.Union[torch.LongTensor]
    ):
        raise NotImplementedError

    def __delitem__(
            self,
            edge_t: _typing.Union[
                None, str, _typing.Tuple[str, str, str], _canonical_edge_type.CanonicalEdgeType
            ]
    ):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> _typing.Iterator[_canonical_edge_type.CanonicalEdgeType]:
        raise NotImplementedError

    def __contains__(
            self,
            edge_type: _typing.Union[
                str, _typing.Tuple[str, str, str], _canonical_edge_type.CanonicalEdgeType
            ]
    ) -> bool:
        raise NotImplementedError


class GraphDataView(_typing.MutableMapping[str, torch.Tensor]):
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
