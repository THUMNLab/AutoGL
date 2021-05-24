import typing as _typing
import torch.utils.data
import torch_geometric
from .target_dependant_sampler import TargetDependantSampler, TargetDependantSampledData


class NeighborSampler(TargetDependantSampler, _typing.Iterable):
    @classmethod
    def __compute_edge_weight(cls, edge_index: torch.LongTensor) -> torch.Tensor:
        __num_nodes = max(int(edge_index[0].max()), int(edge_index[1].max())) + 1
        __out_degree: torch.LongTensor = torch_geometric.utils.degree(
            edge_index[0], __num_nodes
        )
        __in_degree: torch.LongTensor = torch_geometric.utils.degree(
            edge_index[1], __num_nodes
        )
        temp_tensor: torch.Tensor = torch.stack(
            [__out_degree[edge_index[0]], __in_degree[edge_index[1]]]
        )
        temp_tensor: torch.Tensor = torch.pow(temp_tensor, -0.5)
        temp_tensor[torch.isinf(temp_tensor)] = 0.0
        return temp_tensor[0] * temp_tensor[1]

    def __init__(
            self, edge_index: torch.LongTensor,
            target_nodes_indexes: torch.LongTensor,
            sampling_sizes: _typing.Sequence[int],
            batch_size: int = 1, num_workers: int = 0,
            shuffle: bool = True, **kwargs
    ):
        self.__edge_weight: torch.Tensor = self.__compute_edge_weight(edge_index)
        self.__pyg_neighbor_sampler: torch_geometric.data.NeighborSampler = (
            torch_geometric.data.NeighborSampler(
                edge_index, list(sampling_sizes[::-1]), target_nodes_indexes,
                transform=self._transform, batch_size=batch_size,
                num_workers=num_workers, shuffle=shuffle, **kwargs
            )
        )

    def _transform(
        self, batch_size: int, n_id: torch.LongTensor,
        adj_or_adj_list: _typing.Union[
            _typing.Sequence[
                _typing.Tuple[torch.LongTensor, torch.LongTensor, _typing.Tuple[int, int]]
            ],
            _typing.Tuple[torch.LongTensor, torch.LongTensor, _typing.Tuple[int, int]]
        ]
    ) -> TargetDependantSampledData:
        if (
                isinstance(adj_or_adj_list[0], _typing.Tuple) and
                isinstance(adj_or_adj_list, _typing.Sequence) and
                not isinstance(adj_or_adj_list, _typing.Tuple)
        ):
            return TargetDependantSampledData(
                [
                    (current_layer[0], current_layer[1], self.__edge_weight[current_layer[1]])
                    for current_layer in adj_or_adj_list
                ],
                (torch.arange(batch_size, dtype=torch.long).long(), n_id[:batch_size]), n_id
            )
        elif isinstance(adj_or_adj_list, _typing.Tuple) and type(adj_or_adj_list[0]) == torch.Tensor:
            adj_or_adj_list: _typing.Tuple[
                torch.LongTensor, torch.LongTensor, _typing.Tuple[int, int]
            ] = adj_or_adj_list
            return TargetDependantSampledData(
                [(adj_or_adj_list[0], adj_or_adj_list[1], self.__edge_weight[adj_or_adj_list[1]])],
                (torch.arange(batch_size, dtype=torch.long).long(), n_id[:batch_size]), n_id
            )

    def __iter__(self):
        return iter(self.__pyg_neighbor_sampler)

    @classmethod
    def create_basic_sampler(
            cls, edge_index: torch.LongTensor,
            target_nodes_indexes: torch.LongTensor,
            layer_wise_arguments: _typing.Sequence,
            batch_size: int = 1, num_workers: int = 1,
            shuffle: bool = True, *args, **kwargs
    ) -> TargetDependantSampler:
        return cls(
            edge_index, target_nodes_indexes, layer_wise_arguments,
            batch_size, num_workers, shuffle, **kwargs
        )
