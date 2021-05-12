import typing as _typing
import torch.utils.data
import torch_geometric
from .target_dependant_sampler import TargetDependantSampler, TargetDependantSampledData


def _neighbor_sampler_transform(
        batch_size: int, n_id: torch.LongTensor,
        adj_list: _typing.Sequence[
            _typing.Tuple[torch.LongTensor, torch.LongTensor, _typing.Tuple[int, int]]
        ]
) -> TargetDependantSampledData:
    return TargetDependantSampledData(
        [(current_layer[0], current_layer[1], None)for current_layer in adj_list],
        (torch.arange(batch_size), n_id[:batch_size]), n_id
    )


class NeighborSampler(TargetDependantSampler, _typing.Iterable):
    def __init__(
            self, edge_index: torch.LongTensor,
            target_nodes_indexes: torch.LongTensor,
            sampling_sizes: _typing.Sequence[int],
            batch_size: int = 1, num_workers: int = 0,
            shuffle: bool = True, **kwargs
    ):
        self.__pyg_neighbor_sampler: torch_geometric.data.NeighborSampler = (
            torch_geometric.data.NeighborSampler(
                edge_index, list(sampling_sizes[::-1]), target_nodes_indexes,
                transform=_neighbor_sampler_transform, batch_size=batch_size,
                num_workers=num_workers, shuffle=shuffle, **kwargs
            )
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
