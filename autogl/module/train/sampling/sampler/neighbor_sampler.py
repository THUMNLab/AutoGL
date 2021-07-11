import typing as _typing
import torch.utils.data
import torch_geometric
from .target_dependant_sampler import TargetDependantSampler, TargetDependantSampledData


class NeighborSampler(TargetDependantSampler, _typing.Iterable):
    """
    The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ literature, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Arguments
    ------------
    edge_index:
        A :obj:`torch.LongTensor` that defines the underlying graph
        connectivity/message passing flow.
        :obj:`edge_index` holds the indices of a (sparse) adjacency matrix.
        If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
        must be defined as :obj:`[2, num_edges]`, where messages from nodes
        :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
        (in case :obj:`flow="source_to_target"`).
    target_nodes_indexes:
        indexes of target nodes to learn representation.
    sampling_sizes:
        The number of neighbors to sample for each node in each layer.
        If set to :obj:`sampling_sizes[l] = -1`, all neighbors are included in layer :obj:`l`.
    batch_size:
        number of target nodes for each mini-batch.
    num_workers:
        num_workers argument for inner :class:`torch.utils.data.DataLoader`
    shuffle:
        whether to shuffle target nodes for mini-batches.
    """

    class _SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, sequence):
            self.__sequence = sequence

        def __len__(self):
            return len(self.__sequence)

        def __getitem__(self, idx):
            return self.__sequence[idx]

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
        self,
        edge_index: torch.LongTensor,
        target_nodes_indexes: torch.LongTensor,
        sampling_sizes: _typing.Sequence[int],
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs
    ):
        def is_deterministic(__cached: bool = bool(kwargs.get("cached", True))) -> bool:
            if not __cached:
                return False
            _deterministic: bool = True
            for _sampling_size in sampling_sizes:
                if type(_sampling_size) != int:
                    raise TypeError(
                        "The sampling_sizes argument must be a sequence of integer"
                    )
                if _sampling_size >= 0:
                    _deterministic = False
                    break
            return _deterministic

        self.__edge_weight: torch.Tensor = self.__compute_edge_weight(edge_index)
        self.__pyg_neighbor_sampler: torch_geometric.data.NeighborSampler = (
            torch_geometric.data.NeighborSampler(
                edge_index,
                list(sampling_sizes[::-1]),
                target_nodes_indexes,
                transform=self._transform,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                **kwargs
            )
        )

        if is_deterministic():
            pyg_neighbor_sampler: _typing.Iterable = self.__pyg_neighbor_sampler
            self.__cached_sampled_data_list: _typing.Optional[
                _typing.List[TargetDependantSampledData]
            ] = [sampled_data for sampled_data in pyg_neighbor_sampler]
        else:
            self.__cached_sampled_data_list: _typing.Optional[
                _typing.List[TargetDependantSampledData]
            ] = None

    def _transform(
        self,
        batch_size: int,
        n_id: torch.LongTensor,
        adj_or_adj_list: _typing.Union[
            _typing.Sequence[
                _typing.Tuple[
                    torch.LongTensor, torch.LongTensor, _typing.Tuple[int, int]
                ]
            ],
            _typing.Tuple[torch.LongTensor, torch.LongTensor, _typing.Tuple[int, int]],
        ],
    ) -> TargetDependantSampledData:
        if (
            isinstance(adj_or_adj_list[0], _typing.Tuple)
            and isinstance(adj_or_adj_list, _typing.Sequence)
            and not isinstance(adj_or_adj_list, _typing.Tuple)
        ):
            return TargetDependantSampledData(
                [
                    (
                        current_layer[0],
                        current_layer[1],
                        self.__edge_weight[current_layer[1]],
                    )
                    for current_layer in adj_or_adj_list
                ],
                (torch.arange(batch_size, dtype=torch.long).long(), n_id[:batch_size]),
                n_id,
            )
        elif (
            isinstance(adj_or_adj_list, _typing.Tuple)
            and type(adj_or_adj_list[0]) == torch.Tensor
        ):
            adj_or_adj_list: _typing.Tuple[
                torch.LongTensor, torch.LongTensor, _typing.Tuple[int, int]
            ] = adj_or_adj_list
            return TargetDependantSampledData(
                [
                    (
                        adj_or_adj_list[0],
                        adj_or_adj_list[1],
                        self.__edge_weight[adj_or_adj_list[1]],
                    )
                ],
                (torch.arange(batch_size, dtype=torch.long).long(), n_id[:batch_size]),
                n_id,
            )

    def __iter__(self):
        if self.__cached_sampled_data_list is not None and isinstance(
            self.__cached_sampled_data_list, _typing.Sequence
        ):
            return iter(
                torch.utils.data.DataLoader(
                    self._SequenceDataset(self.__cached_sampled_data_list),
                    collate_fn=lambda x: x[0],
                )
            )
        else:
            return iter(self.__pyg_neighbor_sampler)

    @classmethod
    def create_basic_sampler(
        cls,
        edge_index: torch.LongTensor,
        target_nodes_indexes: torch.LongTensor,
        layer_wise_arguments: _typing.Sequence,
        batch_size: int = 1,
        num_workers: int = 1,
        shuffle: bool = True,
        *args,
        **kwargs
    ) -> TargetDependantSampler:
        """
        A static factory method to create instance of :class:`NeighborSampler`

        Arguments
        ------------
        edge_index:
            A :obj:`torch.LongTensor` that defines the underlying graph
            connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a (sparse) adjacency matrix.
            If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
            must be defined as :obj:`[2, num_edges]`, where messages from nodes
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
        target_nodes_indexes:
            indexes of target nodes to learn representation.
        layer_wise_arguments:
            The number of neighbors to sample for each node in each layer.
            If set to :obj:`sampling_sizes[l] = -1`, all neighbors are included in layer :obj:`l`.
        batch_size:
            number of target nodes for each mini-batch.
        num_workers:
            num_workers argument for inner :class:`torch.utils.data.DataLoader`
        shuffle:
            whether to shuffle target nodes for mini-batches.
        """
        return cls(
            edge_index,
            target_nodes_indexes,
            layer_wise_arguments,
            batch_size,
            num_workers,
            shuffle,
            **kwargs
        )
