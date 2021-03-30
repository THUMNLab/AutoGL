import collections
import random
import typing as _typing
import numpy as np
import torch.utils.data


class NeighborSampler(torch.utils.data.DataLoader, collections.Iterable):
    class _NodeIndexesDataset(torch.utils.data.Dataset):
        def __init__(self, node_indexes):
            self.__node_indexes: _typing.Sequence[int] = node_indexes
        
        def __getitem__(self, index) -> int:
            if not 0 <= index < len(self.__node_indexes):
                raise IndexError("Index out of range")
            else:
                return self.__node_indexes[index]
        
        def __len__(self) -> int:
            return len(self.__node_indexes)
    
    def __init__(
            self, data,
            sampling_sizes: _typing.Sequence[int],
            target_node_indexes: _typing.Optional[_typing.Sequence[int]] = None,
            batch_size: _typing.Optional[int] = 1,
            *args, **kwargs
    ):
        self._data = data
        self.__sampling_sizes: _typing.Sequence[int] = sampling_sizes
        
        if not (
                target_node_indexes is not None and
                isinstance(target_node_indexes, _typing.Sequence)
        ):
            if hasattr(data, "train_mask"):
                target_node_indexes: _typing.Sequence[int] = \
                    torch.where(getattr(data, "train_mask"))[0]
            else:
                target_node_indexes: _typing.Sequence[int] = \
                    list(np.arange(0, data.x.shape[0]))
        
        self.__edge_index_map: _typing.Dict[
            int, _typing.Union[torch.Tensor, _typing.Sequence[int]]
        ] = {}
        self.__init_edge_index_map()
        super(NeighborSampler, self).__init__(
            self._NodeIndexesDataset(target_node_indexes),
            batch_size=batch_size if batch_size > 0 else 1,
            collate_fn=self.__sample, *args, **kwargs
        )
    
    def __init_edge_index_map(self):
        self.__edge_index_map.clear()
        all_edge_index: torch.Tensor = getattr(self._data, "edge_index")
        target_node_indexes: torch.Tensor = all_edge_index[1]
        for target_node_index in target_node_indexes.unique().tolist():
            self.__edge_index_map[target_node_index] = torch.where(
                all_edge_index[1] == target_node_index
            )[0]
    
    def __iter__(self):
        return super(NeighborSampler, self).__iter__()
    
    def __sample(
            self, target_nodes_indexes: _typing.List[int]
    ) -> _typing.Tuple[torch.Tensor, _typing.List[torch.Tensor]]:
        """
        Sample a sub-graph with neighborhood sampling
        :param target_nodes_indexes:
        """
        original_edge_index: torch.Tensor = self._data.edge_index
        edges_indexes: _typing.List[torch.Tensor] = []
        
        current_target_nodes_indexes: _typing.List[int] = target_nodes_indexes
        for current_sampling_size in self.__sampling_sizes:
            current_edge_index: _typing.Optional[torch.Tensor] = None
            for current_target_node_index in current_target_nodes_indexes:
                if current_target_node_index in self.__edge_index_map:
                    all_indexes: torch.Tensor = \
                        self.__edge_index_map.get(current_target_node_index)
                else:
                    all_indexes: torch.Tensor = torch.where(
                        original_edge_index[1] == current_target_node_index
                    )[0]
                if all_indexes.numel() < current_sampling_size:
                    sampled_indexes: np.ndarray = np.random.choice(
                        all_indexes.cpu().numpy(), current_sampling_size
                    )
                    if current_edge_index is not None:
                        current_edge_index: torch.Tensor = torch.cat(
                            [current_edge_index, original_edge_index[:, sampled_indexes]], dim=1
                        )
                    else:
                        current_edge_index: torch.Tensor = original_edge_index[:, sampled_indexes]
                else:
                    all_indexes_list = all_indexes.tolist()
                    random.shuffle(all_indexes_list)
                    shuffled_indexes_list: _typing.List[int] = \
                        all_indexes_list[0: current_sampling_size]
                    if current_edge_index is not None:
                        current_edge_index: torch.Tensor = torch.cat(
                            [current_edge_index, original_edge_index[:, shuffled_indexes_list]], dim=1
                        )
                    else:
                        current_edge_index: torch.Tensor = original_edge_index[:, shuffled_indexes_list]
            edges_indexes.append(current_edge_index)
            
            if len(edges_indexes) < len(self.__sampling_sizes):
                next_target_nodes_indexes: torch.Tensor = current_edge_index[0].unique()
                current_target_nodes_indexes = next_target_nodes_indexes.tolist()
        
        return torch.tensor(target_nodes_indexes), edges_indexes[::-1]
