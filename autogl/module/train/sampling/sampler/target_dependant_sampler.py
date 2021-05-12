import torch.utils.data
import typing as _typing


class TargetDependantSampledData:
    class _LayerSampledEdgeData:
        def __init__(
                self, edge_index_for_sampled_graph: torch.Tensor,
                edge_id_in_integral_graph: torch.Tensor,
                edge_weight: _typing.Optional[torch.Tensor]
        ):
            self.__edge_index_for_sampled_graph: torch.Tensor = (
                edge_index_for_sampled_graph
            )
            self.__edge_id_in_integral_graph: torch.Tensor = (
                edge_id_in_integral_graph
            )
            self.__edge_weight: _typing.Optional[torch.Tensor] = edge_weight

        @property
        def edge_index_for_sampled_graph(self) -> torch.LongTensor:
            edge_index_for_sampled_graph: _typing.Any = (
                self.__edge_index_for_sampled_graph
            )
            return edge_index_for_sampled_graph

        @property
        def edge_id_in_integral_graph(self) -> torch.LongTensor:
            edge_id_in_integral_graph: _typing.Any = (
                self.__edge_id_in_integral_graph
            )
            return edge_id_in_integral_graph

        @property
        def edge_weight(self) -> _typing.Optional[torch.Tensor]:
            return self.__edge_weight

    class _TargetNodes:
        @property
        def indexes_in_sampled_graph(self) -> torch.LongTensor:
            indexes_in_sampled_graph: _typing.Any = self.__indexes_in_sampled_graph
            return indexes_in_sampled_graph

        @property
        def indexes_in_integral_graph(self) -> torch.LongTensor:
            indexes_in_integral_graph: _typing.Any = self.__indexes_in_integral_graph
            return indexes_in_integral_graph

        def __init__(
                self,
                indexes_in_sampled_graph: torch.Tensor,
                indexes_in_integral_graph: torch.Tensor,
        ):
            self.__indexes_in_sampled_graph: torch.Tensor = indexes_in_sampled_graph
            self.__indexes_in_integral_graph: torch.Tensor = indexes_in_integral_graph

    @property
    def target_nodes_indexes(self) -> _TargetNodes:
        """ indexes of target nodes in the integral graph """
        return self.__target_nodes_indexes

    @property
    def all_sampled_nodes_indexes(self) -> torch.LongTensor:
        """ indexes of all sampled nodes in the integral graph """
        all_sampled_nodes_indexes: _typing.Any = self.__all_sampled_nodes_indexes
        return all_sampled_nodes_indexes

    @property
    def sampled_edges_for_layers(self) -> _typing.Sequence[_LayerSampledEdgeData]:
        return self.__sampled_edges_for_layers

    def __init__(
            self,
            sampled_edges_for_layers: _typing.Sequence[
                _typing.Tuple[torch.Tensor, torch.Tensor, _typing.Optional[torch.Tensor]]
            ],
            target_nodes_indexes: _typing.Tuple[torch.Tensor, torch.Tensor],
            all_sampled_nodes_indexes: torch.Tensor
    ):
        """

        :param sampled_edges_for_layers: Sequence of tuple (
                                             edge_index_for_sampled_graph,
                                             edge_id_in_integral_graph,
                                             optional edge_weight
                                         )
        :param target_nodes_indexes: (indexes_in_sampled_data, indexes_in_integral_data)
        :param all_sampled_nodes_indexes: torch.Tensor
        """
        self.__sampled_edges_for_layers: _typing.Sequence[
            TargetDependantSampledData._LayerSampledEdgeData
        ] = [
            self._LayerSampledEdgeData(item[0], item[1], item[2])
            for item in sampled_edges_for_layers
        ]
        self.__target_nodes_indexes: TargetDependantSampledData._TargetNodes = (
            self._TargetNodes(target_nodes_indexes[0], target_nodes_indexes[1])
        )
        self.__all_sampled_nodes_indexes: torch.Tensor = all_sampled_nodes_indexes


class TargetDependantSampler(torch.utils.data.DataLoader, _typing.Iterable):
    @classmethod
    def create_basic_sampler(
            cls, edge_index: torch.LongTensor,
            target_nodes_indexes: torch.LongTensor,
            layer_wise_arguments: _typing.Sequence,
            batch_size: int = 1, num_workers: int = 0,
            shuffle: bool = True, *args, **kwargs
    ) -> "TargetDependantSampler":
        """
        :param edge_index: edge index of integral graph
        :param target_nodes_indexes: indexes of target nodes in the integral graph
        :param layer_wise_arguments: layer-wise arguments for sampling
        :param batch_size: batch size for target nodes, default to 1
        :param num_workers: number of workers, default to 0
        :param shuffle: flag for shuffling, default to True
        :param args: remaining positional arguments
        :param kwargs: remaining keyword arguments
        :return: instance of TargetDependantSampler
        """
        raise NotImplementedError
    
    def __iter__(self):
        return super(TargetDependantSampler, self).__iter__()


class BasicLayerWiseTargetDependantSampler(TargetDependantSampler):
    def __init__(
            self, edge_index: torch.LongTensor,
            target_nodes_indexes: torch.LongTensor,
            layer_wise_arguments: _typing.Sequence,
            batch_size: _typing.Optional[int] = 1, num_workers: int = 0,
            shuffle: bool = True, **kwargs
    ):
        self._edge_index: torch.LongTensor = edge_index
        self.__target_nodes_indexes: torch.LongTensor = target_nodes_indexes
        self.__layer_wise_arguments: _typing.Sequence = layer_wise_arguments
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        super(BasicLayerWiseTargetDependantSampler, self).__init__(
            self.__target_nodes_indexes.unique().tolist(),
            batch_size, shuffle, num_workers=num_workers,
            collate_fn=self._collate_fn, **kwargs
        )

    @classmethod
    def create_basic_sampler(
            cls, edge_index: torch.LongTensor,
            target_nodes_indexes: torch.LongTensor,
            layer_wise_arguments: _typing.Sequence,
            batch_size: int = 1, num_workers: int = 0,
            shuffle: bool = True, *args, **kwargs
    ) -> TargetDependantSampler:
        """
        :param edge_index: edge index of integral graph
        :param target_nodes_indexes: indexes of target nodes in the integral graph
        :param layer_wise_arguments: layer-wise arguments for sampling
        :param batch_size: batch size for target nodes
        :param num_workers: number of workers
        :param shuffle: flag for shuffling, default to True
        :param args: remaining positional arguments
        :param kwargs: remaining keyword arguments
        :return: instance of TargetDependantSampler
        """
        return BasicLayerWiseTargetDependantSampler(
            edge_index, target_nodes_indexes, layer_wise_arguments,
            batch_size, num_workers, shuffle, **kwargs
        )

    def _sample_edges_for_layer(
            self, target_nodes_indexes: torch.LongTensor,
            layer_argument: _typing.Any, *args, **kwargs
    ) -> _typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]:
        """
        Sample edges for one layer
        :param target_nodes_indexes: indexes of target nodes
        :param layer_argument: argument for current layer
        :param args: remaining positional arguments
        :param kwargs: remaining keyword arguments
        :return: (edge_id_in_integral_graph, edge_weight)
        """
        raise NotImplementedError

    def _collate_fn(
            self, top_layer_target_nodes_indexes_list: _typing.List[int]
    ) -> TargetDependantSampledData:
        return self.__sample_layers(top_layer_target_nodes_indexes_list)

    def __sample_layers(
            self, top_layer_target_nodes_indexes_list: _typing.Sequence[int]
    ) -> TargetDependantSampledData:
        sampled_edges_for_layers: _typing.List[
            _typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]
        ] = list()
        top_layer_target_nodes_indexes: torch.LongTensor = (
            torch.tensor(top_layer_target_nodes_indexes_list).unique()
        )   # sorted
        target_nodes_indexes: torch.LongTensor = top_layer_target_nodes_indexes
        " Reverse self.__layer_wise_arguments from bottom-up to top-down "
        for layer_argument in self.__layer_wise_arguments[::-1]:
            current_layer_result: _typing.Tuple[
                torch.LongTensor, _typing.Optional[torch.Tensor]
            ] = self._sample_edges_for_layer(target_nodes_indexes, layer_argument)
            __source_nodes_indexes_for_current_layer: torch.Tensor = (
                self._edge_index[0, current_layer_result[0]]
            )
            target_nodes_indexes: torch.LongTensor = (
                __source_nodes_indexes_for_current_layer.unique()
            )
            sampled_edges_for_layers.append(current_layer_result)
        """ Reverse sampled_edges_for_layers from top-down to bottom-up """
        sampled_edges_for_layers: _typing.Sequence[
            _typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]
        ] = sampled_edges_for_layers[::-1]

        sampled_nodes_in_sub_graph: torch.LongTensor = torch.cat(
            [
                self._edge_index[:, current_layer_result[0]].reshape([-1])
                for current_layer_result in sampled_edges_for_layers
            ]
        ).unique()
        __sampled_nodes_in_sub_graph_mapping: _typing.Dict[int, int] = dict(list(zip(
            sampled_nodes_in_sub_graph.tolist(),
            range(sampled_nodes_in_sub_graph.size(0))
        )))

        __sampled_edge_index_for_layers_in_sub_graph: _typing.Sequence[torch.Tensor] = [
            torch.stack([
                torch.tensor(
                    [
                        __sampled_nodes_in_sub_graph_mapping.get(node_index)
                        for node_index in self._edge_index[0, current_layer_result[0]].tolist()
                    ]
                ),
                torch.tensor(
                    [
                        __sampled_nodes_in_sub_graph_mapping.get(node_index)
                        for node_index in self._edge_index[1, current_layer_result[0]].tolist()
                    ]
                ),
            ])
            for current_layer_result in sampled_edges_for_layers
        ]

        return TargetDependantSampledData(
            [
                (temp_tuple[0], temp_tuple[1][0], temp_tuple[1][1]) for temp_tuple
                in zip(__sampled_edge_index_for_layers_in_sub_graph, sampled_edges_for_layers)
            ],
            (
                torch.tensor(
                    [
                        __sampled_nodes_in_sub_graph_mapping.get(current_target_node_index_in_integral_data)
                        for current_target_node_index_in_integral_data
                        in top_layer_target_nodes_indexes.tolist()
                    ]
                ).long(),  # Remap
                top_layer_target_nodes_indexes
            ),
            sampled_nodes_in_sub_graph
        )
