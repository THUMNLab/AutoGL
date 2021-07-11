import torch.utils.data
import typing as _typing


class TargetDependantSampledData:
    """
    A uniform aggregation of sampled data for one mini-batch,
    generally sampler by target-dependent sampler.
    Node-wise Sampling and Layer-wise Sampling techniques are definitely target-dependent,
    for which each sampled subgraph depends on the corresponding target nodes.
    Besides, the Subgraph-wise Sampling mechanism can also be treated as target-dependent,
    however, each set of target nodes for Subgraph-wise Sampling is determined by the sampled graph.

    Parameters
    ------------
    sampled_edges_for_layers:
        A sequence of tuple denoted as
        `( edge_index_for_sampled_graph, edge_id_in_integral_graph, (optional)edge_weight )`,
        where the `edge_index_for_sampled_graph` represents the sampled `edge_index` for sampled subgraph,
        the `edge_id_in_integral_graph` represents
        the corresponding positional indexes for the `edge_index` of integral graph,
        and the optional `edge_weight` for aggregation can also be provided.
    target_nodes_indexes:
        A tuple consists of (`torch.Tensor`, `torch.Tensor`),
        in which the first element represents the indexes of target nodes in sampled subgraph,
        and the second element represents the indexes of target nodes in the integral graph.
    all_sampled_nodes_indexes:
        Indexes of all sampled nodes for mini-batch.

    Attributes
    ------------
    target_nodes_indexes:
        A combined aggregation composed of
        `indexes_in_sampled_graph` and `indexes_in_integral_graph`
    all_sampled_nodes_indexes:
        Indexes of all sampled nodes for mini-batch.
    sampled_edges_for_layers:
        The stored sequence of tuple
        `( edge_index_for_sampled_graph, edge_id_in_integral_graph, (optional)edge_weight )`.
    """

    class _LayerSampledEdgeData:
        def __init__(
            self,
            edge_index_for_sampled_graph: torch.Tensor,
            edge_id_in_integral_graph: torch.Tensor,
            edge_weight: _typing.Optional[torch.Tensor],
        ):
            self.__edge_index_for_sampled_graph: torch.Tensor = (
                edge_index_for_sampled_graph
            )
            self.__edge_id_in_integral_graph: torch.Tensor = edge_id_in_integral_graph
            self.__edge_weight: _typing.Optional[torch.Tensor] = edge_weight

        @property
        def edge_index_for_sampled_graph(self) -> torch.LongTensor:
            edge_index_for_sampled_graph: _typing.Any = (
                self.__edge_index_for_sampled_graph
            )
            return edge_index_for_sampled_graph

        @property
        def edge_id_in_integral_graph(self) -> torch.LongTensor:
            edge_id_in_integral_graph: _typing.Any = self.__edge_id_in_integral_graph
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
        all_sampled_nodes_indexes: torch.Tensor,
    ):
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
    """
    An abstract base class for various target-dependent sampler
    """

    @classmethod
    def create_basic_sampler(
        cls,
        edge_index: torch.LongTensor,
        target_nodes_indexes: torch.LongTensor,
        layer_wise_arguments: _typing.Sequence,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        *args,
        **kwargs
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
    """
    The base class for various Layer-wise Sampling techniques,
    providing basic functionality of composing sampled data for mini-batches.

    Parameters
    ------------
    edge_index:
        edge index of integral graph
    target_nodes_indexes:
        indexes of target nodes in the integral graph
    layer_wise_arguments:
        layer-wise arguments for sampling
    batch_size:
        batch size for target nodes
    num_workers:
        number of workers
    shuffle:
        flag for shuffling, default to True
    kwargs:
        remaining keyword arguments
    """

    def __init__(
        self,
        edge_index: torch.LongTensor,
        target_nodes_indexes: torch.LongTensor,
        layer_wise_arguments: _typing.Sequence,
        batch_size: _typing.Optional[int] = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs
    ):
        self._edge_index: torch.LongTensor = edge_index
        self.__layer_wise_arguments: _typing.Sequence = layer_wise_arguments
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        super(BasicLayerWiseTargetDependantSampler, self).__init__(
            target_nodes_indexes.unique().numpy(),
            batch_size,
            shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs
        )

    @classmethod
    def create_basic_sampler(
        cls,
        edge_index: torch.LongTensor,
        target_nodes_indexes: torch.LongTensor,
        layer_wise_arguments: _typing.Sequence,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        *args,
        **kwargs
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
            edge_index,
            target_nodes_indexes,
            layer_wise_arguments,
            batch_size,
            num_workers,
            shuffle,
            **kwargs
        )

    def _sample_edges_for_layer(
        self,
        __current_layer_target_nodes_indexes: torch.LongTensor,
        __top_layer_target_nodes_indexes: torch.LongTensor,
        layer_argument: _typing.Any,
        *args,
        **kwargs
    ) -> _typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]:
        """
        Sample edges for one specific layer, expected to be implemented in subclass.

        Parameters
        ------------
        __current_layer_target_nodes_indexes:
            target nodes for current layer
        __top_layer_target_nodes_indexes:
            target nodes for top layer
        layer_argument:
            argument for current layer
        args:
            remaining positional arguments
        kwargs:
            remaining keyword arguments

        Returns
        --------
        edge_id_in_integral_graph:
            the corresponding positional indexes for the `edge_index` of integral graph
        edge_weight:
            the optional `edge_weight` for aggregation
        """
        raise NotImplementedError

    def _collate_fn(
        self, top_layer_target_nodes_indexes_list: _typing.List[int]
    ) -> TargetDependantSampledData:
        return self.__sample_layers(
            torch.tensor(top_layer_target_nodes_indexes_list).unique()
        )

    def __sample_layers(
        self, __top_layer_target_nodes_indexes: torch.LongTensor
    ) -> TargetDependantSampledData:
        sampled_edges_for_layers: _typing.List[
            _typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]
        ] = list()
        __current_layer_target_nodes_indexes: torch.LongTensor = (
            __top_layer_target_nodes_indexes
        )
        " Reverse self.__layer_wise_arguments from bottom-up to top-down "
        for layer_argument in self.__layer_wise_arguments[::-1]:
            current_layer_result: _typing.Tuple[
                torch.LongTensor, _typing.Optional[torch.Tensor]
            ] = self._sample_edges_for_layer(
                __current_layer_target_nodes_indexes,
                __top_layer_target_nodes_indexes,
                layer_argument,
            )
            __source_nodes_indexes_for_current_layer: torch.Tensor = self._edge_index[
                0, current_layer_result[0]
            ]
            __current_layer_target_nodes_indexes: torch.LongTensor = (
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
        __sampled_nodes_in_sub_graph_mapping: _typing.Dict[int, int] = dict(
            list(
                zip(
                    sampled_nodes_in_sub_graph.tolist(),
                    range(sampled_nodes_in_sub_graph.size(0)),
                )
            )
        )

        __sampled_edge_index_for_layers_in_sub_graph: _typing.Sequence[torch.Tensor] = [
            torch.stack(
                [
                    torch.tensor(
                        [
                            __sampled_nodes_in_sub_graph_mapping.get(node_index)
                            for node_index in self._edge_index[
                                0, current_layer_result[0]
                            ].tolist()
                        ]
                    ),
                    torch.tensor(
                        [
                            __sampled_nodes_in_sub_graph_mapping.get(node_index)
                            for node_index in self._edge_index[
                                1, current_layer_result[0]
                            ].tolist()
                        ]
                    ),
                ]
            )
            for current_layer_result in sampled_edges_for_layers
        ]

        return TargetDependantSampledData(
            [
                (temp_tuple[0], temp_tuple[1][0], temp_tuple[1][1])
                for temp_tuple in zip(
                    __sampled_edge_index_for_layers_in_sub_graph,
                    sampled_edges_for_layers,
                )
            ],
            (
                torch.tensor(
                    [
                        __sampled_nodes_in_sub_graph_mapping.get(
                            current_target_node_index_in_integral_data
                        )
                        for current_target_node_index_in_integral_data in __top_layer_target_nodes_indexes.tolist()
                        if current_target_node_index_in_integral_data
                        in __sampled_nodes_in_sub_graph_mapping
                    ]
                ).long(),  # Remap
                torch.tensor(
                    [
                        current_target_node_index_in_integral_data
                        for current_target_node_index_in_integral_data in __top_layer_target_nodes_indexes.tolist()
                        if current_target_node_index_in_integral_data
                        in __sampled_nodes_in_sub_graph_mapping
                    ]
                ).long(),
            ),
            sampled_nodes_in_sub_graph,
        )
