import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data
import typing as _typing
import torch_geometric
from . import target_dependant_sampler


class _LayerDependentImportanceSampler(
    target_dependant_sampler.BasicLayerWiseTargetDependantSampler
):
    """
    Obsolete implementation, unused
    """

    class _Utility:
        @classmethod
        def compute_edge_weights(
            cls, __all_edge_index_with_self_loops: torch.Tensor
        ) -> torch.Tensor:
            __out_degree: torch.Tensor = torch_geometric.utils.degree(
                __all_edge_index_with_self_loops[0]
            )
            __in_degree: torch.Tensor = torch_geometric.utils.degree(
                __all_edge_index_with_self_loops[1]
            )

            temp_tensor: torch.Tensor = torch.stack(
                [
                    __out_degree[__all_edge_index_with_self_loops[0]],
                    __in_degree[__all_edge_index_with_self_loops[1]],
                ]
            )
            temp_tensor: torch.Tensor = torch.pow(temp_tensor, -0.5)
            temp_tensor[torch.isinf(temp_tensor)] = 0.0
            return temp_tensor[0] * temp_tensor[1]

        @classmethod
        def get_candidate_source_nodes_probabilities(
            cls,
            all_candidate_edge_indexes: torch.LongTensor,
            all_edge_index_with_self_loops: torch.Tensor,
            all_edge_weights: torch.Tensor,
        ) -> _typing.Tuple[torch.LongTensor, torch.Tensor]:
            """
            :param all_candidate_edge_indexes:
            :param all_edge_index_with_self_loops: integral edge index with self-loops
            :param all_edge_weights:
            :return: (all_source_nodes_indexes, all_source_nodes_probabilities)
            """
            all_candidate_edge_indexes: torch.LongTensor = (
                all_candidate_edge_indexes.unique()
            )
            _all_candidate_edges_weights: torch.Tensor = all_edge_weights[
                all_candidate_edge_indexes
            ]
            all_candidate_source_nodes_indexes: torch.LongTensor = (
                all_edge_index_with_self_loops[0, all_candidate_edge_indexes].unique()
            )
            all_candidate_source_nodes_probabilities: torch.Tensor = torch.tensor(
                [
                    torch.sum(
                        _all_candidate_edges_weights[
                            all_edge_index_with_self_loops[
                                0, all_candidate_edge_indexes
                            ]
                            == _current_source_node_index
                        ]
                    ).item()
                    / torch.sum(_all_candidate_edges_weights).item()
                    for _current_source_node_index in all_candidate_source_nodes_indexes.tolist()
                ]
            )
            assert (
                all_candidate_source_nodes_indexes.size()
                == all_candidate_source_nodes_probabilities.size()
            )
            return (
                all_candidate_source_nodes_indexes,
                all_candidate_source_nodes_probabilities,
            )

        @classmethod
        def filter_selected_edges_by_source_nodes_and_target_nodes(
            cls,
            all_edges_with_self_loops: torch.Tensor,
            selected_source_node_indexes: torch.LongTensor,
            selected_target_node_indexes: torch.LongTensor,
        ) -> torch.Tensor:
            """
            :param all_edges_with_self_loops: all edges with self loops
            :param selected_source_node_indexes: selected source node indexes
            :param selected_target_node_indexes: selected target node indexes
            :return: filtered edge indexes
            """
            selected_edges_mask_for_source_nodes: torch.Tensor = torch.zeros(
                all_edges_with_self_loops.size(1), dtype=torch.bool
            )
            selected_edges_mask_for_source_nodes[
                torch.cat(
                    [
                        torch.where(
                            all_edges_with_self_loops[0]
                            == __current_selected_source_node_index
                        )[0]
                        for __current_selected_source_node_index in selected_source_node_indexes.unique().tolist()
                    ]
                ).unique()
            ] = True
            selected_edges_mask_for_target_nodes: torch.Tensor = torch.zeros(
                all_edges_with_self_loops.size(1), dtype=torch.bool
            )
            selected_edges_mask_for_target_nodes[
                torch.cat(
                    [
                        torch.where(
                            all_edges_with_self_loops[1]
                            == __current_selected_target_node_index
                        )[0]
                        for __current_selected_target_node_index in selected_target_node_indexes.unique().tolist()
                    ]
                )
            ] = True
            return torch.where(
                selected_edges_mask_for_source_nodes
                & selected_edges_mask_for_target_nodes
            )[0]

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
        super().__init__(
            torch_geometric.utils.add_remaining_self_loops(edge_index)[0],
            target_nodes_indexes,
            layer_wise_arguments,
            batch_size,
            num_workers,
            shuffle,
            **kwargs
        )
        self.__all_edge_weights: torch.Tensor = self._Utility.compute_edge_weights(
            self._edge_index
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
        Sample edges for one layer
        :param __current_layer_target_nodes_indexes: target nodes for current layer
        :param __top_layer_target_nodes_indexes: target nodes for top layer
        :param layer_argument: argument for current layer
        :param args: remaining positional arguments
        :param kwargs: remaining keyword arguments
        :return: (edge_id_in_integral_graph, edge_weight)
        """
        if type(layer_argument) != int:
            raise TypeError
        elif not layer_argument > 0:
            raise ValueError
        else:
            sampled_node_size_budget: int = layer_argument

        all_candidate_edge_indexes: torch.LongTensor = torch.cat(
            [
                torch.where(self._edge_index[1] == current_target_node_index)[0]
                for current_target_node_index in __current_layer_target_nodes_indexes.unique().tolist()
            ]
        ).unique()
        (
            __all_candidate_source_nodes_indexes,
            all_candidate_source_nodes_probabilities,
        ) = self._Utility.get_candidate_source_nodes_probabilities(
            all_candidate_edge_indexes,
            self._edge_index,
            self.__all_edge_weights * self.__all_edge_weights,
        )
        assert (
            __all_candidate_source_nodes_indexes.size()
            == all_candidate_source_nodes_probabilities.size()
        )

        """ Sampling """
        if sampled_node_size_budget < __all_candidate_source_nodes_indexes.numel():
            selected_source_node_indexes: torch.LongTensor = (
                __all_candidate_source_nodes_indexes[
                    torch.from_numpy(
                        np.unique(
                            np.random.choice(
                                np.arange(__all_candidate_source_nodes_indexes.numel()),
                                sampled_node_size_budget,
                                p=all_candidate_source_nodes_probabilities.numpy(),
                                replace=False,
                            )
                        )
                    ).unique()
                ].unique()
            )
        else:
            selected_source_node_indexes: torch.LongTensor = (
                __all_candidate_source_nodes_indexes
            )
        selected_source_node_indexes: torch.LongTensor = torch.cat(
            [selected_source_node_indexes, __top_layer_target_nodes_indexes]
        ).unique()

        __selected_edges_indexes: torch.LongTensor = (
            self._Utility.filter_selected_edges_by_source_nodes_and_target_nodes(
                self._edge_index,
                selected_source_node_indexes,
                __current_layer_target_nodes_indexes,
            ).unique()
        )

        non_normalized_selected_edges_weight: torch.Tensor = self.__all_edge_weights[
            __selected_edges_indexes
        ] / torch.tensor(
            [
                all_candidate_source_nodes_probabilities[
                    __all_candidate_source_nodes_indexes == current_source_node_index
                ].item()
                for current_source_node_index in self._edge_index[
                    0, __selected_edges_indexes
                ].tolist()
            ]
        )

        def __normalize_edges_weight_by_target_nodes(
            __edge_index: torch.Tensor, __edge_weight: torch.Tensor
        ) -> torch.Tensor:
            if __edge_index.size(1) != __edge_weight.numel():
                raise ValueError
            for current_target_node_index in __edge_index[1].unique().tolist():
                __current_mask_for_edges: torch.BoolTensor = (
                    __edge_index[1] == current_target_node_index
                )
                __edge_weight[__current_mask_for_edges] = __edge_weight[
                    __current_mask_for_edges
                ] / torch.sum(__edge_weight[__current_mask_for_edges])
            return __edge_weight

        normalized_selected_edges_weight: torch.Tensor = (
            __normalize_edges_weight_by_target_nodes(
                self._edge_index[:, __selected_edges_indexes],
                non_normalized_selected_edges_weight,
            )
        )
        return __selected_edges_indexes, normalized_selected_edges_weight


class LayerDependentImportanceSampler(
    target_dependant_sampler.BasicLayerWiseTargetDependantSampler
):
    """
    The layer-dependent importance sampler from the
    `"Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1911.07323>`_ literature,  which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch training is not feasible.

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
        The number of nodes to sample for each layer.
        It's noteworthy that the target nodes for a specific layer
        always be preserved as source nodes for that layer,
        such that the self loops for those target nodes
        are generally preserved for representation learning.
    batch_size:
        number of target nodes for each mini-batch.
    num_workers:
        num_workers argument for inner :class:`torch.utils.data.DataLoader`
    shuffle:
        whether to shuffle target nodes for mini-batches.
    """

    @classmethod
    def __compute_edge_weight(cls, edge_index: torch.Tensor) -> torch.Tensor:
        __num_nodes: int = max(int(edge_index[0].max()), int(edge_index[1].max())) + 1
        _temp_tensor: torch.Tensor = torch.stack(
            [
                torch_geometric.utils.degree(edge_index[0], __num_nodes)[edge_index[0]],
                torch_geometric.utils.degree(edge_index[1], __num_nodes)[edge_index[1]],
            ]
        )
        _temp_tensor: torch.Tensor = torch.pow(_temp_tensor, -0.5)
        _temp_tensor[torch.isinf(_temp_tensor)] = 0
        return _temp_tensor[0] * _temp_tensor[1]

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
        super(LayerDependentImportanceSampler, self).__init__(
            torch_geometric.utils.add_remaining_self_loops(edge_index)[0],
            target_nodes_indexes,
            layer_wise_arguments,
            batch_size,
            num_workers,
            shuffle,
            **kwargs
        )
        self.__edge_weight: torch.Tensor = self.__compute_edge_weight(self._edge_index)
        self.__integral_normalized_l_matrix: sp.csr_matrix = sp.csr_matrix(
            (
                self.__edge_weight.numpy(),
                (self._edge_index[1].numpy(), self._edge_index[0].numpy()),
            )
        )
        self.__integral_edges_indexes_sparse_matrix: sp.csr_matrix = sp.csr_matrix(
            (
                np.arange(self._edge_index.size(1)),
                (self._edge_index[1].numpy(), self._edge_index[0].numpy()),
            )
        )

    def __sample_edges(
        self,
        __current_layer_target_nodes_indexes: np.ndarray,
        __top_layer_target_nodes_indexes: np.ndarray,
        sampled_source_nodes_budget: int,
    ) -> _typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        :param __current_layer_target_nodes_indexes: indexes of target nodes for current layer
        :param __top_layer_target_nodes_indexes: indexes of target nodes for top layer
        :param sampled_source_nodes_budget: sampled source nodes budget
        :return: (
                    sampled_edges_indexes,
                    sampled_source_nodes_indexes,
                    corresponding probabilities for sampled_source_nodes_indexes
        )
        """
        partial_l_matrix: sp.csr_matrix = self.__integral_normalized_l_matrix[
            __current_layer_target_nodes_indexes, :
        ]
        p: np.ndarray = np.array(
            np.sum(partial_l_matrix.multiply(partial_l_matrix), axis=0)
        )[0]
        p: np.ndarray = p / np.sum(p)
        _number_of_nodes_to_sample = np.min(
            [np.sum(p > 0), sampled_source_nodes_budget]
        )
        _selected_source_nodes: np.ndarray = np.unique(
            np.concatenate(
                [
                    np.random.choice(
                        p.size, _number_of_nodes_to_sample, replace=False, p=p
                    ),
                    __top_layer_target_nodes_indexes,
                ]
            )
        )

        _sampled_edges_indexes_sparse_matrix: sp.csr_matrix = (
            self.__integral_edges_indexes_sparse_matrix[
                __current_layer_target_nodes_indexes, :
            ]
        )
        _sampled_edges_indexes_sparse_matrix: sp.csc_matrix = (
            _sampled_edges_indexes_sparse_matrix.tocsc()[:, _selected_source_nodes]
        )
        _sampled_edges_indexes: np.ndarray = np.unique(
            _sampled_edges_indexes_sparse_matrix.data
        )

        return _sampled_edges_indexes, _selected_source_nodes, p[_selected_source_nodes]

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
        __wrapped_result: _typing.Tuple[
            np.ndarray, np.ndarray, np.ndarray
        ] = self.__sample_edges(
            __current_layer_target_nodes_indexes.numpy(),
            __top_layer_target_nodes_indexes.numpy(),
            layer_argument,
        )
        _sampled_edges_indexes: torch.Tensor = torch.from_numpy(__wrapped_result[0])
        _selected_source_nodes: torch.Tensor = torch.from_numpy(__wrapped_result[1])
        _selected_source_nodes_probabilities: torch.Tensor = torch.from_numpy(
            __wrapped_result[2]
        )

        """ Multiply corresponding discount weights """
        __selected_source_node_probability_mapping: _typing.Dict[int, float] = dict(
            zip(
                _selected_source_nodes.tolist(),
                _selected_source_nodes_probabilities.tolist(),
            )
        )
        _selected_edges_weight: torch.Tensor = self.__edge_weight[
            _sampled_edges_indexes
        ]
        _selected_edges_weight: torch.Tensor = _selected_edges_weight / torch.tensor(
            [
                __selected_source_node_probability_mapping.get(
                    _current_source_node_index
                )
                for _current_source_node_index in self._edge_index[
                    0, _sampled_edges_indexes
                ].tolist()
            ]
        )

        """ Normalize edge weight for selected edges by target nodes """
        for _current_target_node_index in (
            self._edge_index[1, _sampled_edges_indexes].unique().tolist()
        ):
            _current_mask_for_selected_edges: torch.BoolTensor = (
                self._edge_index[1, _sampled_edges_indexes]
                == _current_target_node_index
            )
            _selected_edges_weight[
                _current_mask_for_selected_edges
            ] = _selected_edges_weight[_current_mask_for_selected_edges] / torch.sum(
                _selected_edges_weight[_current_mask_for_selected_edges]
            )

        _sampled_edges_indexes: _typing.Union[
            torch.LongTensor, torch.Tensor
        ] = _sampled_edges_indexes
        return _sampled_edges_indexes, _selected_edges_weight
