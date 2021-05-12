import numpy as np
import torch
import torch.utils.data
import typing as _typing
import torch_geometric
from . import target_dependant_sampler


class LayerDependentImportanceSampler(target_dependant_sampler.BasicLayerWiseTargetDependantSampler):
    class _Utility:
        @classmethod
        def compute_edge_weights(cls, __all_edge_index_with_self_loops: torch.Tensor) -> torch.Tensor:
            __out_degree: torch.Tensor = \
                torch_geometric.utils.degree(__all_edge_index_with_self_loops[0])
            __in_degree: torch.Tensor = \
                torch_geometric.utils.degree(__all_edge_index_with_self_loops[1])

            # temp_tensor: torch.Tensor = torch.zeros_like(__all_edge_index_with_self_loops)
            # temp_tensor[0] = __out_degree[__all_edge_index_with_self_loops[0]]
            # temp_tensor[1] = __in_degree[__all_edge_index_with_self_loops[1]]
            temp_tensor: torch.Tensor = torch.stack(
                [
                    __out_degree[__all_edge_index_with_self_loops[0]],
                    __in_degree[__all_edge_index_with_self_loops[1]]
                ]
            )
            temp_tensor: torch.Tensor = 1.0 / temp_tensor
            temp_tensor[torch.isinf(temp_tensor)] = 0.0
            return temp_tensor[0] * temp_tensor[1]

        @classmethod
        def get_candidate_source_nodes_probabilities(
                cls, all_candidate_edge_indexes: torch.Tensor,
                all_edge_index_with_self_loops: torch.Tensor,
                all_edge_weights: torch.Tensor
        ) -> _typing.Tuple[torch.LongTensor, torch.Tensor]:
            """
            :param all_candidate_edge_indexes:
            :param all_edge_index_with_self_loops: integral edge index with self-loops
            :param all_edge_weights:
            :return: (all_source_nodes_indexes, all_source_nodes_probabilities)
            """
            _all_candidate_edges: torch.Tensor = \
                all_edge_index_with_self_loops[:, all_candidate_edge_indexes]
            _all_candidate_edges_weights: torch.Tensor = \
                all_edge_weights[all_candidate_edge_indexes]

            all_candidate_source_nodes_indexes: torch.LongTensor = _all_candidate_edges[0].unique()
            all_candidate_source_nodes_probabilities: torch.Tensor = torch.tensor(
                [
                    torch.sum(
                        _all_candidate_edges_weights[_all_candidate_edges[0] == _current_source_node_index]
                    ).item() / torch.sum(_all_candidate_edges_weights).item()
                    for _current_source_node_index in all_candidate_source_nodes_indexes.tolist()
                ]
            )
            assert (
                    all_candidate_source_nodes_indexes.size() ==
                    all_candidate_source_nodes_probabilities.size()
            )
            return all_candidate_source_nodes_indexes, all_candidate_source_nodes_probabilities

        @classmethod
        def filter_selected_edges_by_source_nodes_and_target_nodes(
                cls, all_edges_with_self_loops: torch.Tensor,
                selected_source_node_indexes: torch.LongTensor,
                selected_target_node_indexes: torch.LongTensor
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
                torch.cat([
                    torch.where(all_edges_with_self_loops[0] == __current_selected_source_node_index)[0]
                    for __current_selected_source_node_index in selected_source_node_indexes.unique().tolist()
                ]).unique()
            ] = True
            selected_edges_mask_for_target_nodes: torch.Tensor = torch.zeros(
                all_edges_with_self_loops.size(1), dtype=torch.bool
            )
            selected_edges_mask_for_target_nodes[
                torch.cat([
                    torch.where(all_edges_with_self_loops[1] == __current_selected_target_node_index)[0]
                    for __current_selected_target_node_index in selected_target_node_indexes.unique().tolist()
                ])
            ] = True
            return torch.where(
                selected_edges_mask_for_source_nodes & selected_edges_mask_for_target_nodes
            )[0]

    def __init__(
            self, edge_index: torch.LongTensor,
            target_nodes_indexes: torch.LongTensor,
            layer_wise_arguments: _typing.Sequence,
            batch_size: _typing.Optional[int] = 1, num_workers: int = 0,
            shuffle: bool = True, **kwargs
    ):
        super().__init__(
            torch_geometric.utils.add_remaining_self_loops(edge_index)[0],
            target_nodes_indexes, layer_wise_arguments, batch_size, num_workers, shuffle, **kwargs
        )
        self.__all_edge_weights: torch.Tensor = self._Utility.compute_edge_weights(self._edge_index)

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
        if type(layer_argument) != int:
            raise TypeError
        elif not layer_argument > 0:
            raise ValueError
        else:
            sampled_node_size_budget: int = layer_argument

        all_candidate_edge_indexes: torch.LongTensor = torch.cat(
            [
                torch.where(self._edge_index[1] == current_target_node_index)[0]
                for current_target_node_index in target_nodes_indexes.unique().tolist()
            ]
        ).unique()
        __all_candidate_source_nodes_indexes, all_candidate_source_nodes_probabilities = \
            self._Utility.get_candidate_source_nodes_probabilities(
                all_candidate_edge_indexes,
                self._edge_index,
                self.__all_edge_weights
            )
        assert __all_candidate_source_nodes_indexes.size() == all_candidate_source_nodes_probabilities.size()

        """ Sampling """
        if sampled_node_size_budget < __all_candidate_source_nodes_indexes.numel():
            selected_source_node_indexes: torch.LongTensor = __all_candidate_source_nodes_indexes[
                torch.from_numpy(
                    np.unique(np.random.choice(
                        np.arange(__all_candidate_source_nodes_indexes.numel()), sampled_node_size_budget,
                        p=all_candidate_source_nodes_probabilities.numpy()
                    ))
                ).unique()
            ].unique()
        else:
            selected_source_node_indexes: torch.LongTensor = __all_candidate_source_nodes_indexes

        __selected_edges_indexes: torch.LongTensor = (
            self._Utility.filter_selected_edges_by_source_nodes_and_target_nodes(
                self._edge_index,
                selected_source_node_indexes, target_nodes_indexes
            )
        ).unique()

        non_normalized_selected_edges_weight: torch.Tensor = (
                self.__all_edge_weights[__selected_edges_indexes] / (
                    selected_source_node_indexes.numel() * torch.tensor(
                        [
                            all_candidate_source_nodes_probabilities[
                                __all_candidate_source_nodes_indexes == current_source_node_index
                                ].item()
                            for current_source_node_index
                            in self._edge_index[0, __selected_edges_indexes].tolist()
                        ]
                    )
                )
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
                __edge_weight[__current_mask_for_edges] = (
                        __edge_weight[__current_mask_for_edges] / (
                            torch.sum(__edge_weight[__current_mask_for_edges])
                        )
                )
            return __edge_weight

        normalized_selected_edges_weight: torch.Tensor = __normalize_edges_weight_by_target_nodes(
            self._edge_index[:, __selected_edges_indexes],
            non_normalized_selected_edges_weight
        )
        return __selected_edges_indexes, normalized_selected_edges_weight

    # todo: Migrated to the overrode _sample_edges_for_layer method, remove in the future version
    # def __sample_layer(
    #         self, target_nodes_indexes: torch.LongTensor,
    #         sampled_node_size_budget: int
    # ) -> _typing.Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
    #     """
    #     :param target_nodes_indexes:
    #             node indexes for target nodes in the top layer or nodes sampled in upper layer
    #     :param sampled_node_size_budget:
    #     :return: (Tensor, Tensor, LongTensor, LongTensor)
    #     """
    #     all_candidate_edge_indexes: torch.LongTensor = torch.cat(
    #         [
    #             torch.where(self._edge_index[1] == current_target_node_index)[0]
    #             for current_target_node_index in target_nodes_indexes.unique().tolist()
    #         ]
    #     ).unique()
    #     __all_candidate_source_nodes_indexes, all_candidate_source_nodes_probabilities = \
    #         self._Utility.get_candidate_source_nodes_probabilities(
    #             all_candidate_edge_indexes,
    #             self._edge_index,
    #             self.__all_edge_weights
    #         )
    #     assert __all_candidate_source_nodes_indexes.size() == all_candidate_source_nodes_probabilities.size()
    #
    #     """ Sampling """
    #     if sampled_node_size_budget < __all_candidate_source_nodes_indexes.numel():
    #         selected_source_node_indexes: torch.LongTensor = __all_candidate_source_nodes_indexes[
    #             torch.from_numpy(
    #                 np.unique(np.random.choice(
    #                     np.arange(__all_candidate_source_nodes_indexes.numel()), sampled_node_size_budget,
    #                     p=all_candidate_source_nodes_probabilities.numpy()
    #                 ))
    #             ).unique()
    #         ].unique()
    #     else:
    #         selected_source_node_indexes: torch.LongTensor = __all_candidate_source_nodes_indexes
    #
    #     __selected_edges_indexes: torch.LongTensor = (
    #         self._Utility.filter_selected_edges_by_source_nodes_and_target_nodes(
    #             self._edge_index,
    #             selected_source_node_indexes, target_nodes_indexes
    #         )
    #     ).unique()
    #
    #     non_normalized_selected_edges_weight: torch.Tensor = (
    #             self.__all_edge_weights[__selected_edges_indexes] / (
    #                 selected_source_node_indexes.numel() * torch.tensor(
    #                     [
    #                         all_candidate_source_nodes_probabilities[
    #                             __all_candidate_source_nodes_indexes == current_source_node_index
    #                         ].item()
    #                         for current_source_node_index
    #                         in self._edge_index[0, __selected_edges_indexes].tolist()
    #                     ]
    #                 )
    #             )
    #     )
    #
    #     def __normalize_edges_weight_by_target_nodes(
    #             __edge_index: torch.Tensor, __edge_weight: torch.Tensor
    #     ) -> torch.Tensor:
    #         if __edge_index.size(1) != __edge_weight.numel():
    #             raise ValueError
    #         for current_target_node_index in __edge_index[1].unique().tolist():
    #             __current_mask_for_edges: torch.BoolTensor = (
    #                     __edge_index[1] == current_target_node_index
    #             )
    #             __edge_weight[__current_mask_for_edges] = (
    #                 __edge_weight[__current_mask_for_edges] / (
    #                     torch.sum(__edge_weight[__current_mask_for_edges])
    #                 )
    #             )
    #         return __edge_weight
    #
    #     normalized_selected_edges_weight: torch.Tensor = __normalize_edges_weight_by_target_nodes(
    #         self._edge_index[:, __selected_edges_indexes],
    #         non_normalized_selected_edges_weight
    #     )
    #     return (
    #         self._edge_index[:, __selected_edges_indexes],
    #         normalized_selected_edges_weight,
    #         selected_source_node_indexes,
    #         __selected_edges_indexes
    #     )
