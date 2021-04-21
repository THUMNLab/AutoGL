import copy
import typing as _typing
import torch.utils.data
import torch_geometric


class _SubGraphSet(torch.utils.data.Dataset):
    def __init__(self, datalist: _typing.Sequence[_typing.Any], *args, **kwargs):
        self.__graphs: _typing.Sequence[_typing.Any] = datalist
        self.__remaining_args: _typing.Sequence[_typing.Any] = args
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __len__(self) -> int:
        return len(self.__graphs)
    
    def __getitem__(self, index: int) -> _typing.Any:
        if not 0 <= index < len(self.__graphs):
            raise IndexError
        return self.__graphs[index]


class _GraphSAINTSubGraphSampler:
    def __init__(
            self, sampler_class: _typing.Type[torch_geometric.data.GraphSAINTSampler],
            budget: int, num_graphs: int = 1, walk_length: int = 1, num_workers: int = 0
    ):
        """
        :param sampler_class: class of torch_geometric.data.GraphSAINTSampler
        :param budget: general budget
        :param num_graphs: number of sub-graphs to sample, i.e. N in the paper
        :param walk_length: walk length for RandomWalk Sampler
        :param num_workers: how many sub-processes to use for data loading.
                            0 means that the data will be loaded in the main process.
        """
        self.__sampler_class: _typing.Type[
            torch_geometric.data.GraphSAINTSampler
        ] = sampler_class
        self.__budget: int = budget
        self.__num_graphs: int = num_graphs
        self.__walk_length: int = walk_length
        self.__num_workers: int = num_workers if num_workers > 0 else 0
    
    def sample(self, _integral_data) -> _SubGraphSet:
        """
        :param _integral_data: conventional data for an integral graph
        :return: instance of _SubGraphSet
        """
        data = copy.copy(_integral_data)
        data.sampled_node_indexes = torch.arange(data.num_nodes, dtype=torch.int64)
        data.sampled_edge_indexes = torch.arange(data.num_edges, dtype=torch.int64)
        if type(self.__sampler_class) == torch_geometric.data.GraphSAINTRandomWalkSampler:
            _sampler: torch_geometric.data.GraphSAINTRandomWalkSampler = \
                torch_geometric.data.GraphSAINTRandomWalkSampler(
                    data, self.__budget, self.__walk_length, self.__num_graphs,
                    num_workers=self.__num_workers
                )
        else:
            _sampler: torch_geometric.data.GraphSAINTSampler = \
                self.__sampler_class(
                    data, self.__budget, self.__num_graphs,
                    num_workers=self.__num_workers
                )
        """ Sample sub-graphs """
        datalist: list = [d for d in _sampler]
        """ Compute the normalization """
        node_sampled_count = torch.zeros(data.num_nodes, dtype=torch.int64)
        edge_sampled_count = torch.zeros(data.num_edges, dtype=torch.int64)
        concatenated_sampled_nodes: torch.Tensor = torch.cat(
            [sub_graph.sampled_node_indexes for sub_graph in datalist]
        )
        concatenated_sampled_edges: torch.Tensor = torch.cat(
            [sub_graph.sampled_edge_indexes for sub_graph in datalist]
        )
        for current_sampled_node_index in concatenated_sampled_nodes.unique():
            node_sampled_count[current_sampled_node_index] = \
                torch.where(concatenated_sampled_nodes == current_sampled_node_index)[0].size(0)
        for current_sampled_edge_index in concatenated_sampled_edges.unique():
            edge_sampled_count[current_sampled_edge_index] = \
                torch.where(concatenated_sampled_edges == current_sampled_edge_index)[0].size(0)
        _alpha: torch.Tensor = edge_sampled_count / node_sampled_count[data.edge_index[1]]
        _alpha[torch.isnan(_alpha) | torch.isinf(_alpha)] = 0
        _lambda: torch.Tensor = node_sampled_count / self.__num_graphs
        return _SubGraphSet(datalist, **{"alpha": _alpha, "lambda": _lambda})


class GraphSAINTRandomNodeSampler(_GraphSAINTSubGraphSampler):
    def __init__(self, node_budget: int, num_graphs: int = 1):
        super(GraphSAINTRandomNodeSampler, self).__init__(
            torch_geometric.data.GraphSAINTNodeSampler, node_budget, num_graphs
        )


class GraphSAINTRandomEdgeSampler(_GraphSAINTSubGraphSampler):
    def __init__(self, edge_budget: int, num_graphs: int = 1):
        super(GraphSAINTRandomEdgeSampler, self).__init__(
            torch_geometric.data.GraphSAINTNodeSampler, edge_budget, num_graphs
        )


class GraphSAINTRandomWalkSampler(_GraphSAINTSubGraphSampler):
    def __init__(self, edge_budget: int, num_graphs: int = 1, walk_length: int = 4):
        super(GraphSAINTRandomWalkSampler, self).__init__(
            torch_geometric.data.GraphSAINTRandomWalkSampler, edge_budget, num_graphs, walk_length
        )
