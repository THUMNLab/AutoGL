import torch_geometric


class GraphSAINTSamplerFactory:
    @classmethod
    def create_node_sampler(
            cls, data, num_graphs_per_epoch: int, node_budget: int,
            sample_coverage_factor: int = 50, **kwargs
    ) -> torch_geometric.data.GraphSAINTNodeSampler:
        return torch_geometric.data.GraphSAINTNodeSampler(
            data, node_budget,
            num_graphs_per_epoch, sample_coverage_factor, log=False, **kwargs
        )

    @classmethod
    def create_edge_sampler(
            cls, data, num_graphs_per_epoch: int, edge_budget: int,
            sample_coverage_factor: int = 50, **kwargs
    ) -> torch_geometric.data.GraphSAINTEdgeSampler:
        return torch_geometric.data.GraphSAINTEdgeSampler(
            data, edge_budget,
            num_graphs_per_epoch, sample_coverage_factor, log=False, **kwargs
        )

    @classmethod
    def create_random_walk_sampler(
            cls, data, num_graphs_per_epoch: int,
            num_walks: int, walk_length: int,
            sample_coverage_factor: int = 50, **kwargs
    ) -> torch_geometric.data.GraphSAINTRandomWalkSampler:
        return torch_geometric.data.GraphSAINTRandomWalkSampler(
            data, num_walks, walk_length,
            num_graphs_per_epoch, sample_coverage_factor, log=False, **kwargs
        )
