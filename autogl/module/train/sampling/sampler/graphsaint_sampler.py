import torch_geometric


class GraphSAINTSamplerFactory:
    """
    A simple factory class for creating varieties of
    :class:`torch_geometric.data.GraphSAINTSampler`.
    There exists potential sampling performance issues for
    the implementation of :class:`torch_geometric.data.GraphSAINTEdgeSampler`
    provided by PyTorch Geometric. Considering that the ultimate performance of
    GraphSAINT Edge Sampler and GraphSAINT Random Walk Sampler are similar
    according to the original literature
    `"GraphSAINT: Graph Sampling Based Inductive Learning Method"
    <https://arxiv.org/abs/1907.04931>`_ which introduces the GraphSAINT approach,
    nevertheless, when the walk length for GraphSAINT Random Walk Sampler is specified as `2`,
    the Random walk operation is actually selecting edges.
    Therefore an effective implementation for GraphSAINT Edge Sampler is not very urgently needed.
    Meanwhile, the varieties of Subgraph-wise sampling is scheduled to be redesigned and refactored.
    With the aim of abstracting a unified sampling module for representative mainstream varieties of
    Node-wise Sampling, Layer-wise Sampling, and Subgraph-wise Sampling.
    """

    @classmethod
    def create_node_sampler(
        cls,
        data,
        num_graphs_per_epoch: int,
        node_budget: int,
        sample_coverage_factor: int = 50,
        **kwargs
    ) -> torch_geometric.data.GraphSAINTNodeSampler:
        """
        A simple static method for instantiating :class:`torch_geometric.data.GraphSAINTNodeSampler`
        with more explicit arguments.

        Arguments
        ------------
        data:
            The conventional data of integral graph for sampling.
        num_graphs_per_epoch:
            number of subgraphs to sampler per epoch.
        node_budget:
            budget of nodes to sample for one sampled subgraph.
        sample_coverage_factor:
            The average number of samples per node should be used to
            compute normalization statistics.
        **kwargs:
            Additional optional arguments of :class:`torch.utils.data.DataLoader`,
            including :obj:`batch_size` or :obj:`num_workers`.

        Returns
        --------
        Instance of :class:`torch_geometric.data.GraphSAINTNodeSampler`.
        """
        return torch_geometric.data.GraphSAINTNodeSampler(
            data,
            node_budget,
            num_graphs_per_epoch,
            sample_coverage_factor,
            log=False,
            **kwargs
        )

    @classmethod
    def create_edge_sampler(
        cls,
        data,
        num_graphs_per_epoch: int,
        edge_budget: int,
        sample_coverage_factor: int = 50,
        **kwargs
    ) -> torch_geometric.data.GraphSAINTEdgeSampler:
        """
        A simple static method for instantiating :class:`torch_geometric.data.GraphSAINTEdgeSampler`
        with more explicit arguments.

        Arguments
        ------------
        data:
            The conventional data of integral graph for sampling.
        num_graphs_per_epoch:
            number of subgraphs to sampler per epoch.
        edge_budget:
            budget of edges to sample for one sampled subgraph.
        sample_coverage_factor:
            The average number of samples per node should be used to
            compute normalization statistics.
        **kwargs:
            Additional optional arguments of :class:`torch.utils.data.DataLoader`,
            including :obj:`batch_size` or :obj:`num_workers`.

        Returns
        --------
        Instance of :class:`torch_geometric.data.GraphSAINTEdgeSampler`.
        """
        return torch_geometric.data.GraphSAINTEdgeSampler(
            data,
            edge_budget,
            num_graphs_per_epoch,
            sample_coverage_factor,
            log=False,
            **kwargs
        )

    @classmethod
    def create_random_walk_sampler(
        cls,
        data,
        num_graphs_per_epoch: int,
        num_walks: int,
        walk_length: int,
        sample_coverage_factor: int = 50,
        **kwargs
    ) -> torch_geometric.data.GraphSAINTRandomWalkSampler:
        """
        A simple static method for instantiating :class:`torch_geometric.data.GraphSAINTEdgeSampler`
        with more explicit arguments.

        Arguments
        ------------
        data:
            The conventional data of integral graph for sampling.
        num_graphs_per_epoch:
            number of subgraphs to sampler per epoch.
        num_walks:
            The number of random walks for sampling.
        walk_length:
            The length of each random walk.
        sample_coverage_factor:
            The average number of samples per node should be used to
            compute normalization statistics.
        **kwargs:
            Additional optional arguments of :class:`torch.utils.data.DataLoader`,
            including :obj:`batch_size` or :obj:`num_workers`.

        Returns
        --------
        Instance of :class:`torch_geometric.data.GraphSAINTEdgeSampler`.
        """
        return torch_geometric.data.GraphSAINTRandomWalkSampler(
            data,
            num_walks,
            walk_length,
            num_graphs_per_epoch,
            sample_coverage_factor,
            log=False,
            **kwargs
        )
