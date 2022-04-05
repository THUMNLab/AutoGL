import os
import argparse

import tqdm

os.environ["AUTOGL_BACKEND"] = 'pyg'

import torch.nn.functional
import autogl.datasets.utils.conversion
import autogl.module.sampling.graph_sampler
import torch_geometric
from torch_geometric.nn import GCNConv

mock_sampler_configurations = {
    'neighbor_sampler': {
        'num_neighbors': [25, 5],
        'batch_size': 256,
        'shuffle': True
    },
    'graph_saint_node_sampler': {
        'batch_size': 512,
        'num_steps': 10,
        'sample_coverage': 100
    },
    'graph_saint_edge_sampler': {
        'batch_size': 512,
        'num_steps': 10,
        'sample_coverage': 100
    },
    'graph_saint_random_walk_sampler': {
        'batch_size': 512,
        'walk_length': 2,
        'num_steps': 10,
        'sample_coverage': 100
    },
    'cluster_sampler': {
        'num_parts': 50,
        'recursive': False,
        'batch_size': 10,
        'shuffle': True,
        'num_workers': 8
    }
}


class GNN(torch.nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super(GNN, self).__init__()
        self._gcn = GCNConv(input_dimension, output_dimension)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """ This model is a trivial 1-layer GCN """
        return torch.log_softmax(self._gcn.forward(data.x, data.edge_index, data.edge_weight), dim=-1)


__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--sampler', default='cluster_sampler',
        choices=[
            'neighbor_sampler',
            'graph_saint_node_sampler',
            'graph_saint_edge_sampler',
            'graph_saint_random_walk_sampler',
            'cluster_sampler'
        ]
    )
    arguments = argument_parser.parse_args()

    sampler_name = arguments.sampler
    sampler_configurations = mock_sampler_configurations[sampler_name]

    cora_dataset = autogl.datasets.utils.conversion.to_pyg_dataset(
        autogl.datasets.build_dataset_from_name('cora')
    )
    cora_data = cora_dataset[0]
    row, col = cora_data.edge_index
    ''' Normalized edge_weight by in-degree '''
    cora_data.edge_weight = 1. / torch_geometric.utils.degree(col, cora_data.num_nodes)[col]

    gnn_model = GNN(cora_data.x.size(1), int(cora_data.y.max()) + 1).to(__device)

    graph_sampler = autogl.module.sampling.graph_sampler.instantiate_graph_sampler(
        sampler_name, cora_data, sampler_configurations
    )

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)

    ''' train '''
    for _epoch in tqdm.tqdm(range(10)):
        for sampled_subgraph in graph_sampler:
            sampled_batch_data = sampled_subgraph.data
            assert isinstance(sampled_batch_data, torch_geometric.data.Data)
            sampled_batch_data.to(__device)

            if (
                    hasattr(sampled_batch_data, 'node_norm') and
                    hasattr(sampled_batch_data, 'edge_norm') and
                    isinstance(sampled_batch_data.node_norm, torch.Tensor) and
                    isinstance(sampled_batch_data.edge_norm, torch.Tensor) and
                    torch.is_tensor(sampled_batch_data.node_norm) and
                    torch.is_tensor(sampled_batch_data.edge_norm) and
                    sampled_batch_data.node_norm.dim() == sampled_batch_data.edge_norm.dim() == 1 and
                    sampled_batch_data.node_norm.size(0) == sampled_batch_data.x.size(0) and
                    sampled_batch_data.edge_norm.size(0) == sampled_batch_data.edge_index.size(1)
            ):
                sampled_batch_data.edge_weight *= sampled_batch_data.edge_norm
                out = gnn_model(sampled_batch_data)
                loss = torch.nn.functional.nll_loss(out, sampled_batch_data.y, reduction='none')
                loss = (loss * sampled_batch_data.node_norm)[sampled_batch_data.train_mask].sum()
            else:
                out = gnn_model(sampled_batch_data)
                if (
                        hasattr(sampled_batch_data, 'target_nodes_index') and
                        isinstance(sampled_batch_data.target_nodes_index, torch.Tensor) and
                        torch.is_tensor(sampled_batch_data.target_nodes_index)
                ):
                    loss = torch.nn.functional.nll_loss(
                        out[sampled_batch_data.target_nodes_index],
                        sampled_batch_data.y[sampled_batch_data.target_nodes_index]
                    )
                else:
                    loss = torch.nn.functional.nll_loss(
                        out[sampled_batch_data.train_mask],
                        sampled_batch_data.y[sampled_batch_data.train_mask]
                    )

            loss.backward()
            optimizer.step()
