import numpy as np
import networkx as nx
import torch
import autogl
from ._basic import BaseFeatureGenerator
from ..._data_preprocessor_registry import DataPreprocessorUniversalRegistry


@DataPreprocessorUniversalRegistry.register_data_preprocessor("PageRank".lower())
class PageRankFeatureGenerator(BaseFeatureGenerator):
    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight").tolist()
        g = nx.DiGraph()
        g.add_weighted_edges_from(
            [
                (u, v, edge_weight[i])
                for i, (u, v) in enumerate(data.edge_index.t().tolist())
            ]
        )
        page_rank = nx.pagerank(g)
        num_nodes: int = (
            data.x.size(0)
            if data.x is not None and isinstance(data.x, torch.Tensor)
            else (data.edge_index.max().item() + 1)
        )
        pr = np.zeros(num_nodes)
        for i, v in page_rank.items():
            pr[i] = v
        return torch.from_numpy(pr).unsqueeze(-1)
