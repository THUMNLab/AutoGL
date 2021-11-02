from scipy.sparse import csr_matrix
import scipy.sparse as ssp
import networkx as nx
import numpy as np

# @register_feature('pagerank')
from .base import BaseGenerator
from .. import register_feature


@register_feature("pagerank")
class GePageRank(BaseGenerator):
    r"""concat pagerank features"""

    def _transform(self, data):
        graph = nx.DiGraph()
        w = data.edge_weight
        eg = [(u, v, w[i]) for i, (u, v) in enumerate(data.edge_index.T)]
        graph.add_weighted_edges_from(eg)
        pagerank = nx.pagerank(graph)
        pr = np.zeros((data.x.shape[0], 1))
        for i, v in pagerank.items():
            pr[i] = v
        data.x = np.concatenate([data.x, pr], axis=1)
        return data

    def _preprocess(self, data):
        if not hasattr(data, "edge_weight"):
            data.edge_weight = np.ones(data.edge_index.shape[1])

    def _postprocess(self, data):
        del data.edge_weight
