from scipy.sparse import csr_matrix
import scipy.sparse as ssp
import networkx as nx
from .base import BaseGenerator
import numpy as np
from .. import register_feature


class Eigen(object):
    def __init__(self):
        pass

    def normalize_adj(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt = ssp.diags(d_inv_sqrt)
        return adj.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt)

    def forward(self, adj, d, use_eigenvalues=0, adj_norm=1):
        G = nx.from_scipy_sparse_matrix(adj)
        comp = list(nx.connected_components(G))
        results = np.zeros((adj.shape[0], d))
        for i in range(len(comp)):
            node_index = np.array(list(comp[i]))
            d_temp = min(len(node_index) - 2, d)
            if d_temp <= 0:
                continue
            temp_adj = adj[node_index, :][:, node_index].asfptype()
            if adj_norm == 1:
                temp_adj = self.normalize_adj(temp_adj)
            lamb, X = ssp.linalg.eigs(temp_adj, d_temp)
            lamb, X = lamb.real, X.real
            temp_order = np.argsort(lamb)
            lamb, X = lamb[temp_order], X[:, temp_order]
            for i in range(X.shape[1]):
                if np.sum(X[:, i]) < 0:
                    X[:, i] = -X[:, i]
            if use_eigenvalues == 1:
                X = X.dot(np.diag(np.sqrt(np.absolute(lamb))))
            elif use_eigenvalues == 2:
                X = X.dot(np.diag(lamb))
            results[node_index, :d_temp] = X
        return results


@register_feature("eigen")
class GeEigen(BaseGenerator):
    r"""concat Eigen features

    Notes
    -----
    An implementation of [#]_

    References
    ----------
    .. [#] Ziwei Zhang, Peng Cui, Jian Pei, Xin Wang, Wenwu Zhu:
        Eigen-GNN: A Graph Structure Preserving Plug-in for GNNs. CoRR abs/2006.04330 (2020)
        https://arxiv.org/abs/2006.04330


    Parameters
    ----------
    size : int
        EigenGNN hidden size
    """

    def __init__(self, size=32):
        super(GeEigen, self).__init__()
        self.size = size

    def _transform(self, data):

        adj = csr_matrix(
            (data.edge_weight, (data.edge_index[0, :], data.edge_index[1, :])),
            shape=(data.x.shape[0], data.x.shape[0]),
        )
        if np.max(adj - adj.T) > 1e-5:
            adj = adj + adj.T
        mf = Eigen()
        data.x = np.concatenate([data.x, mf.forward(adj, self.size)], axis=1)

        return data

    def _preprocess(self, data):
        if not hasattr(data, "edge_weight"):
            data.edge_weight = np.ones(data.edge_index.shape[1])

    def _postprocess(self, data):
        del data.edge_weight
