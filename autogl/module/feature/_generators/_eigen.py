import autogl
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as ssp
import scipy.sparse.linalg
import networkx as nx
import torch
from ._basic import BaseFeatureGenerator
from .._feature_engineer_registry import FeatureEngineerUniversalRegistry


class _Eigen:
    def __init__(self):
        ...

    @classmethod
    def __normalize_adj(cls, adj):
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt = ssp.diags(d_inv_sqrt)
        return adj.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt)

    def __call__(self, adj, d, use_eigenvalues=0, adj_norm=1):
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
                temp_adj = self.__normalize_adj(temp_adj)
            lamb, X = scipy.sparse.linalg.eigs(temp_adj, d_temp)
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


@FeatureEngineerUniversalRegistry.register_feature_engineer("eigen")
class EigenFeatureGenerator(BaseFeatureGenerator):
    r"""
    concat Eigen features

    Notes
    -----
    An implementation of [#]_

    References
    ----------
    .. [#] Ziwei Zhang, Peng Cui, Jian Pei, Xin Wang, Wenwu Zhu:
        Eigen-GNN: A Graph Structure Preserving Plug-in for GNNs. TKDE (2021)
        https://arxiv.org/abs/2006.04330


    Parameters
    ----------
    size : int
        EigenGNN hidden size
    """
    def __init__(self, size: int = 32):
        super(EigenFeatureGenerator, self).__init__()
        self.__size: int = size

    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        edge_index: np.ndarray = data.edge_index.numpy()
        edge_weight: np.ndarray = getattr(data, "edge_weight").numpy()
        num_nodes: int = (
            data.x.size(0)
            if data.x is not None and isinstance(data.x, torch.Tensor)
            else (data.edge_index.max().item() + 1)
        )
        adj = csr_matrix(
            (edge_weight, (edge_index[0, :], edge_index[1, :])),
            shape=(num_nodes, num_nodes)
        )
        if np.max(adj - adj.T) > 1e-5:
            adj = adj + adj.T
        mf = _Eigen()
        features: np.ndarray = mf(adj, self.__size)
        return torch.from_numpy(features)
