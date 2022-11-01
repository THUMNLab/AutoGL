# Modified from deeprobust Library
import scipy.sparse as sp
import numpy as np
from numba import njit
def truncatedSVD(data, k=50):
    """Truncated SVD on input data.

    Parameters
    ----------
    data :
        input matrix to be decomposed
    k : int
        number of singular values and vectors to compute.

    Returns
    -------
    numpy.array
        reconstructed matrix.
    """
    print('=== GCN-SVD: rank={} ==='.format(k))
    if sp.issparse(data):
        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)
        print("rank_after = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        print("rank_before = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
        print("rank_after = {}".format(len(diag_S.nonzero()[0])))

    return U @ diag_S @ V

from .._data_preprocessor_registry import DataPreprocessorUniversalRegistry
from ._structure_engineer import StructureEngineer,get_feature,get_edges,set_edges,to_adjacency_list,to_adjacency_matrix,LOGGER
import torch
@DataPreprocessorUniversalRegistry.register_data_preprocessor("gcnsvd")
class GCNSVD(StructureEngineer):
    """GCNSVD uses Truncated SVD as preprocessing.See more details in All You Need Is Low (Rank): Defending
    Against Adversarial Attacks on Graphs,
    https://dl.acm.org/doi/abs/10.1145/3336191.3371789.
    """
    def __init__(self, k=50, threshold=0.05, *args, **kwargs):
        """perform rank-k approximation of adjacency matrix via
        truncated SVD

        Parameters
        ----------
        k : int
            number of singular values and vectors to compute.

        threshold : float
            edges with scores larger than threshold will be kept.
        """
        super(GCNSVD, self).__init__(*args, **kwargs)
        self.k=k
        self.threshold=threshold

    def _transform(self,data):
        adj = get_edges(data) # edge list
        LOGGER.info(f'before modified: {adj.shape}')
        adj = to_adjacency_matrix(adj).numpy() # adjacency matrix
        modified_adj = truncatedSVD(adj,self.k) # adjacency matrix
        modified_adj = (modified_adj> self.threshold).astype(int)
        modified_adj = to_adjacency_list(torch.Tensor(modified_adj)) # edge list
        LOGGER.info(f'after modified: {modified_adj.shape}' )
        set_edges(data,modified_adj)
        return data

