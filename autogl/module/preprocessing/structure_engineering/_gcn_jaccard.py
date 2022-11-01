# Modified from deeprobust Library
import scipy.sparse as sp
import numpy as np
from numba import njit
@njit
def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)

            if C < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt

@njit
def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt

@njit
def dropedge_both(A, iA, jA, features, threshold1=2.5, threshold2=0.01):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C1 = np.linalg.norm(features[n1] - features[n2])

            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C2 = inner_product / (np.sqrt(np.square(a).sum() + np.square(b).sum())+ 1e-6)
            if C1 > threshold1 or threshold2 < 0:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt

def drop_dissimilar_edges(features, adj,threshold,binary_feature=True,metric='similarity'):
    """Drop dissimilar edges.(Faster version using numba)
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_triu = sp.triu(adj, format='csr')

    if sp.issparse(features):
        features = features.todense().A # make it easier for njit processing

    if metric == 'distance':
        removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    else:
        if binary_feature:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    print('removed %s edges in the original graph' % removed_cnt)
    modified_adj = adj_triu + adj_triu.transpose()
    return modified_adj

from .._data_preprocessor_registry import DataPreprocessorUniversalRegistry
from ._structure_engineer import StructureEngineer,get_feature,get_edges,set_edges,to_adjacency_list,to_adjacency_matrix,LOGGER
import torch
@DataPreprocessorUniversalRegistry.register_data_preprocessor("gcnjaccard")
class GCNJaccard(StructureEngineer):
    """
    GCNJaccard preprocesses input graph via droppining dissimilar
    edges. See more details in
    Adversarial Examples on Graph Data: Deep Insights into Attack and Defense,
    https://arxiv.org/pdf/1903.01610.pdf.
    """
    def __init__(self, threshold=0.01,binary_feature=True, *args, **kwargs):
        """ drop dissimilar edges with similarity smaller than given threshold

        Parameters
        ----------
        threshold : float
            similarity threshold for dropping edges. If two connected nodes with similarity smaller than threshold, the edge between them will be removed.
        """
        super(GCNJaccard, self).__init__(*args, **kwargs)
        self.threshold=threshold
        self.binary_feature=binary_feature
    def _transform(self,data):
        features = get_feature(data)
        adj = get_edges(data) # edge list 
        LOGGER.info(f'before modified: {adj.shape}')
        adj = to_adjacency_matrix(adj).numpy() # adjacency matrix
        modified_adj = drop_dissimilar_edges(features, adj, self.threshold,self.binary_feature).toarray() # adjacency matrix
        modified_adj = to_adjacency_list(torch.Tensor(modified_adj)) # edge list
        LOGGER.info(f'after modified: {modified_adj.shape}' )
        set_edges(data,modified_adj)
        return data