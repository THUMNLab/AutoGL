from .. import _data_preprocessor


class StructureEngineer(_data_preprocessor.DataPreprocessor):
    ...



import torch
from ....utils import get_logger
LOGGER = get_logger("Structure")

from torch_geometric.utils import to_dense_adj
def get_feature(data):
    """return features : numpy.ndarray
    """
    for fk in 'x feat'.split():
        if fk in data.nodes.data:
            features=data.nodes.data[fk].numpy()
    return features

def get_edges(data):
    return data.edges.connections

def set_edges(data,adj):
    data.data["edge_index"]=adj

def to_adjacency_matrix(adj):
    """
    adj : torch.Tensor [2,E]
    return Tensor [N,N]
    """
    adj = to_dense_adj(adj)[0].long() # adjacency matrix
    return adj
def to_adjacency_list(adj):
    """
    adj : Tensor [N,N]
    return Tensor [2,E]
    """
    adj = torch.stack(adj.nonzero(as_tuple=True)).long() # edge list 
    return adj

from .._data_preprocessor_registry import DataPreprocessorUniversalRegistry
from deeprobust.graph.defense.gcn_preprocess import GCNJaccard as Jaccard
@DataPreprocessorUniversalRegistry.register_data_preprocessor("gcnjaccard")
class GCNJaccard(StructureEngineer):
    """
    GCNJaccard preprocesses input graph via droppining dissimilar
    edges. See more details in
    Adversarial Examples on Graph Data: Deep Insights into Attack and Defense,
    https://arxiv.org/pdf/1903.01610.pdf.
    """
    def __init__(self, threshold=0.01, *args, **kwargs):
        """ drop dissimilar edges with similarity smaller than given threshold

        Parameters
        ----------
        threshold : float
            similarity threshold for dropping edges. If two connected nodes with similarity smaller than threshold, the edge between them will be removed.
        """
        super(GCNJaccard, self).__init__(*args, **kwargs)
        self.engine=Jaccard(2,2,2)
        self.engine.threshold=threshold
    def _transform(self,data):
        features = get_feature(data)
        adj = get_edges(data) # edge list 
        LOGGER.info(f'before modified: {adj.shape}')
        adj = to_adjacency_matrix(adj).numpy() # adjacency matrix
        modified_adj = self.engine.drop_dissimilar_edges(features, adj).toarray() # adjacency matrix
        modified_adj = to_adjacency_list(torch.Tensor(modified_adj)) # edge list
        LOGGER.info(f'after modified: {modified_adj.shape}' )
        set_edges(data,modified_adj)
        return data

from deeprobust.graph.defense.gcn_preprocess import GCNSVD as SVD
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
        self.engine=SVD(2,2,2)
        self.k=k
        self.threshold=threshold

    def _transform(self,data):
        adj = get_edges(data) # edge list
        LOGGER.info(f'before modified: {adj.shape}')
        adj = to_adjacency_matrix(adj).numpy() # adjacency matrix
        modified_adj = self.engine.truncatedSVD(adj,self.k) # adjacency matrix
        modified_adj = (modified_adj> self.threshold).astype(int)
        modified_adj = to_adjacency_list(torch.Tensor(modified_adj)) # edge list
        LOGGER.info(f'after modified: {modified_adj.shape}' )
        set_edges(data,modified_adj)
        return data