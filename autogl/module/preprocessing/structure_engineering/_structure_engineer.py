from .. import _data_preprocessor
class StructureEngineer(_data_preprocessor.DataPreprocessor):
    ...

import torch
from ....utils import get_logger
LOGGER = get_logger("Structure")

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
    num_nodes=adj.max().item()+1
    mat = torch.zeros((num_nodes,num_nodes), dtype=bool)
    mat[tuple(adj)]=1
    return mat

def to_adjacency_list(adj):
    """
    adj : Tensor [N,N]
    return Tensor [2,E]
    """
    adj = torch.stack(adj.nonzero(as_tuple=True)).long() # edge list 
    return adj



