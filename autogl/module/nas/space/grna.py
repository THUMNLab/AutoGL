
from numba.core.errors import reset_terminal
from numpy.lib.arraysetops import isin
from scipy import sparse
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import JumpingKnowledge
from torch_sparse import SparseTensor
import torch_sparse
from torch import Tensor
import numpy as np 
import scipy
import scipy.sparse as sp
from torch_geometric.utils import num_nodes, to_scipy_sparse_matrix,from_scipy_sparse_matrix,remove_self_loops,add_self_loops,add_remaining_self_loops
from numba import njit,jit
from tqdm import tqdm
from scipy.sparse import lil_matrix
from deeprobust.graph.utils import *
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.preprocessing import normalize

from . import register_nas_space
from .base import BaseSpace
from ...model import BaseAutoModel
from ....utils import get_logger
from .operation import gnn_map,act_map
import typing as _typ
from ..backend import *

from torch_geometric.nn import (
    # GATConv,
    GCNConv,
    ChebConv,
    GatedGraphConv,
    ARMAConv,
    SGConv,
)
from .gasso_space import GATConv,GINConv,SAGEConv

def gnn_map(gnn_name, in_dim, out_dim, concat=False, bias=True) -> nn.Module:
    """

    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    """
    if gnn_name == "gat_8":
        return GATConv(in_dim, out_dim, 8, concat=concat, bias=bias)
    elif gnn_name == "gat_6":
        return GATConv(in_dim, out_dim, 6, concat=concat, bias=bias)
    elif gnn_name == "gat_4":
        return GATConv(in_dim, out_dim, 4, concat=concat, bias=bias)
    elif gnn_name == "gat_2":
        return GATConv(in_dim, out_dim, 2, concat=concat, bias=bias)
    elif gnn_name in ["gat_1", "gat"]:
        return GATConv(in_dim, out_dim, 1, concat=concat, bias=bias)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim)
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "gated":
        return GatedGraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "gin":
        return GINConv(torch.nn.Linear(in_dim, out_dim))
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    raise KeyError("Cannot parse key %s" % (gnn_name))


PRE_ROB_OPS={
    'identity': lambda: RobustIdentity(),
    'svd': lambda: SVD(),
    'jaccard': lambda: Jaccard(),
    'gnnguard': lambda: GNNGuard(use_gate=False),
    'vpn': lambda: VPN(),
}

ROB_OPS={
    'identity': lambda: RobustIdentity(),
    'svd': lambda: SVD(),
    'jaccard': lambda: Jaccard(),
    'gnnguard': lambda: GNNGuard(),
    'vpn': lambda: VPN(),
}

# class RobOp(nn.Module):
#     def __init__(self, op, ):
#         super(RobOp, self).__init__()
#         self._op = ROB_OPS[ROB_PRIMITIVES[idx]]()

#     def forward(self, edge_index, edge_weight, features):
#         return self._op(edge_index, edge_weight,  features=features)

class RobustIdentity(nn.Module):
    
    def __init__(self):
        super(RobustIdentity, self).__init__()

    def forward(self, edge_index, edge_weight, features):
        return edge_weight
    
    def check_dense_matrix(self, symmetric=False):
        torch.cuda.empty_cache()
        self.modified_adj= torch.clamp(self.modified_adj,min=0,max=1)
        torch.cuda.empty_cache()

class SVD(RobustIdentity):
    def __init__(self):
        super(SVD, self).__init__()

    def forward(self, edge_index, edge_weight, features,  k=20):
        torch.cuda.empty_cache()
        # print('=== GCN-SVD: rank={} ==='.format(k))
        # if _A.is_sparse:
        # _A = torch.sparse.FloatTensor(edge_index, edge_weight, (features.size(0),features.size(0)))
        # sp_A= torch_sparse_to_scipy_sparse(_A)
        device = edge_index.device
        i = edge_index.cpu().numpy()
        sp_A = sp.csr_matrix((edge_weight.detach().cpu().numpy(), (i[0], i[1])), shape=(features.size(0),features.size(0)))
        row, col = sp_A.tocoo().row, sp_A.tocoo().col
        modified_adj = cal_svd(sp_A, k=k)
        adj_values = torch.tensor(modified_adj[row, col], dtype=torch.float32).to(device)
        adj_values = torch.clamp(adj_values, min=0)
        # else:
        #     sp_A = _A.to_sparse()
        #     modified_adj = cal_svd(torch_sparse_to_scipy_sparse(sp_A),k=k)
        #     modified_adj = torch.from_numpy(modified_adj).to(_A.device)

        #     # sparsify 
        #     self.modified_adj = torch.multiply(sp_A, modified_adj)
        #     self.check_dense_matrix()

        return adj_values

def cal_svd(sp_adj,k):
    adj = sp_adj.asfptype()
    U, S, V = sp.linalg.svds(adj, k=k)
    diag_S = np.diag(S)
    return U @ diag_S @ V



def torch_sparse_to_scipy_sparse(m, return_iv=False):
    '''
    Parameter:
    ----------
    m: torch.sparse matrix
    '''
    i = m.coalesce().indices().detach().cpu().numpy()
    v = m.coalesce().values().detach().cpu().numpy()
    shape = m.coalesce().size()
    
    sp_m = sp.csr_matrix((v, (i[0], i[1])), shape=shape)
    if return_iv:
        return sp_m, i, v
    else:
        return sp_m
        
class Jaccard(RobustIdentity):
    def __init__(self, ):
        super(Jaccard, self).__init__()
    
    def forward(self, edge_index, edge_weight, features, threshold=0.01):
        torch.cuda.empty_cache()
        """Drop dissimilar edges.(Faster version using numba)
        """
        # print("==GCN_Jaccard==")
        self.threshold = threshold
        features = features.detach().cpu().numpy()
        self.binary_feature = (features[features>0]==1).sum()==len(features[features>0])
        # print('Binary feature:',self.binary_feature )
        if sp.issparse(features):
            features = features.todense().A # make it easier for njit processing

        _A = torch.sparse.FloatTensor(edge_index, edge_weight, (features.shape[0],features.shape[0]))
        adj = torch_sparse_to_scipy_sparse(_A)
        adj_triu = sp.triu(adj, format='csr')

        if self.binary_feature:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        
        # print('removed %s edges in the original graph' % removed_cnt)
        modified_adj = adj_triu + adj_triu.transpose()
        
        row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        adj_values = torch.tensor(modified_adj.toarray()[row,col], dtype=torch.float32).to(edge_index.device)

        return adj_values

        # elif isinstance(_A, Tensor):
        #     self.modified_adj = self.drop_dissimilar_edges(features, _A)
        #     self.check_dense_matrix()

        # return self.modified_adj

    def drop_dissimilar_edges(self, features, adj):
        """Drop dissimilar edges. (Slower version)
        """
        # preprocessing based on features
        edges = np.array(adj.nonzero().detach().cpu()).T
        removed_cnt = 0
        for edge in edges:
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue

            if self.binary_feature:
                J = self._jaccard_similarity(features[n1], features[n2])

                if J < self.threshold:
                    adj[n1, n2] = 0
                    adj[n2, n1] = 0
                    removed_cnt += 1
            else:
                # For not binary feature, use cosine similarity
                C = self._cosine_similarity(features[n1], features[n2])
                if C < self.threshold:
                    adj[n1, n2] = 0
                    adj[n2, n1] = 0
                    removed_cnt += 1

        print('removed %s edges in the original graph' % removed_cnt)
        return adj

    def _jaccard_similarity(self, a, b):
        intersection = np.count_nonzero(np.multiply(a,b))
        if (np.count_nonzero(a) + np.count_nonzero(b) - intersection)==0:
            print(f'!Intersection=0!   a:{np.count_nonzero(a)}  b:{np.count_nonzero(b)}  intersection:{intersection}')
            with open('jaccard.txt','a') as f:
                f.write(f'{np.count_nonzero(a)}, {np.count_nonzero(b)}, {intersection}  \n')

            return 0
        J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)
        return J

    def _cosine_similarity(self, a, b):
        inner_product = (a * b).sum()
        C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-10)
        return C


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




class GNNGuard(RobustIdentity):
    def __init__(self, use_gate=True, drop=False):
        super(GNNGuard, self).__init__()
        self.drop = drop
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = Parameter(torch.rand(1)) # creat a generator between [0,1]


    def forward(self, edge_index, edge_weight, features):
        adj_values = self.att_coef(features, edge_index, i=0)
        # adj_2 = self.att_coef(x, adj, i=1)
        if self.use_gate:
            adj_values = self.gate * edge_weight + (1 - self.gate) * adj_values

        adj_values = torch.clamp(adj_values, min=0)
        
        return adj_values

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        # if is_lil == False:
        #     edge_index = edge_index._indices()
        # else:
        #     edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().numpy()[:], edge_index[1].cpu().numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        # sim_matrix = torch.from_numpy(sim_matrix)
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0
        # print('dropped {} edges'.format(1-sim.nonzero()[0].shape[0]/len(sim)))

        # """use jaccard for binary features and cosine for numeric features"""
        # fea_start, fea_end = fea[edge_index[0]], fea[edge_index[1]]
        # isbinray = np.array_equal(fea_copy, fea_copy.astype(bool))  # check is the fea are binary
        # np.seterr(divide='ignore', invalid='ignore')
        # if isbinray:
        #     fea_start, fea_end = fea_start.T, fea_end.T
        #     sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
        # else:
        #     fea_copy[np.isinf(fea_copy)] = 0
        #     fea_copy[np.isnan(fea_copy)] = 0
        #     sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        #     sim = sim_matrix[edge_index[0], edge_index[1]]
        #     sim[sim < 0.01] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')


        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                     att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            # degree = degree.squeeze(-1).squeeze(-1)
            lam = 1 / (degree + 1) # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        att_adj = edge_index
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).cuda()

        # shape = (n_node, n_node)
        # new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return att_edge_weight

    def add_loop_sparse(self, adj, fill_value=1):
        # make identify sparse tensor
        row = torch.range(0, int(adj.shape[0]-1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n.to(self.device)


class VPN(RobustIdentity):
    def __init__(self, r=2):
        super(VPN, self).__init__()
        self.r=r 

        self.theta_1 = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.theta_2 = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.theta_1.data.fill_(1)
        self.theta_2.data.fill_(0)

        if r>=3:
            self.theta_3 = Parameter(torch.FloatTensor(1), requires_grad=True)
            self.theta_3.data.fill_(0)

        elif r==4:
            self.theta_4 = Parameter(torch.FloatTensor(1), requires_grad=True)
            self.theta_4.data.fill_(0)

        self.edge_index=None
        
    def preprocess_adj_alls(self, edge_index, edge_weight, features):
        # print('======VPN======')
        num_nodes = features.size(0)
        self.device=edge_index.device
        self.use_sparse_adj = True

        i = edge_index.cpu().numpy()
        sp_A = sp.csr_matrix((edge_weight.detach().cpu().numpy(), (i[0],i[1])), shape=(num_nodes,num_nodes))
        sp_A = sp_A + sp.eye(num_nodes)
        sp_A[sp_A > 1] = 1
        # A^(r)
        adj_alls =[sp_A]
        for k in range(2, self.r+1):
            adj_k = sp_A ** k
            adj_alls.append(adj_k)

        return adj_alls

    def forward(self, edge_index, edge_weight, features):
        # if (self.edge_index is None) or (edge_index.size()!=self.edge_index.size()) or (edge_index!=self.edge_index).sum()>0:
        #     self.edge_index = edge_index
        self.adj_alls = self.preprocess_adj_alls(edge_index, edge_weight, features)

        # sparsify
        adj_values = self.sparsification(edge_index, edge_weight, self.adj_alls, features.detach().cpu().numpy())
        
        
        #====== dense version
        # # add self loops
        # _A = _A - torch.diag_embed(torch.diag(_A)).to(self.device)
        # _A = _A + torch.diag_embed(torch.ones(num_nodes)).to(self.device)
        # sp_A = _A.to_sparse()

        # # A^(r)
        # adj_alls =[_A]
        # for i in range(1, self.r):
        #     adj_k = torch.sparse.mm(sp_A, adj_alls[-1])
        #     adj_k[adj_k>1]=1.
        #     adj_alls.append(adj_k)
        
        # # sparsify
        # self.modified_adj = self.sparsification(adj_alls, features.detach().cpu().numpy())
        # if not _A.is_sparse:
        #     self.check_dense_matrix()
        # print('VPN:',self.theta_1, self.theta_2)
        adj_values = torch.clamp(adj_values, min=0)
        return adj_values

    def sparsification(self, edge_index, edge_weight, adjs, X, sparse_rate=2.0, metric='euclidean'):
        """
        Parameters
        --------
        adjs: list of torch.Tensor dense matrix /scipy sparse
            [A^(k)]
        x: numpy.array

        Returns:
        --------
        modified_adj: torch 
            modified dense adjacency matrix
        """
        # if sp.issparse(adjs[0]):
        # scipy.sparse
        _A,_A_robust = adjs[0], adjs[-1]
        sparse_ids = generate_sparse_ids(_A, _A_robust, X, sparse_rate=sparse_rate, metric=metric)

        row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        n_nodes = adjs[0].shape[0]

        adj_values = self.theta_1*edge_weight 
        if self.r>=2:
            adj_k = (adjs[1]-adjs[0]).tolil()
            adj_k.setdiag(np.zeros(n_nodes))
            for u in range(n_nodes):
                adj_k[u,sparse_ids[u]]=0
            adj_values = adj_values + self.theta_2 * torch.tensor(adj_k.toarray()[row,col], dtype=torch.float32).to(self.device)
        if self.r>=3:
            adj_k = (adjs[2]-adjs[1]).tolil()
            adj_k.setdiag(np.zeros(n_nodes))
            for u in range(n_nodes):
                adj_k[u,sparse_ids[u]]=0
            adj_values = adj_values + self.theta_3 * torch.tensor(adj_k.toarray()[row,col], dtype=torch.float32).to(self.device)

        # else:
        #     modified_adj = torch.from_numpy(adjs[0].todense()).to(self.device)* self.theta_1
        #     if self.r>=2:
        #         modified_adj += torch.from_numpy(adjs[1].todense()).to(self.device) * self.theta_2
        #     elif self.r>=3:
        #         modified_adj += torch.from_numpy(adjs[2].todense()).to(self.device) * self.theta_3
        #     elif self.r>=4:
        #         modified_adj += torch.from_numpy(adjs[3].todense()).to(self.device) * self.theta_4

        # print('modified adj VPN:', modified_adj)

        return adj_values

def generate_sparse_ids(_A, _A_robust, X, sparse_rate, metric):
    '''
    _A: origin adjacency matrix , sp.csr_matrix
    X: feature matrix. numpy.array
    return: list n
    '''
    sparse_ids = []
    _A.setdiag(np.zeros(_A.shape[0]))
    _D = np.count_nonzero(_A.toarray(), axis=1)

    _A_robust.setdiag(np.zeros(_A_robust.shape[0]))
    _D_r = np.count_nonzero(_A_robust.toarray(), axis=1)

    d_mean = np.mean(_D)
    d_std = np.std(_D)
    if metric=='correlation':
        d_thres = d_mean+2.5*d_std 
    else:
        d_thres = d_mean+2*d_std 
    highD,sparseN,nosparseN = 0,0,0

    for u in range(X.shape[0]):
        neighbors = _A_robust[u,:].nonzero()[1]
        x_neighbors = X[neighbors]
        x_u = X[u]
        
        if _D[u] > d_thres: # high degree nodes, do not power at all
            sparse_t = np.setdiff1d(np.arange(len(_D)), _A[u,:].nonzero()[1])
            sparse_ids.append(sparse_t)
            highD += 1
        elif round(sparse_rate*_D[u]) > _D_r[u]: # sparse rate too high, power all
            sparse_ids.append(np.array([]))
            nosparseN += 1
        else:
            if metric=='correlation':
                dist_2 = np.squeeze(scipy.spatial.distance.cdist(x_u,x_neighbors,'correlation'))
            else:
                dist_2 = np.sum((x_neighbors - x_u) ** 2, axis=1)
        
            nz_sel = int(_D_r[u]-round(sparse_rate*_D[u]))
            sparse_ids.append(neighbors[dist_2.argsort()[-nz_sel:]])
            sparseN += 1
        
    # print('highD {0}, sparified {1}, no sparsified {2}'.format(highD,sparseN,nosparseN))
    return sparse_ids
        

class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.lambd)
def act_map_nn(act):
    return LambdaModule(act_map(act))


GRAPHNAS_DEFAULT_ACT_OPS = [
    # "sigmoid", "tanh", "relu", "linear",
    #  "softplus", "leaky_relu", "relu6", "elu"
    "sigmoid",
    "tanh",
    "relu",
    "linear",
    "elu",
]

@register_nas_space("grnaspace")
class GRNASpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.6,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = ['gcn', "gat_2"],
        # rob_ops: _typ.Tuple = ["identity","svd","jaccard","gnnguard", "vpn"]
        rob_ops: _typ.Tuple = ["identity","svd","jaccard","gnnguard"],
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_ACT_OPS,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.rob_ops = rob_ops
        self.dropout = dropout
        self.act_ops = act_ops

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = None,
        rob_ops: _typ.Tuple = None,
        act_ops: _typ.Tuple = None,
        dropout=None,
    ):
        super().instantiate()
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        self.rob_ops = rob_ops or self.rob_ops
        self.act_ops = act_ops or self.act_ops
        self.dropout = dropout or self.dropout
        for layer in range(self.layer_number):
            setattr(
                self,
                f"op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        op(
                            self.input_dim if layer == 0 else self.hidden_dim,
                            self.output_dim
                            if layer == self.layer_number - 1
                            else self.hidden_dim,
                        )
                        if isinstance(op, type)
                        else gnn_map(
                            op,
                            self.input_dim if layer == 0 else self.hidden_dim,
                            self.output_dim
                            if layer == self.layer_number - 1
                            else self.hidden_dim,
                        )
                        for op in self.ops
                    ],
                ),
            )

            setattr(
                self,
                f"rob_op_{layer}",

                self.setLayerChoice(
                    layer,
                    [
                        op()
                        if isinstance(op, type)
                        else ROB_OPS[op]()
                        for op in self.rob_ops
                    ],
                ),
                
            )
            setattr(
            self,
            "act",
            self.setLayerChoice(
                2 * layer, [act_map_nn(a) for a in self.act_ops], key="act"
            ),
        )
        self._initialized = True

    def forward(self, data):
        x = bk_feat(data)
        edge_weight = data.edge_weight if data.edge_weight is not None else torch.ones(data.edge_index.size(1)).to(data.edge_index.device)
        for layer in range(self.layer_number):
            rob_op = getattr(self, f"rob_op_{layer}")
            edge_weight = rob_op(data.edge_index, edge_weight, x)
            op = getattr(self, f"op_{layer}")
            x = op(x, data.edge_index, edge_weight)
            if layer != self.layer_number - 1:
                # x = F.leaky_relu(x)
                act = getattr(self, "act")
                x = act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) -> BaseAutoModel:
        # return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap().fix(selection)