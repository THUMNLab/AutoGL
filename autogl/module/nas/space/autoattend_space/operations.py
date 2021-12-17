
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ARMAConv, ChebConv, GatedGraphConv, SGConv

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import numpy as np
from ..graph_nas_macro import GeoLayer


class AggAdd(nn.Module):
    def __init__(self, dim, att_head, dropout=0, norm=False, skip_connect=False, *args, **kwargs):
        super(AggAdd, self).__init__()
        self.dropout = dropout
        self.ln_add = nn.BatchNorm1d(
            dim, track_running_stats=True, affine=True)
        self.norm = norm
        self.skip_connect = skip_connect

    def forward(self, x, edge_index, *args, **kwargs):
        # x=[x_stem,[x_sides]]
        norm = self.norm
        x1, x2, x3 = x[0], x[1][0], x[1][1]
        if norm:
            return self.ln_add(x1 + x2)
        else:
            return x1 + x2


class AggAttn(MessagePassing):
    def __init__(self, dim, att_head, dropout=0, norm=False, skip_connect=False, *args, **kwargs):
        super(AggAttn, self).__init__()
        self.dropout = dropout
        self.att_head = att_head
        self.ln_attn = nn.BatchNorm1d(
            dim, track_running_stats=True, affine=True)
        self.norm = norm
        self.skip_connect = skip_connect

    def __repr__(self) -> str:
        return 'AggAttn(att_head={}, dropout={})'.format(self.att_head, self.dropout)

    def forward(self, x, edge_index, *args, **kwargs):
        # x=[x_stem,[x_sides]]
        # use dot-product attn
        x1, x2, x3 = x[0], x[1][0], x[1][1]  # q,k,v
        skip_connect, norm = self.skip_connect, self.norm
        if not skip_connect and not norm:
            return self.propagate(edge_index, x1=x1, x2=x2, x3=x3)

        x = self.propagate(edge_index, x1=x1, x2=x2, x3=x3)
        if not norm:
            return x

        if not skip_connect:
            return self.ln_attn(x)
        return self.ln_attn(x + x1)

    def message(self, x2_j, x1_i, x3_j, index, ptr):
        # x1: query, x2: key, x3: value # torch.Size([10556, 64]) ,index torch.Size([10556])
        node, dim = x1_i.size()
        dim_att = dim // self.att_head
        # torch.Size([10556, 8, 8])
        x2_j = x2_j.view(node, self.att_head, dim_att)
        # torch.Size([10556, 8, 8])
        x1_i = x1_i.view(node, self.att_head, dim_att)
        attn = (x2_j * x1_i).sum(dim=-1) / \
            np.sqrt(dim_att)  # torch.Size([10556, 8])
        attn = softmax(attn, index, ptr)  # torch.Size([10556, 8])
        # torch.Size([10556, 8])
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = x3_j.view(node, self.att_head, dim_att) * attn.unsqueeze(-1)
        out = out.view(-1, dim)
        return out


class GATConv2(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = False,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(GATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj):
        H, C = self.heads, self.out_channels

        x = self.lin(x).view(-1, H, C)
        alpha = (x * self.att).sum(dim=-1)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x,
                             alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Zero(nn.Module):
    def __init__(self, indim, outdim) -> None:
        super().__init__()
        self.outdim = outdim
        self.zero = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, edge_index):
        return torch.zeros(x.size(0), self.outdim).to(x.device) * self.zero

# class Zero(nn.Module):
#     def __init__(self, indim, outdim) -> None:
#         super().__init__()
#         self.outdim = outdim
#         self.ln = nn.Linear(1, 1)

#     def forward(self, x, edge_index):
#         return 0.


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, edge_index):
        return x


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.core = nn.Linear(in_dim, out_dim)

    def forward(self, x, *args):
        return self.core(x)


agg_map = {
    'add': lambda dim, att_head=None, dropout=0, norm=False, skip_connect=False: AggAdd(dim, att_head, dropout, norm, skip_connect),
    'attn': lambda dim, att_head=None, dropout=0, norm=False, skip_connect=False: AggAttn(dim, att_head, dropout, norm, skip_connect),
}
