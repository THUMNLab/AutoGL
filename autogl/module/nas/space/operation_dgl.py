import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import *
import dgl.function as fn
import math
class LinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, g, x, *args,**kwargs):
        return self.linear(x)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class ZeroConv(nn.Module):
    def forward(self,  g, x, *args,**kwargs):
        out = torch.zeros_like(x)
        out.requires_grad = True
        return out

    def __repr__(self):
        return "ZeroConv()"


class Identity(nn.Module):
    def forward(self,  g, x, *args,**kwargs):
        return x

    def __repr__(self):
        return "Identity()"

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class ARMAConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_stacks=1,
                 num_layers=1,
                 activation=None,
                 dropout=0.0,
                 bias=True):
        super(ARMAConv, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = num_stacks
        self.T = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        # init weight
        self.w_0 = nn.ModuleDict({
            str(k): nn.Linear(in_dim, out_dim, bias=False) for k in range(self.K)
        })
        # deeper weight
        self.w = nn.ModuleDict({
            str(k): nn.Linear(out_dim, out_dim, bias=False) for k in range(self.K)
        })
        # v
        self.v = nn.ModuleDict({
            str(k): nn.Linear(in_dim, out_dim, bias=False) for k in range(self.K)
        })
        # bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.K, self.T, 1, self.out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            glorot(self.w_0[str(k)].weight)
            glorot(self.w[str(k)].weight)
            glorot(self.v[str(k)].weight)
        zeros(self.bias)

    def forward(self, g, feats):
        with g.local_scope():
            init_feats = feats
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            output = None

            for k in range(self.K):
                feats = init_feats
                for t in range(self.T):
                    feats = feats * norm
                    g.ndata['h'] = feats
                    g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feats = g.ndata.pop('h')
                    feats = feats * norm

                    if t == 0:
                        feats = self.w_0[str(k)](feats)
                    else:
                        feats = self.w[str(k)](feats)
                    
                    feats += self.dropout(self.v[str(k)](init_feats))
                    feats += self.v[str(k)](self.dropout(init_feats))

                    if self.bias is not None:
                        feats += self.bias[k][t]
                    
                    if self.activation is not None:
                        feats = self.activation(feats)
                    
                if output is None:
                    output = feats
                else:
                    output += feats
                
            return output / self.K 
class GATConvC(GATConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 concat=False
                 ):
        super(GATConvC, self).__init__(in_feats,
                 out_feats,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 activation,
                 allow_zero_in_degree,
                 bias)
        self.concat=concat
    def forward(self, graph, feat, get_attention=False):
        x=super().forward(graph, feat, get_attention=get_attention)
        if get_attention:
            x=x[0]
        out=x
        if self.concat:
            out = out.view(-1, self._num_heads * self._out_feats)
        else:
            out = out.mean(dim=1)
        return out
        
class MChebConv(ChebConv):
    def __init__(self, in_feats, out_feats, k, activation=F.relu,bias=True):
        super().__init__(in_feats, out_feats, k, activation=activation, bias=bias)
    def forward(self, graph, feat):
        lambda_max=dgl.laplacian_lambda_max(graph) # may be very slow
        return super().forward(graph, feat, lambda_max=lambda_max)

def gnn_map(gnn_name, in_dim, out_dim, concat=False, bias=True) -> nn.Module:
    """

    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    """
    # now gat,sage,chebconv may be different from pyg
    if gnn_name == "gat_8":
        return GATConvC(in_dim, out_dim, 8, concat=concat,bias=bias)
    elif gnn_name == "gat_6":
        return GATConvC(in_dim, out_dim, 6, concat=concat, bias=bias)
    elif gnn_name == "gat_4":
        return GATConvC(in_dim, out_dim, 4,  concat=concat,bias=bias)
    elif gnn_name == "gat_2":
        return GATConvC(in_dim, out_dim, 2,   concat=concat,bias=bias)
    elif gnn_name in ["gat_1", "gat"]:
        return GATConvC(in_dim, out_dim, 1, concat=concat, bias=bias)
    elif gnn_name == "gcn":
        return GraphConv(in_dim, out_dim)
    elif gnn_name == "cheb":
        return MChebConv(in_dim, out_dim, k=2, bias=bias)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, bias=bias,aggregator_type='pool')
    elif gnn_name == "gated":
        return GatedGraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "zero":
        return ZeroConv()
    elif gnn_name == "identity":
        return Identity()
    elif hasattr(dgl.nn.conv, gnn_name):
        cls = getattr(dgl.nn.conv, gnn_name)
        assert isinstance(cls, type), "Only support modules, get %s" % (gnn_name)
        kwargs = {
            "in_channels": in_dim,
            "out_channels": out_dim,
            "concat": concat,
            "bias": bias,
        }
        kwargs = {
            key: kwargs[key]
            for key in cls.__init__.__code__.co_varnames
            if key in kwargs
        }
        return cls(**kwargs)
    raise KeyError("Cannot parse key %s" % (gnn_name))



from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from torch.nn import Parameter

import torch
from torch import Tensor
from ..scatter_utils import *
from typing import Optional, Tuple, Union
OptTensor = Optional[Tensor]

def softmax(src: Tensor, index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
            dim: int = 0) -> Tensor:
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def add_remaining_self_loops(
        edge_index: Tensor, edge_attr: OptTensor = None,
        fill_value: Union[float, Tensor, str] = None,
        num_nodes: Optional[int] = None) -> Tuple[Tensor, OptTensor]:
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of non-existing self-loops will
    be added according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    mask = edge_index[0] != edge_index[1]

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], 1.)

        elif isinstance(fill_value, (int, float)):
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:],
                                           fill_value)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
                                reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        inv_mask = ~mask
        loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]

        edge_attr = torch.cat([edge_attr[mask], loop_attr], dim=0)

    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
    return edge_index, edge_attr

class GeoLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        att_type="gat",
        agg_type="sum",
        pool_dim=0,
    ):
        super(GeoLayer,self).__init__()
        if agg_type in ["sum", "mlp"]:
            agg_type="add"
        elif agg_type in ["mean", "max"]:
            pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type
        self.reduce_func={
            'add':fn.sum,
            'mean':fn.mean,
            'max':fn.max
        }.get(self.agg_type,None)

        # GCN weight
        self.gcn_weight = None

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        if self.att_type in ["generalized_linear"]:
            self.general_att_layer = torch.nn.Linear(out_channels, 1, bias=False)

        if self.agg_type in ["mean", "max", "mlp"]:
            if pool_dim <= 0:
                pool_dim = 128
        self.pool_dim = pool_dim
        if pool_dim != 0:
            self.pool_layer = torch.nn.ModuleList()
            self.pool_layer.append(torch.nn.Linear(self.out_channels, self.pool_dim))
            self.pool_layer.append(torch.nn.Linear(self.pool_dim, self.out_channels))
        else:
            pass
        self.reset_parameters()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

        if self.att_type in ["generalized_linear"]:
            glorot(self.general_att_layer.weight)

        if self.pool_dim != 0:
            for layer in self.pool_layer:
                glorot(layer.weight)
                zeros(layer.bias)

    def forward(self, graph,feat):
        """"""
        # add self_loop
        graph=dgl.remove_self_loop(graph)
        graph=dgl.add_self_loop(graph)
        # prepare
        x = feat
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        out = self.propagate(graph, x)
        return out  
    
    def propagate(self,graph,feat):
        # x_i torch.Size([13264, 2, 4])
        # x_j torch.Size([13264, 2, 4])
        # edge_index torch.Size([2, 13264])
        # num_nodes 2708
        # x_i is target; x_j is source 
        edge_index=torch.stack(graph.edges())
        x_i=torch.index_select(feat,0,edge_index[1])
        x_j=torch.index_select(feat,0,edge_index[0])
        num_nodes=graph.num_nodes()
        if self.att_type == "const": # Const OK
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            neighbor = x_j
        elif self.att_type == "gcn": # GCN OK
            if self.gcn_weight is None or self.gcn_weight.size(0) != x_j.size(
                0
            ):  # calculate norm by degree like GCN
                _, norm = self.norm(edge_index, num_nodes, None)
                self.gcn_weight = norm
            neighbor = self.gcn_weight.view(-1, 1, 1) * x_j
        else: # Attention OK
            # Compute attention coefficients.
            alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
            alpha = softmax(alpha, edge_index[0], num_nodes=num_nodes)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            neighbor = x_j *  alpha.view(-1, self.heads, 1)

        if self.pool_dim > 0:
            # neighbor torch.Size([13264, 2, 4])
            for layer in self.pool_layer:
                neighbor = layer(neighbor)
        
        graph.edata['e']=neighbor
        # aggregate : need neighbor as edge, ok
        if self.reduce_func is not None:
            graph.update_all(fn.copy_e('e', 'e'),self.reduce_func('e','h'))
            out=graph.ndata['h']
            graph.edata.pop('e')
        else:
            out=neighbor
            pass

        out = self.update(out)
        return out

    def apply_attention(self, edge_index, num_nodes, x_i, x_j):
        if self.att_type == "gat":
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)

        elif self.att_type == "gat_sym":
            wl = self.att[:, :, : self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels :]  # weight right
            alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
            alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(
                alpha_2, self.negative_slope
            )

        elif self.att_type == "linear":
            wl = self.att[:, :, : self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels :]  # weight right
            al = x_j * wl
            ar = x_j * wr
            alpha = al.sum(dim=-1) + ar.sum(dim=-1)
            alpha = torch.tanh(alpha)
        elif self.att_type == "cos":
            wl = self.att[:, :, : self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels :]  # weight right
            alpha = x_i * wl * x_j * wr
            alpha = alpha.sum(dim=-1)

        elif self.att_type == "generalized_linear":
            wl = self.att[:, :, : self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels :]  # weight right
            al = x_i * wl
            ar = x_j * wr
            alpha = al + ar
            alpha = torch.tanh(alpha)
            alpha = self.general_att_layer(alpha)
        else:
            raise Exception("Wrong attention type:", self.att_type)
        return alpha

    def update(self, aggr_out):# torch.Size([2708, 2, 4])
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels) # torch.Size([2708, 8])
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    def get_param_dict(self):
        params = {}
        key = f"{self.att_type}_{self.agg_type}_{self.in_channels}_{self.out_channels}_{self.heads}"
        weight_key = key + "_weight"
        att_key = key + "_att"
        agg_key = key + "_agg"
        bais_key = key + "_bais"

        params[weight_key] = self.weight
        params[att_key] = self.att
        params[bais_key] = self.bias
        if hasattr(self, "pool_layer"):
            params[agg_key] = self.pool_layer.state_dict()

        return params

    def load_param(self, params):
        key = f"{self.att_type}_{self.agg_type}_{self.in_channels}_{self.out_channels}_{self.heads}"
        weight_key = key + "_weight"
        att_key = key + "_att"
        agg_key = key + "_agg"
        bais_key = key + "_bais"

        if weight_key in params:
            self.weight = params[weight_key]

        if att_key in params:
            self.att = params[att_key]

        if bais_key in params:
            self.bias = params[bais_key]

        if agg_key in params and hasattr(self, "pool_layer"):
            self.pool_layer.load_state_dict(params[agg_key])