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
