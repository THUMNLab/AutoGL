# codes in this file are reproduced from https://github.com/GraphNAS/GraphNAS with some changes.
from torch_geometric.nn.conv import *
import torch
from torch import nn

gnn_list = [
    "gat_8",  # GAT with 8 heads
    "gat_6",  # GAT with 6 heads
    "gat_4",  # GAT with 4 heads
    "gat_2",  # GAT with 2 heads
    "gat_1",  # GAT with 1 heads
    "gcn",  # GCN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "sg",  # simplifying gcn
    "linear",  # skip connection
    "zero",  # skip connection
]
act_list = [
    # "sigmoid", "tanh", "relu", "linear",
    #  "softplus", "leaky_relu", "relu6", "elu"
    "sigmoid", "tanh", "relu", "linear", "elu"
]

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return F.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")

def gnn_map(gnn_name, in_dim, out_dim, concat=False, bias=True) -> nn.Module:
    '''

    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    '''
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
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "zero":
        # return ZeroConv(in_dim, out_dim, bias=bias)
        return Identity()

class Identity(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        return x

class LinearConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class ZeroConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(ZeroConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_dim = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return ZeroConvFunc.apply(torch.zeros([x.size(0), self.out_dim]).to(x.device))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)