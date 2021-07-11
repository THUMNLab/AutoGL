# codes in this file are reproduced from https://github.com/GraphNAS/GraphNAS with some changes.

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    ChebConv,
    SAGEConv,
    GatedGraphConv,
    ARMAConv,
    SGConv,
)
import torch_geometric.nn
import torch
from torch import nn
import torch.nn.functional as F


class LinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class ZeroConv(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        out = torch.zeros_like(x)
        out.requires_grad = True
        return out

    def __repr__(self):
        return "ZeroConv()"


class Identity(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        return x

    def __repr__(self):
        return "Identity()"


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
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "zero":
        return ZeroConv()
    elif gnn_name == "identity":
        return Identity()
    elif hasattr(torch_geometric.nn, gnn_name):
        cls = getattr(torch_geometric.nn, gnn_name)
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
