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


import inspect
import sys
from torch_scatter import scatter_add
import torch_scatter
special_args = [
    "edge_index",
    "edge_index_i",
    "edge_index_j",
    "size",
    "size_i",
    "size_j",
]
__size_error_msg__ = (
    "All tensors which should get mapped to the same source "
    "or target nodes must be of same size in dimension 0."
)

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ["add", "mean", "max"]

    op = getattr(torch_scatter, "scatter_{}".format(name))
    fill_value = -1e9 if name == "max" else 0

    out = op(src, index, 0, None, dim_size)
    if isinstance(out, tuple):
        out = out[0]

    if name == "max":
        out[out == fill_value] = 0

    return out
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    add_remaining_self_loops,
    softmax,
)
class MessagePassing(torch.nn.Module):
    def __init__(self, aggr="add", flow="source_to_target"):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max"]

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [
            (i, arg)
            for i, arg in enumerate(self.__message_args__)
            if arg in special_args
        ]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferrred. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs["edge_index"] = edge_index
        kwargs["size"] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        out = self.message(*message_args)
        if self.aggr in ["add", "mean", "max"]:
            out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        else:
            pass
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out

class GeoLayer(MessagePassing):
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
        if agg_type in ["sum", "mlp"]:
            super(GeoLayer, self).__init__("add")
        elif agg_type in ["mean", "max"]:
            super(GeoLayer, self).__init__(agg_type)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type

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

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # prepare
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        # x [2708,2,4] weight [1433,8]
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):

        # x_i torch.Size([13264, 2, 4])
        # x_j torch.Size([13264, 2, 4])
        # edge_index torch.Size([2, 13264])
        # num_nodes 2708
        if self.att_type == "const":
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            neighbor = x_j
        elif self.att_type == "gcn":
            if self.gcn_weight is None or self.gcn_weight.size(0) != x_j.size(
                0
            ):  # 对于不同的图gcn_weight需要重新计算
                _, norm = self.norm(edge_index, num_nodes, None)
                self.gcn_weight = norm
            neighbor = self.gcn_weight.view(-1, 1, 1) * x_j
        else:
            # Compute attention coefficients.
            alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
            alpha = softmax(alpha, edge_index[0], num_nodes=num_nodes)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            neighbor = x_j * alpha.view(-1, self.heads, 1)
        # pool_layer
        # (0): Linear(in_features=4, out_features=128, bias=True)
        #   (1): Linear(in_features=128, out_features=4, bias=True)
        if self.pool_dim > 0:
            # neighbor torch.Size([13264, 2, 4])
            for layer in self.pool_layer:
                neighbor = layer(neighbor)
        return neighbor

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