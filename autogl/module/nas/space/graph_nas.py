# codes in this file are reproduced from https://github.com/GraphNAS/GraphNAS with some changes.
import typing as _typ
import torch

import torch.nn.functional as F
from nni.nas.pytorch import mutables

from . import register_nas_space
from .base import BaseSpace
from ...model import BaseModel

from torch import nn
from .operation import act_map, gnn_map

GRAPHNAS_DEFAULT_GNN_OPS = [
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

GRAPHNAS_DEFAULT_ACT_OPS = [
    # "sigmoid", "tanh", "relu", "linear",
    #  "softplus", "leaky_relu", "relu6", "elu"
    "sigmoid",
    "tanh",
    "relu",
    "linear",
    "elu",
]


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.lambd)


class StrModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.str = lambd

    def forward(self, *args, **kwargs):
        return self.str

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.str)


def act_map_nn(act):
    return LambdaModule(act_map(act))


def map_nn(l):
    return [StrModule(x) for x in l]


@register_nas_space("graphnas")
class GraphNasNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.9,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_GNN_OPS,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_ACT_OPS,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.act_ops = act_ops
        self.dropout = dropout

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        dropout: _typ.Optional[float] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
    ):
        super().instantiate()
        self.dropout = dropout or self.dropout
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.gnn_ops = gnn_ops or self.gnn_ops
        self.act_ops = act_ops or self.act_ops
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.preproc1 = nn.Linear(self.input_dim, self.hidden_dim)
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for layer in range(2, self.layer_number + 2):
            node_labels.append(f"op_{layer}")
            setattr(
                self,
                f"in_{layer}",
                self.setInputChoice(
                    layer,
                    choose_from=node_labels[:-1],
                    n_chosen=1,
                    return_mask=False,
                    key=f"in_{layer}",
                ),
            )
            setattr(
                self,
                f"op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        gnn_map(op, self.hidden_dim, self.hidden_dim)
                        for op in self.gnn_ops
                    ],
                    key=f"op_{layer}",
                ),
            )
        setattr(
            self,
            "act",
            self.setLayerChoice(
                2 * layer, [act_map_nn(a) for a in self.act_ops], key="act"
            ),
        )
        setattr(
            self,
            "concat",
            self.setLayerChoice(
                2 * layer + 1, map_nn(["add", "product", "concat"]), key="concat"
            ),
        )
        self._initialized = True
        self.classifier1 = nn.Linear(
            self.hidden_dim * self.layer_number, self.output_dim
        )
        self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edges = data.x, data.edge_index  # x [2708,1433] ,[2, 10556]
        x = F.dropout(x, p=self.dropout, training=self.training)
        pprev_, prev_ = self.preproc0(x), self.preproc1(x)
        prev_nodes_out = [pprev_, prev_]
        for layer in range(2, self.layer_number + 2):
            node_in = getattr(self, f"in_{layer}")(prev_nodes_out)
            node_out = getattr(self, f"op_{layer}")(node_in, edges)
            prev_nodes_out.append(node_out)
        act = getattr(self, "act")
        con = getattr(self, "concat")()
        states = prev_nodes_out
        if con == "concat":
            x = torch.cat(states[2:], dim=1)
        else:
            tmp = states[2]
            for i in range(2, len(states)):
                if con == "add":
                    tmp = torch.add(tmp, states[i])
                elif con == "product":
                    tmp = torch.mul(tmp, states[i])
            x = tmp
        x = act(x)
        if con == "concat":
            x = self.classifier1(x)
        else:
            x = self.classifier2(x)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) -> BaseModel:
        # return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap(device).fix(selection)
