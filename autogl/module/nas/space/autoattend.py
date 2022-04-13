# codes in this file are reproduced from AutoAttend with some changes.
from nni.nas.pytorch.mutables import Mutable
import typing as _typ
import torch

import torch.nn.functional as F
from nni.nas.pytorch import mutables

from . import register_nas_space
from .base import BaseSpace
from ..utils import count_parameters, measure_latency

from torch import nn
from .operation import act_map, gnn_map

from ..backend import *

from .autoattend_space.ops1 import OPS as OPS1
from .autoattend_space.ops2 import OPS as OPS2
from .autoattend_space.operations import agg_map
OPS = [OPS1, OPS2]


@register_nas_space("autoattend")
class AutoAttendNodeClassificationSpace(BaseSpace):
    """
    AutoAttend Search Space , please refer to http://proceedings.mlr.press/v139/guan21a.html for details.
    The current implementation is nc (no context weight sharing), 
    we will in future add other types of partial weight sharing proposed in the paper.

    Parameters
    ----------
    ops_type : int 
        0 or 1 , choosing from two sets of ops with index ops_type
    gnn_ops : list of str
        op names for searching, which descripts the compostion of operation pool
    act_op : str
        determine used activation function 
    agg_ops : list of str
        agg op names for searching. Only ['add','attn'] are options, as mentioned in the paper.
    """

    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.9,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops_type=0,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]
                               ] = None,
        act_op="tanh",
        head=8,
        agg_ops=['add', 'attn'],
        # agg_ops=['attn'],

    ):
        super().__init__()

        from autogl.backend import DependentBackend;assert not DependentBackend.is_dgl(),"Now AutoAttend only support pyg"
        
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.dropout = dropout
        self.act_op = act_op
        self.ops_type = ops_type
        self.head = head
        self.agg_ops = agg_ops

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        dropout: _typ.Optional[float] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops_type=None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
        act_op=None,
        head=None,
        agg_ops=None,
        # con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
    ):
        super().instantiate()
        self.dropout = dropout or self.dropout
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.gnn_ops = gnn_ops or self.gnn_ops
        self.act_op = act_op or self.act_op
        self.act = act_map(self.act_op)
        self.head = head or self.head
        self.ops_type = ops_type or self.ops_type
        self.agg_ops = agg_ops or self.agg_ops
        PRIMITIVES = list(OPS[self.ops_type].keys())
        self.gnn_map = lambda x, * \
            args, **kwargs: OPS[self.ops_type][x](*args, **kwargs)
        self.gnn_ops = self.gnn_ops or PRIMITIVES
        self.agg_map = lambda x, * \
            args, **kwargs: agg_map[x](*args, **kwargs)
        # self.preproc0 = nn.Linear(self.input_dim, self.input_dim)

        for layer in range(1, self.layer_number+1):
            # stem path
            layer_stem = f"{layer-1}->{layer}"
            key = f"stem_{layer_stem}"
            self._set_layer_choice(layer_stem, key)

            # input
            key = f"in_{layer}"
            # self._set_input_choice(key,layer, choose_from=node_labels, n_chosen=1, return_mask=False)
            self._set_input_choice(
                key, layer, n_candidates=layer, n_chosen=1, return_mask=False)

            # agg
            key = f"agg_{layer}"
            self._set_agg_choice(layer, key=key)

        for l1 in range(1, self.layer_number+1):
            for l2 in range(l1):
                # from l2 to l1; l2 means input layer; l1 means output layer
                layer = f"{l2}->{l1}"
                # side path
                key = f"side_{layer}"
                for i in range(2):
                    sub_key = f"{key}_{i}"
                    self._set_layer_choice(layer, sub_key)

        self._initialized = True

        # self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)

    def _set_agg_choice(self, layer, key):
        if layer == self.layer_number:
            dim = self.output_dim
        else:
            dim = self.hidden_dim
        agg_head = self.head if layer < self.layer_number else 1
        ops = [self.agg_map(op, dim, agg_head,
                            self.dropout)for op in self.agg_ops]
        choice = self.setLayerChoice(
            layer,
            ops,
            key=key,
        )
        setattr(self, key, choice)
        return choice

    def _set_layer_choice(self, layer, key):
        l1, l2 = list(map(int, layer.split('->')))
        input_dim = output_dim = self.hidden_dim

        input_dim = self.input_dim if l1 == 0 else self.hidden_dim
        output_dim = self.output_dim if l2 == self.layer_number else self.hidden_dim

        opnames = self.gnn_ops.copy()
        if input_dim != output_dim:
            for invalid_op in "ZERO IDEN".split():
                opnames.remove(invalid_op)
        concat = (l2 != self.layer_number)
        if self.ops_type == 0:
            ops = [self.gnn_map(op, input_dim, output_dim,
                                self.dropout, concat=concat)for op in opnames]
        elif self.ops_type == 1:
            ops = [self.gnn_map(op, input_dim, output_dim,
                                self.head, self.dropout, concat=concat)for op in opnames]
        choice = self.setLayerChoice(
            l1,
            ops,
            key=key,
        )
        setattr(self, key, choice)
        return choice

    def _set_input_choice(self, key, layer, **kwargs):
        setattr(self,
                key,
                self.setInputChoice(
                    layer,
                    key=key,
                    **kwargs
                ))

    def forward(self, data):
        x = bk_feat(data)

        def drop(x):
            x = F.dropout(x, p=self.dropout, training=self.training)
            return x
        # prev_ = self.preproc0(x)
        prev_ = x

        side_outs = {}  # {l:[side1,side2...]} collects lth layer's input
        stem_outs = []
        input = prev_
        for layer in range(1, self.layer_number + 1):
            # do layer choice for stem
            layer_stem = f"{layer-1}->{layer}"
            op = getattr(self, f"stem_{layer_stem}")
            stem_out = bk_gconv(op, data, drop(input))
            stem_out = self.act(stem_out)
            stem_outs.append(stem_out)

            # do double layer choice for sides
            for l2 in range(layer, self.layer_number+1):
                l1 = layer-1
                layer_side = f"{l1}->{l2}"

                side_out_list = []
                for i in range(2):
                    op = getattr(self, f'side_{layer_side}_{i}')
                    side_out = bk_gconv(op, data, drop(input))
                    side_out = self.act(side_out)  # torch.Size([2, 2708, 64])
                    side_out_list.append(side_out)
                side_out = torch.stack(side_out_list, dim=0)

                if l2 not in side_outs:
                    side_outs[l2] = []
                side_outs[l2].append(side_out)

            # select input [x1,x2,x3] from side1,side2,stem
            current_side_outs = side_outs[layer]
            side_selected = getattr(self, f"in_{layer}")(current_side_outs)
            input = [stem_outs[-1], side_selected]

            # do agg in [add , attn]
            agg = getattr(self, f"agg_{layer}")
            input = bk_gconv(agg, data, input)

        # x = self.classifier2(input)
        x = input
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device):
        return self.wrap().fix(selection)
