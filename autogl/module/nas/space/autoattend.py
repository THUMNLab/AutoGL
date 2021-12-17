# codes in this file are reproduced from https://github.com/GraphNAS/GraphNAS with some changes.
from nni.nas.pytorch.mutables import Mutable
import typing as _typ
import torch

import torch.nn.functional as F
from nni.nas.pytorch import mutables

from . import register_nas_space
from .base import BaseSpace
from ...model import BaseModel
from ..utils import count_parameters, measure_latency

from torch import nn
from .operation import act_map, gnn_map

from ..backend import *

from .autoattend_space.ops1 import OPS as OPS1
from .autoattend_space.ops2 import OPS as OPS2
from .autoattend_space.operations import agg_map
OPS = [OPS1, OPS2]

from nni.nas.pytorch.mutables import Mutable
class MultiLayerChoice(Mutable):
    def __init__(self, choices, layer, key):
        super(MultiLayerChoice, self).__init__(key)
        self.order = layer
        self.choices = choices

    def forward(self, *args, **kwargs):
        outs = []
        for i in range(len(self.choices)):
            out = self.choices[i](*args, **kwargs)
            outs.append(out)
        outs = torch.stack(outs, dim=0)
        return outs


@register_nas_space("autoattend")
class AutoAttendNodeClassificationSpace(BaseSpace):
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
        agg_ops=['add','attn']
    ):
        super().__init__()
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
        self.agg_ops=agg_ops or self.agg_ops
        PRIMITIVES = list(OPS[self.ops_type].keys())
        self.gnn_map = lambda x, * \
            args, **kwargs: OPS[self.ops_type][x](*args, **kwargs)
        self.gnn_ops = self.gnn_ops or PRIMITIVES
        self.agg_map = lambda x, * \
            args, **kwargs: agg_map[x](*args, **kwargs) 
        # self.con_ops = con_ops or self.con_ops
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim)
        # self.preproc1 = nn.Linear(self.input_dim, self.hidden_dim)
        node_labels = []

        # stem path
        for layer in range(1, self.layer_number+1):
            key = f"stem_{layer}"
            self._set_layer_choice(layer, key)

        # side path
        for layer in range(1, self.layer_number+1):
            choices = []
            key = f"side_{layer}"
            for i in range(2):
                sub_key = f"{key}_{i}"
                choice = self._set_layer_choice(layer, sub_key)
                choices.append(choice)
            setattr(self, key, MultiLayerChoice(choices, layer, key))
            node_labels.append(key)

            # input
            key = f"in_{layer}"
            self._set_input_choice(key,
                layer, choose_from=node_labels, n_chosen=1, return_mask=False)            
        
        # agg
        for layer in range(1, self.layer_number + 1):
            key = f"agg_{layer}"
            self._set_agg_choice(layer, key=key)

        self._initialized = True

        self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)

        print(self)
    def _set_agg_choice(self,layer,key):
        ops=[self.agg_map(op, self.hidden_dim,self.head,self.dropout)for op in self.agg_ops]
        choice = self.setLayerChoice(
            layer,
            ops,
            key=key,
        )
        setattr(self, key, choice)
        return choice
    def _set_layer_choice(self, layer, key):
        if self.ops_type==0:
            ops=[self.gnn_map(op, self.hidden_dim, self.hidden_dim, self.dropout)for op in self.gnn_ops]
        elif self.ops_type==1:
            ops=[self.gnn_map(op, self.hidden_dim, self.hidden_dim, self.head, self.dropout)for op in self.gnn_ops]
        choice = self.setLayerChoice(
            layer,
            ops,
            key=key,
        )
        setattr(self, key, choice)
        return choice

    def _set_input_choice(self, key, layer,**kwargs):
        setattr(self,
                key,
                self.setInputChoice(
                    layer,
                    key=key,
                    **kwargs
                ))

    def forward(self, data):
        x = bk_feat(data)
        x = F.dropout(x, p=self.dropout, training=self.training)
        prev_ = self.preproc0(x)

        side_outs = []
        stem_outs = []
        input = prev_
        for layer in range(1, self.layer_number + 1):
            # do layer choice for stem
            op = getattr(self, f"stem_{layer}")
            print(op)
            stem_out = bk_gconv(op, data, input)
            stem_out = self.act(stem_out)
            # do double layer choice for sides
            op = getattr(self, f'side_{layer}')
            side_out = bk_gconv(op, data, input)
            side_out = self.act(side_out)

            stem_outs.append(stem_out)
            side_outs.append(side_out)

            # select input [x1,x2,x3] from side1,side2,stem
            side_selected = getattr(self, f"in_{layer}")(side_outs)
            input = [stem_outs[-1], side_selected]
            # do agg in [add , attn]
            agg = getattr(self, f"agg_{layer}")
            print(layer,input)
            input = bk_gconv(agg,data,input)

        x = self.classifier2(input)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) -> BaseModel:
        for i in range(1, self.layer_number + 1):
            selection[f'side_{i}'] = None
        # return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap(device).fix(selection)
