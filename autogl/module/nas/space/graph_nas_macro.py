import torch
import typing as _typ
import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_space
from .base import BaseSpace, map_nn
from ...model import BaseAutoModel
from .operation import act_map
from ..utils import count_parameters, measure_latency

from ..backend import *

from operator import *
from .operation import *

@register_nas_space("graphnasmacro")
class GraphNasMacroNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.6,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = None,
        search_act_con=False,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.dropout = dropout
        self.search_act_con = search_act_con

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = None,
        dropout=None,
    ):
        super().instantiate()
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        self.dropout = dropout or self.dropout

        num_feat = self.input_dim
        num_label = self.output_dim

        layer_nums = self.layer_number
        state_num = 5

        # build hidden layer
        for i in range(layer_nums):
            # extract layer information
            setattr(
                self,
                f"attention_{i}",
                self.setLayerChoice(
                    i * state_num + 0,
                    map_nn(
                        [
                            "gat",
                            "gcn",
                            "cos",
                            "const",
                            "gat_sym",
                            "linear",
                            "generalized_linear",
                        ]
                    ),
                    key=f"attention_{i}",
                ),
            )
            setattr(
                self,
                f"aggregator_{i}",
                self.setLayerChoice(
                    i * state_num + 1,
                    map_nn(
                        [
                            "sum",
                            "mean",
                            "max",
                            "mlp",
                        ]
                    ),
                    key=f"aggregator_{i}",
                ),
            )
            setattr(
                self,
                f"act_{i}",
                self.setLayerChoice(
                    i * state_num + 0,
                    map_nn(
                        [
                            "sigmoid",
                            "tanh",
                            "relu",
                            "linear",
                            "softplus",
                            "leaky_relu",
                            "relu6",
                            "elu",
                        ]
                    ),
                    key=f"act_{i}",
                ),
            )
            setattr(
                self,
                f"head_{i}",
                self.setLayerChoice(
                    i * state_num + 0, map_nn([1, 2, 4, 6, 8, 16]), key=f"head_{i}"
                ),
            )
            if i < layer_nums - 1:
                setattr(
                    self,
                    f"out_channels_{i}",
                    self.setLayerChoice(
                        i * state_num + 0,
                        map_nn([4, 8, 16, 32, 64, 128, 256]),
                        key=f"out_channels_{i}",
                    ),
                )

    def parse_model(self, selection, device) -> BaseAutoModel:
        sel_list = []
        for i in range(self.layer_number):
            sel_list.append(
                [
                    "gat",
                    "gcn",
                    "cos",
                    "const",
                    "gat_sym",
                    "linear",
                    "generalized_linear",
                ][selection[f"attention_{i}"]]
            )
            sel_list.append(
                [
                    "sum",
                    "mean",
                    "max",
                    "mlp",
                ][selection[f"aggregator_{i}"]]
            )
            sel_list.append(
                [
                    "sigmoid",
                    "tanh",
                    "relu",
                    "linear",
                    "softplus",
                    "leaky_relu",
                    "relu6",
                    "elu",
                ][selection[f"act_{i}"]]
            )
            sel_list.append([1, 2, 4, 6, 8, 16][selection[f"head_{i}"]])
            if i < self.layer_number - 1:
                sel_list.append(
                    [4, 8, 16, 32, 64, 128, 256][selection[f"out_channels_{i}"]]
                )
        sel_list.append(self.output_dim)
        # sel_list = ['const', 'sum', 'relu6', 2, 128, 'gat', 'sum', 'linear', 2, 7]
        model = GraphNet(
            sel_list,
            self.input_dim,
            self.output_dim,
            self.dropout,
            multi_label=False,
            batch_normal=False,
            layers=self.layer_number,
        ).wrap()
        return model


class GraphNet(BaseSpace):
    def __init__(
        self,
        actions,
        num_feat,
        num_label,
        drop_out=0.6,
        multi_label=False,
        batch_normal=True,
        state_num=5,
        residual=False,
        layers=2,
    ):
        self.residual = residual
        self.batch_normal = batch_normal
        self.layer_nums = layers
        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_label = num_label
        self.input_dim = num_feat
        self.output_dim = num_label
        self.dropout = drop_out

        super().__init__()
        self.build_model(
            actions, batch_normal, drop_out, num_feat, num_label, state_num
        )

    def build_model(
        self, actions, batch_normal, drop_out, num_feat, num_label, state_num
    ):
        if self.residual:
            self.fcs = torch.nn.ModuleList()
        if self.batch_normal:
            self.bns = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.gates = torch.nn.ModuleList()
        self.build_hidden_layers(
            actions,
            batch_normal,
            drop_out,
            self.layer_nums,
            num_feat,
            num_label,
            state_num,
        )

    def build_hidden_layers(
        self,
        actions,
        batch_normal,
        drop_out,
        layer_nums,
        num_feat,
        num_label,
        state_num=6,
    ):

        # build hidden layer
        for i in range(layer_nums):

            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num

            # extract layer information
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            concat = True
            if i == layer_nums - 1:
                concat = False
            if self.batch_normal:
                self.bns.append(torch.nn.BatchNorm1d(in_channels, momentum=0.5))
            self.layers.append(
                GeoLayer(
                    in_channels,
                    out_channels,
                    head_num,
                    concat,
                    dropout=self.dropout,
                    att_type=attention_type,
                    agg_type=aggregator_type,
                )
            )
            self.acts.append(act_map(act))
            if self.residual:
                if concat:
                    self.fcs.append(
                        torch.nn.Linear(in_channels, out_channels * head_num)
                    )
                else:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, data):
        output=bk_feat(data)
        # output, edge_index_all = data.x, data.edge_index  # x [2708,1433] ,[2, 10556]
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)

                # output = act(layer(output, edge_index_all) + fc(output))
                output = act(bk_gconv(layer,data,output) + fc(output))
        else:
            for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                # output = act(layer(output, edge_index_all))
                output = act(bk_gconv(layer,data,output))

        if not self.multi_label:
            output = F.log_softmax(output, dim=1)
        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = self.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                result[key] = self.fcs[i]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f"layer_{i}_fc_{bn.weight.size(0)}"
                result[key] = self.bns[i]
        return result

    def load_param(self, param):
        if param is None:
            return

        for i in range(self.layer_nums):
            self.layers[i].load_param(param["layer_%d" % i])

        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                if key in param:
                    self.fcs[i] = param[key]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f"layer_{i}_fc_{bn.weight.size(0)}"
                if key in param:
                    self.bns[i] = param[key]
