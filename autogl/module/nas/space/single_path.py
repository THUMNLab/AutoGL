from autogl.module.nas.space.operation import gnn_map
import typing as _typ
import torch

import torch.nn.functional as F

from . import register_nas_space
from .base import apply_fixed_architecture
from .base import BaseSpace
from ...model import BaseModel
from ....utils import get_logger

from ...model import AutoGCN


@register_nas_space("singlepath")
class SinglePathNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.2,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = ["GCNConv", "GATConv"],
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.dropout = dropout

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
        for layer in range(self.layer_number):
            setattr(
                self,
                f"op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        op(
                            self.input_dim if layer == 0 else self.hidden_dim,
                            self.output_dim
                            if layer == self.layer_number - 1
                            else self.hidden_dim,
                        )
                        if isinstance(op, type)
                        else gnn_map(
                            op,
                            self.input_dim if layer == 0 else self.hidden_dim,
                            self.output_dim
                            if layer == self.layer_number - 1
                            else self.hidden_dim,
                        )
                        for op in self.ops
                    ],
                ),
            )
        self._initialized = True

    def forward(self, data):
        x, edges = data.x, data.edge_index
        for layer in range(self.layer_number):
            x = getattr(self, f"op_{layer}")(x, edges)
            if layer != self.layer_number - 1:
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) -> BaseModel:
        # return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap(device).fix(selection)
