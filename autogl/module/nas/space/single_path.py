from copy import deepcopy
import typing as _typ
import torch

import torch.nn.functional as F
from nni.nas.pytorch import mutables
from nni.nas.pytorch.fixed import apply_fixed_architecture
from .base import BaseSpace
from ...model import BaseModel
from ....utils import get_logger


class FixedNodeClassificationModel(BaseModel):
    _logger = get_logger("space model")

    def __init__(self, space_model: BaseSpace, selection, device=torch.device("cuda")):
        super().__init__(init=True)
        space_model.instantiate()
        self.init = True
        self.space = []
        self.hyperparams = {}
        self._model = space_model.to(device)
        self.num_features = self._model.input_dim
        self.num_classes = self._model.output_dim
        self.selection = selection
        apply_fixed_architecture(self._model, selection, verbose=False)
        self.params = {"num_class": self.num_classes, "features_num": self.num_features}
        self.device = device

    def to(self, device):
        if isinstance(device, (str, torch.device)):
            self.device = device
        return super().to(device)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def from_hyper_parameter(self, hp):
        """
        receive no hp, just copy self and reset the learnable parameters.
        """

        ret_self = deepcopy(self)
        ret_self._model.instantiate()
        apply_fixed_architecture(ret_self._model, ret_self.selection, verbose=False)
        ret_self.to(self.device)
        return ret_self

    @property
    def model(self):
        return self._model


class SinglePathNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.6,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = None,
        init: bool = False,
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
        dropout = None
    ):
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
                mutables.LayerChoice(
                    [
                        op(
                            self.input_dim if layer == 0 else self.hidden_dim,
                            self.output_dim
                            if layer == self.layer_number - 1
                            else self.hidden_dim,
                        )
                        for op in self.ops
                    ],
                    key=f"{layer}",
                ),
            )
        self._initialized = True

    def forward(self, data):
        x, edges = data.x, data.edge_index
        for layer in range(self.layer_number):
            x = getattr(self, f"op_{layer}")(x, edges)
            if layer != self.layer_number - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def export(self, selection, device) -> BaseModel:
        return FixedNodeClassificationModel(self, selection, device)
