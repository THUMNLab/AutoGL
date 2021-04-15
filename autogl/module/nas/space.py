from nni.nas.pytorch import mutables
from autogl.module.model import BaseModel
import torch.nn as nn
import torch
from autogl.utils import get_logger
from nni.nas.pytorch.fixed import apply_fixed_architecture
from copy import deepcopy

class BaseSpace(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, ops=None, init=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ops = ops
        self._initialized = False
        if init:
            self.instantiate()

    def instantiate(self, input_dim=None, hidden_dim=None, output_dim=None, ops=None):
        """
        instantiate modules in the space
        """
        self.input_dim = input_dim or self.input_dim
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        self._initialized = True

    def forward(self, data):
        pass

class SpaceModel(BaseModel):
    _logger = get_logger('space model')
    def __init__(self, space_model: BaseSpace, selection, device=torch.device('cuda')):
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
        self.params = {
            "num_class": self.num_classes,
            "features_num": self.num_features
        }
        self.device = device

    def to(self, device):
        if isinstance(device, (str, torch.device)):
            self.device = device
        return super().to(device)

    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

    def from_hyper_parameter(self, hp):
        """
        receive no hp, just copy self and reset the learnable parameters.
        """

        """self._model.instantiate()
        apply_fixed_architecture(self._model, self.selection, verbose=False)
        self.to(self.device)
        return self"""

        ret_self = deepcopy(self)
        ret_self._model.instantiate()
        apply_fixed_architecture(ret_self._model, ret_self.selection, verbose=False)
        ret_self.to(self.device)
        return ret_self

    @property
    def model(self):
        return self._model

class GraphSpace(BaseSpace):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, ops=None, init=False):
        super().__init__(input_dim, hidden_dim, output_dim, ops, init)

    def instantiate(self, input_dim=None, hidden_dim=None, output_dim=None, ops=None, *args, **kwargs):
        super().instantiate(input_dim, hidden_dim, output_dim, ops)
        self.op1 = mutables.LayerChoice([op(self.input_dim, self.hidden_dim) for op in self.ops], key = "1")
        self.op2 = mutables.LayerChoice([op(self.hidden_dim, self.output_dim) for op in self.ops], key = "2")

    def forward(self, data):
        x = self.op1(data.x, data.edge_index)
        x = self.op2(x, data.edge_index)
        return x
