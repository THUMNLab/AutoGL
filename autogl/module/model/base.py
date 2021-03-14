"""
auto graph model
a list of models with their hyper parameters
NOTE: neural architecture search (NAS) maybe included here
"""

import torch
import torch.nn.functional as F
from copy import deepcopy


def activate_func(x, func):
    if func == "tanh":
        return torch.tanh(x)
    elif hasattr(F, func):
        return getattr(F, func)(x)
    elif func == "":
        pass
    else:
        raise TypeError("PyTorch does not support activation function {}".format(func))

    return x


class BaseModel(torch.nn.Module):
    def __init__(self, init=False, *args, **kwargs):
        super(BaseModel, self).__init__()

    def get_hyper_parameter(self):
        return deepcopy(self.hyperparams)

    @property
    def hyper_parameter_space(self):
        return self.space

    @hyper_parameter_space.setter
    def hyper_parameter_space(self, space):
        self.space = space

    def initialize(self):
        pass

    def forward(self):
        pass

    def to(self, device):
        if isinstance(device, (str, torch.device)):
            self.device = device
        return super().to(device)

    def from_hyper_parameter(self, hp):
        ret_self = self.__class__(
            num_features=self.num_features,
            num_classes=self.num_classes,
            device=self.device,
            init=False,
        )
        ret_self.hyperparams.update(hp)
        ret_self.params.update(self.params)
        ret_self.initialize()
        return ret_self

    def get_num_classes(self):
        return self.num_classes

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
        self.params["num_class"] = num_classes

    def get_num_features(self):
        return self.num_features

    def set_num_features(self, num_features):
        self.num_features = num_features
        self.params["features_num"] = self.num_features

    def set_num_graph_features(self, num_graph_features):
        assert hasattr(
            self, "num_graph_features"
        ), "Cannot set graph features for tasks other than graph classification"
        self.num_graph_features = num_graph_features
        self.params["num_graph_features"] = num_graph_features
