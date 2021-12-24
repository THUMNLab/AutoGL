import torch
from torch import nn
import torch.nn.functional as F
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

from ..backend import *
if is_dgl():
    from .operation_dgl import gnn_map,GeoLayer
else:
    from .operation_pyg import gnn_map,GeoLayer