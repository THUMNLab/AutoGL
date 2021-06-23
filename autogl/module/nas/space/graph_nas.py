# codes in this file are reproduced from https://github.com/GraphNAS/GraphNAS with some changes.
import typing as _typ
import torch

import torch.nn.functional as F
from nni.nas.pytorch import mutables

from . import register_nas_space
from .base import BaseSpace
from ...model import BaseModel
from ....utils import get_logger

from ...model import AutoGCN
from torch import nn
from .operation import gnn_list, act_list, act_map, gnn_map
from torch.autograd import Function

class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,self.lambd)

class StrModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.str = lambd

    def forward(self, *args,**kwargs):
        return self.str  

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,self.str)

def act_map_nn(act):
    return LambdaModule(act_map(act))

def map_nn(l):
    return [StrModule(x) for x in l]

class ZeroConvFunc(Function):
    @staticmethod
    def forward(ctx,x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return 0

@register_nas_space("graphnas")
class GraphNasNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.9,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = None,
        search_act_con=False
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.dropout = dropout
        self.search_act_con=search_act_con

    def _instantiate(
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
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.preproc1 = nn.Linear(self.input_dim, self.hidden_dim)
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for layer in range(2,self.layer_number+2):
            node_labels.append(f"op_{layer}")
            setattr(self,f"in_{layer}",self.setInputChoice(layer,choose_from=node_labels[:-1], n_chosen=1, return_mask=False,key=f"in_{layer}"))
            setattr(self,f"op_{layer}",self.setLayerChoice(layer,[gnn_map(op,self.hidden_dim,self.hidden_dim)for op in gnn_list],key=f"op_{layer}"))
        if self.search_act_con:
            setattr(self,f"act",self.setLayerChoice(2*layer,[act_map_nn(a)for a in act_list],key=f"act"))
            setattr(self,f"concat",self.setLayerChoice(2*layer+1,map_nn(["add", "product", "concat"]) ,key=f"concat"))
        self._initialized = True
        self.classifier1 = nn.Linear(self.hidden_dim*self.layer_number, self.output_dim)
        self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x, edges = data.x, data.edge_index # x [2708,1433] ,[2, 10556]
        x = F.dropout(x, p=self.dropout, training = self.training)
        pprev_, prev_ = self.preproc0(x), self.preproc1(x)
        prev_nodes_out = [pprev_,prev_]
        for layer in range(2,self.layer_number+2):
            node_in = getattr(self, f"in_{layer}")(prev_nodes_out)
            node_out= getattr(self, f"op_{layer}")(node_in,edges)
            prev_nodes_out.append(node_out)
        if not self.search_act_con:
            x = torch.cat(prev_nodes_out[2:],dim=1)
            x = F.leaky_relu(x)
            x = self.classifier1(x)
        else:
            act=getattr(self, f"act")
            con=getattr(self, f"concat")()
            states=prev_nodes_out
            if con == "concat":
                x=torch.cat(states[2:], dim=1)
            else:
                tmp = states[2]
                for i in range(2,len(states)):
                    if con == "add":
                        tmp = torch.add(tmp, states[i])
                    elif con == "product":
                        tmp = torch.mul(tmp, states[i])
                x=tmp
            x = act(x)
            if con=='concat':
                x=self.classifier1(x)
            else:
                x=self.classifier2(x)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) -> BaseModel:
        #return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap(device).fix(selection)