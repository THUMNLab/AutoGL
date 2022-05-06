import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch.nn.parameter import Parameter
import torch
import math

from . import register_model
from .base import BaseAutoModel, activate_func
from ....utils import get_logger

LOGGER = get_logger("AMGCNModel")

class GCN(nn.Module):
    def __init__(self, num_layers, nfeat, hidden, dropout, act_func='relu'):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(nfeat, hidden[0]))
        for l in range(1,self.num_layers):
            self.convs.append(GraphConv(hidden[l-1], hidden[l]))

        self.dropout = dropout
        self.act_func = act_func

    def forward(self, x, adj):
        for l in range(self.num_layers):
            x = self.convs[l](x, adj)
            if l!=self.num_layers-1:
                x = activate_func(x, self.act_func)
                x = F.dropout(x, self.dropout, training = self.training)

        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class SFGCN(nn.Module):
    def __init__(self, args):
        
        super(SFGCN, self).__init__()

        self.args = args

        missing_keys = list(
            set(
                [
                    "features_num",
                    "num_class",
                    "num_layers",  # num_layers is the layer number of GCN
                    "hidden",      # len(hidden) should be the same as num_layers
                    "dropout",
                    "act",
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))
        if not self.num_layer == len(self.args["hidden"]):
            LOGGER.warn("Warning: layer size does not match the length of hidden units")
        

        self.SGCN1 = GCN(self.args['num_layers'],self.args['features_num'], self.args['hidden'], self.args['hidden'], self.args['act'])
        self.SGCN2 = GCN(self.args['num_layers'],self.args['features_num'], self.args['hidden'], self.args['hidden'], self.args['act'])
        self.CGCN = GCN(self.args['num_layers'],self.args['features_num'], self.args['hidden'], self.args['hidden'], self.args['act'])

        self.dropout = self.args['dropout']
        self.a = nn.Parameter(torch.zeros(size=(self.args['hidden'][-1], 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(self.args['hidden'][-1])
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(self.args['hidden'][-1], self.args['num_class']),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        # return output, att, emb1, com1, com2, emb2, emb
        return output





@register_model("am-gcn-model")
class AutoAMGCN(BaseAutoModel):
    r"""
    AutoAMGCN. The model used in this automodel is AM-GCN, i.e., the graph attentional network from the `"AM-GCN: Adaptive Multi-channel Graph Convolutional
Networks"
    <https://arxiv.org/pdf/2007.02265.pdf>`_ paper. The layer is

    .. math::
        

    Parameters
    ----------
    num_features: `int`.
        The dimension of features.

    num_classes: `int`.
        The number of classes.

    device: `torch.device` or `str`
        The device where model will be running on.

    init: `bool`.
        If True(False), the model will (not) be initialized.

    args: Other parameters.
    """

    def __init__(
        self, num_features=None, num_classes=None, device=None, **args
    ):
        super().__init__(num_features, num_classes, device, **args)
        self.hyper_parameter_space = [
            {
                "parameterName": "num_layers",
                "type": "DISCRETE",
                "feasiblePoints": "2,3,4",
            },
            {
                "parameterName": "hidden",
                "type": "NUMERICAL_LIST",
                "numericalType": "INTEGER",
                "length": 3,
                "minValue": [8, 8, 8],
                "maxValue": [64, 64, 64],
                "scalingType": "LOG",
                "cutPara": ("num_layers",),
                "cutFunc": lambda x: x[0] - 1,
            },
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.8,
                "minValue": 0.2,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
        ]

        self.hyper_parameters = {
            "num_layers": 2,
            "hidden": [32,32],
            "dropout": 0.2,
            "act": "relu",
        }

    def _initialize(self):
        # """AM-GCN model."""
        self._model = SFGCN({
            "features_num": self.input_dimension,
            "num_class": self.output_dimension,
            **self.hyper_parameters
        }).to(self.device)
