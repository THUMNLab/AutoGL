import torch
import typing as _typing

import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn.functional

from . import register_model
from .base import BaseAutoModel, activate_func
from ....utils import get_logger

LOGGER = get_logger("SAGEModel")


class GraphSAGE(torch.nn.Module):

    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.args = args
        self.num_layer = int(self.args["num_layers"])

        missing_keys = list(
            set(
                [
                    "features_num",
                    "num_class",
                    "num_layers",
                    "hidden",
                    "dropout",
                    "act",
                    "agg"
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))

        if not self.num_layer == len(self.args["hidden"]) + 1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")
        
        if self.args["agg"] not in ("gcn", "pool", "mean", "lstm"):
            self.args["agg"] = "gcn"
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(
                self.args["features_num"],
                self.args["hidden"][0],
                aggregator_type=self.args["agg"]
            )
        )
        for i in range(self.num_layer - 2):
            self.convs.append(
                SAGEConv(
                    self.args["hidden"][i] ,
                    self.args["hidden"][i + 1],
                    aggregator_type=self.args["agg"]
                )
            )
            
        self.convs.append(
            SAGEConv(
                self.args["hidden"][-1],
                self.args["num_class"],
                aggregator_type=self.args["agg"]
            )
        )

    def lp_encode(self, data):
        x: torch.Tensor = data.ndata['feat']
        for i in range(len(self.convs) - 2):
            x = self.convs[i](data, x)
            x = activate_func(x, self.args["act"])
        x = self.convs[-2](data, x)
        return x

    def lp_decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def lp_decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
    def forward(self, data):
        try:
            x = data.ndata['feat']
        except:
            print("no x")
            pass

        x = F.dropout(x, p=self.args["dropout"], training=self.training)
        for i in range(self.num_layer):
            x = self.convs[i](data, x)
            if i != self.num_layer - 1:
                x = activate_func(x, self.args["act"])
                x = F.dropout(x, p=self.args["dropout"], training=self.training)

        return F.log_softmax(x, dim=1)



@register_model("sage-model")
class AutoSAGE(BaseAutoModel):
    r"""
    AutoSAGE. The model used in this automodel is GraphSAGE, i.e., the GraphSAGE from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper. The layer is

    .. math::

        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

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

    """

    def __init__(
        self, num_features=None, num_classes=None, device=None, **args
    ):

        super(AutoSAGE, self).__init__(num_features, num_classes, device, **args)

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
                "maxValue": [128, 128, 128],
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
            {
                "parameterName": "agg",
                "type": "CATEGORICAL",
                "feasiblePoints": ["mean", "add", "max"],
            },
        ]

        self.hyper_parameters = {
            "num_layers": 3,
            "hidden": [64, 32],
            "dropout": 0.5,
            "act": "relu",
            "agg": "mean",
        }

    def _initialize(self):
        self._model = GraphSAGE({
            "features_num": self.input_dimension,
            "num_class": self.output_dimension,
            **self.hyper_parameters
        }).to(self.device)
