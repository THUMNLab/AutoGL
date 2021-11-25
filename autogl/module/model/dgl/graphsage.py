import torch
import typing as _typing

import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn.functional

from . import _decoder, register_model
from .base import BaseModel, activate_func
from ....utils import get_logger

LOGGER = get_logger("SAGEModel")


class GraphSAGE(torch.nn.Module):

    def __init__(self, args, **kwargs):
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

        ''' Set decoder '''
        decoder: _typing.Union[str, _typing.Type[_decoder.RepresentationDecoder], None] = kwargs.get("decoder")
        if issubclass(decoder, _decoder.RepresentationDecoder):
            self._decoder: _typing.Optional[_decoder.RepresentationDecoder] = decoder(self.args)
        elif isinstance(decoder, str) and len(decoder.strip()) > 0:
            self._decoder: _typing.Optional[_decoder.RepresentationDecoder] = (
                _decoder.RepresentationDecoderUniversalRegistry.get_representation_decoder(decoder)(
                    self.args
                ) if not ('no' in decoder.lower() or 'null' in decoder.lower()) else None
            )
        else:
            self._decoder: _typing.Optional[_decoder.RepresentationDecoder] = None

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
            x = self.convs[i](data)
            x = activate_func(x, self.args["act"])
        x = self.convs[-2](data)
        return x

    def lp_decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def lp_decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
    def forward(self, data, *args, **kwargs):
        x = data.ndata['feat']

        for i in range(self.num_layer):
            x = self.convs[i](data, x)
            if i != self.num_layer - 1:
                x = activate_func(x, self.args["act"])
                x = F.dropout(x, p=self.args["dropout"], training=self.training)

        if (
                self._decoder is not None and
                isinstance(self._decoder, _decoder.RepresentationDecoder)
        ):
            return self._decoder(data, x, *args, **kwargs)
        else:
            return x



@register_model("sage")
class AutoSAGE(BaseModel):
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
            self, num_features=None, num_classes=None, device=None, init=False,
            decoder: _typing.Union[_typing.Type[_decoder.RepresentationDecoder], str, None] = ...,
            **kwargs
    ):

        super(AutoSAGE, self).__init__()

        self.num_features = num_features if num_features is not None else 0
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.device = device if device is not None else "cpu"
        self.decoder = decoder
        self.params = {
            "features_num": self.num_features,
            "num_class": self.num_classes,
        }
        self.space = [
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

        self.hyperparams = {
            "num_layers": 3,
            "hidden": [64, 32],
            "dropout": 0.5,
            "act": "relu",
            "agg": "mean",
        }

        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        self.model = GraphSAGE({**self.params, **self.hyperparams}, decoder=self.decoder).to(self.device)
