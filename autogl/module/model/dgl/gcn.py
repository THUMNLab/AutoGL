import torch
import typing as _typing
import torch.nn.functional as F
from typing import Optional, Union

from dgl.nn.pytorch.conv import GraphConv
import autogl.data
from . import _decoder, register_model
from .base import BaseModel, activate_func
from ....utils import get_logger


LOGGER = get_logger("GCNModel")


class GCN(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(GCN, self).__init__()
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
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))

        if not self.num_layer == len(self.args["hidden"]) + 1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")
        self.convs = torch.nn.ModuleList()

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
            GraphConv(
                self.args["features_num"],
                self.args["hidden"][0]
            )
        )

        for i in range(self.num_layer - 2):
            self.convs.append(
                GraphConv(
                    self.args["hidden"][0],
                    self.args["hidden"][i + 1]
                )
            )
        self.convs.append(
            GraphConv(
                self.args["hidden"][-1],
                self.args["num_class"]
            )
        )

    def forward(self, data, *args, **kwargs):
        x = data.ndata['feat']
        for i in range(len(self.convs)):
            if i!=0:
                x = F.dropout(x, p=self.args["dropout"], training=self.training)
            x = self.convs[i](data, x)

            if i != self.num_layer - 1:
                x = activate_func(x, self.args["act"])

        if (
                self._decoder is not None and
                isinstance(self._decoder, _decoder.RepresentationDecoder)
        ):
            return self._decoder(data, x, *args, **kwargs)
        else:
            return x


    def cls_encode(self, data) -> torch.Tensor:
        return self(data)

    def cls_decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(x, dim=1)

    def lp_encode(self, data):
        x: torch.Tensor = data.ndata['feat']
        for i in range(len(self.convs) - 2):
            x = self.convs[i](
                autogl.data.Data(x, data.edges())
            )
        x = self.__sequential_encoding_layers[-2](
            autogl.data.Data(x, data.edges()), enable_activation=False
        )
        return x

    def lp_decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def lp_decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


@register_model("gcn")
class AutoGCN(BaseModel):
    r"""
    AutoGCN.
    The model used in this automodel is GCN, i.e., the graph convolutional network from the
    `"Semi-supervised Classification with Graph Convolutional
    Networks" <https://arxiv.org/abs/1609.02907>`_ paper. The layer is

    .. math::

        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Parameters
    ----------
    num_features: ``int``
        The dimension of features.

    num_classes: ``int``
        The number of classes.

    device: ``torch.device`` or ``str``
        The device where model will be running on.

    init: `bool`.
        If True(False), the model will (not) be initialized.
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        device: Union[str, torch.device] = 'cpu',
        init: bool = False,
        decoder: _typing.Union[_typing.Type[_decoder.RepresentationDecoder], str, None] = ...,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.decoder = decoder
        self.params = {
            "features_num": self.num_features,
            "num_class": self.num_classes,
        }
        self.space = [
            {
                "parameterName": "add_self_loops",
                "type": "CATEGORICAL",
                "feasiblePoints": [1],
            },
            {
                "parameterName": "normalize",
                "type": "CATEGORICAL",
                "feasiblePoints": [1],
            },
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
        ]

        # initial point of hp search
        self.hyperparams = {
            "num_layers": 3,
            "hidden": [128, 64],
            "dropout": 0.,
            "act": "relu",
        }

        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        self.model = GCN({**self.params, **self.hyperparams}, decoder=self.decoder).to(self.device)
