import torch
import torch.nn.functional as F
from typing import Sequence, Optional, Union, Tuple
from numbers import Real

from dgl.nn.pytorch.conv import GraphConv
from dgl import remove_self_loop, add_self_loop
import autogl.data
from . import register_model
from .base import BaseAutoModel, activate_func, ClassificationSupportedSequentialModel
from ....utils import get_logger


LOGGER = get_logger("GCNModel")


class GCN(torch.nn.Module):
    def __init__(self, args):
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

        
        self.convs.append(
            GraphConv(
                self.args["features_num"],
                self.args["hidden"][0]
            )
        )

        for i in range(self.num_layer - 2):
            self.convs.append(
                GraphConv(
                    self.args["hidden"][i],
                    self.args["hidden"][i + 1]
                )
            )
        self.convs.append(
            GraphConv(
                self.args["hidden"][-1],
                self.args["num_class"]
            )
        )

    def forward(self, data):
        x = data.ndata['feat']
        for i in range(len(self.convs)):
            if i!=0:
                x = F.dropout(x, p=self.args["dropout"], training=self.training)
            x = self.convs[i](data, x)

            if i != self.num_layer - 1:
                x = activate_func(x, self.args["act"])

        return F.log_softmax(x, dim=1)


    def cls_encode(self, data) -> torch.Tensor:
        return self(data)

    def cls_decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(x, dim=1)

    def lp_encode(self, data):
        # discard the last layer, only use the layer before

        x = data.ndata['feat']
        for i in range(len(self.convs) - 1):
            if i != 0:
                x = F.dropout(x, p=self.args["dropout"], training=self.training)
            x = self.convs[i](data, x)

            if i != len(self.convs) - 2:
                x = activate_func(x, self.args["act"])

        return x

    def lp_decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def lp_decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


@register_model("gcn-model")
class AutoGCN(BaseAutoModel):
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
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        device: Union[str, torch.device] = 'cpu',
        **kwargs
    ) -> None:
        super().__init__(input_dimension, output_dimension, device, **kwargs)
        
        self.hyper_parameter_space = [
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
        self.hyper_parameters = {
            "num_layers": 3,
            "hidden": [128, 64],
            "dropout": 0.,
            "act": "relu",
        }

    def _initialize(self):
        self._model = GCN({
            "features_num": self.input_dimension,
            "num_class": self.output_dimension,
            **self.hyper_parameters
        }).to(self.device)
