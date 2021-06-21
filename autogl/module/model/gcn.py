import torch
import torch.nn.functional
import torch_geometric
from torch_geometric.nn import GCNConv
import typing as _typing
from . import register_model
from .base import BaseModel, activate_func, ClassificationModel
from ...utils import get_logger

LOGGER = get_logger("GCNModel")


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_features: _typing.Sequence[int],
        dropout: float,
        activation_name: str,
    ):
        super().__init__()
        self.__convolution_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        num_layers: int = len(hidden_features) + 1
        if num_layers == 1:
            self.__convolution_layers.append(GCNConv(num_features, num_classes))
        else:
            self.__convolution_layers.append(GCNConv(num_features, hidden_features[0]))
            for i in range(len(hidden_features)):
                self.__convolution_layers.append(
                    GCNConv(hidden_features[i], hidden_features[i + 1])
                    if i + 1 < len(hidden_features)
                    else GCNConv(hidden_features[i], num_classes)
                )
        self.__dropout: float = dropout
        self.__activation_name: str = activation_name

    def __layer_wise_forward(self, data):
        # todo: Implement this forward method
        #         in case that data.edge_indexes property is provided
        #         for Layer-wise and Node-wise sampled training
        raise NotImplementedError

    def __basic_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: _typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer_index in range(len(self.__convolution_layers)):
            x: torch.Tensor = self.__convolution_layers[layer_index](
                x, edge_index, edge_weight
            )
            if layer_index + 1 < len(self.__convolution_layers):
                x = activate_func(x, self.__activation_name)
                x = torch.nn.functional.dropout(
                    x, p=self.__dropout, training=self.training
                )
        return torch.nn.functional.log_softmax(x, dim=1)

    def forward(self, data) -> torch.Tensor:
        if hasattr(data, "edge_indexes") and getattr(data, "edge_indexes") is not None:
            return self.__layer_wise_forward(data)
        else:
            if not (hasattr(data, "x") and hasattr(data, "edge_index")):
                raise AttributeError
            if not (
                type(getattr(data, "x")) == torch.Tensor
                and type(getattr(data, "edge_index")) == torch.Tensor
            ):
                raise TypeError
            x: torch.Tensor = getattr(data, "x")
            edge_index: torch.LongTensor = getattr(data, "edge_index")
            if (
                hasattr(data, "edge_weight")
                and type(getattr(data, "edge_weight")) == torch.Tensor
                and getattr(data, "edge_weight").size() == (edge_index.size(1),)
            ):
                edge_weight: _typing.Optional[torch.Tensor] = getattr(
                    data, "edge_weight"
                )
            else:
                edge_weight: _typing.Optional[torch.Tensor] = None
            return self.__basic_forward(x, edge_index, edge_weight)

    def encode(self, data):
        x = data.x
        num_layers = len(self.__convolution_layers)
        for i in range(num_layers - 1):
            x = self.__convolution_layers[i](x, data.train_pos_edge_index)
            if i != num_layers - 2:
                x = activate_func(x, self.__activation_name)
                # x = F.dropout(x, p=self.args["dropout"], training=self.training)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# @register_model("gcn")
# class AutoGCN(ClassificationModel):
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
        num_features: int = ...,
        num_classes: int = ...,
        device: _typing.Union[str, torch.device] = ...,
        init: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device

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
        ]

        # initial point of hp search
        # self.hyperparams = {
        #     "num_layers": 2,
        #     "hidden": [16],
        #     "dropout": 0.2,
        #     "act": "leaky_relu",
        # }

        self.hyperparams = {
            "num_layers": 3,
            "hidden": [128, 64],
            "dropout": 0,
            "act": "relu",
        }

        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        self.model = GCN(
            self.num_features,
            self.num_classes,
            self.hyperparams.get("hidden"),
            self.hyperparams.get("dropout"),
            self.hyperparams.get("act"),
        ).to(self.device)
