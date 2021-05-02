import typing as _typing
import torch
import torch.nn.functional
from torch_geometric.nn.conv import SAGEConv

from . import register_model
from .base import ClassificationModel, activate_func


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_features: _typing.Sequence[int],
        dropout: float,
        activation_name: str,
        aggr: str = "mean"
    ):
        super(GraphSAGE, self).__init__()
        if type(aggr) != str:
            raise TypeError
        if aggr not in ("add", "max", "mean"):
            aggr = "mean"

        self.__convolution_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        num_layers: int = len(hidden_features) + 1
        if num_layers == 1:
            self.__convolution_layers.append(
                SAGEConv(num_features, num_classes, aggr=aggr)
            )
        else:
            self.__convolution_layers.append(
                SAGEConv(num_features, hidden_features[0], aggr=aggr)
            )
            for i in range(len(hidden_features)):
                if i + 1 < len(hidden_features):
                    self.__convolution_layers.append(
                        SAGEConv(hidden_features[i], hidden_features[i + 1], aggr=aggr)
                    )
                else:
                    self.__convolution_layers.append(
                        SAGEConv(hidden_features[i], num_classes, aggr=aggr)
                    )
        self.__dropout: float = dropout
        self.__activation_name: str = activation_name

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

    def __layer_wise_forward(
            self, x: torch.Tensor,
            edge_indexes: _typing.Sequence[torch.Tensor],
            edge_weights: _typing.Sequence[_typing.Optional[torch.Tensor]]
    ) -> torch.Tensor:
        assert len(edge_indexes) == len(edge_weights) == len(self.__convolution_layers)
        for edge_index in edge_indexes:
            if type(edge_index) != torch.Tensor:
                raise TypeError
            if edge_index.size(0) != 2:
                raise ValueError
        for edge_weight in edge_weights:
            if not (edge_weight is None or type(edge_weight) == torch.Tensor):
                raise TypeError

        for layer_index in range(len(self.__convolution_layers)):
            x: torch.Tensor = self.__convolution_layers[layer_index](
                x, edge_indexes[layer_index]
            )
            if layer_index + 1 < len(self.__convolution_layers):
                x = activate_func(x, self.__activation_name)
                x = torch.nn.functional.dropout(x, p=self.__dropout, training=self.training)
        return torch.nn.functional.log_softmax(x, dim=1)

    def forward(self, data) -> torch.Tensor:
        if (
                hasattr(data, "edge_indexes") and
                isinstance(getattr(data, "edge_indexes"), _typing.Sequence) and
                len(getattr(data, "edge_indexes")) == len(self.__convolution_layers)
        ):
            edge_indexes: _typing.Sequence[torch.Tensor] = getattr(data, "edge_indexes")
            if (
                hasattr(data, "edge_weights") and
                isinstance(getattr(data, "edge_weights"), _typing.Sequence) and
                len(getattr(data, "edge_weights")) == len(self.__convolution_layers)
            ):
                edge_weights: _typing.Sequence[_typing.Optional[torch.Tensor]] = (
                    getattr(data, "edge_weights")
                )
            else:
                edge_weights: _typing.Sequence[_typing.Optional[torch.Tensor]] = [
                    None for _ in range(len(self.__convolution_layers))
                ]
            return self.__layer_wise_forward(
                getattr(data, "x"), edge_indexes, edge_weights
            )
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


@register_model("sage")
class AutoSAGE(ClassificationModel):
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
            self,
            num_features: int = ...,
            num_classes: int = ...,
            device: _typing.Union[str, torch.device] = ...,
            init: bool = False,
            **kwargs
    ):
        default_hp_space: _typing.Sequence[_typing.Dict[str, _typing.Any]] = [
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
                "parameterName": "aggr",
                "type": "CATEGORICAL",
                "feasiblePoints": ["mean", "add", "max"],
            },
        ]
        super(AutoSAGE, self).__init__(
            num_features, num_classes, device=device,
            hyper_parameter_space=default_hp_space, init=init, **kwargs
        )

    def _initialize(self):
        """ Initialize model """
        self.model = GraphSAGE(
            self.num_features,
            self.num_classes,
            self.hyper_parameter.get("hidden"),
            self.hyper_parameter.get("dropout"),
            self.hyper_parameter.get("act"),
            self.hyper_parameter.get("aggr")
        ).to(self.device)
