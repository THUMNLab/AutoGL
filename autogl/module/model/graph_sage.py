import typing as _typing
import torch
import torch.nn.functional
from torch_geometric.nn.conv import SAGEConv

import autogl.data
from . import register_model
from .base import (
    ClassificationModel, activate_func,
    SequentialGraphNeuralNetwork
)


class GraphSAGE(SequentialGraphNeuralNetwork):
    class _SAGELayer(torch.nn.Module):
        def __init__(
                self, input_channels: int, output_channels: int, aggr: str,
                activation_name: _typing.Optional[str] = ...,
                dropout_probability: _typing.Optional[float] = ...
        ):
            super().__init__()
            self._convolution: SAGEConv = SAGEConv(
                input_channels, output_channels, aggr=aggr
            )
            if (
                    activation_name is not Ellipsis and
                    activation_name is not None and
                    type(activation_name) == str
            ):
                self._activation_name: _typing.Optional[str] = activation_name
            else:
                self._activation_name: _typing.Optional[str] = None
            if (
                    dropout_probability is not Ellipsis and
                    dropout_probability is not None and
                    type(dropout_probability) == float
            ):
                if dropout_probability < 0:
                    dropout_probability = 0
                if dropout_probability > 1:
                    dropout_probability = 1
                self._dropout: _typing.Optional[torch.nn.Dropout] = (
                    torch.nn.Dropout(dropout_probability)
                )
            else:
                self._dropout: _typing.Optional[torch.nn.Dropout] = None

        def forward(self, data) -> torch.Tensor:
            x: torch.Tensor = getattr(data, "x")
            edge_index: torch.Tensor = getattr(data, "edge_index")
            if type(x) != torch.Tensor or type(edge_index) != torch.Tensor:
                raise TypeError

            x: torch.Tensor = self._convolution.forward(x, edge_index)
            if self._activation_name is not None:
                x: torch.Tensor = activate_func(x, self._activation_name)
            if self._dropout is not None:
                x: torch.Tensor = self._dropout.forward(x)
            return x

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

        if len(hidden_features) == 0:
            self.__sequential_module_list: torch.nn.ModuleList = torch.nn.ModuleList(
                (self._SAGELayer(num_features, num_classes, aggr),)
            )
        else:
            self.__sequential_module_list: torch.nn.ModuleList = torch.nn.ModuleList()
            self.__sequential_module_list.append(self._SAGELayer(
                num_features, hidden_features[0], aggr, activation_name, dropout
            ))
            for i in range(len(hidden_features)):
                if i + 1 < len(hidden_features):
                    self.__sequential_module_list.append(self._SAGELayer(
                        hidden_features[i], hidden_features[i + 1], aggr,
                        activation_name, dropout
                    ))
                else:
                    self.__sequential_module_list.append(self._SAGELayer(
                        hidden_features[i], num_classes, aggr
                    ))

    @property
    def encoder_sequential_modules(self) -> torch.nn.ModuleList:
        return self.__sequential_module_list

    def encode(self, data) -> torch.Tensor:
        if (
            hasattr(data, "edge_indexes") and
            isinstance(getattr(data, "edge_indexes"), _typing.Sequence) and
            len(getattr(data, "edge_indexes")) == len(self.__sequential_module_list)
        ):
            for __edge_index in getattr(data, "edge_indexes"):
                if type(__edge_index) != torch.Tensor:
                    raise TypeError
            """ Layer-wise encode """
            x: torch.Tensor = getattr(data, "x")
            for i, __edge_index in enumerate(getattr(data, "edge_indexes")):
                _intermediate_data: autogl.data.Data = autogl.data.Data(
                    x=x, edge_index=__edge_index
                )
                x: torch.Tensor = self.encoder_sequential_modules[i](_intermediate_data)
            return x
        else:
            for i in range(len(self.encoder_sequential_modules)):
                data.x = self.encoder_sequential_modules[i](data)
            return data.x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(x, dim=1)


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
        default_hp = {
            "num_layers": 3,
            "hidden": [64, 32],
            "dropout": 0.5,
            "act": "relu",
            "agg": "mean",
        }
        super(AutoSAGE, self).__init__(
            num_features, num_classes, device=device,
            hyper_parameter_space=default_hp_space,
            hyper_parameter=default_hp, init=init, **kwargs
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
