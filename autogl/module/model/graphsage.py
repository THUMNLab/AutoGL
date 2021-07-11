import torch
import typing as _typing

from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional
import autogl.data
from . import register_model
from .base import BaseModel, activate_func, ClassificationSupportedSequentialModel
from ...utils import get_logger

LOGGER = get_logger("SAGEModel")


class GraphSAGE(ClassificationSupportedSequentialModel):
    class _SAGELayer(torch.nn.Module):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            aggr: str,
            activation_name: _typing.Optional[str] = ...,
            dropout_probability: _typing.Optional[float] = ...,
        ):
            super().__init__()
            self._convolution: SAGEConv = SAGEConv(
                input_channels, output_channels, aggr=aggr
            )
            if (
                activation_name is not Ellipsis
                and activation_name is not None
                and type(activation_name) == str
            ):
                self._activation_name: _typing.Optional[str] = activation_name
            else:
                self._activation_name: _typing.Optional[str] = None
            if (
                dropout_probability is not Ellipsis
                and dropout_probability is not None
                and type(dropout_probability) == float
            ):
                if dropout_probability < 0:
                    dropout_probability = 0
                if dropout_probability > 1:
                    dropout_probability = 1
                self._dropout: _typing.Optional[torch.nn.Dropout] = torch.nn.Dropout(
                    dropout_probability
                )
            else:
                self._dropout: _typing.Optional[torch.nn.Dropout] = None

        def forward(self, data, enable_activation: bool = True) -> torch.Tensor:
            x: torch.Tensor = getattr(data, "x")
            edge_index: torch.Tensor = getattr(data, "edge_index")
            if type(x) != torch.Tensor or type(edge_index) != torch.Tensor:
                raise TypeError

            x: torch.Tensor = self._convolution.forward(x, edge_index)
            if self._activation_name is not None and enable_activation:
                x: torch.Tensor = activate_func(x, self._activation_name)
            if self._dropout is not None:
                x: torch.Tensor = self._dropout.forward(x)
            return x

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_features: _typing.Sequence[int],
        activation_name: str,
        layers_dropout: _typing.Union[
            _typing.Optional[float], _typing.Sequence[_typing.Optional[float]]
        ] = None,
        aggr: str = "mean",
    ):
        super().__init__()
        if not type(num_features) == type(num_classes) == int:
            raise TypeError
        if not isinstance(hidden_features, _typing.Sequence):
            raise TypeError
        for hidden_feature in hidden_features:
            if type(hidden_feature) != int:
                raise TypeError
            elif hidden_feature <= 0:
                raise ValueError
        if isinstance(layers_dropout, _typing.Sequence):
            if len(layers_dropout) != (len(hidden_features) + 1):
                raise TypeError
            for d in layers_dropout:
                if d is not None and type(d) != float:
                    raise TypeError
            _layers_dropout: _typing.Sequence[_typing.Optional[float]] = layers_dropout
        elif layers_dropout is None or type(layers_dropout) == float:
            _layers_dropout: _typing.Sequence[_typing.Optional[float]] = [
                layers_dropout for _ in range(len(hidden_features))
            ] + [None]
        else:
            raise TypeError
        if not type(activation_name) == type(aggr) == str:
            raise TypeError
        if aggr not in ("add", "max", "mean"):
            aggr = "mean"

        if len(hidden_features) == 0:
            self.__sequential_encoding_layers: torch.nn.ModuleList = (
                torch.nn.ModuleList(
                    [
                        self._SAGELayer(
                            num_features,
                            num_classes,
                            aggr,
                            activation_name,
                            _layers_dropout[0],
                        )
                    ]
                )
            )
        else:
            self.__sequential_encoding_layers: torch.nn.ModuleList = (
                torch.nn.ModuleList(
                    [
                        self._SAGELayer(
                            num_features,
                            hidden_features[0],
                            aggr,
                            activation_name,
                            _layers_dropout[0],
                        )
                    ]
                )
            )
            for i in range(len(hidden_features)):
                if i + 1 < len(hidden_features):
                    self.__sequential_encoding_layers.append(
                        self._SAGELayer(
                            hidden_features[i],
                            hidden_features[i + 1],
                            aggr,
                            activation_name,
                            _layers_dropout[i + 1],
                        )
                    )
                else:
                    self.__sequential_encoding_layers.append(
                        self._SAGELayer(
                            hidden_features[i],
                            num_classes,
                            aggr,
                            _layers_dropout[i + 1],
                        )
                    )

    @property
    def sequential_encoding_layers(self) -> torch.nn.ModuleList:
        return self.__sequential_encoding_layers

    def cls_encode(self, data) -> torch.Tensor:
        if (
            hasattr(data, "edge_indexes")
            and isinstance(getattr(data, "edge_indexes"), _typing.Sequence)
            and len(getattr(data, "edge_indexes"))
            == len(self.__sequential_encoding_layers)
        ):
            for __edge_index in getattr(data, "edge_indexes"):
                if type(__edge_index) != torch.Tensor:
                    raise TypeError
            """ Layer-wise encode """
            x: torch.Tensor = getattr(data, "x")
            for i, __edge_index in enumerate(getattr(data, "edge_indexes")):
                x: torch.Tensor = self.__sequential_encoding_layers[i](
                    autogl.data.Data(x=x, edge_index=__edge_index)
                )
            return x
        else:
            x: torch.Tensor = getattr(data, "x")
            for i in range(len(self.__sequential_encoding_layers)):
                x = self.__sequential_encoding_layers[i](
                    autogl.data.Data(x, getattr(data, "edge_index"))
                )
            return x

    def cls_decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(x, dim=1)

    def lp_encode(self, data):
        x: torch.Tensor = getattr(data, "x")
        for i in range(len(self.__sequential_encoding_layers) - 2):
            x = self.__sequential_encoding_layers[i](
                autogl.data.Data(x, getattr(data, "edge_index"))
            )
        x = self.__sequential_encoding_layers[-2](
            autogl.data.Data(x, getattr(data, "edge_index")), enable_activation=False
        )
        return x

    def lp_decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def lp_decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


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
        self, num_features=None, num_classes=None, device=None, init=False, **args
    ):

        super(AutoSAGE, self).__init__()

        self.num_features = num_features if num_features is not None else 0
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.device = device if device is not None else "cpu"
        self.init = True

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
        self.model = GraphSAGE(
            self.num_features,
            self.num_classes,
            self.hyperparams.get("hidden"),
            self.hyperparams.get("act", "relu"),
            self.hyperparams.get("dropout", None),
            self.hyperparams.get("agg", "mean"),
        ).to(self.device)
