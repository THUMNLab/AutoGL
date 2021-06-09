import torch
import torch.nn.functional
from torch_geometric.nn.conv import GCNConv
import typing as _typing
import autogl.data
from . import register_model
from .base import activate_func, ClassificationModel, SequentialGraphNeuralNetwork
from ...utils import get_logger

LOGGER = get_logger("GCNModel")


class GCN(SequentialGraphNeuralNetwork):
    class _GCNLayer(torch.nn.Module):
        def __init__(
                self, input_channels: int, output_channels: int,
                add_self_loops: bool = True, normalize: bool = True,
                activation_name: _typing.Optional[str] = ...,
                dropout_probability: _typing.Optional[float] = ...
        ):
            super().__init__()
            self._convolution: GCNConv = GCNConv(
                input_channels, output_channels,
                add_self_loops=bool(add_self_loops),
                normalize=bool(normalize)
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
            edge_index: torch.LongTensor = getattr(data, "edge_index")
            edge_weight: _typing.Optional[torch.Tensor] = getattr(data, "edge_weight")
            """ Validate the arguments """
            if not type(x) == type(edge_index) == torch.Tensor:
                raise TypeError
            if edge_weight is not None and (
                    type(edge_weight) != torch.Tensor or
                    edge_index.size() != (2, edge_weight.size(0))
            ):
                edge_weight: _typing.Optional[torch.Tensor] = None

            x: torch.Tensor = self._convolution.forward(x, edge_index, edge_weight)
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
            dropout: _typing.Union[float, _typing.Sequence[_typing.Optional[float]]],
            activation_name: str,
            add_self_loops: bool = True,
            normalize: bool = True
    ):
        if isinstance(dropout, _typing.Sequence):
            if len(dropout) != len(hidden_features) + 1:
                raise TypeError(
                    "When the dropout argument is a sequence, "
                    "The sequence length must equal to the number of layers to construct."
                )
            for _dropout in dropout:
                if _dropout is not None and type(_dropout) != float:
                    raise TypeError(
                        "When the dropout argument is a sequence, "
                        "every item in the sequence must be float or None"
                    )
            dropout_list: _typing.Sequence[_typing.Optional[float]] = dropout
        elif type(dropout) == float:
            if dropout < 0:
                dropout = 0
            if dropout > 1:
                dropout = 1
            dropout_list: _typing.Sequence[_typing.Optional[float]] = [
                dropout for _ in range(len(hidden_features) + 1)
            ]
        else:
            raise TypeError(
                "The provided dropout argument must be a float "
                "or a sequence in which each item is either float or None."
            )
        super().__init__()
        if len(hidden_features) == 0:
            self.__sequential_module_list: torch.nn.ModuleList = torch.nn.ModuleList(
                (
                    self._GCNLayer(
                        num_features, num_classes, add_self_loops, normalize,
                        dropout_probability=dropout_list[0]
                    ),
                )
            )
        else:
            self.__sequential_module_list: torch.nn.ModuleList = torch.nn.ModuleList()
            self.__sequential_module_list.append(self._GCNLayer(
                num_features, hidden_features[0], add_self_loops,
                normalize, activation_name, dropout_list[0]
            ))
            for hidden_feature_index in range(len(hidden_features)):
                if hidden_feature_index + 1 < len(hidden_features):
                    self.__sequential_module_list.append(self._GCNLayer(
                        hidden_features[hidden_feature_index],
                        hidden_features[hidden_feature_index + 1],
                        add_self_loops, normalize, activation_name,
                        dropout_list[hidden_feature_index + 1]
                    ))
                else:
                    self.__sequential_module_list.append(self._GCNLayer(
                        hidden_features[hidden_feature_index], num_classes,
                        add_self_loops, normalize,
                        dropout_list[-1]
                    ))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(x, dim=1)

    @property
    def encoder_sequential_modules(self) -> torch.nn.ModuleList:
        return self.__sequential_module_list

    def __extract_edge_indexes_and_weights(self, data) -> _typing.Union[
        _typing.Sequence[_typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]],
        _typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]
    ]:
        def __compose_edge_index_and_weight(
                _edge_index: torch.LongTensor,
                _edge_weight: _typing.Optional[torch.Tensor] = None
        ) -> _typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]:
            if type(_edge_index) != torch.Tensor or _edge_index.dtype != torch.int64:
                raise TypeError
            if _edge_weight is not None and (
                    type(_edge_weight) != torch.Tensor or
                    _edge_index.size() != (2, _edge_weight.size(0))
            ):
                _edge_weight: _typing.Optional[torch.Tensor] = None
            return _edge_index, _edge_weight

        if not (
                hasattr(data, "edge_indexes") and
                isinstance(getattr(data, "edge_indexes"), _typing.Sequence) and
                len(getattr(data, "edge_indexes")) == len(self.__sequential_module_list)
        ):
            return __compose_edge_index_and_weight(
                getattr(data, "edge_index"), getattr(data, "edge_weight", None)
            )
        for __edge_index in getattr(data, "edge_indexes"):
            if type(__edge_index) != torch.Tensor or __edge_index.dtype != torch.int64:
                return __compose_edge_index_and_weight(
                    getattr(data, "edge_index"), getattr(data, "edge_weight", None)
                )

        if (
                hasattr(data, "edge_weights") and
                isinstance(getattr(data, "edge_weights"), _typing.Sequence) and
                len(getattr(data, "edge_weights")) == len(self.__sequential_module_list)
        ):
            return [
                __compose_edge_index_and_weight(_edge_index, _edge_weight)
                for _edge_index, _edge_weight
                in zip(getattr(data, "edge_indexes"), getattr(data, "edge_weights"))
            ]
        else:
            return [
                __compose_edge_index_and_weight(__edge_index)
                for __edge_index in getattr(data, "edge_indexes")
            ]

    def encode(self, data) -> torch.Tensor:
        edge_indexes_and_weights: _typing.Union[
            _typing.Sequence[_typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]],
            _typing.Tuple[torch.LongTensor, _typing.Optional[torch.Tensor]]
        ] = self.__extract_edge_indexes_and_weights(data)

        if (
                (not isinstance(edge_indexes_and_weights, tuple))
                and isinstance(edge_indexes_and_weights[0], tuple)
        ):
            """ edge_indexes_and_weights is sequence of (edge_index, edge_weight) """
            assert len(edge_indexes_and_weights) == len(self.__sequential_module_list)
            x: torch.Tensor = getattr(data, "x")
            for _edge_index_and_weight, gcn in zip(
                    edge_indexes_and_weights, self.__sequential_module_list
            ):
                _temp_data = autogl.data.Data(x=x, edge_index=_edge_index_and_weight[0])
                _temp_data.edge_weight = _edge_index_and_weight[1]
                x = gcn(_temp_data)
            return x
        else:
            """ edge_indexes_and_weights is (edge_index, edge_weight) """
            x = getattr(data, "x")
            for gcn in self.__sequential_module_list:
                _temp_data = autogl.data.Data(x=x, edge_index=edge_indexes_and_weights[0])
                _temp_data.edge_weight = edge_indexes_and_weights[1]
                x = gcn(_temp_data)
            return x


@register_model("gcn")
class AutoGCN(ClassificationModel):
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
        default_hp_space: _typing.Sequence[_typing.Dict[str, _typing.Any]] = [
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
        default_hp = {
            "add_self_loops": 1,
            "normalize": 1,
            "num_layers": 2,
            "hidden": [16],
            "dropout": 0.2,
            "act": "leaky_relu",
        }

        super(AutoGCN, self).__init__(
            num_features, num_classes, device=device,
            hyper_parameter_space=default_hp_space,
            hyper_parameter=default_hp, init=init,
            **kwargs
        )

    def _initialize(self):
        self.model = GCN(
            self.num_features,
            self.num_classes,
            self.hyper_parameter.get("hidden"),
            self.hyper_parameter.get("dropout"),
            self.hyper_parameter.get("act"),
            add_self_loops=bool(self.hyper_parameter.get("add_self_loops", True)),
            normalize=bool(self.hyper_parameter.get("normalize", True))
        ).to(self.device)
