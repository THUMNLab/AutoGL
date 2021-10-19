import typing as _typing
import torch.nn.functional
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul

from . import register_model
from .base import ClassificationModel, ClassificationSupportedSequentialModel


class _GraphSAINTAggregationLayers:
    class MultiOrderAggregationLayer(torch.nn.Module):
        class Order0Aggregator(torch.nn.Module):
            def __init__(
                self,
                input_dimension: int,
                output_dimension: int,
                bias: bool = True,
                activation: _typing.Optional[str] = "ReLU",
                batch_norm: bool = True,
            ):
                super().__init__()
                if not type(input_dimension) == type(output_dimension) == int:
                    raise TypeError
                if not (input_dimension > 0 and output_dimension > 0):
                    raise ValueError
                if not type(bias) == bool:
                    raise TypeError
                self.__linear_transform = torch.nn.Linear(
                    input_dimension, output_dimension, bias
                )
                self.__linear_transform.reset_parameters()
                if type(activation) == str:
                    if activation.lower() == "ReLU".lower():
                        self.__activation = torch.nn.functional.relu
                    elif activation.lower() == "elu":
                        self.__activation = torch.nn.functional.elu
                    elif hasattr(torch.nn.functional, activation) and callable(
                        getattr(torch.nn.functional, activation)
                    ):
                        self.__activation = getattr(torch.nn.functional, activation)
                    else:
                        self.__activation = lambda x: x
                else:
                    self.__activation = lambda x: x
                if type(batch_norm) != bool:
                    raise TypeError
                else:
                    self.__optional_batch_normalization: _typing.Optional[
                        torch.nn.BatchNorm1d
                    ] = (
                        torch.nn.BatchNorm1d(output_dimension, 1e-8)
                        if batch_norm
                        else None
                    )

            def forward(
                self,
                x: _typing.Union[
                    torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor]
                ],
                _edge_index: torch.Tensor,
                _edge_weight: _typing.Optional[torch.Tensor] = None,
                _size: _typing.Optional[_typing.Tuple[int, int]] = None,
            ) -> torch.Tensor:
                __output: torch.Tensor = self.__linear_transform(x)
                if self.__activation is not None and callable(self.__activation):
                    __output: torch.Tensor = self.__activation(__output)
                if self.__optional_batch_normalization is not None and isinstance(
                    self.__optional_batch_normalization, torch.nn.BatchNorm1d
                ):
                    __output: torch.Tensor = self.__optional_batch_normalization(
                        __output
                    )
                return __output

        class Order1Aggregator(MessagePassing):
            def __init__(
                self,
                input_dimension: int,
                output_dimension: int,
                bias: bool = True,
                activation: _typing.Optional[str] = "ReLU",
                batch_norm: bool = True,
            ):
                super().__init__(aggr="add")
                if not type(input_dimension) == type(output_dimension) == int:
                    raise TypeError
                if not (input_dimension > 0 and output_dimension > 0):
                    raise ValueError
                if not type(bias) == bool:
                    raise TypeError
                self.__linear_transform = torch.nn.Linear(
                    input_dimension, output_dimension, bias
                )
                self.__linear_transform.reset_parameters()
                if type(activation) == str:
                    if activation.lower() == "ReLU".lower():
                        self.__activation = torch.nn.functional.relu
                    elif activation.lower() == "elu":
                        self.__activation = torch.nn.functional.elu
                    elif hasattr(torch.nn.functional, activation) and callable(
                        getattr(torch.nn.functional, activation)
                    ):
                        self.__activation = getattr(torch.nn.functional, activation)
                    else:
                        self.__activation = lambda x: x
                else:
                    self.__activation = lambda x: x
                if type(batch_norm) != bool:
                    raise TypeError
                else:
                    self.__optional_batch_normalization: _typing.Optional[
                        torch.nn.BatchNorm1d
                    ] = (
                        torch.nn.BatchNorm1d(output_dimension, 1e-8)
                        if batch_norm
                        else None
                    )

            def forward(
                self,
                x: _typing.Union[
                    torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor]
                ],
                _edge_index: torch.Tensor,
                _edge_weight: _typing.Optional[torch.Tensor] = None,
                _size: _typing.Optional[_typing.Tuple[int, int]] = None,
            ) -> torch.Tensor:

                if type(x) == torch.Tensor:
                    x: _typing.Tuple[torch.Tensor, torch.Tensor] = (x, x)

                __output = self.propagate(
                    _edge_index, x=x, edge_weight=_edge_weight, size=_size
                )
                __output: torch.Tensor = self.__linear_transform(__output)
                if self.__activation is not None and callable(self.__activation):
                    __output: torch.Tensor = self.__activation(__output)
                if self.__optional_batch_normalization is not None and isinstance(
                    self.__optional_batch_normalization, torch.nn.BatchNorm1d
                ):
                    __output: torch.Tensor = self.__optional_batch_normalization(
                        __output
                    )
                return __output

            def message(
                self, x_j: torch.Tensor, edge_weight: _typing.Optional[torch.Tensor]
            ) -> torch.Tensor:
                return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

            def message_and_aggregate(
                self,
                adj_t: SparseTensor,
                x: _typing.Union[
                    torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor]
                ],
            ) -> torch.Tensor:
                return matmul(adj_t, x[0], reduce=self.aggr)

        @property
        def integral_output_dimension(self) -> int:
            return (self._order + 1) * self._each_order_output_dimension

        def __init__(
            self,
            _input_dimension: int,
            _each_order_output_dimension: int,
            _order: int,
            bias: bool = True,
            activation: _typing.Optional[str] = "ReLU",
            batch_norm: bool = True,
            _dropout: _typing.Optional[float] = ...,
        ):
            super().__init__()
            if not (
                type(_input_dimension) == type(_order) == int
                and type(_each_order_output_dimension) == int
            ):
                raise TypeError
            if _input_dimension <= 0 or _each_order_output_dimension <= 0:
                raise ValueError
            if _order not in (0, 1):
                raise ValueError("Unsupported order number")
            self._input_dimension: int = _input_dimension
            self._each_order_output_dimension: int = _each_order_output_dimension
            self._order: int = _order
            if type(bias) != bool:
                raise TypeError
            self.__order0_transform = self.Order0Aggregator(
                self._input_dimension,
                self._each_order_output_dimension,
                bias,
                activation,
                batch_norm,
            )
            if _order == 1:
                self.__order1_transform = self.Order1Aggregator(
                    self._input_dimension,
                    self._each_order_output_dimension,
                    bias,
                    activation,
                    batch_norm,
                )
            else:
                self.__order1_transform = None
            if _dropout is not None and type(_dropout) == float:
                if _dropout < 0:
                    _dropout = 0
                if _dropout > 1:
                    _dropout = 1
                self.__optional_dropout: _typing.Optional[
                    torch.nn.Dropout
                ] = torch.nn.Dropout(_dropout)
            else:
                self.__optional_dropout: _typing.Optional[torch.nn.Dropout] = None

        def _forward(
            self,
            x: _typing.Union[torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor]],
            edge_index: torch.Tensor,
            edge_weight: _typing.Optional[torch.Tensor] = None,
            size: _typing.Optional[_typing.Tuple[int, int]] = None,
        ) -> torch.Tensor:
            if self.__order1_transform is not None and isinstance(
                self.__order1_transform, self.Order1Aggregator
            ):
                __output: torch.Tensor = torch.cat(
                    [
                        self.__order0_transform(x, edge_index, edge_weight, size),
                        self.__order1_transform(x, edge_index, edge_weight, size),
                    ],
                    dim=1,
                )
            else:
                __output: torch.Tensor = self.__order0_transform(
                    x, edge_index, edge_weight, size
                )
            if self.__optional_dropout is not None and isinstance(
                self.__optional_dropout, torch.nn.Dropout
            ):
                __output: torch.Tensor = self.__optional_dropout(__output)
            return __output

        def forward(self, data) -> torch.Tensor:
            x: torch.Tensor = getattr(data, "x")
            if type(x) != torch.Tensor:
                raise TypeError
            edge_index: torch.LongTensor = getattr(data, "edge_index")
            if type(edge_index) != torch.Tensor:
                raise TypeError
            edge_weight: _typing.Optional[torch.Tensor] = getattr(
                data, "edge_weight", None
            )
            if edge_weight is not None and type(edge_weight) != torch.Tensor:
                raise TypeError
            return self._forward(x, edge_index, edge_weight)

    class WrappedDropout(torch.nn.Module):
        def __init__(self, dropout_module: torch.nn.Dropout):
            super().__init__()
            self.__dropout_module: torch.nn.Dropout = dropout_module

        def forward(self, tenser_or_data) -> torch.Tensor:
            if type(tenser_or_data) == torch.Tensor:
                return self.__dropout_module(tenser_or_data)
            elif (
                hasattr(tenser_or_data, "x")
                and type(getattr(tenser_or_data, "x")) == torch.Tensor
            ):
                return self.__dropout_module(getattr(tenser_or_data, "x"))
            else:
                raise TypeError


class GraphSAINTMultiOrderAggregationModel(ClassificationSupportedSequentialModel):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        _output_dimension_for_each_order: int,
        _layers_order_list: _typing.Sequence[int],
        _pre_dropout: float,
        _layers_dropout: _typing.Union[float, _typing.Sequence[float]],
        activation: _typing.Optional[str] = "ReLU",
        bias: bool = True,
        batch_norm: bool = True,
        normalize: bool = True,
    ):
        super(GraphSAINTMultiOrderAggregationModel, self).__init__()
        if type(_output_dimension_for_each_order) != int:
            raise TypeError
        if not _output_dimension_for_each_order > 0:
            raise ValueError
        self._layers_order_list: _typing.Sequence[int] = _layers_order_list

        if isinstance(_layers_dropout, _typing.Sequence):
            if len(_layers_dropout) != len(_layers_order_list):
                raise ValueError
            else:
                self._layers_dropout: _typing.Sequence[float] = _layers_dropout
        elif type(_layers_dropout) == float:
            if _layers_dropout < 0:
                _layers_dropout = 0
            if _layers_dropout > 1:
                _layers_dropout = 1
            self._layers_dropout: _typing.Sequence[float] = [
                _layers_dropout for _ in _layers_order_list
            ]
        else:
            raise TypeError
        if type(_pre_dropout) != float:
            raise TypeError
        else:
            if _pre_dropout < 0:
                _pre_dropout = 0
            if _pre_dropout > 1:
                _pre_dropout = 1
        self.__sequential_encoding_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            (
                _GraphSAINTAggregationLayers.WrappedDropout(
                    torch.nn.Dropout(_pre_dropout)
                ),
                _GraphSAINTAggregationLayers.MultiOrderAggregationLayer(
                    num_features,
                    _output_dimension_for_each_order,
                    _layers_order_list[0],
                    bias,
                    activation,
                    batch_norm,
                    _layers_dropout[0],
                ),
            )
        )
        for _layer_index in range(1, len(_layers_order_list)):
            self.__sequential_encoding_layers.append(
                _GraphSAINTAggregationLayers.MultiOrderAggregationLayer(
                    self.__sequential_encoding_layers[-1].integral_output_dimension,
                    _output_dimension_for_each_order,
                    _layers_order_list[_layer_index],
                    bias,
                    activation,
                    batch_norm,
                    _layers_dropout[_layer_index],
                )
            )
        self.__apply_normalize: bool = normalize
        self.__linear_transform: torch.nn.Linear = torch.nn.Linear(
            self.__sequential_encoding_layers[-1].integral_output_dimension,
            num_classes,
            bias,
        )
        self.__linear_transform.reset_parameters()

    def cls_decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.__apply_normalize:
            x: torch.Tensor = torch.nn.functional.normalize(x, p=2, dim=1)
        return torch.nn.functional.log_softmax(self.__linear_transform(x), dim=1)

    def cls_encode(self, data) -> torch.Tensor:
        if type(getattr(data, "x")) != torch.Tensor:
            raise TypeError
        if type(getattr(data, "edge_index")) != torch.Tensor:
            raise TypeError
        if (
            getattr(data, "edge_weight", None) is not None
            and type(getattr(data, "edge_weight")) != torch.Tensor
        ):
            raise TypeError
        for encoding_layer in self.__sequential_encoding_layers:
            setattr(data, "x", encoding_layer(data))
        return getattr(data, "x")

    @property
    def sequential_encoding_layers(self) -> torch.nn.ModuleList:
        return self.__sequential_encoding_layers


@register_model("GraphSAINTAggregationModel")
class GraphSAINTAggregationModel(ClassificationModel):
    def __init__(
        self,
        num_features: int = ...,
        num_classes: int = ...,
        device: _typing.Union[str, torch.device] = ...,
        init: bool = False,
        **kwargs
    ):
        super(GraphSAINTAggregationModel, self).__init__(
            num_features, num_classes, device=device, init=init, **kwargs
        )
        # todo: Initialize with default hyper parameter space and hyper parameter

    def _initialize(self):
        """ Initialize model """
        self.model = GraphSAINTMultiOrderAggregationModel(
            self.num_features,
            self.num_classes,
            self.hyper_parameter.get("output_dimension_for_each_order"),
            self.hyper_parameter.get("layers_order_list"),
            self.hyper_parameter.get("pre_dropout"),
            self.hyper_parameter.get("layers_dropout"),
            self.hyper_parameter.get("activation", "ReLU"),
            bool(self.hyper_parameter.get("bias", True)),
            bool(self.hyper_parameter.get("batch_norm", True)),
            bool(self.hyper_parameter.get("normalize", True)),
        ).to(self.device)
