import torch.nn.functional
import typing as _typing
import dgl
from dgl.nn.pytorch.glob import (
    SumPooling, AvgPooling, MaxPooling, SortPooling
)
from .. import base_decoder, decoder_registry
from ... import _utils
from ...encoders import base_encoder


class _LogSoftmaxDecoder(torch.nn.Module):
    def forward(
            self, features: _typing.Sequence[torch.Tensor],
            _graph: dgl.DGLGraph, *__args, **__kwargs
    ):
        return torch.nn.functional.log_softmax(features[-1], dim=-1)


@decoder_registry.DecoderUniversalRegistry.register_decoder('gin')
@decoder_registry.DecoderUniversalRegistry.register_decoder('gin_decoder')
class LogSoftmaxDecoderMaintainer(base_decoder.BaseAutoDecoderMaintainer):
    def _initialize(self, encoder, *args, **kwargs) -> _typing.Optional[bool]:
        self._decoder = _LogSoftmaxDecoder().to(self.device)
        return True


class _GINDecoder(torch.nn.Module):
    def __init__(
            self, input_dimensions: _typing.Sequence[int],
            hidden_dimension: int, output_dimension: int,
            act: _typing.Optional[str], dropout: _typing.Optional[float],
            graph_pooling_type: str, gf_dimension: _typing.Optional[int],
    ):
        super(_GINDecoder, self).__init__()
        _input_dimension: int = input_dimensions[-1]
        if isinstance(gf_dimension, int) and gf_dimension > 0:
            _input_dimension += gf_dimension
            self.__gf_dimension: _typing.Optional[int] = gf_dimension
        else:
            self.__gf_dimension: _typing.Optional[int] = None

        self._act: _typing.Optional[str] = act
        self._dropout: _typing.Optional[float] = dropout
        self._fc1: torch.nn.Linear = torch.nn.Linear(
            _input_dimension, hidden_dimension
        )
        self._fc2: torch.nn.Linear = torch.nn.Linear(
            hidden_dimension, output_dimension
        )
        if not isinstance(graph_pooling_type, str):
            raise TypeError
        elif graph_pooling_type.lower() == 'sum':
            self.__pool = SumPooling()
        elif graph_pooling_type.lower() == 'mean':
            self.__pool = AvgPooling()
        elif graph_pooling_type.lower() == 'max':
            self.__pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(
            self, features: _typing.Sequence[torch.Tensor],
            graph: dgl.DGLGraph, *__args, **__kwargs
    ) -> torch.Tensor:
        feature: torch.Tensor = features[-1]
        if (
                isinstance(self.__gf_dimension, int) and self.__gf_dimension > 0 and
                hasattr(graph, 'gf') and isinstance(graph.gf, torch.Tensor)
        ):
            gf: torch.Tensor = getattr(graph, 'gf')
            if not (gf.dim() == 2 and gf.size(1) == self.__gf_dimension):
                raise ValueError
            feature: torch.Tensor = torch.cat([feature, gf], dim=-1)
        feature: torch.Tensor = self._fc1(feature)
        feature: torch.Tensor = _utils.activation.activation_func(feature, self._act)
        if (
                isinstance(self._dropout, float)
                and 0 <= self._dropout <= 1
        ):
            feature: torch.Tensor = torch.nn.functional.dropout(
                feature, self._dropout, self.training
            )
        feature = self._fc2(feature)
        feature = self.__pool(graph, feature)
        return torch.nn.functional.log_softmax(feature, dim=-1)


@decoder_registry.DecoderUniversalRegistry.register_decoder('GIN'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('GINPool'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('GINPool_decoder'.lower())
class GINDecoderMaintainer(base_decoder.BaseAutoDecoderMaintainer):
    def _initialize(self, encoder: base_encoder.AutoHomogeneousEncoderMaintainer, *args, **kwargs) -> _typing.Optional[bool]:
        self._decoder = _GINDecoder(
            list(encoder.get_output_dimensions()),
            self.hyper_parameters["hidden"], self.output_dimension,
            self.hyper_parameters["act"], self.hyper_parameters["dropout"],
            self.hyper_parameters["graph_pooling_type"],
            gf_dimension=(
                getattr(self, "num_graph_features")
                if (
                        hasattr(self, "num_graph_features") and
                        isinstance(getattr(self, "num_graph_features"), int) and
                        getattr(self, "num_graph_features") > 0
                )
                else None
            )
        ).to(self.device)
        return True

    def __init__(
            self, output_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(GINDecoderMaintainer, self).__init__(
            output_dimension, device, *args, **kwargs
        )
        self.hyper_parameter_space = (
            {
                "parameterName": "hidden",
                "type": "INTEGER",
                "maxValue": 64,
                "minValue": 8,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.9,
                "minValue": 0.1,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "graph_pooling_type",
                "type": "CATEGORICAL",
                "feasiblePoints": ["sum", "mean", "max"],
            }
        )
        self.hyper_parameters = {
            "hidden": 64,
            "dropout": 0.5,
            "act": "relu",
            "graph_pooling_type": "sum"
        }


class _TopKPoolDecoder(torch.nn.Module):
    def __init__(
            self, input_dimensions: _typing.Iterable[int],
            output_dimension: int, dropout: float
    ):
        super(_TopKPoolDecoder, self).__init__()
        k: int = min(len(list(input_dimensions)), 3)
        self.__pool: SortPooling = SortPooling(k)
        self.__linear_predictions: torch.nn.ModuleList = (
            torch.nn.ModuleList()
        )
        for layer, dimension in enumerate(input_dimensions):
            self.__linear_predictions.append(
                torch.nn.Linear(dimension * k, output_dimension)
            )
        self._dropout = torch.nn.Dropout(dropout)

    def forward(
            self, features: _typing.Sequence[torch.Tensor],
            graph: dgl.DGLGraph, *__args, **__kwargs
    ):
        cumulative_result = 0
        for i, h in enumerate(features):
            cumulative_result += self._dropout(self.__linear_predictions[i](self.__pool(graph, h)))
        return cumulative_result


@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK_decoder'.lower())
class TopKDecoderMaintainer(base_decoder.BaseAutoDecoderMaintainer):
    def _initialize(
            self, encoder: base_encoder.AutoHomogeneousEncoderMaintainer, *args, **kwargs
    ) -> _typing.Optional[bool]:
        self._decoder = _TopKPoolDecoder(
            encoder.get_output_dimensions(),
            self.output_dimension,
            self.hyper_parameters["float"]
        ).to(self.device)
        return True

    def __init__(
            self, output_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(TopKDecoderMaintainer, self).__init__(
            output_dimension, device, *args, **kwargs
        )
        self.hyper_parameter_space = [
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.9,
                "minValue": 0.1,
                "scalingType": "LINEAR",
            }
        ]
        self.hyper_parameters = {
            "dropout": 0.5
        }
