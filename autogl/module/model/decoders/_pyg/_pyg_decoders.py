import torch.nn.functional
import typing as _typing
import torch_geometric
from torch_geometric.nn.glob import global_add_pool
from ...encoders import base_encoder
from .. import base_decoder, decoder_registry
from ... import _utils


class _LogSoftmaxDecoder(torch.nn.Module):
    def forward(self, features: _typing.Sequence[torch.Tensor], *__args, **__kwargs) -> torch.Tensor:
        return torch.nn.functional.log_softmax(features[-1])


@decoder_registry.DecoderUniversalRegistry.register_decoder('log_softmax')
@decoder_registry.DecoderUniversalRegistry.register_decoder('log_softmax_decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('LogSoftmax'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('LogSoftmax_decoder'.lower())
class LogSoftmaxDecoderMaintainer(base_decoder.BaseAutoDecoderMaintainer):
    def _initialize(self, *args, **kwargs) -> _typing.Optional[bool]:
        self._decoder = _LogSoftmaxDecoder().to(self.device)
        return True


class _GINDecoder(torch.nn.Module):
    def __init__(
            self, _final_dimension: int, hidden_dimension: int, output_dimension: int,
            _act: _typing.Optional[str], _dropout: _typing.Optional[float],
            num_graph_features: _typing.Optional[int]
    ):
        super(_GINDecoder, self).__init__()
        if (
                isinstance(num_graph_features, int)
                and num_graph_features > 0
        ):
            _final_dimension += num_graph_features
            self.__num_graph_features: _typing.Optional[int] = num_graph_features
        else:
            self.__num_graph_features: _typing.Optional[int] = None
        self._fc1: torch.nn.Linear = torch.nn.Linear(
            _final_dimension, hidden_dimension
        )
        self._fc2: torch.nn.Linear = torch.nn.Linear(
            hidden_dimension, output_dimension
        )
        self._act: _typing.Optional[str] = _act
        self._dropout: _typing.Optional[float] = _dropout

    def forward(
            self, features: _typing.Sequence[torch.Tensor],
            data: torch_geometric.data.Data, *__args, **__kwargs
    ):
        feature = features[-1]
        feature = global_add_pool(feature, data.batch)
        if (
                isinstance(self.__num_graph_features, int)
                and self.__num_graph_features > 0
        ):
            if (
                    hasattr(data, 'gf') and
                    isinstance(data.gf, torch.Tensor) and data.gf.dim() == 2 and
                    data.gf.size() == (feature.size(0), self.__num_graph_features)
            ):
                graph_features: torch.Tensor = data.gf
            else:
                raise ValueError(
                    f"The provided data is expected to contain property 'gf' "
                    f"with {self.__num_graph_features} dimensions as graph feature"
                )
            feature: torch.Tensor = torch.cat([feature, graph_features], dim=-1)
        feature: torch.Tensor = self._fc1(feature)
        feature: torch.Tensor = _utils.activation.activation_func(feature, self._act)
        if isinstance(self._dropout, float) and 0 <= self._dropout <= 1:
            feature: torch.Tensor = torch.nn.functional.dropout(
                feature, self._dropout, self.training
            )
        feature: torch.Tensor = self._fc2(feature)
        return torch.nn.functional.log_softmax(feature, dim=-1)


@decoder_registry.DecoderUniversalRegistry.register_decoder('GIN'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('GINPool'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('GINPool_decoder'.lower())
class GINDecoderMaintainer(base_decoder.BaseAutoDecoderMaintainer):
    def _initialize(self, encoder: base_encoder.AutoHomogeneousEncoderMaintainer, *args, **kwargs) -> _typing.Optional[bool]:
        if (
                isinstance(getattr(self, "num_graph_features"), int) and
                getattr(self, "num_graph_features") > 0
        ):
            num_graph_features: _typing.Optional[int] = getattr(self, "num_graph_features")
        else:
            num_graph_features: _typing.Optional[int] = None
        self._decoder = _GINDecoder(
            tuple(encoder.get_output_dimensions())[-1],
            self.hyper_parameters['hidden'], self.output_dimension,
            self.hyper_parameters['act'], self.hyper_parameters['dropout'],
            num_graph_features
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
        self.hyper_parameter_space = [
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
            }
        ]
        self.hyper_parameters = {
            "hidden": 32,
            "act": "relu",
            "dropout": 0.5
        }
