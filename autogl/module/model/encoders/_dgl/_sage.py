import torch.nn.functional
import typing as _typing
import dgl
from dgl.nn.pytorch.conv import SAGEConv
from .. import base_encoder, encoder_registry
from ... import _utils


class _SAGE(torch.nn.Module):
    def __init__(
            self, input_dimension: int,
            dimensions: _typing.Sequence[int],
            act: _typing.Optional[str],
            dropout: _typing.Optional[float],
            agg: str
    ):
        super(_SAGE, self).__init__()
        if agg not in ("gcn", "pool", "mean", "lstm"):
            raise ValueError("Unsupported aggregator type")
        self.__convolution_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer, _dimension in enumerate(dimensions):
            self.__convolution_layers.append(
                SAGEConv(
                    input_dimension if layer == 0 else dimensions[layer - 1],
                    _dimension, agg
                )
            )
        self._act: _typing.Optional[str] = act
        self._dropout: _typing.Optional[float] = dropout

    def forward(self, graph: dgl.DGLGraph, *args, **kwargs):
        x: torch.Tensor = graph.ndata['feat']
        x = torch.nn.functional.dropout(x, self._dropout, self.training)
        results: _typing.MutableSequence[torch.Tensor] = [x]
        for _layer in range(len(self.__convolution_layers)):
            x = self.__convolution_layers[_layer](graph, x)
            if _layer < len(self.__convolution_layers) - 1:
                x = _utils.activation.activation_func(x, self._act)
                x = torch.nn.functional.dropout(x, self._dropout, self.training)
            results.append(x)
        return results


@encoder_registry.EncoderUniversalRegistry.register_encoder('sage')
@encoder_registry.EncoderUniversalRegistry.register_encoder('sage_encoder')
class SAGEEncoderMaintainer(base_encoder.AutoHomogeneousEncoderMaintainer):
    def __init__(
            self,
            input_dimension: _typing.Optional[int] = ...,
            final_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(SAGEEncoderMaintainer, self).__init__(
            input_dimension, final_dimension, device, *args, **kwargs
        )
        self.hyper_parameter_space = [
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
        self.hyper_parameters = {
            "num_layers": 3,
            "hidden": [64, 32],
            "dropout": 0.5,
            "act": "relu",
            "agg": "mean",
        }

    def _initialize(self) -> _typing.Optional[bool]:
        dimensions = list(self.hyper_parameters["hidden"])
        if (
                self.final_dimension not in (Ellipsis, None) and
                isinstance(self.final_dimension, int) and
                self.final_dimension > 0
        ):
            dimensions.append(self.final_dimension)
        self._encoder = _SAGE(
            self.input_dimension, dimensions,
            self.hyper_parameters["act"],
            self.hyper_parameters["dropout"],
            self.hyper_parameters["agg"]
        ).to(self.device)
        return True
