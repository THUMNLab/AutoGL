import torch.nn.functional
import typing as _typing
import dgl
from dgl.nn.pytorch.conv import GraphConv
from .. import base_encoder, encoder_registry
from ... import _utils


class _GCN(torch.nn.Module):
    def __init__(
            self, input_dimension: int,
            dimensions: _typing.Sequence[int],
            act: _typing.Optional[str],
            dropout: _typing.Optional[float]
    ):
        super(_GCN, self).__init__()
        self.__convolution_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer, _dimension in enumerate(dimensions):
            self.__convolution_layers.append(
                GraphConv(
                    input_dimension if layer == 0 else dimensions[layer - 1],
                    _dimension
                )
            )
        self._act: _typing.Optional[str] = act
        self._dropout: _typing.Optional[float] = dropout

    def forward(self, graph: dgl.DGLGraph, *__args, **__kwargs):
        x: torch.Tensor = graph.ndata['feat']
        results: _typing.MutableSequence[torch.Tensor] = [x]
        for _layer in range(len(self.__convolution_layers)):
            x = self.__convolution_layers[_layer](graph, x)
            if _layer < len(self.__convolution_layers) - 1:
                x = _utils.activation.activation_func(x, self._act)
                x = torch.nn.functional.dropout(x, self._dropout, self.training)
            results.append(x)
        return results


@encoder_registry.EncoderUniversalRegistry.register_encoder('gcn')
@encoder_registry.EncoderUniversalRegistry.register_encoder('gcn_encoder')
class GCNEncoderMaintainer(base_encoder.AutoHomogeneousEncoderMaintainer):
    def __init__(
            self,
            input_dimension: _typing.Optional[int] = ...,
            final_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(GCNEncoderMaintainer, self).__init__(
            input_dimension, final_dimension, device, *args, **kwargs
        )
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
        self.hyper_parameters = {
            "num_layers": 3,
            "hidden": [128, 64],
            "dropout": 0.,
            "act": "relu",
        }

    def _initialize(self) -> _typing.Optional[bool]:
        dimensions = list(self.hyper_parameters["hidden"])
        if (
                self.final_dimension not in (Ellipsis, None) and
                isinstance(self.final_dimension, int) and
                self.final_dimension > 0
        ):
            dimensions.append(self.final_dimension)
        self._encoder = _GCN(
            self.input_dimension, dimensions,
            self.hyper_parameters["act"],
            self.hyper_parameters["dropout"]
        ).to(self.device)
        return True