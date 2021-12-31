import torch.nn.functional
import typing as _typing
import torch_geometric
from torch_geometric.nn.conv import GCNConv
from .. import base_encoder, encoder_registry
from ... import _utils


class _GCN(torch.nn.Module):
    def __init__(
            self, input_dimension: int, dimensions: _typing.Sequence[int],
            _act: _typing.Optional[str], _dropout: _typing.Optional[float]
    ):
        super(_GCN, self).__init__()
        self._act: _typing.Optional[str] = _act
        self._dropout: _typing.Optional[float] = _dropout
        self.__convolution_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer, output_dimension in enumerate(dimensions):
            self.__convolution_layers.append(
                GCNConv(input_dimension if layer == 0 else dimensions[layer - 1], output_dimension)
            )

    def forward(
            self, data: torch_geometric.data.Data, *__args, **__kwargs
    ) -> _typing.Sequence[torch.Tensor]:
        x: torch.Tensor = data.x
        edge_index: torch.LongTensor = data.edge_index
        if (
                hasattr(data, "edge_weight") and
                isinstance(getattr(data, "edge_weight"), torch.Tensor)
                and torch.is_tensor(data.edge_weight)
        ):
            edge_weight: _typing.Optional[torch.Tensor] = data.edge_weight
        else:
            edge_weight: _typing.Optional[torch.Tensor] = None
        results: _typing.MutableSequence[torch.Tensor] = [x]
        for layer, convolution_layer in enumerate(self.__convolution_layers):
            x = convolution_layer(x, edge_index, edge_weight)
            if layer < len(self.__convolution_layers) - 1:
                x: torch.Tensor = _utils.activation.activation_func(x, self._act)
                if isinstance(self._dropout, float) and 0 <= self._dropout <= 1:
                    x = torch.nn.functional.dropout(x, self._dropout, self.training)
            results.append(x)
        return results


@encoder_registry.EncoderUniversalRegistry.register_encoder('gcn')
@encoder_registry.EncoderUniversalRegistry.register_encoder('gcn_encoder')
class GCNEncoderMaintainer(base_encoder.AutoHomogeneousEncoderMaintainer):
    def _initialize(self) -> _typing.Optional[bool]:
        dimensions = list(self.hyper_parameters['hidden'])
        if (
                self.final_dimension not in (Ellipsis, None)
                and isinstance(self.final_dimension, int)
                and self.final_dimension > 0
        ):
            dimensions.append(self.final_dimension)
        self._encoder = _GCN(
            self.input_dimension, dimensions,
            self.hyper_parameters['act'], self.hyper_parameters['dropout']
        )
        return True

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
            "num_layers": 2,
            "hidden": [16],
            "dropout": 0.2,
            "act": "leaky_relu",
        }
