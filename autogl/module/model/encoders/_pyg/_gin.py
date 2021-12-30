import typing as _typing
import torch.nn.functional
import torch_geometric
from torch_geometric.nn.conv import GINConv
from .. import base_encoder, encoder_registry
from ... import _utils


class _GIN(torch.nn.Module):
    def __init__(
            self, input_dimension: int,
            dimensions: _typing.Sequence[int],
            _act: str, _dropout: float,
            mlp_layers: int, _eps: str
    ):
        super(_GIN, self).__init__()

        self._act: str = _act

        def _get_act() -> torch.nn.Module:
            if _act == 'leaky_relu':
                return torch.nn.LeakyReLU()
            elif _act == 'relu':
                return torch.nn.ReLU()
            elif _act == 'elu':
                return torch.nn.ELU()
            elif _act == 'tanh':
                return torch.nn.Tanh()
            elif _act == 'PReLU'.lower():
                return torch.nn.PReLU()
            else:
                return torch.nn.ReLU()

        convolution_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        batch_normalizations: torch.nn.ModuleList = torch.nn.ModuleList()

        __mlp_layers = [torch.nn.Linear(input_dimension, dimensions[0])]
        for _ in range(mlp_layers - 1):
            __mlp_layers.append(_get_act())
            __mlp_layers.append(torch.nn.Linear(dimensions[0], dimensions[0]))
        convolution_layers.append(
            GINConv(torch.nn.Sequential(*__mlp_layers), train_eps=_eps == "True")
        )
        batch_normalizations.append(torch.nn.BatchNorm1d(dimensions[0]))

        num_layers: int = len(dimensions)
        for layer in range(num_layers - 1):
            __mlp_layers = [torch.nn.Linear(dimensions[layer], dimensions[layer + 1])]
            for _ in range(mlp_layers - 1):
                __mlp_layers.append(_get_act())
                __mlp_layers.append(
                    torch.nn.Linear(dimensions[layer + 1], dimensions[layer + 1])
                )
            convolution_layers.append(
                GINConv(torch.nn.Sequential(*__mlp_layers), train_eps=_eps == "True")
            )
            batch_normalizations.append(
                torch.nn.BatchNorm1d(dimensions[layer + 1])
            )

        self.__convolution_layers: torch.nn.ModuleList = convolution_layers
        self.__batch_normalizations: torch.nn.ModuleList = batch_normalizations

    def forward(
            self, data: torch_geometric.data.Data, *__args, **__kwargs
    ) -> _typing.Sequence[torch.Tensor]:
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index

        results: _typing.MutableSequence[torch.Tensor] = [x]
        num_layers = len(self.__convolution_layers)
        for layer in range(num_layers):
            x: torch.Tensor = self.__convolution_layers[layer](x, edge_index)
            x: torch.Tensor = _utils.activation.activation_func(x, self._act)
            x: torch.Tensor = self.__batch_normalizations[layer](x)
            results.append(x)
        return results


@encoder_registry.EncoderUniversalRegistry.register_encoder('gin')
@encoder_registry.EncoderUniversalRegistry.register_encoder('gin-encoder')
class GINEncoderMaintainer(base_encoder.AutoHomogeneousEncoderMaintainer):
    def _initialize(self) -> _typing.Optional[bool]:
        dimensions = list(self.hyper_parameters['hidden'])
        if (
                self.final_dimension not in (Ellipsis, None)
                and isinstance(self.final_dimension, int)
                and self.final_dimension > 0
        ):
            dimensions.append(self.final_dimension)
        self._encoder = _GIN(
            self.input_dimension, dimensions,
            self.hyper_parameters['act'],
            self.hyper_parameters['dropout'],
            self.hyper_parameters['mlp_layers'],
            self.hyper_parameters['eps']
        ).to(self.device)
        return True

    def __init__(
            self,
            input_dimension: _typing.Optional[int] = ...,
            final_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(GINEncoderMaintainer, self).__init__(
            input_dimension, final_dimension, device, *args, **kwargs
        )
        self.hyper_parameter_space = [
            {
                "parameterName": "num_layers",
                "type": "DISCRETE",
                "feasiblePoints": "4,5,6",
            },
            {
                "parameterName": "hidden",
                "type": "NUMERICAL_LIST",
                "numericalType": "INTEGER",
                "length": 5,
                "minValue": [8, 8, 8, 8, 8],
                "maxValue": [64, 64, 64, 64, 64],
                "scalingType": "LOG",
                "cutPara": ("num_layers",),
                "cutFunc": lambda x: x[0] - 1,
            },
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.9,
                "minValue": 0.1,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
            {
                "parameterName": "eps",
                "type": "CATEGORICAL",
                "feasiblePoints": ["True", "False"],
            },
            {
                "parameterName": "mlp_layers",
                "type": "DISCRETE",
                "feasiblePoints": "2,3,4",
            },
        ]
        self.hyper_parameters = {
            "num_layers": 3,
            "hidden": [64, 32],
            "dropout": 0.5,
            "act": "relu",
            "eps": "True",
            "mlp_layers": 2,
        }
