import dgl
import torch.nn.functional
import typing as _typing
from dgl.nn.pytorch.conv import GINConv
from .. import base_encoder, encoder_registry
from ... import _utils


class ApplyNodeFunc(torch.nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = torch.nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = torch.nn.functional.relu(h)
        return h


class MLP(torch.nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = torch.nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = torch.nn.functional.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class _GIN(torch.nn.Module):
    def __init__(
            self, input_dimension: int,
            dimensions: _typing.Sequence[int],
            num_mlp_layers: int,
            act: _typing.Optional[str],
            _eps: str, neighbor_pooling_type: str
    ):
        super(_GIN, self).__init__()
        self.__num_layers: int = len(dimensions)

        self._act: _typing.Optional[str] = act

        self.__gin_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.__batch_normalizations: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer in range(self.__num_layers):
            mlp = MLP(
                num_mlp_layers,
                input_dimension if layer == 0 else dimensions[layer - 1],
                dimensions[layer], dimensions[layer]
            )
            self.__gin_layers.append(
                GINConv(
                    ApplyNodeFunc(mlp), neighbor_pooling_type, 0,
                    _eps.lower() == "true"
                )
            )
            self.__batch_normalizations.append(
                torch.nn.BatchNorm1d(dimensions[layer])
            )

    def forward(self, graph: dgl.DGLGraph, *__args, **__kwargs) -> _typing.Sequence[torch.Tensor]:
        x: torch.Tensor = graph.ndata['feat']

        features: _typing.MutableSequence[torch.Tensor] = [x]
        for layer in range(self.__num_layers):
            x: torch.Tensor = self.__gin_layers[layer](graph, x)
            x: torch.Tensor = self.__batch_normalizations[layer](x)
            x: torch.Tensor = _utils.activation.activation_func(x, self._act)
            features.append(x)

        return features


@encoder_registry.EncoderUniversalRegistry.register_encoder('gin')
@encoder_registry.EncoderUniversalRegistry.register_encoder('gin_encoder')
class GINEncoderMaintainer(base_encoder.AutoHomogeneousEncoderMaintainer):
    def __init__(
            self,
            input_dimension: _typing.Optional[int] = ...,
            final_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(GINEncoderMaintainer, self).__init__(
            input_dimension, final_dimension,
            device, *args, **kwargs
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
                "maxValue": [64, 64, 64],
                "scalingType": "LOG",
                "cutPara": ("num_layers",),
                "cutFunc": lambda x: x[0] - 1,
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
            {
                "parameterName": "neighbor_pooling_type",
                "type": "CATEGORICAL",
                "feasiblePoints": ["sum", "mean", "max"],
            }
        ]
        self.hyper_parameters = {
            "num_layers": 3,
            "hidden": [64, 64],
            "act": "relu",
            "eps": "False",
            "mlp_layers": 2,
            "neighbor_pooling_type": "sum"
        }

    def _initialize(self) -> _typing.Optional[bool]:
        dimensions = list(self.hyper_parameters['hidden'])
        if (
                self.final_dimension not in (Ellipsis, None) and
                isinstance(self.final_dimension, int) and
                self.final_dimension > 0
        ):
            dimensions.append(self.final_dimension)
        self._encoder = _GIN(
            self.input_dimension, dimensions, self.hyper_parameters["mlp_layers"],
            self.hyper_parameters["act"], self.hyper_parameters["eps"],
            self.hyper_parameters["neighbor_pooling_type"]
        ).to(self.device)
        return True
