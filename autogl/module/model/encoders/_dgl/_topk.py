import dgl
import torch.nn.functional
import typing as _typing
from dgl.nn.pytorch.conv import GraphConv
from .. import base_encoder, encoder_registry


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
    """ MLP with linear output """
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


class _TopK(torch.nn.Module):
    def __init__(self, input_dimension: int, dimensions: _typing.Sequence[int]):
        super(_TopK, self).__init__()
        self.__gcn_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.__batch_normalizations: torch.nn.ModuleList = torch.nn.ModuleList()
        self.__num_layers = len(dimensions)
        for layer in range(self.__num_layers):
            self.__gcn_layers.append(
                GraphConv(
                    input_dimension if layer == 0 else dimensions[layer - 1],
                    dimensions[layer]
                )
            )
            self.__batch_normalizations.append(
                torch.nn.BatchNorm1d(dimensions[layer])
            )

    def forward(self, graph: dgl.DGLGraph, *__args, **__kwargs) -> _typing.Sequence[torch.Tensor]:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        h = graph.ndata['feat']
        hidden_rep = [h]
        for i in range(self.__num_layers):
            h = self.__gcn_layers[i](graph, h)
            h = self.__batch_normalizations[i](h)
            h = torch.nn.functional.relu(h)
            hidden_rep.append(h)
        return hidden_rep


@encoder_registry.EncoderUniversalRegistry.register_encoder('TopK'.lower())
@encoder_registry.EncoderUniversalRegistry.register_encoder('TopK_encoder'.lower())
class AutoTopKMaintainer(base_encoder.AutoHomogeneousEncoderMaintainer):
    def __init__(
            self,
            input_dimension: _typing.Optional[int] = ...,
            final_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(AutoTopKMaintainer, self).__init__(
            input_dimension, final_dimension,
            device, *args, **kwargs
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
            }
        ]
        self.hyper_parameters = {
            "num_layers": 5,
            "hidden": [64, 64, 64, 64]
        }

    def _initialize(self) -> _typing.Optional[bool]:
        dimensions = list(self.hyper_parameters["hidden"])
        if (
                self.final_dimension not in (Ellipsis, None) and
                isinstance(self.final_dimension, int) and
                self.final_dimension > 0
        ):
            dimensions.append(self.final_dimension)
        self._encoder = _TopK(
            self.input_dimension, dimensions
        ).to(self.device)
        return True
