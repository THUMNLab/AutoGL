import logging
import torch.nn.functional
import typing as _typing
import dgl
from dgl.nn.pytorch.conv import GATConv
from .. import base_encoder, encoder_registry
from ... import _utils


class GATUtils:
    @classmethod
    def to_total_hidden_dimensions(
            cls, per_head_dimensions: _typing.Sequence[int],
            num_hidden_heads: int, num_output_heads: int, concat_last: bool = False,
    ):
        return [
            d * (num_hidden_heads if layer < (len(per_head_dimensions) - 1) else (num_output_heads if concat_last else 1))
            for layer, d in enumerate(per_head_dimensions)
        ]


class GAT(torch.nn.Module):
    def __init__(
            self, input_dimension: int, dimensions: _typing.Sequence[int],
            num_hidden_heads: int, num_output_heads: int,
            act: _typing.Optional[str], dropout: _typing.Optional[float], concat_last: bool = True
    ):
        super(GAT, self).__init__()
        dimensions = GATUtils.to_total_hidden_dimensions(
            dimensions, num_hidden_heads, num_output_heads, concat_last=concat_last
        )
        num_layers: int = len(dimensions)
        self.__convolutions: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer in range(len(dimensions)):
            num_heads: int = (
                num_hidden_heads
                if layer < num_layers - 1
                else num_output_heads
            )
            if dimensions[layer] % num_heads != 0:
                raise ValueError
            self.__convolutions.append(
                GATConv(
                    dimensions[layer - 1] if layer > 0 else input_dimension,
                    dimensions[layer] // num_heads, num_heads,
                    dropout, dropout
                )
            )
        self.__activation: _typing.Optional[str] = act
        self.__concat_last = concat_last

    def forward(
            self, graph: dgl.DGLGraph, *__args, **__kwargs
    ) -> _typing.Iterable[torch.Tensor]:
        num_layers = len(self.__convolutions)
        x: torch.Tensor = graph.ndata['feat']
        results = [x]
        for layer in range(num_layers):
            if layer < num_layers - 1 or self.__concat_last:
                x = self.__convolutions[layer](graph, x).flatten(1)
            else:
                x = self.__convolutions[layer](graph, x).mean(1)
            if layer < num_layers - 1:
                x = _utils.activation.activation_func(x, self.__activation)
            results.append(x)
        logging.debug("{:d} layer, each layer shape {:s}".format(len(results), " ".join([str(x.shape) for x in results])))
        return results


@encoder_registry.EncoderUniversalRegistry.register_encoder('gat')
@encoder_registry.EncoderUniversalRegistry.register_encoder('gat_encoder')
class GATEncoderMaintainer(base_encoder.AutoHomogeneousEncoderMaintainer):
    r"""
    AutoGAT. The model used in this automodel is GAT, i.e., the graph attentional network from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper. The layer is

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j}
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    
    Parameters
    ----------
    input_dimension: `Optional[int]`
        The dimension of input features.
    final_dimension: `Optional[int]`
        The dimension of final features.
    device: `torch.device` or `str` or `int`
        The device where model will be running on.
    kwargs:
        Other parameters.
    """

    def __init__(
            self,
            input_dimension: _typing.Optional[int] = ...,
            final_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(GATEncoderMaintainer, self).__init__(
            input_dimension, final_dimension, device, *args, **kwargs
        )
        self.hyper_parameters: _typing.Mapping[str, _typing.Any] = {
            "num_layers": 2,
            "hidden": [32],
            "heads": 4,
            "dropout": 0.2,
            "act": "leaky_relu",
        }
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
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.8,
                "minValue": 0.2,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "heads",
                "type": "DISCRETE",
                "feasiblePoints": "2,4,8,16",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
        ]

    def get_output_dimensions(self) -> _typing.Iterable[int]:
        temp = list(self.hyper_parameters["hidden"])
        concat_last = True
        if (
                self.final_dimension not in (Ellipsis, None) and
                isinstance(self.final_dimension, int) and
                self.final_dimension > 0
        ):
            temp.append(self.final_dimension)
            concat_last = False
        output_dimensions = [self.input_dimension]
        output_dimensions.extend(
            GATUtils.to_total_hidden_dimensions(
                temp,
                self.hyper_parameters.get("num_hidden_heads", self.hyper_parameters["heads"]),
                self.hyper_parameters.get("num_output_heads", 1), concat_last=concat_last
            )
        )
        return output_dimensions

    def _initialize(self) -> _typing.Optional[bool]:
        dimensions = list(self.hyper_parameters["hidden"])
        concat_last = True
        if (
                self.final_dimension not in (Ellipsis, None) and
                isinstance(self.final_dimension, int) and
                self.final_dimension > 0
        ):
            dimensions.append(self.final_dimension)
            concat_last = False
        logging.debug("current dimensions %s", dimensions)
        self._encoder: torch.nn.Module = GAT(
            self.input_dimension, dimensions,
            self.hyper_parameters.get("num_hidden_heads", self.hyper_parameters["heads"]),
            self.hyper_parameters.get("num_output_heads", 1),
            self.hyper_parameters.get("act"),
            self.hyper_parameters.get("dropout"),
            concat_last
        ).to(self.device)
        return True
