import logging
import torch.nn.functional
import typing as _typing
import torch_geometric
from torch_geometric.nn.conv import GATConv
from .. import base_encoder, encoder_registry
from ... import _utils


class GATUtils:
    @classmethod
    def to_total_hidden_dimensions(
            cls, per_head_output_dimensions: _typing.Sequence[int],
            num_hidden_heads: int, num_output_heads: int, concat_last: bool = False,
    ) -> _typing.Sequence[int]:
        return [
            d * (num_hidden_heads if layer < (len(per_head_output_dimensions) - 1) else (num_output_heads if concat_last else 1))
            for layer, d in enumerate(per_head_output_dimensions)
        ]


class _GAT(torch.nn.Module):
    def __init__(
            self, input_dimension: int,
            per_head_output_dimensions: _typing.Sequence[int],
            num_hidden_heads: int, num_output_heads: int,
            _dropout: float, _act: _typing.Optional[str],
            concat_last: bool = True
    ):
        super(_GAT, self).__init__()
        self._dropout: float = _dropout
        self._act: _typing.Optional[str] = _act
        total_output_dimensions: _typing.Sequence[int] = (
            GATUtils.to_total_hidden_dimensions(
                per_head_output_dimensions, num_hidden_heads, num_output_heads, concat_last = concat_last
            )
        )
        num_layers = len(per_head_output_dimensions)
        self.__convolution_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer in range(len(per_head_output_dimensions)):
            self.__convolution_layers.append(
                GATConv(
                    input_dimension if layer == 0 else total_output_dimensions[layer - 1],
                    per_head_output_dimensions[layer],
                    num_hidden_heads if layer < num_layers - 1 else num_output_heads,
                    dropout=_dropout,
                    concat=True if layer < num_layers - 1 or concat_last else False
                )
            )

    def forward(self, data: torch_geometric.data.Data, *__args, **__kwargs):
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
        for layer, _gat in enumerate(self.__convolution_layers):
            x: torch.Tensor = torch.nn.functional.dropout(
                x, self._dropout, self.training
            )
            x: torch.Tensor = _gat(x, edge_index, edge_weight)
            if layer < len(self.__convolution_layers) - 1:
                x: torch.Tensor = _utils.activation.activation_func(x, self._act)
            results.append(x)
        logging.debug("{:d} layer, each layer shape {:s}".format(len(results), " ".join([str(x.shape) for x in results])))
        return results


@encoder_registry.EncoderUniversalRegistry.register_encoder('gat')
@encoder_registry.EncoderUniversalRegistry.register_encoder('gat_encoder')
class GATEncoderMaintainer(base_encoder.AutoHomogeneousEncoderMaintainer):
    def _initialize(self) -> _typing.Optional[bool]:
        dimensions = list(self.hyper_parameters['hidden'])
        concat_last = True
        if (
                self.final_dimension not in (Ellipsis, None)
                and isinstance(self.final_dimension, int)
                and self.final_dimension > 0
        ):
            dimensions.append(self.final_dimension)
            concat_last = False
        logging.debug("current dimensions %s", dimensions)
        self._encoder = _GAT(
            self.input_dimension,
            dimensions,
            self.hyper_parameters.get('num_hidden_heads'),
            self.hyper_parameters.get('num_output_heads'),
            self.hyper_parameters['dropout'],
            self.hyper_parameters['act'],
            concat_last=concat_last
        )
        return True

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
                self.hyper_parameters.get("num_output_heads", 1),
                concat_last = concat_last
            )
        )
        return output_dimensions

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
                "parameterName": "num_hidden_heads",
                "type": "DISCRETE",
                "feasiblePoints": "2,4,8,16",
            },
            {
                "parameterName": "num_output_heads",
                "type": "DISCRETE",
                "feasiblePoints": "2,4,8,16",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
        ]
        self.hyper_parameters = {
            "num_layers": 2,
            "hidden": [32],
            "num_hidden_heads": 4,
            "num_output_heads": 4,
            "dropout": 0.2,
            "act": "leaky_relu",
        }
