import torch.nn.functional
import typing as _typing
import torch_geometric
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn.glob import (
    global_add_pool, global_max_pool, global_mean_pool
)
from ...encoders import base_encoder
from .. import base_decoder, decoder_registry
from ... import _utils


class _LogSoftmaxDecoder(torch.nn.Module):
    def forward(self, features: _typing.Sequence[torch.Tensor], *__args, **__kwargs) -> torch.Tensor:
        return torch.nn.functional.log_softmax(features[-1], dim=1)


@decoder_registry.DecoderUniversalRegistry.register_decoder('log_softmax')
@decoder_registry.DecoderUniversalRegistry.register_decoder('log_softmax_decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('LogSoftmax'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('LogSoftmax_decoder'.lower())
class LogSoftmaxDecoderMaintainer(base_decoder.BaseDecoderMaintainer):
    def _initialize(self, *args, **kwargs) -> _typing.Optional[bool]:
        self._decoder = _LogSoftmaxDecoder().to(self.device)
        return True


class _SumPoolMLPDecoder(torch.nn.Module):
    def __init__(
            self, _final_dimension: int, hidden_dimension: int, output_dimension: int,
            _act: _typing.Optional[str], _dropout: _typing.Optional[float],
            num_graph_features: _typing.Optional[int]
    ):
        super(_SumPoolMLPDecoder, self).__init__()
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


@decoder_registry.DecoderUniversalRegistry.register_decoder('SumPoolMLP'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('SumPoolMLPDecoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('SumPoolMLP_Decoder'.lower())
class SumPoolMLPDecoderMaintainer(base_decoder.BaseDecoderMaintainer):
    def _initialize(self, encoder: base_encoder.AutoHomogeneousEncoderMaintainer, *args, **kwargs) -> _typing.Optional[bool]:
        if (
                isinstance(getattr(self, "num_graph_features"), int) and
                getattr(self, "num_graph_features") > 0
        ):
            num_graph_features: _typing.Optional[int] = getattr(self, "num_graph_features")
        else:
            num_graph_features: _typing.Optional[int] = None
        self._decoder = _SumPoolMLPDecoder(
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
        super(SumPoolMLPDecoderMaintainer, self).__init__(
            output_dimension, device, *args, **kwargs
        )
        self.num_graph_features = kwargs.get("num_graph_features", 0)
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


class _DiffPoolDecoder(torch.nn.Module):
    def __init__(
            self, input_dimension: int, output_dimension: int,
            _ratio: _typing.Union[float, int], _dropout: _typing.Optional[float],
            _act: _typing.Optional[str], num_graph_features: _typing.Optional[int]
    ):
        super(_DiffPoolDecoder, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.ratio: _typing.Union[float, int] = _ratio
        self._act: _typing.Optional[str] = _act
        self.dropout: _typing.Optional[float] = _dropout
        self.num_graph_features: _typing.Optional[int] = num_graph_features

        self.conv1 = GraphConv(self.input_dimension, 128)
        self.pool1 = TopKPooling(128, ratio=self.ratio)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=self.ratio)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=self.ratio)

        if (
                isinstance(self.num_graph_features, int)
                and self.num_graph_features > 0
        ):
            self.lin1 = torch.nn.Linear(256 + self.num_graph_features, 128)
        else:
            self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, self.output_dimension)

    def forward(
            self, features: _typing.Sequence[torch.Tensor],
            data: torch_geometric.data.Data, *__args, **__kwargs
    ):
        x: torch.Tensor = features[-1]
        edge_index: torch.LongTensor = data.edge_index
        batch = data.batch
        if (
                self.num_graph_features is not None and
                isinstance(self.num_graph_features, int) and
                self.num_graph_features > 0
        ):
            if not (
                    hasattr(data, 'gf') and
                    isinstance(data.gf, torch.Tensor) and
                    data.gf.size() == (x.size(0), self.num_graph_features)
            ):
                raise ValueError(
                    f"The provided data is expected to contain property 'gf' "
                    f"with {self.num_graph_features} dimensions as graph feature"
                )

        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = torch.nn.functional.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = x1 + x2 + x3
        if (
                isinstance(self.num_graph_features, int)
                and self.num_graph_features > 0
        ):
            x = torch.cat([x, data.gf], dim=-1)
        x = self.lin1(x)
        x = _utils.activation.activation_func(x, self._act)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = _utils.activation.activation_func(x, self._act)
        x = torch.nn.functional.log_softmax(self.lin3(x), dim=-1)
        return x


@decoder_registry.DecoderUniversalRegistry.register_decoder('DiffPool'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('DiffPoolDecoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('DiffPool_decoder'.lower())
class DiffPoolDecoderMaintainer(base_decoder.BaseDecoderMaintainer):
    def _initialize(
            self, encoder: base_encoder.AutoHomogeneousEncoderMaintainer, *args, **kwargs
    ) -> _typing.Optional[bool]:
        if (
                isinstance(getattr(self, "num_graph_features"), int) and
                getattr(self, "num_graph_features") > 0
        ):
            num_graph_features: _typing.Optional[int] = getattr(self, "num_graph_features")
        else:
            num_graph_features: _typing.Optional[int] = None
        self._decoder = _DiffPoolDecoder(
            list(encoder.get_output_dimensions())[-1],
            self.output_dimension,
            self.hyper_parameters['ratio'], self.hyper_parameters['dropout'],
            self.hyper_parameters['act'], num_graph_features
        ).to(self.device)
        return True

    def __init__(
            self, output_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(DiffPoolDecoderMaintainer, self).__init__(
            output_dimension, device, *args, **kwargs
        )
        self.num_graph_features = kwargs.get("num_graph_features", 0)
        self.hyper_parameter_space = [
            {
                "parameterName": "ratio",
                "type": "DOUBLE",
                "maxValue": 0.9,
                "minValue": 0.1,
                "scalingType": "LINEAR",
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
        ]
        self.hyper_parameters = {
            "ratio": 0.8,
            "dropout": 0.5,
            "act": "relu"
        }

class _DotProductLinkPredictonDecoder(torch.nn.Module):
    def forward(self,
        features: _typing.Sequence[torch.Tensor],
        graph: torch_geometric.data.Data,
        pos_edge: torch.Tensor,
        neg_edge: torch.Tensor,
        **__kwargs
    ):
        z = features[-1]
        edge_index = torch.cat([pos_edge, neg_edge], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

@decoder_registry.DecoderUniversalRegistry.register_decoder('lpdecoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('dotproduct'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('lp-decoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('dot-product'.lower())
class DotProductLinkPredictionDecoderMaintainer(base_decoder.BaseDecoderMaintainer):
    def _initialize(self, *args, **kwargs):
        self._decoder = _DotProductLinkPredictonDecoder()
