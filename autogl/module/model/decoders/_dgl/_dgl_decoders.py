import torch.nn.functional
import typing as _typing
import dgl
from dgl.nn.pytorch.glob import (
    SumPooling, AvgPooling, MaxPooling, SortPooling
)
from .. import base_decoder, decoder_registry
from ...encoders import base_encoder
from ... import _utils


class _LogSoftmaxDecoder(torch.nn.Module):
    def forward(
            self, features: _typing.Sequence[torch.Tensor],
            _graph: dgl.DGLGraph, *__args, **__kwargs
    ):
        return torch.nn.functional.log_softmax(features[-1], dim=-1)


@decoder_registry.DecoderUniversalRegistry.register_decoder('log_softmax')
@decoder_registry.DecoderUniversalRegistry.register_decoder('log-softmax-decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('log-softmax_decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('log_softmax-decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('log_softmax_decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('LogSoftmax'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('LogSoftmax-decoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('LogSoftmax_decoder'.lower())
class LogSoftmaxDecoderMaintainer(base_decoder.BaseDecoderMaintainer):
    def _initialize(self, encoder, *args, **kwargs) -> _typing.Optional[bool]:
        self._decoder = _LogSoftmaxDecoder().to(self.device)
        return True


class _JKSumPoolDecoder(torch.nn.Module):
    def __init__(
            self, input_dimensions: _typing.Sequence[int],
            output_dimension: int, dropout: float,
            graph_pooling_type: str
    ):
        super(_JKSumPoolDecoder, self).__init__()
        self._linear_transforms: torch.nn.ModuleList = torch.nn.ModuleList()
        for input_dimension in input_dimensions:
            self._linear_transforms.append(
                torch.nn.Linear(input_dimension, output_dimension)
            )
        self._dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)
        if not isinstance(graph_pooling_type, str):
            raise TypeError
        elif graph_pooling_type.lower() == 'sum':
            self.__pool = SumPooling()
        elif graph_pooling_type.lower() == 'mean':
            self.__pool = AvgPooling()
        elif graph_pooling_type.lower() == 'max':
            self.__pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(
            self, features: _typing.Sequence[torch.Tensor],
            graph: dgl.DGLGraph, *__args, **__kwargs
    ):
        if len(features) != len(self._linear_transforms):
            raise ValueError
        score_over_layer = 0
        for i, feature in enumerate(features):
            score_over_layer += self._dropout(self._linear_transforms[i](self.__pool(graph, feature)))
        return score_over_layer


@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLP')
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLPDecoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLP-decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLP-Decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLP_decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLP_Decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLP'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLPDecoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLP-Decoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('JKSumPoolMLP_Decoder'.lower())
class JKSumPoolDecoderMaintainer(base_decoder.BaseDecoderMaintainer):
    def _initialize(self, encoder: base_encoder.AutoHomogeneousEncoderMaintainer, *args, **kwargs) -> _typing.Optional[bool]:
        self._decoder = _JKSumPoolDecoder(
            list(encoder.get_output_dimensions()), self.output_dimension,
            self.hyper_parameters["dropout"],
            self.hyper_parameters["graph_pooling_type"]
        ).to(self.device)
        return True

    def __init__(
            self, output_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(JKSumPoolDecoderMaintainer, self).__init__(
            output_dimension, device, *args, **kwargs
        )
        self.hyper_parameter_space = (
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.9,
                "minValue": 0.1,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "graph_pooling_type",
                "type": "CATEGORICAL",
                "feasiblePoints": ["sum", "mean", "max"],
            }
        )
        self.hyper_parameters = {
            "dropout": 0.5,
            "graph_pooling_type": "sum"
        }

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
        self._sumpool = SumPooling()
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
            data: dgl.DGLGraph, *__args, **__kwargs
    ):
        feature = features[-1]
        feature = self._sumpool(data, feature)
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


class _TopKPoolDecoder(torch.nn.Module):
    def __init__(
            self, input_dimensions: _typing.Iterable[int],
            output_dimension: int, dropout: float
    ):
        super(_TopKPoolDecoder, self).__init__()
        k: int = min(len(list(input_dimensions)), 3)
        self.__pool: SortPooling = SortPooling(k)
        self.__linear_predictions: torch.nn.ModuleList = (
            torch.nn.ModuleList()
        )
        for layer, dimension in enumerate(input_dimensions):
            self.__linear_predictions.append(
                torch.nn.Linear(dimension * k, output_dimension)
            )
        self._dropout = torch.nn.Dropout(dropout)

    def forward(
            self, features: _typing.Sequence[torch.Tensor],
            graph: dgl.DGLGraph, *__args, **__kwargs
    ):
        cumulative_result = 0
        for i, h in enumerate(features):
            cumulative_result += self._dropout(self.__linear_predictions[i](self.__pool(graph, h)))
        return cumulative_result


@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK')
@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK-decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK-Decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK_decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK_Decoder')
@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK-decoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('TopK_decoder'.lower())
class TopKDecoderMaintainer(base_decoder.BaseDecoderMaintainer):
    def _initialize(
            self, encoder: base_encoder.AutoHomogeneousEncoderMaintainer, *args, **kwargs
    ) -> _typing.Optional[bool]:
        self._decoder = _TopKPoolDecoder(
            encoder.get_output_dimensions(),
            self.output_dimension,
            self.hyper_parameters["dropout"]
        ).to(self.device)
        return True

    def __init__(
            self, output_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(TopKDecoderMaintainer, self).__init__(
            output_dimension, device, *args, **kwargs
        )
        self.hyper_parameter_space = [
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.9,
                "minValue": 0.1,
                "scalingType": "LINEAR",
            }
        ]
        self.hyper_parameters = {
            "dropout": 0.5
        }


class _DotProductLinkPredictionDecoder(torch.nn.Module):
    def forward(
            self,
            features: _typing.Sequence[torch.Tensor],
            graph: dgl.DGLGraph,
            pos_edge: torch.Tensor,
            neg_edge: torch.Tensor,
            **__kwargs
    ):
        z = features[-1]
        edge_index = torch.cat([pos_edge, neg_edge], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


@decoder_registry.DecoderUniversalRegistry.register_decoder('LPDecoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('DOTProduct'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('DOTProductDecoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('lp-decoder'.lower())
@decoder_registry.DecoderUniversalRegistry.register_decoder('dot-product'.lower())
class DotProductLinkPredictionDecoderMaintainer(base_decoder.BaseDecoderMaintainer):
    def _initialize(self, *args, **kwargs):
        self._decoder = _DotProductLinkPredictionDecoder()
