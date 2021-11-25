import dgl
from dgl.nn.pytorch.glob import (
    AvgPooling, MaxPooling, SortPooling, SumPooling
)
import torch.nn.functional
import typing as _typing
from . import _decoder
from ._decoder_registry import RepresentationDecoderUniversalRegistry


def activate_func(x, func):
    if func == "tanh":
        return torch.tanh(x)
    elif hasattr(torch.nn.functional, func):
        return getattr(torch.nn.functional, func)(x)
    elif func == "":
        pass
    else:
        raise TypeError("PyTorch does not support activation function {}".format(func))
    return x


@RepresentationDecoderUniversalRegistry.register_representation_decoder("log_softmax")
class LogSoftmaxDecoder(_decoder.RepresentationDecoder):
    def __call__(
            self, graph: dgl.DGLGraph,
            features: _typing.Union[torch.Tensor, _typing.Sequence[torch.Tensor]],
            *args, **kwargs
    ) -> torch.Tensor:
        if isinstance(features, torch.Tensor) and torch.is_tensor(features):
            return torch.nn.functional.log_softmax(features, dim=-1)
        elif isinstance(features, _typing.Sequence):
            if len(features) == 0:
                raise ValueError
            if not (
                    isinstance(features[0], torch.Tensor)
                    and torch.is_tensor(features[0])
            ):
                raise TypeError
            return torch.nn.functional.log_softmax(features[0], dim=-1)
        else:
            raise TypeError


@RepresentationDecoderUniversalRegistry.register_representation_decoder("gin")
@RepresentationDecoderUniversalRegistry.register_representation_decoder("gin_decoder")
class GINDecoder(_decoder.RepresentationDecoder):
    def __init__(self, hyper_parameters: _typing.Mapping[str, _typing.Any], *args, **kwargs):
        super(GINDecoder, self).__init__(hyper_parameters, *args, **kwargs)
        num_layers: int = hyper_parameters["num_layers"]

        temp: _typing.Tuple[_typing.Optional[int], _typing.Optional[int]] = (
            hyper_parameters.get("num_classes"), hyper_parameters.get("num_class")
        )
        if all([(not isinstance(i, int)) for i in temp]):
            raise ValueError("num_classes or num_class not exists")
        else:
            num_classes: int = temp[0] if isinstance(temp[0], int) else temp[1]
            if not num_classes > 0:
                raise ValueError("num_classes must be positive integer")

        num_graph_features: int = hyper_parameters["num_graph_features"]
        graph_pooling_type: str = hyper_parameters["graph_pooling_type"]
        hidden: _typing.Sequence[int] = hyper_parameters["hidden"]
        self.__activation: str = hyper_parameters["act"]
        self.__dropout: float = hyper_parameters["dropout"]
        self.__fc1: torch.nn.Linear = torch.nn.Linear(
            hidden[num_layers - 3] + num_graph_features,
            hidden[num_layers - 3]
        )
        self.__fc2: torch.nn.Linear = torch.nn.Linear(
            hidden[num_layers - 2], num_classes
        )
        if graph_pooling_type == 'sum':
            self.__pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.__pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.__pool = MaxPooling()
        else:
            raise ValueError

    def __call__(
            self, graph: dgl.DGLGraph,
            features: _typing.Union[torch.Tensor, _typing.Sequence[torch.Tensor]],
            *args, **kwargs
    ) -> torch.Tensor:
        if isinstance(features, torch.Tensor) and torch.is_tensor(features):
            feature: torch.Tensor = features
        elif isinstance(features, _typing.Sequence):
            if len(features) == 0:
                raise ValueError
            if not (
                isinstance(features[0], torch.Tensor)
                and torch.is_tensor(features[0])
            ):
                raise TypeError
            else:
                feature: torch.Tensor = features[0]
        else:
            raise TypeError
        feature: torch.Tensor = self.__fc1(feature)
        feature: torch.Tensor = activate_func(feature, self.__activation)
        feature: torch.Tensor = torch.nn.functional.dropout(
            feature, self.__dropout, training=self.training
        )
        feature: torch.Tensor = self.__fc2(feature)
        feature: torch.Tensor = self.__pool.forward(graph, feature)
        return torch.nn.functional.log_softmax(feature, dim=1)


@RepresentationDecoderUniversalRegistry.register_representation_decoder("TopK".lower())
@RepresentationDecoderUniversalRegistry.register_representation_decoder("TopKPool".lower())
@RepresentationDecoderUniversalRegistry.register_representation_decoder("TopK_decoder".lower())
class TopKPoolDecoder(_decoder.RepresentationDecoder):
    def __init__(self, hyper_parameters: _typing.Mapping[str, _typing.Any], *args, **kwargs):
        super(TopKPoolDecoder, self).__init__(hyper_parameters, *args, **kwargs)

        temp: _typing.Tuple[_typing.Optional[int], _typing.Optional[int]] = (
            hyper_parameters.get("num_features"), hyper_parameters.get("features_num")
        )
        if all([(not isinstance(i, int)) for i in temp]):
            raise ValueError("num_features or features_num not exists")
        else:
            num_features: int = temp[0] if isinstance(temp[0], int) else temp[1]
            if not num_features > 0:
                raise ValueError("Illegal number of features")
        temp: _typing.Tuple[_typing.Optional[int], _typing.Optional[int]] = (
            hyper_parameters.get("num_classes"), hyper_parameters.get("num_class")
        )
        if all([(not isinstance(i, int)) for i in temp]):
            raise ValueError("num_classes or num_class not exists")
        else:
            num_classes: int = temp[0] if isinstance(temp[0], int) else temp[1]
            if not num_classes > 0:
                raise ValueError("num_classes must be positive integer")

        num_layers: int = hyper_parameters["num_layers"]
        hidden_dimension: int = hyper_parameters["hidden"][0]
        dropout: float = hyper_parameters.get("dropout", 0)
        k: int = hyper_parameters.get("k", 3)

        self.__num_layers: int = num_layers
        self.__pool = SortPooling(k)
        self.__linear_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.__linear_layers.append(
                torch.nn.Linear(num_features * k, num_classes) if layer == 0
                else torch.nn.Linear(hidden_dimension * k, num_classes)
            )
        self.__dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)

    def __call__(
            self, graph: dgl.DGLGraph,
            features: _typing.Union[torch.Tensor, _typing.Sequence[torch.Tensor]],
            *args, **kwargs
    ) -> torch.Tensor:
        if not isinstance(features, _typing.Sequence):
            raise TypeError
        if not len(features) == self.__num_layers:
            raise ValueError
        if not all([torch.is_tensor(feature) for feature in features]):
            raise TypeError

        score_over_layer = torch.zeros(1)
        for i, feature in enumerate(features):
            pooled_h = self.__pool(graph, feature)
            score_over_layer += self.__dropout(self.__linear_layers[i](pooled_h))

        return score_over_layer


@RepresentationDecoderUniversalRegistry.register_representation_decoder("lp_decoder")
@RepresentationDecoderUniversalRegistry.register_representation_decoder("LinkPrediction_decoder".lower())
@RepresentationDecoderUniversalRegistry.register_representation_decoder("Link_Prediction_decoder".lower())
class LinkPredictionDecoder(_decoder.RepresentationDecoder):
    def __call__(self, graph: dgl.DGLGraph, features: _typing.Union[torch.Tensor, _typing.Sequence[torch.Tensor]], *args, **kwargs) -> torch.Tensor:
        z: torch.Tensor = graph.ndata['feat']
        if (
                "pos_edge_index" in kwargs and "neg_edge_index" in kwargs and
                isinstance(kwargs["pos_edge_index"], torch.Tensor) and
                isinstance(kwargs["neg_edge_index"], torch.Tensor) and
                torch.is_tensor(kwargs["pos_edge_index"]) and
                torch.is_tensor(kwargs["neg_edge_index"])
        ):
            pos_edge_index: torch.Tensor = kwargs["pos_edge_index"]
            neg_edge_index: torch.Tensor = kwargs["neg_edge_index"]
        else:
            raise ValueError

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits
