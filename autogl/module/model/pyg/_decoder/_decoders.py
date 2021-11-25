import torch.nn.functional
import typing as _typing
import torch_geometric
from torch_geometric.nn.glob import global_mean_pool, global_max_pool
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
@RepresentationDecoderUniversalRegistry.register_representation_decoder("log_softmax_decoder")
class LogSoftmaxDecoder(_decoder.RepresentationDecoder):
    def __init__(self, hyper_parameters: _typing.Mapping[str, _typing.Any], *args, **kwargs):
        kwargs["hyper_parameters"] = hyper_parameters
        kwargs["node_classification_compatible"] = True
        kwargs["graph_classification_compatible"] = False
        kwargs["link_prediction_compatible"] = False
        super(LogSoftmaxDecoder, self).__init__(args, kwargs)

    def __call__(self, data: torch_geometric.data.Data, *args, **kwargs) -> torch.Tensor:
        if (
                hasattr(data, 'x') and
                isinstance(getattr(data, 'x'), torch.Tensor) and
                torch.is_tensor(getattr(data, 'x'))
        ):
            return torch.nn.functional.log_softmax(getattr(data, 'x'), dim=-1)
        else:
            raise TypeError


@RepresentationDecoderUniversalRegistry.register_representation_decoder("gin")
@RepresentationDecoderUniversalRegistry.register_representation_decoder("gin_decoder")
class GINDecoder(_decoder.RepresentationDecoder):
    def __init__(self, hyper_parameters: _typing.Mapping[str, _typing.Any], *args, **kwargs):
        kwargs["hyper_parameters"] = hyper_parameters
        kwargs["node_classification_compatible"] = False
        kwargs["graph_classification_compatible"] = True
        kwargs["link_prediction_compatible"] = False
        super(GINDecoder, self).__init__(*args, **kwargs)
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
        self.__activation: str = hyper_parameters["act"]
        self.__dropout: float = hyper_parameters.get("dropout", 0)

        self.fc1 = torch.nn.Linear(
            hyper_parameters["hidden"][num_layers - 3] + num_graph_features,
            hyper_parameters["hidden"][num_layers - 2],
            )
        self.fc2 = torch.nn.Linear(
            hyper_parameters["hidden"][num_layers - 2], num_classes
        )

    def __call__(self, data: torch_geometric.data.Data, *args, **kwargs) -> torch.Tensor:
        if (
                hasattr(data, 'x') and isinstance(getattr(data, 'x'), torch.Tensor) and
                torch.is_tensor(getattr(data, 'x'))
        ):
            x: torch.Tensor = data.x
        else:
            raise TypeError
        x = self.fc1(x)
        x = activate_func(x, self.__activation)
        x = torch.nn.functional.dropout(x, p=self.__dropout, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


@RepresentationDecoderUniversalRegistry.register_representation_decoder("DiffPool".lower())
@RepresentationDecoderUniversalRegistry.register_representation_decoder("DiffPool_decoder".lower())
class DiffPoolDecoder(_decoder.RepresentationDecoder):
    """
    https://arxiv.org/abs/1905.05178
    https://arxiv.org/abs/1905.02850
    """
    def __init__(self, hyper_parameters: _typing.Mapping[str, _typing.Any], *args, **kwargs):
        kwargs["hyper_parameters"] = hyper_parameters
        kwargs["node_classification_compatible"] = False
        kwargs["graph_classification_compatible"] = True
        kwargs["link_prediction_compatible"] = False
        super(DiffPoolDecoder, self).__init__(*args, **kwargs)
        temp: _typing.Tuple[_typing.Optional[int], _typing.Optional[int]] = (
            hyper_parameters.get("num_classes"), hyper_parameters.get("num_class")
        )
        if all([(not isinstance(i, int)) for i in temp]):
            raise ValueError("num_classes or num_class not exists")
        else:
            num_classes: int = temp[0] if isinstance(temp[0], int) else temp[1]
            if not num_classes > 0:
                raise ValueError("num_classes must be positive integer")
        temp: _typing.Tuple[_typing.Optional[int], _typing.Optional[int]] = (
            hyper_parameters.get("num_features"), hyper_parameters.get("features_num")
        )
        if all([(not isinstance(i, int)) for i in temp]):
            raise ValueError("num_features or features_num not exists")
        else:
            num_features: int = temp[0] if isinstance(temp[0], int) else temp[1]
            if not num_features > 0:
                raise ValueError("num_features must be positive integer")
        num_graph_features: int = hyper_parameters["num_graph_features"]
        ratio: float = hyper_parameters["ratio"]
        self.__activation: str = hyper_parameters["act"]
        self.__dropout: float = hyper_parameters.get("dropout", 0)
        self.conv1 = torch_geometric.nn.GraphConv(self.num_features, 128)
        self.pool1 = torch_geometric.nn.TopKPooling(128, ratio=ratio)
        self.conv2 = torch_geometric.nn.GraphConv(128, 128)
        self.pool2 = torch_geometric.nn.TopKPooling(128, ratio=ratio)
        self.conv3 = torch_geometric.nn.GraphConv(128, 128)
        self.pool3 = torch_geometric.nn.TopKPooling(128, ratio=ratio)

        self.lin1 = torch.nn.Linear(256 + num_graph_features, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)

    def __call__(self, data: torch_geometric.data.Data, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.num_graph_features > 0:
            graph_feature: _typing.Optional[torch.Tensor] = data.gf
        else:
            graph_feature = None

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
                self.num_graph_features > 0 and
                torch.is_tensor(graph_feature) and
                isinstance(graph_feature, torch.Tensor)
        ):
            x = torch.cat([x, graph_feature], dim=-1)
        x = self.lin1(x)
        x = activate_func(x, self.__activation)
        x = torch.nn.functional.dropout(x, p=self.__dropout, training=self.training)
        x = self.lin2(x)
        x = activate_func(x, self.__activation)
        x = torch.nn.functional.log_softmax(self.lin3(x), dim=-1)
        return x


@RepresentationDecoderUniversalRegistry.register_representation_decoder("lp_decoder")
@RepresentationDecoderUniversalRegistry.register_representation_decoder("LinkPrediction_decoder".lower())
@RepresentationDecoderUniversalRegistry.register_representation_decoder("Link_Prediction_decoder".lower())
class LinkPredictionDecoder(_decoder.RepresentationDecoder):
    def __call__(self, data: torch_geometric.data.Data, *args, **kwargs) -> torch.Tensor:
        z: torch.Tensor = data.x
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
