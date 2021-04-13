import typing as _typing
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

from . import register_model
from .base import BaseModel, activate_func


class GraphSAGE(torch.nn.Module):
    def __init__(
            self, num_features: int, num_classes: int,
            hidden_features: _typing.Sequence[int],
            dropout: float, activation_name: str,
            aggr: str = "mean", **kwargs
    ):
        super(GraphSAGE, self).__init__()
        if type(aggr) != str:
            raise TypeError
        if aggr not in ("add", "max", "mean"):
            aggr = "mean"
        
        self.__convolution_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        
        num_layers: int = len(hidden_features) + 1
        if num_layers == 1:
            self.__convolution_layers.append(
                SAGEConv(num_features, num_classes, aggr=aggr)
            )
        else:
            self.__convolution_layers.append(
                SAGEConv(num_features, hidden_features[0], aggr=aggr)
            )
            for i in range(len(hidden_features)):
                if i + 1 < len(hidden_features):
                    self.__convolution_layers.append(
                        SAGEConv(hidden_features[i], hidden_features[i + 1], aggr=aggr)
                    )
                else:
                    self.__convolution_layers.append(
                        SAGEConv(hidden_features[i], num_classes, aggr=aggr)
                    )
        self.__dropout: float = dropout
        self.__activation_name: str = activation_name
    
    def __full_forward(self, data):
        x: torch.Tensor = getattr(data, "x")
        edge_index: torch.Tensor = getattr(data, "edge_index")
        for layer_index in range(len(self.__convolution_layers)):
            x: torch.Tensor = self.__convolution_layers[layer_index](x, edge_index)
            if layer_index + 1 < len(self.__convolution_layers):
                x = activate_func(x, self.__activation_name)
                x = F.dropout(x, p=self.__dropout, training=self.training)
        return F.log_softmax(x, dim=1)
    
    def __distributed_forward(self, data):
        x: torch.Tensor = getattr(data, "x")
        edge_indexes: _typing.Sequence[torch.Tensor] = getattr(data, "edge_indexes")
        if len(edge_indexes) != len(self.__convolution_layers):
            raise AttributeError
        for layer_index in range(len(self.__convolution_layers)):
            x: torch.Tensor = self.__convolution_layers[layer_index](x, edge_indexes[layer_index])
            if layer_index + 1 < len(self.__convolution_layers):
                x = activate_func(x, self.__activation_name)
                x = F.dropout(x, p=self.__dropout, training=self.training)
        return F.log_softmax(x, dim=1)
    
    def forward(self, data):
        if (
                hasattr(data, "edge_indexes") and
                isinstance(getattr(data, "edge_indexes"), _typing.Sequence) and
                len(getattr(data, "edge_indexes")) == len(self.__convolution_layers)
        ):
            return self.__distributed_forward(data)
        else:
            return self.__full_forward(data)


@register_model("sage")
class AutoSAGE(BaseModel):
    def __init__(
            self, num_features: int = 1, num_classes: int = 1,
            device: _typing.Optional[torch.device] = torch.device("cpu"),
            init: bool = False, **kwargs
    ):
        super(AutoSAGE, self).__init__(init)
        self.__num_features: int = num_features
        self.__num_classes: int = num_classes
        self.__device: torch.device = device if device is not None else torch.device("cpu")
        
        self.hyperparams = {
            "num_layers": 3,
            "hidden": [64, 32],
            "dropout": 0.5,
            "act": "relu",
            "aggr": "mean",
        }
        self.params = {
            "num_features": self.__num_features,
            "num_classes": self.__num_classes
        }
        
        self._model: GraphSAGE = GraphSAGE(
            self.__num_features, self.__num_classes, [64, 32], 0.5, "relu"
        )
        
        self._initialized: bool = False
        if init:
            self.initialize()
    
    @property
    def model(self) -> GraphSAGE:
        return self._model
    
    def initialize(self):
        """ Initialize model """
        if not self._initialized:
            self._model: GraphSAGE = GraphSAGE(
                self.__num_features, self.__num_classes,
                hidden_features=self.hyperparams["hidden"],
                activation_name=self.hyperparams["act"],
                **self.hyperparams
            ).to(self.__device)
            self._initialized = True
