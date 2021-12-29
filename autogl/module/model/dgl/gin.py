import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, LeakyReLU, Tanh, ELU
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from torch.nn import BatchNorm1d
from . import register_model
from .base import BaseAutoModel, activate_func
from copy import deepcopy
from ....utils import get_logger

LOGGER = get_logger("GINModel")


def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
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
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)



class GIN(torch.nn.Module):
    """GIN model"""
    def __init__(self, args):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.args = args

        missing_keys = list(
            set(
                [
                    "features_num",
                    "num_class",
                    "num_graph_features",
                    "num_layers",
                    "hidden",
                    "dropout",
                    "act",
                    "mlp_layers",
                    "eps",
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))

        self.num_graph_features = self.args["num_graph_features"]
        self.num_layers = self.args["num_layers"]
        assert self.num_layers > 2, "Number of layers in GIN should not less than 3"
        if not self.num_layers == len(self.args["hidden"]) + 1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")

        self.eps = True if self.args["eps"]=="True" else False
        self.num_mlp_layers = self.args["mlp_layers"]
        input_dim = self.args["features_num"]
        hidden = self.args["hidden"]
        neighbor_pooling_type = self.args["neighbor_pooling_type"]
        graph_pooling_type = self.args["graph_pooling_type"]
        if self.args["act"] == "leaky_relu":
            act = LeakyReLU()
        elif self.args["act"] == "relu":
            act = ReLU()
        elif self.args["act"] == "elu":
            act = ELU()
        elif self.args["act"] == "tanh":
            act = Tanh()
        else:
            act = ReLU()
        final_dropout = self.args["dropout"]
        output_dim = self.args["num_class"]

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(self.num_mlp_layers, input_dim, hidden[layer], hidden[layer])
            else:
                mlp = MLP(self.num_mlp_layers, hidden[layer-1], hidden[layer], hidden[layer])

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden[layer]))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden[layer-1], output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, data):
        x = data.ndata.pop('feat')

        if self.num_graph_features > 0:
            graph_feature = data.gf

        # list of hidden representation at each layer (including input)
        hidden_rep = [x]

        for i in range(self.num_layers - 1):
            x = self.ginlayers[i](data, x)
            x = self.batch_norms[i](x)
            x = activate_func(x, self.args["act"])
            hidden_rep.append(x)

        score_over_layer = 0
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(data, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer


@register_model("gin-model")
class AutoGIN(BaseAutoModel):
    r"""
    AutoGIN. The model used in this automodel is GIN, i.e., the graph isomorphism network from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper. The layer is

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Parameters
    ----------
    num_features: `int`.
        The dimension of features.

    num_classes: `int`.
        The number of classes.

    device: `torch.device` or `str`
        The device where model will be running on.

    init: `bool`.
        If True(False), the model will (not) be initialized.
    """

    def __init__(
        self,
        num_features=None,
        num_classes=None,
        device=None,
        num_graph_features=0,
        **args
    ):

        super().__init__(num_features, num_classes, device, num_graph_features=num_graph_features, **args)
        self.num_graph_features = num_graph_features
        
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
                "feasiblePoints": [True, False],
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
            },
            {
                "parameterName": "graph_pooling_type",
                "type": "CATEGORICAL",
                "feasiblePoints": ["sum", "mean", "max"],
            },
        ]

        self.hyper_parameters = {
            "num_layers": 5,
            "hidden": [64,64,64,64],
            "dropout": 0.5,
            "act": "relu",
            "eps": "False",
            "mlp_layers": 2,
            "neighbor_pooling_type": "sum",
            "graph_pooling_type": "sum"
        }
    
    def from_hyper_parameter(self, hp, **kwargs):
        return super().from_hyper_parameter(hp, num_graph_features=self.num_graph_features, **kwargs)

    def _initialize(self):
        # """Initialize model."""

        self._model = GIN({
            "features_num": self.input_dimension,
            "num_class": self.output_dimension,
            "num_graph_features": self.num_graph_features,
            **self.hyper_parameters
        }).to(self.device)