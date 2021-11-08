import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, LeakyReLU, Tanh, ELU
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SortPooling
from torch.nn import BatchNorm1d
from . import register_model
from .base import BaseModel, activate_func
from copy import deepcopy
from ....utils import get_logger

LOGGER = get_logger("TopkModel")


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



class Topkpool(torch.nn.Module):
    """Topkpool model"""
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

        """
        super(Topkpool, self).__init__()
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
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))
        #if not self.num_layer == len(self.args["hidden"]) + 1:
        #    LOGGER.warn("Warning: layer size does not match the length of hidden units")


        self.num_graph_features = self.args["num_graph_features"]
        self.num_layers = self.args["num_layers"]
        assert self.num_layers > 2, "Number of layers in GIN should not less than 3"

        self.num_mlp_layers = self.args["mlp_layers"]
        input_dim = self.args["features_num"]
        hidden_dim = self.args["hidden"][0]
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
        self.gcnlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.gcnlayers.append(GraphConv(input_dim, hidden_dim))
            else:
                self.gcnlayers.append(GraphConv(hidden_dim, hidden_dim))

            if layer == 0:
                mlp = MLP(self.num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(self.num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            #self.gcnlayers.append(GraphConv(input_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        #TopKPool
        k = 3
        self.pool = SortPooling(k)

        for layer in range(self.num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim * k, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim * k, output_dim))

        self.drop = nn.Dropout(final_dropout)


    #def forward(self, g, h):
    def forward(self, data):
        g, _ = data
        h = g.ndata.pop('attr')
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.gcnlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            #import pdb; pdb.set_trace()
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer


@register_model("topkpool")
class AutoTopkpool(BaseModel):
    r"""
    AutoTopkpool. The model used in this automodel is from https://arxiv.org/abs/1905.05178, https://arxiv.org/abs/1905.02850
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
        init=False,
        num_graph_features=None,
        **args
    ):
        super(AutoTopkpool, self).__init__()
        LOGGER.debug(
            "topkpool __init__ get params num_graph_features {}".format(
                num_graph_features
            )
        )
        self.num_features = num_features if num_features is not None else 0
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.num_graph_features = (
            int(num_graph_features) if num_graph_features is not None else 0
        )
        self.device = device if device is not None else "cpu"
        self.init = True

        self.params = {
            "features_num": self.num_features,
            "num_class": self.num_classes,
            "num_graph_features": self.num_graph_features,
        }
        self.space = [
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
                "parameterName": "mlp_layers",
                "type": "DISCRETE",
                "feasiblePoints": "2,3,4",
            },
        ]

        self.hyperparams = {
            "num_layers": 5,
            "hidden": [64,64,64,64],
            "dropout": 0.5,
            "act": "relu",
            "mlp_layers": 2
        }

        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        LOGGER.debug("topkpool initialize with parameters {}".format(self.params))
        self.model = Topkpool({**self.params, **self.hyperparams}).to(self.device)

