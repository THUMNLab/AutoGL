import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SortPooling
from . import register_model
from .base import BaseAutoModel
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
        assert self.num_layers == len(self.args["hidden"]) + 1, "Warning: layer size does not match the length of hidden units"

        input_dim = self.args["features_num"]
        hidden = self.args["hidden"]
        final_dropout = self.args["dropout"]
        output_dim = self.args["num_class"]

        # List of MLPs
        self.gcnlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.gcnlayers.append(GraphConv(input_dim, hidden[layer]))
            else:
                self.gcnlayers.append(GraphConv(hidden[layer-1], hidden[layer]))

            #self.gcnlayers.append(GraphConv(input_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden[layer]))

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
                    nn.Linear(hidden[layer-1] * k, output_dim))

        self.drop = nn.Dropout(final_dropout)


    #def forward(self, g, h):
    def forward(self, data):
        h = data.ndata.pop('feat')
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.gcnlayers[i](data, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(data, h)
            #import pdb; pdb.set_trace()
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer


@register_model("topkpool-model")
class AutoTopkpool(BaseAutoModel):
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
        num_graph_features=0,
        **args
    ):
        super().__init__(num_features, num_classes, device, num_graph_features=num_graph_features)
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
                "parameterName": "num_layers",
                "type": "INTEGER",
                "minValue": 7,
                "maxValue": 2,
                "scalingType": "LINEAR"
            },
            {
                "parameterName": "mlp_layers",
                "type": "DISCRETE",
                "feasiblePoints": "2,3,4",
            },
        ]

        self.hyper_parameters = {
            "num_layers": 5,
            "hidden": [64,64,64,64],
            "dropout": 0.5,
            "act": "relu",
            "mlp_layers": 2
        }

    def from_hyper_parameter(self, hp, **kwargs):
        return super().from_hyper_parameter(hp, num_graph_features=self.num_graph_features, **kwargs)

    def _initialize(self):
        self._model = Topkpool({
            "features_num": self.input_dimension,
            "num_class": self.output_dimension,
            "num_graph_features": self.num_graph_features,
            **self.hyper_parameters
        }).to(self.device)

