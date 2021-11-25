import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as _typing
from torch.nn import ReLU, LeakyReLU, Tanh, ELU
from dgl.nn.pytorch.conv import GINConv
from . import _decoder, register_model
from .base import BaseModel, activate_func
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
    def __init__(self, args, **kwargs):
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

        self.eps = self.args["eps"]
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

        for layer in range(self.num_layers - 2):
            if layer == 0:
                mlp = MLP(self.num_mlp_layers, input_dim, hidden[layer], hidden[layer])
            else:
                mlp = MLP(self.num_mlp_layers, hidden[layer-1], hidden[layer], hidden[layer])

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden[layer]))

        ''' Set decoder '''
        decoder: _typing.Union[str, _typing.Type[_decoder.RepresentationDecoder], None] = kwargs.get("decoder")
        if issubclass(decoder, _decoder.RepresentationDecoder):
            self._decoder: _typing.Optional[_decoder.RepresentationDecoder] = decoder(self.args)
        elif isinstance(decoder, str) and len(decoder.strip()) > 0:
            self._decoder: _typing.Optional[_decoder.RepresentationDecoder] = (
                _decoder.RepresentationDecoderUniversalRegistry.get_representation_decoder(decoder)(
                    self.args
                ) if not ('no' in decoder.lower() or 'null' in decoder.lower()) else None
            )
        else:
            self._decoder: _typing.Optional[_decoder.RepresentationDecoder] = None

    def forward(self, data, *args, **kwargs):
        g = data
        x = g.ndata.pop('feat')

        if self.num_graph_features > 0:
            graph_feature = data.gf

        # list of hidden representation at each layer (including input)
        # hidden_rep = [h]

        for i in range(self.num_layers - 2):
            x = self.ginlayers[i](g, x)
            x = activate_func(x, self.args["act"])
            x = self.batch_norms[i](x)
            # h = F.relu(h)
            # hidden_rep.append(h)
        if self.num_graph_features > 0:
            x = torch.cat([x, graph_feature], dim=-1)
        if (
                self._decoder is not None and
                isinstance(self._decoder, _decoder.RepresentationDecoder)
        ):
            return self._decoder(g, x, *args, **kwargs)
        else:
            return x


@register_model("gin")
class AutoGIN(BaseModel):
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
        init=False,
        num_graph_features=None,
        decoder: _typing.Union[_typing.Type[_decoder.RepresentationDecoder], str, None] = ...,
        **kwargs
    ):

        super(AutoGIN, self).__init__()
        self.num_features = num_features if num_features is not None else 0
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.num_graph_features = (
            int(num_graph_features) if num_graph_features is not None else 0
        )
        self.device = device if device is not None else "cpu"
        self.decoder = decoder
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
                "parameterName": "eps",
                "type": "CATEGORICAL",
                "feasiblePoints": ["True", "False"],
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

        self.hyperparams = {
            "num_layers": 5,
            "hidden": [64,64,64,64],
            "dropout": 0.5,
            "act": "relu",
            "eps": "False",
            "mlp_layers": 2,
            "neighbor_pooling_type": "sum",
            "graph_pooling_type": "sum"
        }

        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        # """Initialize model."""
        if self.initialized:
            return
        self.initialized = True
        self.model = GIN({**self.params, **self.hyperparams}, decoder=self.decoder).to(self.device)
