import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, LeakyReLU, Tanh, ELU
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import BatchNorm1d
from . import register_model
from .base import BaseModel, activate_func
from copy import deepcopy
from ...utils import get_logger

LOGGER = get_logger("GINModel")


def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args


class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.args = args
        self.num_layer = int(self.args["num_layers"])
        assert self.num_layer > 2, "Number of layers in GIN should not less than 3"

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
        if not self.num_layer == len(self.args["hidden"]) + 1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")
        self.num_graph_features = self.args["num_graph_features"]

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

        train_eps = True if self.args["eps"] == "True" else False

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        nn = [Linear(self.args["features_num"], self.args["hidden"][0])]
        for _ in range(self.args["mlp_layers"] - 1):
            nn.append(act)
            nn.append(Linear(self.args["hidden"][0], self.args["hidden"][0]))
        # nn.append(BatchNorm1d(self.args['hidden'][0]))
        self.convs.append(GINConv(Sequential(*nn), train_eps=train_eps))
        self.bns.append(BatchNorm1d(self.args["hidden"][0]))

        for i in range(self.num_layer - 3):
            nn = [Linear(self.args["hidden"][i], self.args["hidden"][i + 1])]
            for _ in range(self.args["mlp_layers"] - 1):
                nn.append(act)
                nn.append(
                    Linear(self.args["hidden"][i + 1], self.args["hidden"][i + 1])
                )
            # nn.append(BatchNorm1d(self.args['hidden'][i+1]))
            self.convs.append(GINConv(Sequential(*nn), train_eps=train_eps))
            self.bns.append(BatchNorm1d(self.args["hidden"][i + 1]))

        self.fc1 = Linear(
            self.args["hidden"][self.num_layer - 3] + self.num_graph_features,
            self.args["hidden"][self.num_layer - 2],
        )
        self.fc2 = Linear(
            self.args["hidden"][self.num_layer - 2], self.args["num_class"]
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.num_graph_features > 0:
            graph_feature = data.gf

        for i in range(self.num_layer - 2):
            x = self.convs[i](x, edge_index)
            x = activate_func(x, self.args["act"])
            x = self.bns[i](x)

        x = global_add_pool(x, batch)
        if self.num_graph_features > 0:
            x = torch.cat([x, graph_feature], dim=-1)
        x = self.fc1(x)
        x = activate_func(x, self.args["act"])
        x = F.dropout(x, p=self.args["dropout"], training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


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
        **args
    ):

        super(AutoGIN, self).__init__()
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
                "parameterName": "eps",
                "type": "CATEGORICAL",
                "feasiblePoints": ["True", "False"],
            },
            {
                "parameterName": "mlp_layers",
                "type": "DISCRETE",
                "feasiblePoints": "2,3,4",
            },
        ]

        self.hyperparams = {
            "num_layers": 3,
            "hidden": [64, 32],
            "dropout": 0.5,
            "act": "relu",
            "eps": "True",
            "mlp_layers": 2,
        }

        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        # """Initialize model."""
        if self.initialized:
            return
        self.initialized = True
        self.model = GIN({**self.params, **self.hyperparams}).to(self.device)
