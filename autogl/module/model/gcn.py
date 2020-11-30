import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from . import register_model
from .base import BaseModel, activate_func
from ...utils import get_logger

LOGGER = get_logger("GCNModel")


def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_layer = int(self.args["num_layers"])

        missing_keys = list(set(["features_num", "num_class", "num_layers", "hidden", "dropout", "act"]) - set(self.args.keys()))
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ','.join(missing_keys))

        if not self.num_layer == len(self.args["hidden"]) + 1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.args["features_num"], self.args["hidden"][0]))
        for i in range(self.num_layer - 2):
            self.convs.append(
                GCNConv(self.args["hidden"][i], self.args["hidden"][i + 1])
            )
        self.convs.append(
            GCNConv(self.args["hidden"][self.num_layer - 2], self.args["num_class"])
        )

    def forward(self, data):
        try:
            x = data.x
        except:
            print("no x")
            pass
        try:
            edge_index = data.edge_index
        except:
            print("no index")
            pass
        try:
            edge_weight = data.edge_weight
        except:
            edge_weight = None
            pass

        for i in range(self.num_layer):
            x = self.convs[i](x, edge_index, edge_weight)
            if i != self.num_layer - 1:
                x = activate_func(x, self.args["act"])
                x = F.dropout(x, p=self.args["dropout"], training=self.training)
        return F.log_softmax(x, dim=1)


@register_model("gcn")
class AutoGCN(BaseModel):
    r"""
    AutoGCN.
    The model used in this automodel is GCN, i.e., the graph convolutional network from the
    `"Semi-supervised Classification with Graph Convolutional
    Networks" <https://arxiv.org/abs/1609.02907>`_ paper. The layer is

    .. math::

        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Parameters
    ----------
    num_features: ``int``
        The dimension of features.

    num_classes: ``int``
        The number of classes.

    device: ``torch.device`` or ``str``
        The device where model will be running on.

    init: `bool`.
        If True(False), the model will (not) be initialized.
    """

    def __init__(
        self, num_features=None, num_classes=None, device=None, init=False, **args
    ):

        super(AutoGCN, self).__init__()

        self.num_features = num_features if num_features is not None else 0
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.device = device if device is not None else "cpu"
        self.init = True

        self.params = {
            "features_num": self.num_features,
            "num_class": self.num_classes,
        }
        self.space = [
            {
                "parameterName": "num_layers",
                "type": "DISCRETE",
                "feasiblePoints": "2,3,4",
            },
            {
                "parameterName": "hidden",
                "type": "NUMERICAL_LIST",
                "numericalType": "INTEGER",
                "length": 3,
                "minValue": [8, 8, 8],
                "maxValue": [128, 128, 128],
                "scalingType": "LOG",
                "cutPara": ("num_layers",),
                "cutFunc": lambda x: x[0] - 1,
            },
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.8,
                "minValue": 0.2,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
        ]

        # initial point of hp search
        self.hyperparams = {
            "num_layers": 2,
            "hidden": [16],
            "dropout": 0.2,
            "act": "leaky_relu",
        }

        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        # """Initialize model."""
        if self.initialized:
            return
        self.initialized = True
        self.model = GCN({**self.params, **self.hyperparams}).to(self.device)
