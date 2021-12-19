import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from . import register_model
from .base import BaseAutoModel, activate_func
from ....utils import get_logger

LOGGER = get_logger("TopkModel")


def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args


class Topkpool(torch.nn.Module):
    def __init__(self, args):
        super(Topkpool, self).__init__()
        self.args = args

        missing_keys = list(
            set(
                [
                    "features_num",
                    "num_class",
                    "num_graph_features",
                    "ratio",
                    "dropout",
                    "act",
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))

        self.num_features = self.args["features_num"]
        self.num_classes = self.args["num_class"]
        self.ratio = self.args["ratio"]
        self.dropout = self.args["dropout"]
        self.num_graph_features = self.args["num_graph_features"]

        self.conv1 = GraphConv(self.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=self.ratio)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=self.ratio)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=self.ratio)

        self.lin1 = torch.nn.Linear(256 + self.num_graph_features, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.num_graph_features > 0:
            graph_feature = data.gf

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        if self.num_graph_features > 0:
            x = torch.cat([x, graph_feature], dim=-1)
        x = self.lin1(x)
        x = activate_func(x, self.args["act"])
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = activate_func(x, self.args["act"])
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


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
        init=False,
        num_graph_features=0,
        **args
    ):
        super().__init__(num_features, num_classes, device, num_graph_features=num_graph_features, **args)
        self.num_graph_features = num_graph_features
        self.hyper_parameter_space = [
            {
                "parameterName": "ratio",
                "type": "DOUBLE",
                "maxValue": 0.9,
                "minValue": 0.1,
                "scalingType": "LINEAR",
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
        ]

        self.hyper_parameters = {"ratio": 0.8, "dropout": 0.5, "act": "relu"}


    def from_hyper_parameter(self, hp, **kwargs):
        return super().from_hyper_parameter(hp, num_graph_features=self.num_graph_features, **kwargs)

    def _initialize(self):
        self._model = Topkpool({
            "features_num": self.input_dimension,
            "num_class": self.output_dimension,
            "num_graph_features": self.num_graph_features,
            **self.hyper_parameters
        }).to(self.device)
