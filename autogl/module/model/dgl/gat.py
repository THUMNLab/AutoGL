import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from . import register_model
from .base import BaseAutoModel, activate_func
from ....utils import get_logger

LOGGER = get_logger("GATModel")


def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args


class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.args = args

        missing_keys = list(
            set(
                [
                    "features_num",
                    "num_class",
                    "num_layers",
                    "hidden",
                    "heads",
                    "dropout",
                    "act",
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))

        self.num_layer = int(self.args["num_layers"])
        if not self.num_layer == len(self.args["hidden"]) + 1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")
        self.convs = torch.nn.ModuleList()

        self.convs.append(
            GATConv(
                self.args["features_num"],
                self.args["hidden"][0],
                num_heads =self.args["heads"],
                feat_drop=self.args.get("feat_drop", self.args["dropout"]),
                attn_drop=self.args["dropout"],
            )
        )
        last_dim = self.args["hidden"][0] * self.args["heads"]
        for i in range(self.num_layer - 2):
            self.convs.append(
                GATConv(
                    last_dim,
                    self.args["hidden"][i + 1],
                    num_heads=self.args["heads"],
                    feat_drop=self.args.get("feat_drop", self.args["dropout"]),
                    attn_drop=self.args["dropout"],
                )
            )
            last_dim = self.args["hidden"][i + 1] * self.args["heads"]
        self.convs.append(
            GATConv(
                last_dim,
                self.args["num_class"],
                num_heads=1,
                feat_drop=self.args.get("feat_drop", self.args["dropout"]),
                attn_drop=self.args["dropout"],
            )
        )

    def forward(self, data):
        try:
            x = data.ndata['feat']
        except:
            print("no x")
            pass

        # data = dgl.remove_self_loop(data)
        # data = dgl.add_self_loop(data)
        
        for i in range(self.num_layer-1):
            x = self.convs[i](data, x).flatten(1)
            x = activate_func(x, self.args["act"])

        x = self.convs[-1](data, x).mean(1)

        return F.log_softmax(x, dim=1)

    def lp_encode(self, data):
        x = data.ndata['feat']
        for i in range(self.num_layer - 1):
            x = self.convs[i](data, x).flatten(1)
            if i != self.num_layer - 2:
                x = activate_func(x, self.args["act"])
                
        return x

    def lp_decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def lp_decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


@register_model("gat-model")
class AutoGAT(BaseAutoModel):
    r"""
    AutoGAT. The model used in this automodel is GAT, i.e., the graph attentional network from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper. The layer is

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j}

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

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

    args: Other parameters.
    """

    def __init__(
        self, input_dimension=None, output_dimension=None, device=None, **args
    ):
        super().__init__(input_dimension, output_dimension, device, **args)

        self.hyper_parameter_space = [
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
                "maxValue": [64, 64, 64],
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
                "parameterName": "heads",
                "type": "DISCRETE",
                "feasiblePoints": "2,4,8,16",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
        ]

        self.hyper_parameters = {
            "num_layers": 2,
            "hidden": [32],
            "heads": 4,
            "dropout": 0.2,
            "act": "leaky_relu",
        }

    def _initialize(self):
        # """Initialize model."""
        self._model = GAT({
            "features_num": self.input_dimension,
            "num_class": self.output_dimension,
            **self.hyper_parameters
        }).to(self.device)
