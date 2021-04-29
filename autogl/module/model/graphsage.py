import torch
from . import register_model
from .base import BaseModel, activate_func

from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from ...utils import get_logger

LOGGER = get_logger("SAGEModel")


class SAGEConv(MessagePassing):
    r"""Modified from SAGEConv in Pytorch Geometric <https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/sage_conv.py>
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        normalize: bool = False,
        bias: bool = True,
        aggr: str = "mean",
        **kwargs
    ):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args


class GraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.args = args
        agg = self.args["agg"]
        self.num_layer = int(self.args["num_layers"])
        if not self.num_layer == len(self.args["hidden"]) + 1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")

        missing_keys = list(set(["features_num", "num_class", "num_layers",
                    "hidden", "dropout", "act", "agg"]) - set(self.args.keys()))
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ','.join(missing_keys))
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(self.args["features_num"], self.args["hidden"][0], aggr=agg)
        )
        for i in range(self.num_layer - 2):
            self.convs.append(
                SAGEConv(self.args["hidden"][i], self.args["hidden"][i + 1], aggr=agg)
            )
        self.convs.append(
            SAGEConv(
                self.args["hidden"][self.num_layer - 2],
                self.args["num_class"],
                aggr=agg,
            )
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

    def encode(self, data):
        x = data.x
        for i in range(self.num_layer - 1):
            x = self.convs[i](x, data.train_pos_edge_index)
            if i != self.num_layer - 2:
                x = activate_func(x, self.args["act"])
                # x = F.dropout(x, p=self.args["dropout"], training=self.training)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


@register_model("sage")
class AutoSAGE(BaseModel):
    r"""
    AutoSAGE. The model used in this automodel is GraphSAGE, i.e., the GraphSAGE from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper. The layer is

    .. math::

        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

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
        self, num_features=None, num_classes=None, device=None, init=False, **args
    ):

        super(AutoSAGE, self).__init__()

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
            {
                "parameterName": "agg",
                "type": "CATEGORICAL",
                "feasiblePoints": ["mean", "add", "max"],
            },
        ]

        self.hyperparams = {
            "num_layers": 3,
            "hidden": [64, 32],
            "dropout": 0.5,
            "act": "relu",
            "agg": "mean",
        }

        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        # """Initialize model."""
        if self.initialized:
            return
        self.initialized = True
        self.model = GraphSAGE({**self.params, **self.hyperparams}).to(self.device)
