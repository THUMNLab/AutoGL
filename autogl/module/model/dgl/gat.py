import torch
import typing as _typing
from dgl.nn.pytorch.conv import GATConv
from . import _decoder, register_model
from .base import BaseModel, activate_func
from ....utils import get_logger

LOGGER = get_logger("GATModel")


def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args


class GAT(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(GAT, self).__init__()
        self.args = args
        self.num_layer = int(self.args["num_layers"])

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

        if not self.num_layer == len(self.args["hidden"]) + 1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")
        self.convs = torch.nn.ModuleList()

        num_output_heads: int = self.args.get("num_output_heads", 1)

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
                num_heads=num_output_heads,
                feat_drop=self.args.get("feat_drop", self.args["dropout"]),
                attn_drop=self.args["dropout"],
            )
        )

    def forward(self, data, *args, **kwargs):
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

        x = self.convs[-1](data, x).flattern(1)
        if self._decoder is not None:
            return self._decoder(data, x, *args, **kwargs)
        else:
            return x

    def lp_encode(self, data):
        x = data.ndata['feat']
        for i in range(self.num_layer - 1):
            x = self.convs[i](data).flatten(1)
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


@register_model("gat")
class AutoGAT(BaseModel):
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
            self, num_features=None, num_classes=None, device=None, init=False,
            decoder: _typing.Union[_typing.Type[_decoder.RepresentationDecoder], str, None] = ...,
            **kwargs
    ):
        super(AutoGAT, self).__init__()
        self.num_features = num_features if num_features is not None else 0
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.device = device if device is not None else "cpu"
        self.init = True
        self.decoder = decoder

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

        self.hyperparams = {
            "num_layers": 2,
            "hidden": [32],
            "heads": 4,
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
        self.model = GAT({**self.params, **self.hyperparams}, decoder=self.decoder).to(self.device)
