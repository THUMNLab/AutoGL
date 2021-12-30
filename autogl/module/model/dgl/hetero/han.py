import logging
import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GATConv
from functools import partial

from ..._utils import activation
from .. import register_model
from .base import BaseHeteroModelMaintainer
from .....utils import get_logger

LOGGER = get_logger("HANModel")

def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout, act):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=act,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)


class HAN(nn.Module):
    def __init__(self, args):
        super(HAN, self).__init__()
        self.args = args
        missing_keys = list(
            set(
                [
                    "meta_paths",
                    "num_features",
                    "num_class",
                    "out_key",
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

        act = partial(activation.activation_func, function_name=self.args.get("act", "relu"))
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(self.args["meta_paths"], self.args["num_features"], self.args["hidden"][0], self.args["heads"][0], self.args["dropout"], act))
        for l in range(1, len(self.args["hidden"])):
            self.layers.append(HANLayer(self.args["meta_paths"], self.args["hidden"][l-1] * self.args["heads"][l-1],
                                        self.args["hidden"][l], self.args["heads"][l], self.args["dropout"], act))
        self.predict = nn.Linear(self.args["hidden"][-1] * self.args["heads"][-1], self.args["num_class"])

    def forward(self, g):
        h = g.nodes[self.args["out_key"]].data['feat']
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

@register_model("han")
class AutoHAN(BaseHeteroModelMaintainer):
    r"""
    AutoHAN.
    The model used in this automodel is HAN, i.e., the graph convolutional network from the
    `"Heterogenous Graph Attention Network"
    <https://arxiv.org/pdf/1903.07293.pdf>`_ paper.


    Parameters
    ----------
    num_features: `int`.
        The dimension of features.

    num_classes: `int`.
        The number of classes.

    device: `torch.device` or `str`.
        The device where model will be running on.

    init: `bool`.
        If True(False), the model will (not) be initialized.

    dataset: `autogl.datasets`.
        Hetero Graph Dataset in autogl.
    """
    def __init__(
        self, num_features=None, num_classes=None, device=None, init=False, dataset=None, **args
    ):
        super().__init__(num_features, num_classes, device, dataset, **args)

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
                "type": "NUMERICAL_LIST",
                "numericalType": "INTEGER",
                "scalingType": "LOG",
                "length": 3,
                "minValue": [1, 1, 1],
                "maxValue": [16, 16, 16],
                "cutPara": ("num_layers",),
                "cutFunc": lambda x: x[0] - 1,
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh","gelu"],  # add F.gelu()
            },
        ]

        self.hyper_parameters = {
            "num_layers": 2,
            "hidden": [256],
            "heads": [8],
            "dropout": 0.2,
            "act": "gelu",
        }

        if init is True:
            self.initialize()

    def _initialize(self):
        self._model = HAN(dict(
            num_features=self.input_dimension,
            num_class=self.output_dimension,
            out_key=self.out_key,
            meta_paths=self.meta_paths,
            **self.hyper_parameters
        )).to(self.device)

    def from_dataset(self, dataset):
        self.register_parameter("out_key", dataset.schema["target_node_type"])
        self.register_parameter("meta_paths", dataset.schema.meta_paths)
