import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GATConv

from .. import register_model
from ..base import BaseModel, activate_func, ClassificationSupportedSequentialModel
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

        if self.args["act"] == "leaky_relu":
            act = F.leaky_relu
        elif self.args["act"] == "relu":
            act = F.relu
        elif self.args["act"] == "elu":
            act = F.elu
        elif self.args["act"] == "tanh":
            act = F.tanh
        elif self.args["act"] == "gelu":
            act = F.gelu
        else:
            act = F.relu
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(self.args["meta_paths"], self.args["num_features"], self.args["hidden"][0], self.args["heads"][0], self.args["dropout"], act))
        for l in range(1, len(self.args["heads"])):
            self.layers.append(HANLayer(self.args["meta_paths"], self.args["hidden"][l-1] * self.args["heads"][l-1],
                                        self.args["hidden"], self.args["heads"][l], self.args["dropout"], act))
        self.predict = nn.Linear(self.args["hidden"][-1] * self.args["heads"][-1], self.args["num_class"])

    def forward(self, g, out_key):
        h = g.nodes[out_key].data['feat']
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

@register_model("han")
class AutoHAN(BaseModel):

    def __init__(
        #self,  dataset = None, meta_paths=None, num_features=None, num_classes=None, device=None, init=True, **args
        self,  G = None, meta_paths=None, num_features=None, num_classes=None, device=None, init=True, **args
    ):
        super(AutoHAN, self).__init__()
        self.meta_paths = meta_paths
        #self.meta_paths = dataset.get_metapaths()
        self.num_features = num_features if num_features is not None else 0
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.device = device if device is not None else "cpu"
        self.init = init

        self.params = {
            "meta_paths": self.meta_paths,
            "num_features": self.num_features,
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
                "type": "NUMERICAL_LIST",
                "numericalType": "INTEGER",
                "feasiblePoints": "[8]",
                "cutPara": ("num_layers",),
                "cutFunc": lambda x: x[0] - 1,
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh","gelu"],  # add F.gelu()
            },
        ]

        self.hyperparams = {
            "num_layers": 2,
            "hidden": [256],
            "heads": [8],
            "dropout": 0.2,
            "act": "gelu",
        }


        if G is not None:
            self.from_dataset(G)
        self.initialized = False
        if init is True:
            self.initialize()

    def initialize(self):
        # """Initialize model."""
        if self.initialized:
            return
        self.initialized = True
        print(self.params)
        self.model = HAN({**self.params, **self.hyperparams}).to(self.device)


    def from_dataset(self, dataset):
        G = dataset
        #G = dataset[0]
        node_dict = {}
        edge_dict = {}
        for ntype in G.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in G.etypes:
            edge_dict[etype] = len(edge_dict)


    def from_hyper_parameter(self, hp):
        ret_self = self.__class__(
            meta_path=self.meta_path,
            num_features=self.num_features,
            num_classes=self.num_classes,
            device=self.device,
            init=False,
        )
        ret_self.hyperparams.update(hp)
        ret_self.params.update(self.params)
        ret_self.initialize()
        return ret_self

