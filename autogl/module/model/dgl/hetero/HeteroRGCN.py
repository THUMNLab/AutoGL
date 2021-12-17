import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax

from .. import register_model
from ..base import BaseModel, activate_func, ClassificationSupportedSequentialModel
from .....utils import get_logger

LOGGER = get_logger("HGTModel")

def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroRGCN(nn.Module):
    def __init__(self, args):

        super(HeteroRGCN, self).__init__()
        self.args = args 
        self.edge_type = self.args["edge_type"]
        missing_keys = list(
            set(
                [
                    "features_num",
                    "num_class",
                    "num_layers",
                    "hidden",
                    "act",
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))

        self.num_layers = int(self.args["num_layers"])

        if not self.num_layers == len(self.args["hidden"])-1:
            LOGGER.warn("Warning: layer size does not match the length of hidden units")

        self.layers  = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(self.args["features_num"], self.args["hidden"][0], self.edge_type))
        for i in range(self.num_layers-2):
            self.layers.append(HeteroRGCNLayer(self.args["hidden"][i], self.args["hidden"][i+1], self.edge_type))
        self.layers.append(HeteroRGCNLayer(self.args["hidden"][-1], self.args["num_class"], self.edge_type))
            
    def forward(self, G, out_key):
        h_dict = {ntype : G.nodes[ntype].data['feat'] for ntype in G.ntypes}
        for l in range(self.num_layers):
            h_dict = self.layers[l](G, h_dict)
            if l!=self.num_layers-1:
                h_dict = {k : activate_func(h, self.args["act"]) for k, h in h_dict.items()}    

        return h_dict[out_key]

@register_model("HeteroRGCN")
class AutoHeteroRGCN(BaseModel):
    r"""
    AutoHAN.
    The model used in this automodel is HeteroRGCN, i.e., the relational graph convolutional network from the
    `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        
    Parameters
    ----------
    G: ``autogl.data``
        The Hetero Graph Data.
    
    meta_paths: ``List[List[str]]``
        List of meth paths, each as a list of edge types.

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
        self,  G = None, meta_paths = None, num_features=None, num_classes=None, device=None, init=False, **args
    ):
        super(AutoHeteroRGCN, self).__init__()
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
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],  
            },
        ]

        self.hyperparams = {
            "num_layers": 2,
            "hidden": [256],
            "act": "leaky_rely",
        }

        if dataset is not None:
            self.from_dataset(dataset)
        self.initialized = False
        if init is True:
            self.initialize()



    def initialize(self):
        # """Initialize model."""
        if self.initialized:
            return
        self.initialized = True
        print(self.params)
        self.model = HeteroRGCN({**self.params, **self.hyperparams}).to(self.device)


    def from_dataset(self, dataset):
        self.params["out_key"] = dataset.schema['target_node_type'] 
        self.params["edge_type"] = dataset[0].etypes


    def from_hyper_parameter(self, hp):
        ret_self = self.__class__(
            G=self.G,
            num_features=self.num_features,
            num_classes=self.num_classes,
            device=self.device,
            init=False,
        )
        ret_self.hyperparams.update(hp)
        ret_self.params.update(self.params)
        ret_self.initialize()
        return ret_self
