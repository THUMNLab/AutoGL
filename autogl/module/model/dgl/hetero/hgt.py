import logging
import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax

from .. import register_model
from .base import BaseHeteroModelMaintainer
from ..base import activate_func
from .....utils import get_logger

LOGGER = get_logger("HGTModel")

def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False,
                 out_key = None):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.out_key = out_key
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

class HGT(nn.Module):
    def __init__(self, args):

        super(HGT, self).__init__()
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
                    "use_norm",
                    "node_dict",
                    "edge_dict",
                    "out_key"
                ]
            )
            - set(self.args.keys())
        )
        if len(missing_keys) > 0:
            raise Exception("Missing keys: %s." % ",".join(missing_keys))

        self.node_dict = self.args["node_dict"]
        self.edge_dict = self.args["edge_dict"]
        self.out_key = self.args["out_key"]
        self.gcs = nn.ModuleList()
        self.num_layers = int(self.args["num_layers"])

        hidden = self.args["hidden"]*self.args["heads"]

        self.adapt_ws  = nn.ModuleList()
        for t in range(len(self.node_dict)):
            self.adapt_ws.append(nn.Linear(self.args["features_num"], hidden))

        for i in range(self.num_layers):
            self.gcs.append(HGTLayer(hidden, hidden, self.node_dict, self.edge_dict, \
                self.args["heads"], use_norm = self.args["use_norm"], dropout = self.args["dropout"]))
            
        self.out = nn.Linear(hidden, self.args["num_class"])

    def forward(self, G):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = activate_func(self.adapt_ws[n_id](G.nodes[ntype].data['feat']), self.args["act"])
        for i in range(self.num_layers):
            h = self.gcs[i](G, h)
        return self.out(h[self.out_key])

@register_model("hgt")
class AutoHGT(BaseHeteroModelMaintainer):
    r"""
    AutoHGT.
    The model used in this automodel is HGT, i.e., the graph convolutional network from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.
        
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
                "type": "INTEGER", 
                "minValue": 8, 
                "maxValue": 128,
                "scalingType": "LOG"
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
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh","gelu"],  # add F.gelu()
            },
            {
                "parameterName": "use_norm",
                "type": "CATEGORICAL",
                "feasiblePoints": [True, False], 
            },
        ]

        self.hyper_parameters = {
            "num_layers": 2,
            "hidden": 64,
            "heads": 4,
            "dropout": 0.2,
            "act": "gelu",
            "use_norm": True
        }

        if init is True:
            self.initialize()

    def _initialize(self):
        self._model = HGT(dict(
            features_num=self.input_dimension, 
            num_class=self.output_dimension, 
            out_key=self.out_key,
            node_dict=self.node_dict,
            edge_dict=self.edge_dict,
            **self.hyper_parameters
        )).to(self.device)

    def from_dataset(self, dataset):
        G: dgl.DGLGraph = dataset[0]
        # generate edge and node dict
        self.register_parameter("out_key", dataset.schema["target_node_type"])
        self.register_parameter("node_dict", dict(zip(G.ntypes, range(len(G.ntypes)))))
        self.register_parameter("edge_dict", dict(zip(G.etypes, range(len(G.etypes)))))
