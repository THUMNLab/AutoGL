# codes in this file are reproduced from https://github.com/GraphNAS/GraphNAS with some changes.
import typing as _typ
import torch

import torch.nn.functional as F
from nni.nas.pytorch import mutables

from . import register_nas_space
from .base import BaseSpace
from autogl.module.model.pyg.base import activate_func
from ...model import BaseAutoModel
from ..utils import count_parameters, measure_latency
from torch_geometric.data.batch import Batch

from torch import nn
from torch.nn import Linear
from .operation import act_map, gnn_map

from ..backend import *

# POOL
from torch_geometric.nn import GINConv, global_add_pool

GRAPHNAS_DEFAULT_GNN_OPS = [
    "gat_8",  # GAT with 8 heads
    "gat_6",  # GAT with 6 heads
    "gat_4",  # GAT with 4 heads
    "gat_2",  # GAT with 2 heads
    "gat_1",  # GAT with 1 heads
    "gcn",  # GCN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "sg",  # simplifying gcn
    "linear",  # skip connection
    "zero",  # skip connection
]

GRAPHNAS_DEFAULT_ACT_OPS = [
    # "sigmoid", "tanh", "relu", "linear",
    #  "softplus", "leaky_relu", "relu6", "elu"
    "sigmoid",
    "tanh",
    "relu",
    "linear",
    "elu",
]

GRAPHNAS_DEFAULT_CON_OPS=["add", "product", "concat"]
# GRAPHNAS_DEFAULT_CON_OPS=[ "concat"] # for darts

class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.lambd)


class StrModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.str = lambd

    def forward(self, *args, **kwargs):
        return self.str

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.str)


def act_map_nn(act):
    return LambdaModule(act_map(act))


def map_nn(l):
    return [StrModule(x) for x in l]


@register_nas_space("graphnas")
class GraphNasNodeClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.9,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_GNN_OPS,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_ACT_OPS,
        con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_CON_OPS
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.act_ops = act_ops
        self.con_ops = con_ops
        self.dropout = dropout

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        dropout: _typ.Optional[float] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
        con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
    ):
        super().instantiate()
        self.dropout = dropout or self.dropout
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.gnn_ops = gnn_ops or self.gnn_ops
        self.act_ops = act_ops or self.act_ops
        self.con_ops = con_ops or self.con_ops
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.preproc1 = nn.Linear(self.input_dim, self.hidden_dim)
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for layer in range(2, self.layer_number + 2):
            node_labels.append(f"op_{layer}")
            setattr(
                self,
                f"in_{layer}",
                self.setInputChoice(
                    layer,
                    choose_from=node_labels[:-1],
                    n_chosen=1,
                    return_mask=False,
                    key=f"in_{layer}",
                ),
            )
            setattr(
                self,
                f"op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        gnn_map(op, self.hidden_dim, self.hidden_dim)
                        for op in self.gnn_ops
                    ],
                    key=f"op_{layer}",
                ),
            )
        setattr(
            self,
            "act",
            self.setLayerChoice(
                2 * layer, [act_map_nn(a) for a in self.act_ops], key="act"
            ),
        )
        # for DARTS, len(con_ops) can only <=1, for dimension problems
        if len(self.con_ops)>1:
            setattr(
                self,
                "concat",
                self.setLayerChoice(
                    2 * layer + 1, map_nn(self.con_ops), key="concat"
                ),
            )
        self._initialized = True
        self.classifier1 = nn.Linear(
            self.hidden_dim * self.layer_number, self.output_dim
        )
        self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        # x, edges = data.x, data.edge_index  # x [2708,1433] ,[2, 10556]
        x= bk_feat(data)

        x = F.dropout(x, p=self.dropout, training=self.training)
        pprev_, prev_ = self.preproc0(x), self.preproc1(x)
        prev_nodes_out = [pprev_, prev_]
        for layer in range(2, self.layer_number + 2):
            node_in = getattr(self, f"in_{layer}")(prev_nodes_out)
            op=getattr(self, f"op_{layer}")
            node_out = bk_gconv(op,data,node_in)
            prev_nodes_out.append(node_out)
        act = getattr(self, "act")
        if len(self.con_ops)>1:
            con = getattr(self, "concat")()
        elif len(self.con_ops)==1:
            con=self.con_ops[0]
        else:
            con="concat"

        states = prev_nodes_out
        if con == "concat":
            x = torch.cat(states[2:], dim=1)
        else:
            tmp = states[2]
            for i in range(3, len(states)):
                if con == "add":
                    tmp = torch.add(tmp, states[i])
                elif con == "product":
                    tmp = torch.mul(tmp, states[i])
            x = tmp
        x = act(x)
        if con == "concat":
            x = self.classifier1(x)
        else:
            x = self.classifier2(x)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) -> BaseAutoModel:
        # return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap().fix(selection)

@register_nas_space("gclgraphnas")
class GraphNasGraphClassificationSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.9,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_GNN_OPS,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_ACT_OPS,
        con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_CON_OPS
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.act_ops = act_ops
        self.con_ops = con_ops
        self.dropout = dropout

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        dropout: _typ.Optional[float] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
        con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
    ):
        super().instantiate()
        self.dropout = dropout or self.dropout
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.gnn_ops = gnn_ops or self.gnn_ops
        self.act_ops = act_ops or self.act_ops
        self.con_ops = con_ops or self.con_ops
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.preproc1 = nn.Linear(self.input_dim, self.hidden_dim)
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for layer in range(2, self.layer_number + 2):
            node_labels.append(f"op_{layer}")
            setattr(
                self,
                f"in_{layer}",
                self.setInputChoice(
                    layer,
                    choose_from=node_labels[:-1],
                    n_chosen=1,
                    return_mask=False,
                    key=f"in_{layer}",
                ),
            )
            setattr(
                self,
                f"op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        gnn_map(op, self.hidden_dim, self.hidden_dim)
                        for op in self.gnn_ops
                    ],
                    key=f"op_{layer}",
                ),
            )
        setattr(
            self,
            "act",
            self.setLayerChoice(
                2 * layer, [act_map_nn(a) for a in self.act_ops], key="act"
            ),
        )
        # for DARTS, len(con_ops) can only <=1, for dimension problems
        if len(self.con_ops)>1:
            setattr(
                self,
                "concat",
                self.setLayerChoice(
                    2 * layer + 1, map_nn(self.con_ops), key="concat"
                ),
            )
        self._initialized = True
        self.classifier1 = nn.Linear(
            self.hidden_dim * self.layer_number, self.output_dim
        )
        self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc1 = Linear(2*self.hidden_dim,self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.output_dim)


    def forward(self, data):
        # print("nas_1")
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print("nas_10")
        pprev_, prev_ = self.preproc0(x), self.preproc1(x)
        prev_nodes_out = [pprev_, prev_]
        # print("nas_11")
        for layer in range(2, self.layer_number + 2):
            # print("nas_12")
            node_in = getattr(self, f"in_{layer}")(prev_nodes_out)
            op=getattr(self, f"op_{layer}")
            node_out = bk_gconv(op,data,node_in)
            prev_nodes_out.append(node_out)
        
        # print("nas_2")
        if len(self.con_ops)>1:
            con = getattr(self, "concat")()
        elif len(self.con_ops)==1:
            con=self.con_ops[0]
        else:
            con="concat"

        states = prev_nodes_out
        if con == "concat":
            x = torch.cat(states[2:], dim=1)
        else:
            tmp = states[2]
            for i in range(3, len(states)):
                if con == "add":
                    tmp = torch.add(tmp, states[i])
                elif con == "product":
                    tmp = torch.mul(tmp, states[i])
            x = tmp
        # print("nas_3")
        # Graph Pooling
        x = x.sum(dim=0, keepdim=True)
        
        x = self.fc1(x)
        x = activate_func(x, "leaky_relu")
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)

        # if con == "concat":
        #     x = self.classifier1(x)
        # else:
        #     x = self.classifier2(x)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) -> BaseAutoModel:
        # return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap().fix(selection)

@register_nas_space("gclgraphnas2")
class GraphNasGraphClassificationSpace2(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.9,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        metadata = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_GNN_OPS,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_ACT_OPS,
        con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_CON_OPS,
        device = "cpu",
    ):
        super().__init__()

        # new
        self.edge_index,self.inner_edge_indexs,self.cross_edge_indexs,self.num_nodes = metadata
        self.edge_index=self.edge_index.to(device)
        self.inner_edge_indexs=[i.to(device) for i in self.inner_edge_indexs]
        self.cross_edge_indexs=[i.to(device) for i in self.cross_edge_indexs]
        self.device = device 

        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.act_ops = act_ops
        self.con_ops = con_ops
        self.dropout = dropout

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        dropout: _typ.Optional[float] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
        con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
    ):
        super().instantiate()
        self.dropout = dropout or self.dropout
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.gnn_ops = gnn_ops or self.gnn_ops
        self.act_ops = act_ops or self.act_ops
        self.con_ops = con_ops or self.con_ops
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
        self.preproc1 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]

        for layer in range(1, self.layer_number + 2):
            node_labels.append(f"op_{layer}")
            setattr(
                self,
                f"in_{layer}", # name
                self.setInputChoice(
                    layer,
                    choose_from=node_labels[:-1], # 连接之前的所有gnn cell
                    n_chosen=1, 
                    return_mask=False,
                    key=f"in_{layer}", # value
                ),
            )

            setattr(
                self,
                f"op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        gnn_map(op, self.hidden_dim, self.hidden_dim)
                        for op in self.gnn_ops
                    ],
                    key=f"op_{layer}",
                ),
            )
            op=getattr(self, f"op_{layer}").to(self.device)
        setattr(
            self,
            "act",
            self.setLayerChoice(
                2 * layer, [act_map_nn(a) for a in self.act_ops], key="act"
            ),
        )
        # for DARTS, len(con_ops) can only <=1, for dimension problems
        if len(self.con_ops)>1:
            setattr(
                self,
                "concat",
                self.setLayerChoice(
                    2 * layer + 1, map_nn(self.con_ops), key="concat"
                ),
            )
        self._initialized = True
        self.classifier1 = nn.Linear(
            self.hidden_dim * self.layer_number, self.output_dim
        ).to(self.device)
        self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        # self.fc1 = Linear(2*self.hidden_dim,self.hidden_dim).to(self.device)
        # self.fc2 = Linear(self.hidden_dim, self.output_dim).to(self.device)
        self.fc1 = Linear(64,64).to(self.device)
        self.fc2 = Linear(64, self.output_dim).to(self.device)
        def crossconv():
            from torch_scatter import scatter
            def pool(x,edge_index,num_nodes,shift):
                res = x[edge_index[0]-shift] 
                res = scatter(res, edge_index[1], dim=0, dim_size=num_nodes, reduce='mean')
                return res
            return pool 
        self.cross_convs=[crossconv() for i in range(len(self.cross_edge_indexs))]

    # 主要改这个！！！！
    def forward(self, data, batch): # batch = 0,1,2,...
        data.x , data.edge_index = data.x.to(self.device) , data.edge_index.to(self.device)
        batch_size = shift = data.batch[0]
        # print(data) # Batch(batch=[578], edge_index=[2, 8458], ptr=[2], x=[578, 2], y=[1])

        x = bk_feat(data)
        x = F.dropout(x, p=self.dropout, training=self.training)
        pprev_, prev_ = self.preproc0(x), self.preproc1(x)
        prev_nodes_out = [pprev_, prev_]

        # 第一层卷积
        op=getattr(self, f"op_{1}")
        node_out = bk_gconv(op,data,prev_nodes_out[-1]).to(self.device)
        prev_nodes_out.append(node_out)

        for layer in range(2, self.layer_number + 2):
            # 第一层先pool到第二层，然后第二层进行卷积，得到第二层的特征
            num_nodes1=self.num_nodes[layer-2] # 第一层的节点个数
            num_nodes2=self.num_nodes[layer-1] # 第二层的节点个数
 
            cross_edge_index=self.cross_edge_indexs[layer-2] # [2, 2469]
            inner_edge_index=self.inner_edge_indexs[layer-2]

            # 层间池化
            node_temp = self.cross_convs[layer-2](prev_nodes_out[-1],cross_edge_index,num_nodes2,shift*batch)
           
            # 层内卷积
            op=getattr(self, f"op_{layer}")
            node_out = op(node_temp, inner_edge_index)
            prev_nodes_out.append(node_out)
        
        if len(self.con_ops)>1:
            con = getattr(self, "concat")()
        elif len(self.con_ops)==1:
            con=self.con_ops[0]
        else:
            con="concat"

        states = prev_nodes_out
        
        if con == "concat":
            # x = torch.cat(states[2:], dim=1)
            x = torch.cat(states[2:], dim=0)
        else:
            tmp = states[2]
            for i in range(3, len(states)):
                if con == "add":
                    tmp = torch.add(tmp, states[i])
                elif con == "product":
                    tmp = torch.mul(tmp, states[i])
            x = tmp

        # Graph Pooling
        x = x.sum(dim=0, keepdim=True)
        
        x = self.fc1(x)
        x = activate_func(x, "leaky_relu")
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)



    def parse_model(self, selection, device) -> BaseAutoModel:
        # return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap().fix(selection)