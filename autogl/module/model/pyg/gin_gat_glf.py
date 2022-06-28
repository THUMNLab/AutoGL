import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_scatter import scatter

from . import register_model
from .base import BaseAutoModel, activate_func
from ....utils import get_logger

LOGGER = get_logger("GATModel")


def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args



def construct_batch(batch_size,num_nodes):
    batch=torch.cat([torch.ones(num_nodes).long()*i for i in range(batch_size)],dim=0)
    return batch

def construct_inner_edge(edge_index,batch_size,num_nodes):
    shift=torch.LongTensor([[num_nodes,num_nodes]]).T.to(edge_index.device)
    e=torch.cat([edge_index+shift*i for i in range(batch_size)],dim=-1)
    return e

def construct_cross_edge(edge_index,batch_size,num_nodes1,num_nodes2):
    
    shift=torch.LongTensor([[num_nodes1,num_nodes2]]).T.to(edge_index.device)
    e=torch.cat([edge_index+shift*i for i in range(batch_size)],dim=-1)
    return e

def construct_cross_edge_both(edge_index,batch_size,num_nodes1,num_nodes2):
    shift0 = num_nodes1+num_nodes2
    shift1=torch.LongTensor([[0,num_nodes1]]).T.to(edge_index.device)
    edge_index=torch.cat([edge_index+shift0*i for i in range(batch_size)],dim=-1)
    e = torch.cat([edge_index+shift1 for i in range(batch_size)],dim=-1)
    return e

class GATPool(torch.nn.Module):
    def __init__(self, args):
        super(GATPool, self).__init__()
        self.args = args
        self.edge_index,self.inner_edge_indexs,self.cross_edge_indexs,self.num_nodes=self.args["metadata"]

        missing_keys = list(
            set(
                [
                    "features_num",
                    "num_classes",
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

        def crossconv0():
            def pool(x,edge_index,num_nodes):
                res = x[edge_index[0]]
                res = scatter(res, edge_index[1], dim=0, dim_size=num_nodes, reduce='mean') 
                return res
            return pool

        self.inner_convs = torch.nn.ModuleList()
        self.cross_convs1 = torch.nn.ModuleList()

        self.conv1 = GINConv(Sequential(Linear(self.args["features_num"], self.args["hidden"][0]), BatchNorm1d(self.args["hidden"][0]), ReLU(), Linear(self.args["hidden"][0], self.args["hidden"][0]), ReLU()))

        self.cross_convs0=[crossconv0() for i in range(len(self.num_nodes)-2)]
        
        for l in range(len(self.num_nodes)-2):

            self.cross_convs1.append(
                 GATConv(self.args["hidden"][l],self.args["hidden"][l],heads=self.args["heads"], concat=False, dropout=self.args["dropout"], add_self_loops=False)
            )
            if l==len(self.num_nodes)-3:
                self.inner_convs.append(
                    GINConv(Sequential(Linear(self.args["hidden"][l], self.args["hidden"][l]), BatchNorm1d(self.args["hidden"][l]), ReLU(), Linear(self.args["hidden"][l], self.args["hidden"][l]), ReLU()))
                )
            else:
                self.inner_convs.append(
                    GINConv(Sequential(Linear(self.args["hidden"][l], self.args["hidden"][l+1]), BatchNorm1d(self.args["hidden"][l+1]), ReLU(), Linear(self.args["hidden"][l+1], self.args["hidden"][l+1]), ReLU()))
                )
        
        
        self.lin1 = Linear(self.num_nodes[-2]*self.args["hidden"][-1], self.args["hidden"][-1])
        self.lin2 = Linear(self.args["hidden"][-1], self.args["num_classes"])

        self.cache={}

    # @timing
    def forward(self, x, batch, return_attention_weights=False):
        batch_size=batch[-1].item()+1
        attentions = []

        # self.edge_index,self.inner_edge_indexs,self.cross_edge_indexs,self.num_nodes=metadata
        edge_index=self.construct_inner_edge(self.edge_index,batch_size,self.num_nodes[0],-1)

        x = self.conv1(x, edge_index)

        for i in range(len(self.num_nodes)-2):
            num_nodes1=self.num_nodes[i]
            num_nodes2=self.num_nodes[i+1]
            cross_edge_index=self.cross_edge_indexs[i]
            cross_edge_index0=self.construct_cross_edge(cross_edge_index,batch_size,num_nodes1,num_nodes2,i)
            cross_edge_index1=self.construct_cross_edge_both(cross_edge_index,batch_size,num_nodes1,num_nodes2,i)
            
            inner_edge_index=self.inner_edge_indexs[i]
            inner_edge_index=self.construct_inner_edge(inner_edge_index,batch_size,num_nodes2,i)

            x0 = x # last layerx 
            x0 = x0.reshape(batch_size, num_nodes1,-1)
            x1 = self.cross_convs0[i](x,cross_edge_index0,num_nodes2*batch_size)   # get new layer x初始值
            x1 = x1.reshape(batch_size, num_nodes2,-1)
            x = torch.cat([x0,x1], dim=1)
            x = x.reshape(batch_size*(num_nodes1+num_nodes2),-1)

            if return_attention_weights:
                x, (edge_index_, alpha) = self.cross_convs1[i](x, cross_edge_index1, return_attention_weights=return_attention_weights)
                attentions.append((edge_index_, alpha))
            else:
                print('cross_edge_index1:',cross_edge_index1)
                print('x:',x.size())
                x = self.cross_convs1[i](x, cross_edge_index1)
            
            # 生成新一层数据的mask
            mask =  (torch.arange(num_nodes2+num_nodes1) >= num_nodes1)
            mask = mask.repeat(batch_size)
            x = x[mask]
            # x = F.elu(x)
            x = activate_func(x, self.args["act"])

            x = self.inner_convs[i](x,inner_edge_index)
        
        x = x.view(batch_size,-1)
        x = activate_func(self.lin1(x), self.args["act"])
        x = self.lin2(x)

        if return_attention_weights:
            return x,attentions
        else:
            return x
    
    # @timing
    def construct_cross_edge(self,edge_index,batch_size,num_nodes1,num_nodes2,index):
        cindex = ('c0',batch_size,index)
        if cindex not in self.cache:
            edge_index=construct_cross_edge(edge_index,batch_size,num_nodes1,num_nodes2)
            self.cache[cindex]=edge_index
        else:
            edge_index=self.cache[cindex]
        return edge_index
    # @timing
    def construct_cross_edge_both(self,edge_index,batch_size,num_nodes1,num_nodes2,index):
        cindex = ('c1',batch_size,index)
        if cindex not in self.cache:
            edge_index=construct_cross_edge_both(edge_index,batch_size,num_nodes1,num_nodes2)
            self.cache[cindex]=edge_index
        else:
            edge_index=self.cache[cindex]
        return edge_index
    # @timing
    def construct_inner_edge(self,edge_index,batch_size,num_nodes,index):
        cindex = ('i',batch_size,index)
        if cindex not in self.cache:
            edge_index=construct_inner_edge(edge_index,batch_size,num_nodes)
            self.cache[cindex]=edge_index
        else:
            edge_index=self.cache[cindex]
        return edge_index

    # def lp_encode(self, data):
    #     x = data.x
    #     for i in range(self.num_layer - 1):
    #         x = self.convs[i](x, data.edge_index) # Jie
    #         if i != self.num_layer - 2:
    #             x = activate_func(x, self.args["act"])
    #             # x = F.dropout(x, p=self.args["dropout"], training=self.training)
    #     return x



@register_model("gat-pooling-model")
class AutoGATPooling(BaseAutoModel):
    r"""

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
        self, num_features=None, num_classes=None, device=None, **args
    ):
        super().__init__(num_features, num_classes, device, **args)
        self.hyper_parameter_space = [
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
                "maxValue": 0.6,
                "minValue": 0,
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
            "hidden": [16,16,16],
            "heads": 4,
            "dropout": 0,
            "act": "relu",
        }

    def _initialize(self):
        # """Initialize model."""
        self._model = GATPool({
            "features_num": self.input_dimension,
            "num_classes": self.output_dimension,
            "metadata":None,
            **self.hyper_parameters
        }).to(self.device)
