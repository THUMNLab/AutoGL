import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize
import scipy.sparse as sp


from ..base import BaseAutoModel, activate_func
from .. import register_model
from .....utils import get_logger

LOGGER = get_logger("GNNGuard_GCN")

class GNNGuard_GCN(nn.Module):

    def __init__(self, args):
        super(GNNGuard_GCN, self).__init__()

        self.args = args
        self.num_layer = int(self.args["num_layers"])
        if 'attention' not in self.args:
            self.attention = True
        else:
            self.attention = self.args['attention']=='True'

        """GCN from geometric"""
        """network from torch-geometric, """
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(
                self.args["features_num"], 
                self.args["hidden"][0],
                dropout=self.args["dropout"]
                )
        )
        for i in range(self.num_layer-2):
            self.convs.append(
            GCNConv(
                self.args["hidden"][i], 
                self.args["hidden"][i+1],
                dropout=self.args["dropout"]
                )
            )
        self.convs.append(
            GCNConv(
                self.args["hidden"][-1],
                self.args["num_class"],
                dropout=self.args["dropout"],
            )
        )

        self.gate = nn.Parameter(torch.rand(self.num_layers-1))


    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        # add self-loop
        edge_index = add_remaining_self_loops(edge_index)

        # convert edge index to torch sparse matrix
        row, col = edge_index[0], edge_index[1]
        i = torch.stack((row, col), dim=0)
        v = torch.ones(len(row), dtype=torch.float32)
        adj = torch.sparse.FloatTensor(i, v, shape=(data.x.size(0),data.x.size(0))).to(data.x.device)

        for i in range(self.num_layer):
            
            if i==0:
                # if attention=True, use attention mechanism
                if self.attention:
                    adj = self.att_coef(x, adj, i=0)
                    adj_values = adj._values()

                    # if has edge weight: multiply edge_weight and adj_values
                    if data.edge_weight is not None:
                        adj_values = data.edge_weight*adj_values
            else:  
                if self.attention:
                    adj_2 = self.att_coef(x, adj, i=i)
                    adj_values = self.gate[i-1] * adj._values() + (1 - self.gate) * adj_2._values()

                    if data.edge_weight is not None:
                        adj_values = data.edge_weight*adj_values
                else:
                    adj_values = adj._values()

            x = self.convs[i](x, edge_index, edge_weight=adj_values)
            if i != self.num_layer - 1:
                x = activate_func(x, self.args["act"])
                x = F.dropout(x, p=self.args["dropout"], training=self.training)

        return F.log_softmax(x, dim=1)

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
        # row, col = edge_index[0], edge_index[1]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        # sim_matrix = torch.from_numpy(sim_matrix)
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0
        # print('dropped {} edges'.format(1-sim.nonzero()[0].shape[0]/len(sim)))

        # """use jaccard for binary features and cosine for numeric features"""
        # fea_start, fea_end = fea[edge_index[0]], fea[edge_index[1]]
        # isbinray = np.array_equal(fea_copy, fea_copy.astype(bool))  # check is the fea are binary
        # np.seterr(divide='ignore', invalid='ignore')
        # if isbinray:
        #     fea_start, fea_end = fea_start.T, fea_end.T
        #     sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
        # else:
        #     fea_copy[np.isinf(fea_copy)] = 0
        #     fea_copy[np.isnan(fea_copy)] = 0
        #     sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        #     sim = sim_matrix[edge_index[0], edge_index[1]]
        #     sim[sim < 0.01] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')


        """add learnable dropout, make character vector"""
        # if self.drop:
        #     character = np.vstack((att_dense_norm[row, col].A1,
        #                              att_dense_norm[col, row].A1))
        #     character = torch.from_numpy(character.T)
        #     drop_score = self.drop_learn_1(character)
        #     drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
        #     mm = torch.nn.Threshold(0.5, 0)
        #     drop_score = mm(drop_score)
        #     mm_2 = torch.nn.Threshold(-0.49, 1)
        #     drop_score = mm_2(-drop_score)
        #     drop_decision = drop_score.clone().requires_grad_()
        #     # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
        #     drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
        #     drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
        #     att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            # degree = degree.squeeze(-1).squeeze(-1)
            lam = 1 / (degree + 1) # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        att_adj = edge_index
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj



@register_model("GNNGuard_GCN-model")
class AutoGNNGuard_GCN(BaseAutoModel):
    r"""
    
    .. math::
        
    .. math::
        

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
            {
                "parameterName": "attention",
                "type": "CATEGORICAL",
                "feasiblePoints": ['True', 'False'],
            },

        ]

        self.hyper_parameters = {
            "num_layers": 2,
            "hidden": [32],
            "dropout": 0.2,
            "act": "leaky_relu",
            "attention": "True"
        }

    def _initialize(self):
        # """Initialize model."""
        self._model = GNNGuard_GCN({
            "features_num": self.input_dimension,
            "num_class": self.output_dimension,
            **self.hyper_parameters
        }).to(self.device)
