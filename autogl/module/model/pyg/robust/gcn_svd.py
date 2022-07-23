from tkinter import TRUE
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import typing as _typing
import math
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from numba import njit

from .. import register_model
from . import utils
from ..gcn import GCN
from ..base import BaseAutoModel
from .....utils import get_logger

LOGGER = get_logger("GCNSVDModel")


### ========================== ###

class GCN4Robust(GCN):
    # 在已有gcn的基础上增加robust的部分
    def __init__(self, nfeat, nclass, nhid, activation, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, add_self_loops = True, normalize = TRUE):
        super(GCN4Robust, self).__init__(nfeat, nclass, nhid, activation, dropout=dropout, add_self_loops = add_self_loops, normalize = normalize)
    
    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

class GCNSVD(GCN4Robust):
    def __init__(self, nfeat, nclass, nhid, activation, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, add_self_loops = True, normalize = True):
        super(GCNSVD, self).__init__(nfeat, nclass, nhid, activation, dropout, lr, weight_decay, with_relu, with_bias, add_self_loops, normalize)

    def fit(self, features, adj, labels, idx_train, idx_val=None, k=50, train_iters=200, initialize=True, verbose=True, **kwargs):
        modified_adj = self.truncatedSVD(adj, k=k)
        self.k = k
        # modified_adj_tensor = utils.sparse_mx_to_torch_sparse_tensor(self.modified_adj)
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)

        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters, initialize=initialize, verbose=verbose)

    def truncatedSVD(self, data, k=50):
        print('=== GCN-SVD: rank={} ==='.format(k))
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            print("rank_after = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            print("rank_before = {}".format(len(S.nonzero()[0])))
            diag_S = np.diag(S)
            print("rank_after = {}".format(len(diag_S.nonzero()[0])))

        return U @ diag_S @ V

    def predict(self, features=None, adj=None):

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            adj = self.truncatedSVD(adj, k=self.k)
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

@register_model("gcnsvd-model")
class AutoGCNSVD(BaseAutoModel):
    def __init__(
        self,
        num_features: int = ...,
        num_classes: int = ...,
        device: _typing.Union[str, torch.device] = ...,
        **kwargs
    ) -> None:
        super().__init__(num_features, num_classes, device, **kwargs)
        self.hyper_parameter_space = [
            {
                "parameterName": "add_self_loops",
                "type": "CATEGORICAL",
                "feasiblePoints": [1],
            },
            {
                "parameterName": "normalize",
                "type": "CATEGORICAL",
                "feasiblePoints": [1],
            },
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
        ]

        self.hyper_parameters = {
            "num_layers": 3,
            "hidden": [128, 64],
            "dropout": 0,
            "act": "relu",
        }

    def _initialize(self):
        self._model = GCNSVD(
            self.input_dimension,
            self.output_dimension,
            self.hyper_parameters.get("hidden"),
            self.hyper_parameters.get("act"),
            self.hyper_parameters.get("dropout", None),
            bool(self.hyper_parameters.get("add_self_loops", True)),
            bool(self.hyper_parameters.get("normalize", True)),
        ).to(self.device)
