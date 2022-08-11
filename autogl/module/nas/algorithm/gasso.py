# "Graph differentiable architecture search with structure optimization" NeurIPS 21'

import logging

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_algo
from .base import BaseNAS
from ..estimator.base import BaseEstimator
from ..space import BaseSpace
from ..utils import replace_layer_choice, replace_input_choice
from ...model.base import BaseAutoModel

from torch.autograd import Variable
import numpy as np
import time
import copy
import torch.optim as optim
import scipy.sparse as sp

_logger = logging.getLogger(__name__)

@register_nas_algo("gasso")
class Gasso(BaseNAS):
    """
    GASSO trainer.

    Parameters
    ----------
    num_epochs : int
        Number of epochs planned for training.
    warmup_epochs : int
        Number of epochs planned for warming up.
    workers : int
        Workers for data loading.
    model_lr : float
        Learning rate to optimize the model.
    model_wd : float
        Weight decay to optimize the model.
    arch_lr : float
        Learning rate to optimize the architecture.
    stru_lr : float
        Learning rate to optimize the structure.
    lamb : float
        The parameter to control the influence of hidden feature smoothness
    device : str or torch.device
        The device of the whole process
    """
    def __init__(
        self,
        num_epochs=250,
        warmup_epochs=10,
        model_lr=0.01,
        model_wd=1e-4,
        arch_lr = 0.03,
        stru_lr = 0.04,
        lamb = 0.6,
        device="auto",
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.arch_lr = arch_lr
        self.stru_lr = stru_lr
        self.lamb = lamb

    def train_stru(self, model, optimizer, data):
        # forward
        model.train()
        data[0].adj = self.adjs
        logits = model(data[0]).detach()
        loss = 0
        for adj in self.adjs:
            e1 = adj[0][0]
            e2 = adj[0][1]
            ew = adj[1]
            diff = (logits[e1] - logits[e2]).pow(2).sum(1)
            smooth = (diff * torch.sigmoid(ew)).sum()
            dist = (ew * ew).sum()
            loss += self.lamb * smooth + dist
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        del logits

    def _infer(self, model: BaseSpace, dataset, estimator: BaseEstimator, mask="train"):
        dataset[0].adj = self.adjs
        metric, loss = estimator.infer(model, dataset, mask=mask)
        return metric, loss

    def prepare(self, dset):
        """Train Pro-GNN.
        """
        data = dset[0]
        self.ews = []
        self.edges = data.edge_index.to(self.device)
        edge_weight = torch.ones(self.edges.size(1)).to(self.device)

        self.adjs = []
        for i in range(self.steps):
            edge_weight = Variable(edge_weight * 1.0, requires_grad = True).to(self.device)
            self.ews.append(edge_weight)
            self.adjs.append((self.edges, edge_weight))

    def fit(self, data):
        self.optimizer = optim.Adam(self.space.parameters(), lr=self.model_lr, weight_decay=self.model_wd)
        self.arch_optimizer = optim.Adam(self.space.arch_parameters(),
                                              lr=self.arch_lr, betas=(0.5, 0.999))
        self.stru_optimizer = optim.SGD(self.ews, lr=self.stru_lr)

        # Train model
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")

        t_total = time.time()
        for epoch in range(self.num_epochs):
            self.space.train()
            self.optimizer.zero_grad()
            _, loss = self._infer(self.space, data, self.estimator, "train")
            loss.backward()
            self.optimizer.step()

            if epoch <20:
                continue
            self.train_stru(self.space, self.stru_optimizer, data)
            
            self.arch_optimizer.zero_grad()
            _, loss = self._infer(self.space, data, self.estimator, "train")
            loss.backward()
            self.arch_optimizer.step()

            self.space.eval()
            train_acc, _ = self._infer(self.space, data, self.estimator, "train")
            val_acc, val_loss = self._infer(self.space, data, self.estimator, "val")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_performance = val_acc
                self.space.keep_prediction()
            #print("acc:" + str(train_acc) + " val_acc" + str(val_acc))

        return best_performance, min_val_loss

    def search(self, space: BaseSpace, dataset, estimator):
        self.estimator = estimator
        self.space = space.to(self.device)
        self.steps = space.steps
        self.prepare(dataset)
        perf, val_loss = self.fit(dataset)
        return space.parse_model(None, self.device)