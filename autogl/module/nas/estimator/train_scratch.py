import torch.nn as nn
import torch.nn.functional as F

from ..space import BaseSpace
from .base import BaseEstimator
import torch

from autogl.module.train import NodeClassificationFullTrainer
class TrainEstimator(BaseEstimator):
    def __init__(self):
        self.estimator=OneShotEstimator()
    def infer(self,model: BaseSpace, dataset, mask="train"):
        # self.trainer.model=model
        # self.trainer.device=model.device
        self.trainer=NodeClassificationFullTrainer(
                    model=model,
                    optimizer=torch.optim.Adam,
                    lr=0.005,
                    max_epoch=300,
                    early_stopping_round=30,
                    weight_decay=5e-4,
                    device="auto",
                    init=False,
                    feval=['acc'],
                    loss="nll_loss",
                    lr_scheduler_type=None)
        self.trainer.train(dataset)
        with torch.no_grad():
            return self.estimator.infer(model,dataset,mask)
