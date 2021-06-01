import torch.nn as nn
import torch.nn.functional as F

from ..space import BaseSpace
from .base import BaseEstimator
import torch

class OneShotEstimator(BaseEstimator):
    """
    One shot estimator.

    Use model directly to get estimations.
    """

    def infer(self, model: BaseSpace, dataset, mask="train"):
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        pred = model(dset)[getattr(dset, f"{mask}_mask")]
        y = dset.y[getattr(dset, f'{mask}_mask')]
        loss = F.nll_loss(pred, y)
        acc=sum(pred.max(1)[1]==y).item()/y.size(0)
        return acc, loss

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
                    lr=0.01,
                    max_epoch=200,
                    early_stopping_round=200,
                    weight_decay=5e-4,
                    device="auto",
                    init=False,
                    feval=['acc'],
                    loss="nll_loss",
                    lr_scheduler_type=None)
        self.trainer.train(dataset)
        with torch.no_grad():
            return self.estimator.infer(model,dataset,mask)
