import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_estimator
from ..space import BaseSpace
from .base import BaseEstimator
from .one_shot import OneShotEstimator
import torch

from autogl.module.train import NodeClassificationFullTrainer

@register_nas_estimator("scratch")
class TrainEstimator(BaseEstimator):
    def __init__(self):
        self.estimator=OneShotEstimator()

    def infer(self, model: BaseSpace, dataset, mask="train"):
        # self.trainer.model=model
        # self.trainer.device=model.device
        boxmodel = model.wrap()
        self.trainer=NodeClassificationFullTrainer(
                    model=boxmodel,
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
            return self.estimator.infer(boxmodel.model, dataset, mask)
