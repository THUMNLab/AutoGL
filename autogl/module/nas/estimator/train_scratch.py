import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_estimator
from ..space import BaseSpace
from .base import BaseEstimator
from .one_shot import OneShotEstimator
import torch

from autogl.module.train import NodeClassificationFullTrainer, Acc


@register_nas_estimator("scratch")
class TrainEstimator(BaseEstimator):
    """
    An estimator which trans from scratch
    """

    def __init__(self, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation
        self.estimator = OneShotEstimator(self.loss_f, self.evaluation)

    def infer(self, model: BaseSpace, dataset, mask="train"):
        # self.trainer.model=model
        # self.trainer.device=model.device
        boxmodel = model.wrap()
        self.trainer = NodeClassificationFullTrainer(
            model=boxmodel,
            optimizer=torch.optim.Adam,
            lr=0.005,
            max_epoch=300,
            early_stopping_round=30,
            weight_decay=5e-4,
            device="auto",
            init=False,
            feval=self.evaluation,
            loss=self.loss_f,
            lr_scheduler_type=None,
        )
        try:
            self.trainer.train(dataset)
            with torch.no_grad():
                return self.estimator.infer(boxmodel.model, dataset, mask)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                INF = 100
                fin = [-INF if eva.is_higher_better else INF for eva in self.evaluation]
                return fin, 0
            else:
                raise e
