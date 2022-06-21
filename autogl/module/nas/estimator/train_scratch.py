from . import register_nas_estimator
from ..space import BaseSpace
from .base import BaseEstimator
from .one_shot import OneShotEstimator, GCLOneShotEstimator2, OneShotEstimator_HardwareAware
import torch

from autogl.module.train import GraphClassificationFullTrainer, GraphClassificationFullTrainer_PPI, Acc


@register_nas_estimator("scratch")
class TrainEstimator(BaseEstimator):
    """
    An estimator which trans from scratch

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    """

    def __init__(self, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation
        self.estimator = OneShotEstimator(self.loss_f, self.evaluation)

    def infer(self, model: BaseSpace, dataset, mask="train"):
        boxmodel = model.wrap()
        self.trainer = GraphClassificationFullTrainer(
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

@register_nas_estimator("gclscratch2")
class GCLTrainEstimator2(BaseEstimator):
    """
    An estimator which trans from scratch

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    """

    def __init__(self, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.loss_f = loss_f
        self.evaluation = evaluation
        self.estimator = GCLOneShotEstimator2(self.loss_f, self.evaluation)

    def infer(self, model: BaseSpace, dataset, mask="train", batch_size=64, max_epoch=100, lr=1e-4,device="auto"):
        boxmodel = model.wrap()
        self.trainer = GraphClassificationFullTrainer_PPI(
            model=boxmodel,
            optimizer=torch.optim.Adam,
            lr=lr,
            max_epoch=max_epoch,
            batch_size=batch_size,
            early_stopping_round=30,
            weight_decay=5e-4,
            device=device,
            init=False,
            feval=self.evaluation,
            loss=self.loss_f,
            lr_scheduler_type=None,
        )
        try:
            self.trainer.train(dataset, batch_size=batch_size)
            with torch.no_grad():
                return self.estimator.infer(boxmodel.model, dataset, mask, batch_size)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                INF = 100
                fin = [-INF if eva.is_higher_better else INF for eva in self.evaluation]
                return fin, 0
            else:
                raise e


@register_nas_estimator("scratch_hardware")
class TrainEstimator_HardwareAware(TrainEstimator):
    """
    An hardware-aware estimator which trans from scratch
    """

    def __init__(
        self,
        loss_f="nll_loss",
        evaluation=[Acc()],
        hardware_evaluation="parameter",
        hardware_metric_weight=0,
    ):
        super().__init__(loss_f, evaluation)
        self.estimator = OneShotEstimator_HardwareAware(
            self.loss_f, self.evaluation, hardware_evaluation, hardware_metric_weight
        )
