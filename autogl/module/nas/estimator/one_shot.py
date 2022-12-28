import torch
from deeprobust.graph.global_attack import Random,DICE
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import time
import torch.nn.functional as F

from . import register_nas_estimator
from ..space import BaseSpace
from .base import BaseEstimator
from ..backend import *
from ...train.evaluation import Acc
from ..utils import get_hardware_aware_metric

@register_nas_estimator("oneshot")
class OneShotEstimator(BaseEstimator):
    """
    One shot estimator.

    Use model directly to get estimations.

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

    def infer(self, model: BaseSpace, dataset, mask="train"):
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        mask = bk_mask(dset, mask)

        pred = model(dset)[mask]
        label = bk_label(dset)
        y = label[mask]

        loss = getattr(F, self.loss_f)(pred, y)
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()
        
        y = y.cpu()
        metrics = [eva.evaluate(probs, y) for eva in self.evaluation]
        return metrics, loss


@register_nas_estimator("oneshot_hardware")
class OneShotEstimator_HardwareAware(OneShotEstimator):
    """
    One shot hardware-aware estimator.

    Use model directly to get estimations with some hardware-aware metrics.

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    hardware_evaluation : str or runable
        The hardware-aware metrics. Can be 'parameter' or 'latency'. Or you can define a special metric by a runable function
    hardware_metric_weight : float
        The weight of hardware-aware metric, which will be a bias added to metrics
    """

    def __init__(
        self,
        loss_f="nll_loss",
        evaluation=[Acc()],
        hardware_evaluation="parameter",
        hardware_metric_weight=0,
    ):
        super().__init__(loss_f, evaluation)
        self.hardware_evaluation = hardware_evaluation
        self.hardware_metric_weight = hardware_metric_weight

    def infer(self, model: BaseSpace, dataset, mask="train"):
        metrics, loss = super().infer(model, dataset, mask)
        if isinstance(self.hardware_evaluation, str):
            hardware_metric = get_hardware_aware_metric(model, self.hardware_evaluation)
        else:
            hardware_metric = self.hardware_evaluation(model)
        metrics = [x - hardware_metric * self.hardware_metric_weight for x in metrics]
        metrics.append(hardware_metric)
        return metrics, loss


