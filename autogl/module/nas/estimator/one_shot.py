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
    """
    def __init__(self, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation

    def infer(self, model: BaseSpace, dataset, mask="train"):
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        mask=bk_mask(dset,mask)

        pred = model(dset)[mask]
        label=bk_label(dset)
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

    Use model directly to get estimations.
    """
    def __init__(self, loss_f="nll_loss", evaluation=[Acc()], hardware_evaluation="parameter"):
        super().__init__(loss_f, evaluation)
        self.hardware_evaluation = hardware_evaluation

    def infer(self, model: BaseSpace, dataset, mask="train"):
        metrics, loss = super().infer(model, dataset, mask)
        if isinstance(self.hardware_evaluation, str):
            metrics.append(get_hardware_aware_metric(model, self.hardware_evaluation))
        else:
            metrics.append(self.hardware_evaluation(model))
        return metrics, loss
