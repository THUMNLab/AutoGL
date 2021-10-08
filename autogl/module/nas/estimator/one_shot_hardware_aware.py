import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_estimator
from ..space import BaseSpace
from .base import BaseEstimator
from ..backend import *
from ...train.evaluation import Evaluation, Acc

# @register_nas_estimator("oneshot")
# class OneShotEstimator(BaseEstimator):
#     """
#     One shot estimator.

#     Use model directly to get estimations.
#     """

#     def infer(self, model: BaseSpace, dataset, mask="train"):
#         device = next(model.parameters()).device
#         dset = dataset[0].to(device)
#         pred = model(dset)[getattr(dset, f"{mask}_mask")]
#         y = dset.y[getattr(dset, f"{mask}_mask")]
#         loss = getattr(F, self.loss_f)(pred, y)
#         # acc=sum(pred.max(1)[1]==y).item()/y.size(0)
#         probs = F.softmax(pred, dim=1).detach().cpu().numpy()
#         y = y.cpu()
#         metrics = [eva.evaluate(probs, y) for eva in self.evaluation]
#         return metrics, loss

@register_nas_estimator("oneshot_hardware")
class OneShotEstimator_HardwareAware(BaseEstimator):
    """
    One shot estimator.

    Use model directly to get estimations.
    """
    def __init__(self, loss_f="nll_loss", evaluation=[Acc()], hardware_evaluation="parameter"):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation
        self.hardware_evaluation=hardware_evaluation

    def infer(self, model: BaseSpace, dataset, mask="train"):
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        mask=bk_mask(dset,mask)

        pred = model(dset)[mask]
        label=bk_label(dset)
        y = label[mask]

        loss = getattr(F, self.loss_f)(pred, y)
        # acc=sum(pred.max(1)[1]==y).item()/y.size(0)
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()
        y = y.cpu()
        model_info = model.get_model_info()
        # print(model_info)
        # print(self.hardware_evaluation)
        metrics = [eva.evaluate(probs, y) for eva in self.evaluation]
        metrics.append(model_info[self.hardware_evaluation]())
        return metrics, loss
