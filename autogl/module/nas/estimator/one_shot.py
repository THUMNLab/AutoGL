import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_estimator
from ..space import BaseSpace
from .base import BaseEstimator


@register_nas_estimator("oneshot")
class OneShotEstimator(BaseEstimator):
    """
    One shot estimator.

    Use model directly to get estimations.
    """

    def infer(self, model: BaseSpace, dataset, mask="train"):
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        pred = model(dset)[getattr(dset, f"{mask}_mask")]
        y = dset.y[getattr(dset, f"{mask}_mask")]
        loss = getattr(F, self.loss_f)(pred, y)
        # acc=sum(pred.max(1)[1]==y).item()/y.size(0)
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()
        y = y.cpu()
        metrics = [eva.evaluate(probs, y) for eva in self.evaluation]
        return metrics, loss
