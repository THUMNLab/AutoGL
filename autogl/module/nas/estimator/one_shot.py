import torch.nn as nn
import torch.nn.functional as F

from ..space import BaseSpace
from .base import BaseEstimator


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
        return loss, loss
