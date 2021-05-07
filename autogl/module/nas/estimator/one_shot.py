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
