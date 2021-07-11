# Modified from NNI

import logging

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_algo
from .base import BaseNAS
from ..estimator.base import BaseEstimator
from ..space import BaseSpace
from ..utils import replace_layer_choice, replace_input_choice
from ...model.base import BaseModel
from nni.retiarii.oneshot.pytorch.darts import DartsLayerChoice, DartsInputChoice

_logger = logging.getLogger(__name__)


@register_nas_algo("darts")
class Darts(BaseNAS):
    """
    DARTS trainer.

    Parameters
    ----------
    num_epochs : int
        Number of epochs planned for training.
    workers : int
        Workers for data loading.
    gradient_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    model_lr : float
        Learning rate to optimize the model.
    model_wd : float
        Weight decay to optimize the model.
    arch_lr : float
        Learning rate to optimize the architecture.
    arch_wd : float
        Weight decay to optimize the architecture.
    device : str or torch.device
        The device of the whole process
    """

    def __init__(
        self,
        num_epochs=5,
        workers=4,
        gradient_clip=5.0,
        model_lr=1e-3,
        model_wd=5e-4,
        arch_lr=3e-4,
        arch_wd=1e-3,
        device="cuda",
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.workers = workers
        self.gradient_clip = gradient_clip
        self.model_optimizer = torch.optim.Adam
        self.arch_optimizer = torch.optim.Adam
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.arch_lr = arch_lr
        self.arch_wd = arch_wd

    def search(self, space: BaseSpace, dataset, estimator):
        model_optim = self.model_optimizer(
            space.parameters(), self.model_lr, weight_decay=self.model_wd
        )

        nas_modules = []
        replace_layer_choice(space, DartsLayerChoice, nas_modules)
        replace_input_choice(space, DartsInputChoice, nas_modules)
        space = space.to(self.device)

        ctrl_params = {}
        for _, m in nas_modules:
            if m.name in ctrl_params:
                assert (
                    m.alpha.size() == ctrl_params[m.name].size()
                ), "Size of parameters with the same label should be same."
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        arch_optim = self.arch_optimizer(
            list(ctrl_params.values()), self.arch_lr, weight_decay=self.arch_wd
        )

        for epoch in range(self.num_epochs):
            self._train_one_epoch(
                epoch, space, dataset, estimator, model_optim, arch_optim
            )

        selection = self.export(nas_modules)
        return space.parse_model(selection, self.device)

    def _train_one_epoch(
        self,
        epoch,
        model: BaseSpace,
        dataset,
        estimator,
        model_optim: torch.optim.Optimizer,
        arch_optim: torch.optim.Optimizer,
    ):
        model.train()

        # phase 1. architecture step
        arch_optim.zero_grad()
        # only no unroll here
        _, loss = self._infer(model, dataset, estimator, "val")
        loss.backward()
        arch_optim.step()

        # phase 2: child network step
        model_optim.zero_grad()
        metric, loss = self._infer(model, dataset, estimator, "train")
        loss.backward()
        # gradient clipping
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        model_optim.step()

    def _infer(self, model: BaseSpace, dataset, estimator: BaseEstimator, mask="train"):
        metric, loss = estimator.infer(model, dataset, mask=mask)
        return metric, loss

    @torch.no_grad()
    def export(self, nas_modules) -> dict:
        result = dict()
        for name, module in nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
