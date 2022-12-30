# "Adversarially Robust Neural Architecture Search for Graph Neural Networks"

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
from ...model.base import BaseAutoModel
from .spos import Evolution, UniformSampler, Spos
from ..utils import (
    AverageMeterGroup,
    replace_layer_choice,
    replace_input_choice,
    get_module_order,
    sort_replaced_module,
    PathSamplingLayerChoice,
    PathSamplingInputChoice,
)


from tqdm import tqdm, trange
from torch.autograd import Variable
import numpy as np
import time
import copy
import torch.optim as optim
import scipy.sparse as sp


_logger = logging.getLogger(__name__)


@register_nas_algo("grna")
class GRNA(Spos):
    """
    GRNA trainer.

    Parameters
    ----------
    n_warmup : int
        Number of epochs for training super network.
    model_lr : float
        Learning rate for super network.
    model_wd : float
        Weight decay for super network.
    
    Other parameters see Evolution
    """

    def __init__(
        self,
        n_warmup=1000,
        grad_clip=5.0,
        disable_progress=False,
        optimize_mode='maximize', 
        population_size=100, 
        sample_size=25, 
        cycles=20000,
        mutation_prob=0.05,
        device="cuda",
    ):
        super().__init__(n_warmup,
        grad_clip,
        disable_progress,
        optimize_mode, 
        population_size, 
        sample_size, 
        cycles,
        mutation_prob,
        device)

    def _prepare(self):
        # replace choice
        self.nas_modules = []
        k2o = get_module_order(self.model)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)
        # to device
        self.model = self.model.to(self.device)
        self.model_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.model_lr, weight_decay=self.model_wd
        )
        # controller
        self.controller = UniformSampler(self.nas_modules)

        # Evolution
        self.evolve = Evolution(
            optimize_mode='maximize', 
            population_size=self.population_size, 
            sample_size=self.sample_size, 
            cycles=self.cycles, 
            mutation_prob=self.mutation_prob,
            disable_progress=self.disable_progress
            )

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.model, self.dataset, mask=mask)
        return metric[0], loss
