# codes in this file are reproduced from https://github.com/microsoft/nni with some changes.
import copy
from logging import Logger
from numpy.core.fromnumeric import sort

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_algo
from .base import BaseNAS
from ..space import BaseSpace
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
from ....utils import get_logger

import numpy as np
LOGGER = get_logger("SPOS")

import collections
import dataclasses
import random


@dataclasses.dataclass
class Individual:
    """
    A class that represents an individual.
    Holds two attributes, where ``x`` is the model and ``y`` is the metric (e.g., accuracy).
    """
    x: dict
    y: float


class Evolution:
    """
    Algorithm for regularized evolution (i.e. aging evolution).
    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image Classifier Architecture Search".
    
    Parameters
    ----------
    optimize_mode : str
        Can be one of "maximize" and "minimize". Default: maximize.
    population_size : int
        The number of individuals to keep in the population. Default: 100.
    cycles : int
        The number of cycles (trials) the algorithm should run for. Default: 20000.
    sample_size : int
        The number of individuals that should participate in each tournament. Default: 25.
    mutation_prob : float
        Probability that mutation happens in each dim. Default: 0.05
    """

    def __init__(self, optimize_mode='maximize', population_size=100, sample_size=25, cycles=20000,
                 mutation_prob=0.05,disable_progress=False):
        assert optimize_mode in ['maximize', 'minimize']
        assert sample_size < population_size
        self.optimize_mode = optimize_mode
        self.population_size = population_size
        self.sample_size = sample_size
        self.cycles = cycles
        self.mutation_prob = mutation_prob
        self.disable_progress= disable_progress
        self._worst = float('-inf') if self.optimize_mode == 'maximize' else float('inf')

        self._success_count = 0
        self._population = collections.deque()
        self._running_models = []
        self._polling_interval = 2.
        self._history = []

    def best_parent(self,sample_size=None):
        """get the config of the best parent 
        """
        samples = [p for p in self._population]  # copy population
        random.shuffle(samples)
        if sample_size is not None:
            samples = list(samples)[:sample_size]
        if self.optimize_mode == 'maximize':
            parent = max(samples, key=lambda sample: sample.y)
        else:
            parent = min(samples, key=lambda sample: sample.y)
        return parent.x

    def _prepare(self):
        self.uniform=UniformSampler(self.nas_modules)
        self.mutation=MutationSampler(self.nas_modules,self.mutation_prob)

    def _get_metric(self,config):
        for name, module in self.nas_modules:
            module.sampled = config[name]
        # todo: this may be computational expensive 
        # model=self.model.parse_model(config,self.device) 
        with torch.no_grad():
            metric, loss = self.estimator.infer(self.model, self.dataset, mask='val')
        return metric[0]

    def search(self, space: BaseSpace,nas_modules,dset,estimator,device):
        self.model = space
        self.dataset = dset 
        self.estimator = estimator
        self.nas_modules = nas_modules
        self.device = device

        self._prepare()
        LOGGER.info('Initializing the first population.')
        with tqdm(range(self.population_size), disable=self.disable_progress) as bar:
            for i in bar:
                config = self.uniform.resample()
                metric=self._get_metric(config)
                individual = Individual(config, metric)
                # LOGGER.debug('Individual created: %s', str(individual))
                self._population.append(individual)
                self._history.append(individual)
                bar.set_postfix(metric=metric,max=max(x.y for x in self._population),min=min(x.y for x in self._population))

        LOGGER.info('Running mutations.')
        with tqdm(range(self.cycles), disable=self.disable_progress) as bar:
            for i in bar:
                parent=self.best_parent(self.sample_size)
                config=self.mutation.resample(parent)
                metric=self._get_metric(config) # todo : add aging factor
                individual = Individual(config, metric)
                LOGGER.debug('Individual created: %s', str(individual))
                self._population.append(individual)
                self._history.append(individual)
                if len(self._population) > self.population_size:
                    self._population.popleft()
                bar.set_postfix(metric=metric,max_h=max(x.y for x in self._history),max=max(x.y for x in self._population),min=min(x.y for x in self._population))
        
        # todo: origin is best in history | or the population may need to be retrained
        self._history.sort(key=lambda x: x.y)
        # best=self.best_parent()
        if self.optimize_mode == 'maximize':
            best=self._history[-1].x
        else:
            best=self._history[0].x
        return best



class MutationSampler:
    """uniform mutator

    Parameters
    ----------
    nas_modules: 
        nas_modules in NAS algorithms , including choices of modules
    mutation_prob: float
        probability of doing mutation in each choice.
    parent : dict
        parent individual's choices
    """
    def __init__(self,nas_modules,mutation_prob):
        selection_range = {}
        for k, v in nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range
        self.mutation_prob = mutation_prob

    def resample(self, parent):
        search_space=self.selection_dict
        child = {}
        for k, v in parent.items():
            if random.uniform(0, 1) < self.mutation_prob:
                child[k] = np.random.choice(range(search_space[k])) # do not exclude the original operator
            else:
                child[k] = v
        return child

class UniformSampler:
    """Uniform Sampler

    Parameters
    ----------
    nas_modules: 
        nas_modules in NAS algorithms , including choices of modules
    """
    def __init__(self,nas_modules):
        selection_range = {}
        for k, v in nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range
    def resample(self):
        selection = {}
        for k, v in self.selection_dict.items():
            selection[k] = np.random.choice(range(v))
        return selection

@register_nas_algo("spos")
class Spos(BaseNAS):
    """
    SPOS trainer.

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
        super().__init__(device)
        self.model_lr=5e-3
        self.model_wd=5e-4
        self.n_warmup = n_warmup
        self.disable_progress= disable_progress
        self.grad_clip = grad_clip
        self.optimize_mode = optimize_mode
        self.population_size = population_size
        self.sample_size = sample_size
        self.cycles = cycles
        self.mutation_prob = mutation_prob

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
        self.controller=UniformSampler(self.nas_modules)

        # Evolution
        self.evolve = Evolution(
            optimize_mode='maximize', 
            population_size=self.population_size, 
            sample_size=self.sample_size, 
            cycles=self.cycles, 
            mutation_prob=self.mutation_prob,
            disable_progress=self.disable_progress
            )

    def search(self, space: BaseSpace, dset, estimator):
        self.model = space
        self.dataset = dset  
        self.estimator = estimator

        self._prepare()
        self._train() # train using uniform sampling
        self._search() # search using evolutionary algorithm

        selection = self.export()

        # here may sample N , retrain N ,and get best
        print(selection)
        return space.parse_model(selection, self.device)

    def _search(self):
        self.best_config=self.evolve.search(
            self.model,
            self.nas_modules,
            self.dataset,
            self.estimator,
            self.device,
            )


    def _train(self):
        with tqdm(range(self.n_warmup), disable=self.disable_progress) as bar:
            for i in bar:
                acc, l1 = self._train_one_epoch(i)
                with torch.no_grad():
                    val_acc, val_loss = self._infer("val")
                bar.set_postfix(loss=l1, acc=acc, val_acc=val_acc, val_loss=val_loss.item())

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.model_optim.zero_grad()
        self._resample() # uniform sampling
        metric, loss = self._infer(mask="train")
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.model_optim.step()
        return metric, loss.item()

    def _resample(self):
        result=self.controller.resample()
        for name, module in self.nas_modules:
            module.sampled = result[name]

    def export(self):
        return self.best_config

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.model, self.dataset, mask=mask)
        return metric[0], loss
