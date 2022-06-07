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
)
from tqdm import tqdm
from .rl import PathSamplingLayerChoice, PathSamplingInputChoice
import numpy as np
from ....utils import get_logger

LOGGER = get_logger("random_search_NAS")


@register_nas_algo("random")
class RandomSearch(BaseNAS):
    """
    Uniformly random architecture search

    Parameters
    ----------
    device : str or torch.device
        The device of the whole process, e.g. "cuda", torch.device("cpu")
    num_epochs : int
        Number of epochs planned for training.
    disable_progeress: boolean
        Control whether show the progress bar.
    """

    def __init__(self, device="auto", num_epochs=400, disable_progress=False, hardware_metric_limit=None):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.disable_progress = disable_progress
        self.hardware_metric_limit = hardware_metric_limit

    def search(self, space: BaseSpace, dset, estimator):
        self.estimator = estimator
        self.dataset = dset
        self.space = space

        self.nas_modules = []
        k2o = get_module_order(self.space)
        replace_layer_choice(self.space, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.space, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)
        selection_range = {}
        for k, v in self.nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range

        # space_size=np.prod(list(selection_range.values()))

        arch_perfs = []
        cache = {}
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                selection = self.sample()
                vec = tuple(list(selection.values()))
                if vec not in cache:
                    self.arch = space.parse_model(selection, self.device)
                    metric, loss, hardware_metric = self._infer(mask="val")
                    if self.hardware_metric_limit is None or hardware_metric[0] < self.hardware_metric_limit:
                        arch_perfs.append([metric, selection])
                    cache[vec] = metric
                bar.set_postfix(acc=metric, max_acc=max(cache.values()))
        selection = arch_perfs[np.argmax([x[0] for x in arch_perfs])][1]
        arch = space.parse_model(selection, self.device)
        return arch

    def sample(self):
        # uniformly sample
        selection = {}
        for k, v in self.selection_dict.items():
            selection[k] = np.random.choice(range(v))
        return selection

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.arch._model, self.dataset, mask=mask)
        return metric[0], loss, metric[1:]

@register_nas_algo("gclrandom")
class GCLRandomSearch(BaseNAS):
    """
    Uniformly random architecture search

    Parameters
    ----------
    device : str or torch.device
        The device of the whole process, e.g. "cuda", torch.device("cpu")
    num_epochs : int
        Number of epochs planned for training.
    disable_progeress: boolean
        Control whether show the progress bar.
    """

    def __init__(self, device="auto", num_epochs=400, disable_progress=False, hardware_metric_limit=None):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.disable_progress = disable_progress
        self.hardware_metric_limit = hardware_metric_limit

    def search(self, space: BaseSpace, dset, estimator):
        self.estimator = estimator
        self.dataset = dset
        self.space = space
        # print("random_search_0")
        self.nas_modules = []
        k2o = get_module_order(self.space)
        replace_layer_choice(self.space, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.space, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)
        selection_range = {}
        for k, v in self.nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range

        # space_size=np.prod(list(selection_range.values()))

        arch_perfs = []
        cache = {}
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                # print("random_search_1")
                selection = self.sample()
                vec = tuple(list(selection.values()))
                if vec not in cache:
                    self.arch = space.parse_model(selection, self.device)
                    # print("random_search_2")
                    metric, loss = self._infer(mask="test")
                    arch_perfs.append([metric, selection])
                    cache[vec] = metric
                bar.set_postfix(acc=metric, max_acc=max(cache.values()))
        selection = arch_perfs[np.argmax([x[0] for x in arch_perfs])][1]
        arch = space.parse_model(selection, self.device)
        return arch

    def sample(self):
        # uniformly sample
        selection = {}
        for k, v in self.selection_dict.items():
            selection[k] = np.random.choice(range(v))
        return selection

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.arch._model, self.dataset, mask=mask)
        return metric, loss

@register_nas_algo("gclrandom2")
class GCLRandomSearch2(BaseNAS):
    """
    Uniformly random architecture search

    Parameters
    ----------
    device : str or torch.device
        The device of the whole process, e.g. "cuda", torch.device("cpu")
    num_epochs : int
        Number of epochs planned for training.
    disable_progeress: boolean
        Control whether show the progress bar.
    """

    def __init__(self, batch, device="auto", num_epochs=400, disable_progress=False, hardware_metric_limit=None):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.batch = batch
        self.disable_progress = disable_progress
        self.hardware_metric_limit = hardware_metric_limit

    def search(self, space: BaseSpace, dset, estimator):
        self.estimator = estimator
        self.dataset = dset
        # self.meta_data = meta_data
        self.space = space
        # print("random_search_0")
        self.nas_modules = []
        k2o = get_module_order(self.space)
        replace_layer_choice(self.space, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.space, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)
        selection_range = {}
        for k, v in self.nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range

        # space_size=np.prod(list(selection_range.values()))

        arch_perfs = []
        cache = {}
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                selection = self.sample()
                vec = tuple(list(selection.values()))
                if vec not in cache:
                    self.arch = space.parse_model(selection, self.device)
                    metric, loss = self._infer(mask="test")
                    arch_perfs.append([metric, selection])
                    cache[vec] = metric
                bar.set_postfix(acc=metric, max_acc=max(cache.values()))
        selection = arch_perfs[np.argmax([x[0] for x in arch_perfs])][1]
        arch = space.parse_model(selection, self.device)
        return arch

    def sample(self):
        # uniformly sample
        selection = {}
        for k, v in self.selection_dict.items():
            selection[k] = np.random.choice(range(v))
        return selection

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.arch._model, self.dataset, mask=mask)
        return metric, loss
