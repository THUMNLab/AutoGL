# Modified from NNI

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nas import BaseNAS
from .utils import AverageMeterGroup, replace_layer_choice, replace_input_choice


_logger = logging.getLogger(__name__)


class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.key
        self.op_choices = nn.ModuleDict(layer_choice.named_children())
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

    def forward(self, *args, **kwargs):
        op_results = torch.stack(
            [op(*args, **kwargs) for op in self.op_choices.values()]
        )
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == "alpha":
                continue
            yield name, p

    def export(self):
        return torch.argmax(self.alpha).item()


class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.key
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == "alpha":
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[: self.n_chosen]


class DartsTrainer(BaseNAS):
    """
    DARTS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    grad_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    learning_rate : float
        Learning rate to optimize the model.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    arc_learning_rate : float
        Learning rate of architecture parameters.
    unrolled : float
        ``True`` if using second order optimization, else first order optimization.
    """

    """def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False):"""

    def __init__(self, *args, **kwargs):
        self.num_epochs = kwargs.get("num_epochs", 5)
        self.workers = 4
        self.device = "cuda"
        self.log_frequency = None

        # for _, module in self.nas_modules:
        #    module.to(self.device)

        # use the same architecture weight for modules with duplicated names

    def search(self, space, dset, trainer):
        """
        main process
        """
        self.model = space
        self.dataset = dset
        self.trainer = trainer
        self.model_optim = torch.optim.SGD(
            self.model.parameters(), lr=0.01, weight_decay=3e-4
        )

        self.nas_modules = []
        replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
        replace_input_choice(self.model, DartsInputChoice, self.nas_modules)

        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert (
                    m.alpha.size() == ctrl_params[m.name].size()
                ), "Size of parameters with the same label should be same."
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        self.ctrl_optim = torch.optim.Adam(
            list(ctrl_params.values()), 3e-4, betas=(0.5, 0.999), weight_decay=1.0e-3
        )
        self.grad_clip = 5.0

        for step in range(self.num_epochs):
            self._train_one_epoch(step)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info(
                    "Epoch [%s/%s] Step [%s/%s]  %s",
                    epoch + 1,
                    self.num_epochs,
                    step + 1,
                    len(self.train_loader),
                    meters,
                )

        return self.export()

    def _train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()

        # phase 1. architecture step
        self.ctrl_optim.zero_grad()
        # only no unroll here
        _, loss = self._infer()
        loss.backward()
        self.ctrl_optim.step()

        # phase 2: child network step
        self.model_optim.zero_grad()
        metric, loss = self._infer()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )  # gradient clipping
        self.model_optim.step()

    def _infer(self):
        metric, loss = self.trainer.infer(self.model, self.dataset)
        return metric, loss

    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
