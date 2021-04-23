# codes in this file are reproduced from https://github.com/microsoft/nni with some changes.
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNAS
from ..space import BaseSpace
from ..utils import AverageMeterGroup, replace_layer_choice, replace_input_choice
from nni.nas.pytorch.fixed import apply_fixed_architecture
_logger = logging.getLogger(__name__)
def _get_mask(sampled, total):
    multihot = [i == sampled or (isinstance(sampled, list) and i in sampled) for i in range(total)]
    return torch.tensor(multihot, dtype=torch.bool)  # pylint: disable=not-callable

class PathSamplingLayerChoice(nn.Module):
    """
    Mixed module, in which fprop is decided by exactly one or multiple (sampled) module.
    If multiple module is selected, the result will be sumed and returned.

    Attributes
    ----------
    sampled : int or list of int
        Sampled module indices.
    mask : tensor
        A multi-hot bool 1D-tensor representing the sampled mask.
    """

    def __init__(self, layer_choice):
        super(PathSamplingLayerChoice, self).__init__()
        self.op_names = []
        for name, module in layer_choice.named_children():
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, 'There has to be at least one op to choose from.'
        self.sampled = None  # sampled can be either a list of indices or an index

    def forward(self, *args, **kwargs):
        assert self.sampled is not None, 'At least one path needs to be sampled before fprop.'
        if isinstance(self.sampled, list):
            return sum([getattr(self, self.op_names[i])(*args, **kwargs) for i in self.sampled])  # pylint: disable=not-an-iterable
        else:
            return getattr(self, self.op_names[self.sampled])(*args, **kwargs)  # pylint: disable=invalid-sequence-index

    def __len__(self):
        return len(self.op_names)

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))


class PathSamplingInputChoice(nn.Module):
    """
    Mixed input. Take a list of tensor as input, select some of them and return the sum.

    Attributes
    ----------
    sampled : int or list of int
        Sampled module indices.
    mask : tensor
        A multi-hot bool 1D-tensor representing the sampled mask.
    """

    def __init__(self, input_choice):
        super(PathSamplingInputChoice, self).__init__()
        self.n_candidates = input_choice.n_candidates
        self.n_chosen = input_choice.n_chosen
        self.sampled = None

    def forward(self, input_tensors):
        if isinstance(self.sampled, list):
            return sum([input_tensors[t] for t in self.sampled])  # pylint: disable=not-an-iterable
        else:
            return input_tensors[self.sampled]

    def __len__(self):
        return self.n_candidates

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))


class StackedLSTMCell(nn.Module):
    def __init__(self, layers, size, bias):
        super().__init__()
        self.lstm_num_layers = layers
        self.lstm_modules = nn.ModuleList([nn.LSTMCell(size, size, bias=bias)
                                           for _ in range(self.lstm_num_layers)])

    def forward(self, inputs, hidden):
        prev_h, prev_c = hidden
        next_h, next_c = [], []
        for i, m in enumerate(self.lstm_modules):
            curr_h, curr_c = m(inputs, (prev_h[i], prev_c[i]))
            next_c.append(curr_c)
            next_h.append(curr_h)
            # current implementation only supports batch size equals 1,
            # but the algorithm does not necessarily have this limitation
            inputs = curr_h[-1].view(1, -1)
        return next_h, next_c


class ReinforceField:
    """
    A field with ``name``, with ``total`` choices. ``choose_one`` is true if one and only one is meant to be
    selected. Otherwise, any number of choices can be chosen.
    """

    def __init__(self, name, total, choose_one):
        self.name = name
        self.total = total
        self.choose_one = choose_one

    def __repr__(self):
        return f'ReinforceField(name={self.name}, total={self.total}, choose_one={self.choose_one})'


class ReinforceController(nn.Module):
    """
    A controller that mutates the graph with RL.

    Parameters
    ----------
    fields : list of ReinforceField
        List of fields to choose.
    lstm_size : int
        Controller LSTM hidden units.
    lstm_num_layers : int
        Number of layers for stacked LSTM.
    tanh_constant : float
        Logits will be equal to ``tanh_constant * tanh(logits)``. Don't use ``tanh`` if this value is ``None``.
    skip_target : float
        Target probability that skipconnect will appear.
    temperature : float
        Temperature constant that divides the logits.
    entropy_reduction : str
        Can be one of ``sum`` and ``mean``. How the entropy of multi-input-choice is reduced.
    """

    def __init__(self, fields, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5,
                 skip_target=0.4, temperature=None, entropy_reduction='sum'):
        super(ReinforceController, self).__init__()
        self.fields = fields
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.skip_target = skip_target

        self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
        self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]),  # pylint: disable=not-callable
                                         requires_grad=False)
        assert entropy_reduction in ['sum', 'mean'], 'Entropy reduction must be one of sum and mean.'
        self.entropy_reduction = torch.sum if entropy_reduction == 'sum' else torch.mean
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.soft = nn.ModuleDict({
            field.name: nn.Linear(self.lstm_size, field.total, bias=False) for field in fields
        })
        self.embedding = nn.ModuleDict({
            field.name: nn.Embedding(field.total, self.lstm_size) for field in fields
        })

    def resample(self):
        self._initialize()
        result = dict()
        for field in self.fields:
            result[field.name] = self._sample_single(field)
        return result

    def _initialize(self):
        self._inputs = self.g_emb.data
        self._c = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self._h = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self.sample_log_prob = 0
        self.sample_entropy = 0
        self.sample_skip_penalty = 0

    def _lstm_next_step(self):
        self._h, self._c = self.lstm(self._inputs, (self._h, self._c))

    def _sample_single(self, field):
        self._lstm_next_step()
        logit = self.soft[field.name](self._h[-1])
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        if field.choose_one:
            sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            log_prob = self.cross_entropy_loss(logit, sampled)
            self._inputs = self.embedding[field.name](sampled)
        else:
            logit = logit.view(-1, 1)
            logit = torch.cat([-logit, logit], 1)  # pylint: disable=invalid-unary-operand-type
            sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip_prob = torch.sigmoid(logit)
            kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl
            log_prob = self.cross_entropy_loss(logit, sampled)
            sampled = sampled.nonzero().view(-1)
            if sampled.sum().item():
                self._inputs = (torch.sum(self.embedding[field.name](sampled.view(-1)), 0) / (1. + torch.sum(sampled))).unsqueeze(0)
            else:
                self._inputs = torch.zeros(1, self.lstm_size, device=self.embedding[field.name].weight.device)

        sampled = sampled.detach().numpy().tolist()
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        if len(sampled) == 1:
            sampled = sampled[0]
        return sampled


class Enas(BaseNAS):
    """
    ENAS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    reward_function : callable
        Receives logits and ground truth label, return a tensor, which will be feeded to RL controller as reward.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    grad_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    entropy_weight : float
        Weight of sample entropy loss.
    skip_weight : float
        Weight of skip penalty loss.
    baseline_decay : float
        Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
    ctrl_lr : float
        Learning rate for RL controller.
    ctrl_steps_aggregate : int
        Number of steps that will be aggregated into one mini-batch for RL controller.
    ctrl_steps : int
        Number of mini-batches for each epoch of RL controller learning.
    ctrl_kwargs : dict
        Optional kwargs that will be passed to :class:`ReinforceController`.
    """

    def __init__(self, device='cuda', workers=4,log_frequency=None,
                 grad_clip=5., entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999,
                 ctrl_lr=0.00035, ctrl_steps_aggregate=20, ctrl_kwargs=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.device=device
        self.num_epochs = kwargs.get("num_epochs", 5)
        self.workers = workers
        self.log_frequency = log_frequency
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.
        self.ctrl_steps_aggregate = ctrl_steps_aggregate
        self.grad_clip = grad_clip
        self.workers = workers
        self.ctrl_kwargs=ctrl_kwargs
        self.ctrl_lr=ctrl_lr

    def search(self, space: BaseSpace, dset, estimator):
        self.model = space
        self.dataset = dset#.to(self.device)
        self.estimator = estimator
        self.model_optim = torch.optim.SGD(
            self.model.parameters(), lr=0.01, weight_decay=3e-4
        )
        # replace choice
        self.nas_modules = []
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        # to device
        self.model = self.model.to(self.device)
        # fields
        self.nas_fields = [ReinforceField(name, len(module),
                                          isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1)
                           for name, module in self.nas_modules]
        self.controller = ReinforceController(self.nas_fields, **(self.ctrl_kwargs or {}))
        self.ctrl_optim = torch.optim.Adam(self.controller.parameters(), lr=self.ctrl_lr)
        # train
        for i in range(self.num_epochs):
            self._train_model(i)
            self._train_controller(i)
        
        selection=self.export()
        return space.export(selection,self.device)

    def _train_model(self, epoch): 
        self.model.train()
        self.controller.eval()
        self.model_optim.zero_grad()
        self._resample()
        metric,loss=self._infer()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.model_optim.step()

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        self.ctrl_optim.zero_grad()
        for ctrl_step in range(self.ctrl_steps_aggregate):
            self._resample()
            with torch.no_grad():
                metric,loss=self._infer()
            reward =-metric  # todo : now metric is loss 
            if self.entropy_weight:
                reward += self.entropy_weight * self.controller.sample_entropy.item()
            self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
            loss = self.controller.sample_log_prob * (reward - self.baseline)
            if self.skip_weight:
                loss += self.skip_weight * self.controller.sample_skip_penalty
            loss /= self.ctrl_steps_aggregate
            loss.backward()
        
            if (ctrl_step + 1) % self.ctrl_steps_aggregate == 0:
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip)
                self.ctrl_optim.step()
                self.ctrl_optim.zero_grad()

            if self.log_frequency is not None and ctrl_step % self.log_frequency == 0:
                _logger.info('RL Epoch [%d/%d] Step [%d/%d]  %s', epoch + 1, self.num_epochs,
                                ctrl_step + 1, self.ctrl_steps_aggregate)

    def _resample(self):
        result = self.controller.resample()
        for name, module in self.nas_modules:
            module.sampled = result[name]

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()

    def _infer(self):
        metric, loss = self.estimator.infer(self.model, self.dataset)
        return metric, loss
