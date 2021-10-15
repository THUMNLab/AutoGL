# codes in this file are reproduced from https://github.com/microsoft/nni with some changes.
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
from nni.nas.pytorch.fixed import apply_fixed_architecture
from tqdm import tqdm
from datetime import datetime
import numpy as np
from ....utils import get_logger

LOGGER = get_logger("random_search_NAS")


def _get_mask(sampled, total):
    multihot = [
        i == sampled or (isinstance(sampled, list) and i in sampled)
        for i in range(total)
    ]
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
        assert self.op_names, "There has to be at least one op to choose from."
        self.sampled = None  # sampled can be either a list of indices or an index

    def forward(self, *args, **kwargs):
        assert (
            self.sampled is not None
        ), "At least one path needs to be sampled before fprop."
        if isinstance(self.sampled, list):
            return sum(
                [getattr(self, self.op_names[i])(*args, **kwargs) for i in self.sampled]
            )  # pylint: disable=not-an-iterable
        else:
            return getattr(self, self.op_names[self.sampled])(
                *args, **kwargs
            )  # pylint: disable=invalid-sequence-index

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
            return sum(
                [input_tensors[t] for t in self.sampled]
            )  # pylint: disable=not-an-iterable
        else:
            return input_tensors[self.sampled]

    def __len__(self):
        return self.n_candidates

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return f"PathSamplingInputChoice(n_candidates={self.n_candidates}, chosen={self.sampled})"


class StackedLSTMCell(nn.Module):
    def __init__(self, layers, size, bias):
        super().__init__()
        self.lstm_num_layers = layers
        self.lstm_modules = nn.ModuleList(
            [nn.LSTMCell(size, size, bias=bias) for _ in range(self.lstm_num_layers)]
        )

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
        return f"ReinforceField(name={self.name}, total={self.total}, choose_one={self.choose_one})"


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

    def __init__(
        self,
        fields,
        lstm_size=64,
        lstm_num_layers=1,
        tanh_constant=1.5,
        skip_target=0.4,
        temperature=None,
        entropy_reduction="sum",
    ):
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
        self.skip_targets = nn.Parameter(
            torch.tensor(
                [1.0 - self.skip_target, self.skip_target]
            ),  # pylint: disable=not-callable
            requires_grad=False,
        )
        assert entropy_reduction in [
            "sum",
            "mean",
        ], "Entropy reduction must be one of sum and mean."
        self.entropy_reduction = torch.sum if entropy_reduction == "sum" else torch.mean
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.soft = nn.ModuleDict(
            {
                field.name: nn.Linear(self.lstm_size, field.total, bias=False)
                for field in fields
            }
        )
        self.embedding = nn.ModuleDict(
            {field.name: nn.Embedding(field.total, self.lstm_size) for field in fields}
        )

    def resample(self):
        self._initialize()
        result = dict()
        for field in self.fields:
            result[field.name] = self._sample_single(field)
        return result

    def _initialize(self):
        self._inputs = self.g_emb.data
        self._c = [
            torch.zeros(
                (1, self.lstm_size),
                dtype=self._inputs.dtype,
                device=self._inputs.device,
            )
            for _ in range(self.lstm_num_layers)
        ]
        self._h = [
            torch.zeros(
                (1, self.lstm_size),
                dtype=self._inputs.dtype,
                device=self._inputs.device,
            )
            for _ in range(self.lstm_num_layers)
        ]
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
            logit = torch.cat(
                [-logit, logit], 1
            )  # pylint: disable=invalid-unary-operand-type
            sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip_prob = torch.sigmoid(logit)
            kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl
            log_prob = self.cross_entropy_loss(logit, sampled)
            sampled = sampled.nonzero().view(-1)
            if sampled.sum().item():
                self._inputs = (
                    torch.sum(self.embedding[field.name](sampled.view(-1)), 0)
                    / (1.0 + torch.sum(sampled))
                ).unsqueeze(0)
            else:
                self._inputs = torch.zeros(
                    1, self.lstm_size, device=self.embedding[field.name].weight.device
                )

        sampled = sampled.detach().numpy().tolist()
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (
            log_prob * torch.exp(-log_prob)
        ).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        if len(sampled) == 1:
            sampled = sampled[0]
        return sampled


@register_nas_algo("rl")
class RL(BaseNAS):
    """
    RL in GraphNas.

    Parameters
    ----------
    num_epochs : int
        Number of epochs planned for training.
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
    n_warmup : int
        Number of epochs for training super network.
    model_lr : float
        Learning rate for super network.
    model_wd : float
        Weight decay for super network.
    disable_progress: boolean
        Control whether show the progress bar.
    """

    def __init__(
        self,
        num_epochs=5,
        device="cuda",
        log_frequency=None,
        grad_clip=5.0,
        entropy_weight=0.0001,
        skip_weight=0.8,
        baseline_decay=0.999,
        ctrl_lr=0.00035,
        ctrl_steps_aggregate=20,
        ctrl_kwargs=None,
        n_warmup=100,
        model_lr=5e-3,
        model_wd=5e-4,
        disable_progress=True,
    ):
        super().__init__(device)
        self.device = device
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.0
        self.ctrl_steps_aggregate = ctrl_steps_aggregate
        self.grad_clip = grad_clip
        self.ctrl_kwargs = ctrl_kwargs
        self.ctrl_lr = ctrl_lr
        self.n_warmup = n_warmup
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.disable_progress = disable_progress

    def search(self, space: BaseSpace, dset, estimator):
        self.model = space
        self.dataset = dset  # .to(self.device)
        self.estimator = estimator
        # replace choice
        self.nas_modules = []

        k2o = get_module_order(self.model)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)

        # to device
        self.model = self.model.to(self.device)
        # fields
        self.nas_fields = [
            ReinforceField(
                name,
                len(module),
                isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1,
            )
            for name, module in self.nas_modules
        ]
        self.controller = ReinforceController(
            self.nas_fields, **(self.ctrl_kwargs or {})
        )
        self.ctrl_optim = torch.optim.Adam(
            self.controller.parameters(), lr=self.ctrl_lr
        )
        # train
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                l2 = self._train_controller(i)
                bar.set_postfix(reward_controller=l2)

        selection = self.export()
        arch = space.parse_model(selection, self.device)
        # print(selection,arch)
        return arch

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        self.ctrl_optim.zero_grad()
        rewards = []
        with tqdm(
            range(self.ctrl_steps_aggregate), disable=self.disable_progress
        ) as bar:
            for ctrl_step in bar:
                self._resample()
                metric, loss = self._infer(mask="val")
                bar.set_postfix(acc=metric, loss=loss.item())
                LOGGER.info(f"{self.arch}\n{self.selection}\n{metric},{loss}")
                reward = metric
                rewards.append(reward)
                if self.entropy_weight:
                    reward += (
                        self.entropy_weight * self.controller.sample_entropy.item()
                    )
                self.baseline = self.baseline * self.baseline_decay + reward * (
                    1 - self.baseline_decay
                )
                loss = self.controller.sample_log_prob * (reward - self.baseline)
                if self.skip_weight:
                    loss += self.skip_weight * self.controller.sample_skip_penalty
                loss /= self.ctrl_steps_aggregate
                loss.backward()

                if (ctrl_step + 1) % self.ctrl_steps_aggregate == 0:
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.controller.parameters(), self.grad_clip
                        )
                    self.ctrl_optim.step()
                    self.ctrl_optim.zero_grad()

                if (
                    self.log_frequency is not None
                    and ctrl_step % self.log_frequency == 0
                ):
                    LOGGER.info(
                        "RL Epoch [%d/%d] Step [%d/%d]  %s",
                        epoch + 1,
                        self.num_epochs,
                        ctrl_step + 1,
                        self.ctrl_steps_aggregate,
                    )
        return sum(rewards) / len(rewards)

    def _resample(self):
        result = self.controller.resample()
        self.arch = self.model.parse_model(result, device=self.device)
        self.selection = result

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.arch._model, self.dataset, mask=mask)
        return metric[0], loss


@register_nas_algo("graphnas")
class GraphNasRL(BaseNAS):
    """
    RL in GraphNas.

    Parameters
    ----------
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    num_epochs : int
        Number of epochs planned for training.
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
    n_warmup : int
        Number of epochs for training super network.
    model_lr : float
        Learning rate for super network.
    model_wd : float
        Weight decay for super network.
    topk : int
        Number of architectures kept in training process.
    disable_progeress: boolean
        Control whether show the progress bar.
    """

    def __init__(
        self,
        device="cuda",
        num_epochs=10,
        log_frequency=None,
        grad_clip=5.0,
        entropy_weight=0.0001,
        skip_weight=0,
        baseline_decay=0.95,
        ctrl_lr=0.00035,
        ctrl_steps_aggregate=100,
        ctrl_kwargs=None,
        n_warmup=100,
        model_lr=5e-3,
        model_wd=5e-4,
        topk=5,
        disable_progress=True,
    ):
        super().__init__(device)
        self.device = device
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.ctrl_steps_aggregate = ctrl_steps_aggregate
        self.grad_clip = grad_clip
        self.ctrl_kwargs = ctrl_kwargs
        self.ctrl_lr = ctrl_lr
        self.n_warmup = n_warmup
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.hist = []
        self.topk = topk
        self.disable_progress = disable_progress

    def search(self, space: BaseSpace, dset, estimator):
        self.model = space
        self.dataset = dset  # .to(self.device)
        self.estimator = estimator
        # replace choice
        self.nas_modules = []

        k2o = get_module_order(self.model)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)

        # to device
        self.model = self.model.to(self.device)
        # fields
        self.nas_fields = [
            ReinforceField(
                name,
                len(module),
                isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1,
            )
            for name, module in self.nas_modules
        ]
        self.controller = ReinforceController(
            self.nas_fields,
            lstm_size=100,
            temperature=5.0,
            tanh_constant=2.5,
            **(self.ctrl_kwargs or {}),
        )
        self.ctrl_optim = torch.optim.Adam(
            self.controller.parameters(), lr=self.ctrl_lr
        )
        # train
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                l2 = self._train_controller(i)
                bar.set_postfix(reward_controller=l2)

        # selection=self.export()

        selections = [x[1] for x in self.hist]
        candidiate_accs = [-x[0] for x in self.hist]
        # print('candidiate accuracies',candidiate_accs)
        selection = self._choose_best(selections)
        arch = space.parse_model(selection, self.device)
        # print(selection,arch)
        return arch

    def _choose_best(self, selections):
        # graphnas use top 5 models, can evaluate 20 times epoch and choose the best.
        results = []
        for selection in selections:
            accs = []
            for i in tqdm(range(20), disable=self.disable_progress):
                self.arch = self.model.parse_model(selection, device=self.device)
                metric, loss = self._infer(mask="val")
                accs.append(metric)
            result = np.mean(accs)
            LOGGER.info(
                "selection {} \n acc {:.4f} +- {:.4f}".format(
                    selection, np.mean(accs), np.std(accs) / np.sqrt(20)
                )
            )
            results.append(result)
        best_selection = selections[np.argmax(results)]
        return best_selection

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        self.ctrl_optim.zero_grad()
        rewards = []
        baseline = None
        # diff: graph nas train 100 and derive 100 for every epoch(10 epochs), we just train 100(20 epochs). totol num of samples are same (2000)
        with tqdm(
            range(self.ctrl_steps_aggregate), disable=self.disable_progress
        ) as bar:
            for ctrl_step in bar:
                self._resample()
                metric, loss = self._infer(mask="val")

                # bar.set_postfix(acc=metric,loss=loss.item())
                LOGGER.debug(f"{self.arch}\n{self.selection}\n{metric},{loss}")
                # diff: not do reward shaping as in graphnas code
                reward = metric
                self.hist.append([-metric, self.selection])
                if len(self.hist) > self.topk:
                    self.hist.sort(key=lambda x: x[0])
                    self.hist.pop()
                rewards.append(reward)

                if self.entropy_weight:
                    reward += (
                        self.entropy_weight * self.controller.sample_entropy.item()
                    )

                if not baseline:
                    baseline = reward
                else:
                    baseline = baseline * self.baseline_decay + reward * (
                        1 - self.baseline_decay
                    )

                loss = self.controller.sample_log_prob * (reward - baseline)
                self.ctrl_optim.zero_grad()
                loss.backward()

                self.ctrl_optim.step()

                bar.set_postfix(acc=metric, max_acc=max(rewards))
        return sum(rewards) / len(rewards)

    def _resample(self):
        result = self.controller.resample()
        self.arch = self.model.parse_model(result, device=self.device)
        self.selection = result

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.arch._model, self.dataset, mask=mask)
        return metric[0], loss
