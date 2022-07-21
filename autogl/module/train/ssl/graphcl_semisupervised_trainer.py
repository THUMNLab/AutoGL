# codes in this file are reproduced from <https://github.com/divelab/DIG> with some changes.
import os
import torch
import logging

import torch.nn.functional as F
import torch.utils.data

import numpy as np
import typing as _typing
import torch.multiprocessing as mp

from typing import Union, Tuple, Sequence, Type, Callable
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
from tqdm import trange
from copy import deepcopy
from dig.sslgraph.evaluation.eval_graph import k_fold

from .base import BaseContrastiveTrainer, get_view_by_name
from .. import register_trainer
from ..base import EarlyStopping, _DummyModel
from ..evaluation import Evaluation, get_feval, Acc
from ...model import (
    BaseAutoModel,
    BaseEncoderMaintainer, 
    BaseDecoderMaintainer,
    DecoderUniversalRegistry
)
from ....datasets import utils
from ....utils import get_logger

LOGGER = get_logger("graphcl semi-supervised trainer")

@register_trainer("GraphCLSemisupervisedTrainer")
class GraphCLSemisupervisedTrainer(BaseContrastiveTrainer):
    def __init__(
        self, 
        model: Union[Tuple[BaseEncoderMaintainer, BaseDecoderMaintainer], BaseEncoderMaintainer, BaseAutoModel, str] = None,
        prediction_model_head: Union[BaseDecoderMaintainer, str, None] = None,
        num_features: Union[int, None] = None,
        num_classes: Union[int,None] = None,
        num_graph_features: Union[int, None] = None,
        device: Union[torch.device, str] = "auto",
        feval: Union[
            Sequence[str], Sequence[Type[Evaluation]]
        ] = (Acc,),
        loss: Union[str, Callable] = "NCE",
        f_loss: Union[str, Callable] = "nll_loss",
        views_fn: Union[
            Sequence[str], Sequence[Callable], None
        ] = None,
        aug_ratio: Union[float, Sequence[float]] = 0.2,
        z_dim: Union[int, None] = 128,
        neg_by_crpt: bool = False,
        tau: int = 0.5,
        model_path: Union[str, None] = "./models",
        num_workers: int = 0,
        batch_size: int = 128,
        p_optim: Union[torch.optim.Optimizer, str] = "Adam",
        p_lr: float = 0.0001,
        p_lr_scheduler_type: str = None,
        p_weight_decay: int = 0,
        p_epoch: int = 100,
        p_early_stopping_round: int = 20,
        f_optim: Union[torch.optim.Optimizer, str] = "Adam",
        f_lr: float = 0.001,
        f_lr_scheduler_type: str = None,
        f_weight_decay: int = 0,
        f_epoch: int = 100,
        f_early_stopping_round: int = 20,
        init: bool = False,
        *args,
        **kwargs,
    ):
        """
        This trainer implements the semi-supervised task training method of GraphCL for graph-level tasks with pretraining and finetuning.
        GraphCL is a contrastive method proposed in the paper `Graph Contrastive Learning with Augmentations` in nips 2020.
        <https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html>
        
        Parameters
        ----------
        model:
            Models can be `str`, `autogl.module.model.encoders.BaseEncoderMaintainer` or a tuple of (encoder, decoder) 
            if need to specify both encoder and decoder. Encoder can be `str` or
            `autogl.module.model.encoders.BaseEncoderMaintainer`, and decoder can be `str`
            or `autogl.module.model.decoders.BaseDecoderMaintainer`.
            If only encoder is specified, decoder will be default to "sumpoolmlp"
        prediction_model: `BaseAutoModel`, `str` or None
            A model used to finetuning
            Only required if `node_level` = True.
        num_features: `int` or None
            The number of features in dataset.
        num_classes: `int` or None
            The number of classes in dataset. Only required when doing semi-supervised tasks.
        num_graph_features: `int` or None
            The number of graph level features in dataset.
        device: `torch.device` or `str`
            The device this trainer will use.
            When `device` = "auto", if GPU exists in the device and dependency is installed, 
            the trainer will give priority to GPU, otherwise CPU will be used
        feval: a sequence of `str` or a sequence of `Evaluation`
            The evaluation methods.
        loss: `str` or `Callable`
            The loss function or the learning objective of contrastive model.
        f_loss: `str` or `Callable`
            The loss function during finetuning stage.
        views_fn: a list of `str`, a list of `Callable` or None
            List of functions or augmentation methods to generate views from give graphs.
            In GraphCL, the number of views_fn should be 2.
            If the element of `views_fn` is str, the options should be "dropN", "permE", "subgraph", "maskN", "random2", "random3", "random4" or None.
        aug_ratio: a list of `float` or a `float`
            The ratio of augmentations, a number between [0, 1).
            If aug_ratio is set as a float, the value will shared by all views_fn.
            If aug_ratio is set as a list of float, the value of this list and views_fn one to one correspondence.
        z_dim: `int`
            The dimension of graph-level representations
        neg_by_crpt: `bool`
            The mode to obtain negative samples. Only required when `loss` = "JSE"
        tau: `int`
            The temperature parameter in InfoNCE loss. Only used when `loss` = "NCE"
        model_path: `str` or None
            The directory to restore the saved model.
            If `model_path` = None, the model will not be saved.
        num_workers: `int`
            Number of workers.
        batch_size: `int`
            Batch size for pretraining and inference.
        p_optim: `str` or `torch.optim.Optimizer`
            Optimizer for pretraining.
        p_lr: `float`
            Pretraining learning rate.
        p_lr_scheduler_type: `str`
            Scheduler type for pretraining learning rate.
        p_weight_decay: `float`
            Pretraining weight decay rate.
        p_epoch: `int`
            Pretraining epochs number.
        p_early_stopping: `int`
            Pretraining early stopping round.
        f_optim: `str` or `torch.optim.Optimizer`
            Optimizer for finetuning.
        f_lr: `float`
            Finetuning learning rate.
        f_lr_scheduler_type: `str`
            Scheduler type for finetuning learning rate.
        f_weight_decay: `float`
            Finetuning weight decay rate.
        f_epoch: `int`
            Finetuning epochs number.
        f_early_stopping: `int`
            Finetuning early stopping round.
        init: `bool`
            Whether to initialize the model.
        """
        # set encoder and decoder
        if isinstance(model, Tuple):
            encoder, decoder = model
        elif isinstance(model, BaseAutoModel):
            raise ValueError("The GraphCL trainer must need an encoder and a decoder, so `model` shouldn't be an instance of `BaseAutoModel`")
        else:
            encoder, decoder = model, "sumpoolmlp"
        # set augmentation methods
        if isinstance(views_fn, list):
            if isinstance(aug_ratio, float):
                aug_ratio = [aug_ratio] * len(views_fn)
            assert len(aug_ratio) == len(views_fn)
        self.views_fn_opt = views_fn
        self.aug_ratio = aug_ratio
        views_fn = self._get_views_fn(views_fn, aug_ratio)
        # init contrastive learning
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            decoder_node=None,
            num_features=num_features,
            num_graph_features=num_graph_features,
            views_fn=views_fn,
            graph_level=True,
            node_level=False,
            device=device,
            feval=feval,
            loss=loss,
            z_dim=z_dim,
            z_node_dim=None,
            neg_by_crpt=neg_by_crpt,
            tau=tau,
            model_path=model_path
        )
        self._prediction_model_head = None
        self.num_classes = num_classes
        self.prediction_model_head = prediction_model_head
        # recording the information about dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers > 0:
            mp.set_start_method("fork", force=True)
        # recording the information about optimizer in pretraining stage and finetuning stage
        # pretraining stage
        self.p_opt_received = p_optim
        self.p_optimizer = self._get_optimizer(p_optim)
        self.p_lr_scheduler_type = p_lr_scheduler_type
        self.p_lr = p_lr
        self.p_epoch = p_epoch
        self.p_weight_decay = p_weight_decay
        self.p_early_stopping_round = (
            p_early_stopping_round if p_early_stopping_round is not None else 100
        )
        self.p_early_stopping = EarlyStopping(
            patience=self.p_early_stopping_round, verbose=False
        )
        # finetuning stage
        self.f_opt_received = f_optim
        self.f_optimizer = self._get_optimizer(f_optim)
        self.f_lr_scheduler_type = f_lr_scheduler_type
        self.f_lr = f_lr
        self.f_epoch = f_epoch
        self.f_weight_decay = f_weight_decay
        self.f_early_stopping_round = (
            f_early_stopping_round if f_early_stopping_round is not None else 100
        )
        self.f_early_stopping = EarlyStopping(
            patience=self.f_early_stopping_round, verbose=False
        )
        if hasattr(F, f_loss):
            self.f_loss = getattr(F, f_loss)
        elif callable(f_loss):
            self.f_loss = f_loss
        else:
            raise NotImplementedError(f"The loss {f_loss} is not supported yet.")
        # parameters that record the results
        self.valid_result = None
        self.valid_result_prob = None
        self.valid_score = None

        # hyper parameter space
        self.hyper_parameter_space = [
            {
                "parameterName": "batch_size",
                "type": "INTEGER",
                "maxValue": 128,
                "minValue": 32,
                "scalingType": "LOG",
            },
            {
                "parameterName": "p_epoch",
                "type": "INTEGER",
                "maxValue": 150,
                "minValue": 50,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "p_early_stopping_round",
                "type": "INTEGER",
                "maxValue": 30,
                "minValue": 10,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "p_lr",
                "type": "DOUBLE",
                "maxValue": 1e-2,
                "minValue": 1e-4,
                "scalingType": "LOG",
            },
            {
                "parameterName": "p_weight_decay",
                "type": "DOUBLE",
                "maxValue": 5e-3,
                "minValue": 5e-8,
                "scalingType": "LOG"
            },
            {
                "parameterName": "f_epoch",
                "type": "INTEGER",
                "maxValue": 150,
                "minValue": 50,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "f_early_stopping_round",
                "type": "INTEGER",
                "maxValue": 100,
                "minValue": 10,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "f_lr",
                "type": "DOUBLE",
                "maxValue": 1e-2,
                "minValue": 1e-4,
                "scalingType": "LOG",
            },
            {
                "parameterName": "f_weight_decay",
                "type": "DOUBLE",
                "maxValue": 5e-3,
                "minValue": 0,
                "scalingType": "LOG"
            },
        ]
        self.hyper_parameters = {
            "batch_size": self.batch_size,
            "p_epoch": self.p_epoch,
            "p_early_stopping_round": self.p_early_stopping_round,
            "p_lr": self.p_lr,
            "p_weight_decay": self.p_weight_decay,
            "f_epoch": self.f_epoch,
            "f_early_stopping_round": self.f_early_stopping_round,
            "f_lr": self.f_lr,
            "f_weight_decay": self.f_weight_decay,
        }
        self.args = args
        self.kwargs = kwargs
        if init:
            self.initialize()

    def _get_embed(self, view):
        if self.neg_by_crpt:
            view_crpt = self._corrupt_graph(view)
            if self.node_level and self.graph_level:
                z_g, z_n = self.encoder.encoder(view)
                z_g_crpt, z_n_crpt = self.encoder.encoder(view_crpt)
                z = (torch.cat([z_g, z_g_crpt], 0),
                     torch.cat([z_n, z_n_crpt], 0))
            else:
                z = self.encoder.encoder(view)
                z_crpt = self.encoder.encoder(view_crpt)
                z = torch.cat([z, z_crpt], 0)
        else:
            z = self.encoder.encoder(view)
        return z

    def _compose_model(self, pretrain=False):
        if pretrain:
            return _DummyModel(self.encoder, self.decoder).to(self.device)
        else:
            return _DummyModel(self.encoder, self.prediction_model_head).to(self.device)

    def _initialize(self):
        self.encoder.initialize()
        if self.decoder is not None:
            self.decoder.initialize(self.encoder)
        if self.prediction_model_head is not None:
            self.prediction_model_head.initialize(self.encoder)

    @classmethod
    def get_task_name(cls):
        return "GraphCLSemisupervisedTrainer"
    
    def _get_optimizer(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam": 
                optimizer = torch.optim.Adam
            elif optimizer.lower() == "sgd": 
                optimizer = torch.optim.SGD
            else: 
                raise ValueError("Currently not support optimizer {}".format(optimizer))
        elif isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            optimizer = optimizer
        else:
            raise ValueError("Currently not support optimizer {}".format(optimizer))
        return optimizer

    def _get_views_fn(self, views_fn, aug_ratio):
        # GraphCL only need two kinds of augmentation methods
        assert (views_fn is None) or (len(views_fn) == 2)
        final_views_fn = []
        for i, view in enumerate(views_fn):
            if isinstance(view, str):
                assert view in ["dropN", "permE", "subgraph", "maskN", "random2", "random3", "random4"]
                final_views_fn.append(get_view_by_name(view, aug_ratio[i]))
        return final_views_fn

    def _get_scheduler(self, stage, optimizer):
        if stage == 'pretraining':
            lr_scheduler_type = self.p_lr_scheduler_type
        else:
            lr_scheduler_type = self.f_lr_scheduler_type
        if type(lr_scheduler_type) == str and lr_scheduler_type == "steplr":
            scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        elif type(lr_scheduler_type) == str and lr_scheduler_type == "multisteplr":
            scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        elif type(lr_scheduler_type) == str and lr_scheduler_type == "exponentiallr":
            scheduler = ExponentialLR(optimizer, gamma=0.1)
        elif (
            type(lr_scheduler_type) == str and lr_scheduler_type == "reducelronplateau"
        ):
            scheduler = ReduceLROnPlateau(optimizer, "min")
        else:
            scheduler = None
        return scheduler

    def _get_loader_loss(self, loader, optimizer=None, scheduler=None, mode="train"):
        epoch_loss = 0.0
        last_loss = 0.0
        if mode == "train":
            for data in loader:
                optimizer.zero_grad()
                if None in self.views_fn:
                    # For view fn that returns multiple views
                    views = []
                    for v_fn in self.views_fn:
                        if v_fn is not None:
                            views += [*v_fn(data)]
                else:
                    views = [v_fn(data) for v_fn in self.views_fn]
                zs = []
                for view in views:
                    z = self._get_embed(view.to(self.device))
                    zs.append(self.decoder.decoder(z, view.to(self.device)))
                loss = self.loss(zs, neg_by_crpt=self.neg_by_crpt, tau=self.tau)
                loss.backward()
                optimizer.step()
                if self.p_lr_scheduler_type:
                    scheduler.step()
                epoch_loss += loss.item()
                last_loss = loss.item()
        else:
            for data in loader:
                if None in self.views_fn:
                    # For view fn that returns multiple views
                    views = []
                    for v_fn in self.views_fn:
                        if v_fn is not None:
                            views += [*v_fn(data)]
                else:
                    views = [v_fn(data) for v_fn in self.views_fn]
                zs = []
                for view in views:
                    z = self._get_embed(view.to(self.device))
                    zs.append(self.decoder.decoder(z, view.to(self.device)))
                loss = self.loss(zs, neg_by_crpt=self.neg_by_crpt, tau=self.tau)
                epoch_loss += loss.item()
                last_loss = loss.item()
        return epoch_loss, last_loss

    @property
    def num_classes(self):
        return self._num_classes
    
    @num_classes.setter
    def num_classes(self, num_classes):
        self._num_classes = num_classes
        if self.prediction_model_head is not None:
            self.prediction_model_head.output_dimension = num_classes

    @property
    def prediction_model_head(self):
        return self._prediction_model_head
    
    @prediction_model_head.setter
    def prediction_model_head(self, head: _typing.Union[BaseDecoderMaintainer, str, None]):
        if isinstance(self.encoder, BaseAutoModel):
            raise ValueError("Encoder shouldn't be a `BaseAutoModel` in GraphCLSemisupervisedTrainer.")
        if isinstance(head, str):
            self._prediction_model_head = DecoderUniversalRegistry.get_decoder(head)(
                self.num_classes,
                input_dim=self.last_dim,
                num_graph_features=self.num_graph_features,
                device=self.device,
                init=self.initialized
            )
        elif isinstance(head, BaseDecoderMaintainer) or head is None:
            self._prediction_model_head = head
        else:
            raise NotImplementedError(f"Sorry. The head {head} is not supported yet.")
        self.num_features = self.num_features
        self.last_dim = self.last_dim
        self.num_graph_features = self.num_graph_features
        self.num_classes = self.num_classes

    def _train_pretraining_only(self, dataset):
        """
        Pretraining stage
        As a matter of fact, it trains encoder, and decoder is just an auxiliary task
        """
        import torch_geometric
        if int(torch_geometric.__version__.split('.')[0]) >= 2:
            # version 2.x
            from torch_geometric.loader import DataLoader
        else:
            from torch_geometric.data import DataLoader
        pre_train_loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        optimizer = self.p_optimizer(
            self.encoder.encoder.parameters(), lr=self.p_lr, weight_decay=self.p_weight_decay
        )
        optimizer.add_param_group({"params": self.decoder.decoder.parameters()})
        scheduler = self._get_scheduler("pretraining", optimizer)
        with trange(self.p_epoch) as t:
            for epoch in t:
                self.encoder.encoder.train()
                self.decoder.decoder.train()
                t.set_description(f"Pretraining: epoch {epoch + 1}")
                epoch_loss, last_loss = self._get_loader_loss(pre_train_loader, optimizer, scheduler, "train")
                t.set_postfix(loss="{:.6f}".format(float(last_loss)))
                self.p_early_stopping(epoch_loss, self.encoder.encoder)
                if self.p_early_stopping.early_stop:
                    LOGGER.debug("Early stopping at", epoch)
                    break
        
        self.p_early_stopping.load_checkpoint(self.encoder.encoder)
    
    def _train_finetuning_only(self, dataset):
        """
        Finetuning stage.
        In this stage, we actually trained a combination model with the encoder trained
        during pretraining stage and another projection head with downstream task.
        """
        model = self._compose_model()
        fine_train_loader = utils.graph_get_split(dataset, "train", batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        try:
            fine_val_loader = utils.graph_get_split(dataset, "val", batch_size=self.batch_size, num_workers=self.num_workers)
        except ValueError:
            fine_val_loader = utils.graph_get_split(dataset, "train", batch_size=self.batch_size, num_workers=self.num_workers)
        optimizer = self.f_optimizer(model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
        scheduler = self._get_scheduler("finetune", optimizer)
        with trange(self.f_epoch) as t:
            epoch_loss = 0.0
            for epoch in t:
                t.set_description(f"Finetuning epoch {epoch + 1}")
                model.train()
                for data in fine_train_loader:
                    optimizer.zero_grad()
                    data = data.to(self.device)
                    out = model(data)
                    loss = self.f_loss(out, data.y)
                    loss.backward()
                    optimizer.step()
                    if self.f_lr_scheduler_type:
                        scheduler.step()
                    epoch_loss += loss.item()
                if fine_val_loader is not None:
                    eval_func = (
                        self.feval if not isinstance(self.feval, list) else self.feval[0]
                    )
                    val_loss = self._evaluate(fine_val_loader, eval_func)

                    if eval_func.is_higher_better():
                        val_loss = -val_loss
                    self.f_early_stopping(val_loss, model)
                    if self.f_early_stopping.early_stop:
                        LOGGER.debug("Early stopping at", epoch)
                        break
                else:
                    self.f_early_stopping(epoch_loss, model)
                    if self.f_early_stopping.early_stop:
                        LOGGER.debug("Early stop at", epoch)
                        break
        self.f_early_stopping.load_checkpoint(model)

    def _train_only(self, dataset):
        """
        Training the model on the given dataset.
        """
        self.encoder.encoder.to(self.device)
        self.decoder.decoder.to(self.device)
        self._train_pretraining_only(dataset)
        self._train_finetuning_only(dataset)

    def train(self, dataset, keep_valid_result=True):
        """
        The function of training on the given dataset and keeping valid result.

        Parameters
        ----------
        dataset: The graph dataset used to be trained.
        keep_valid_result: `bool`
            if True, save the validation result after training.

        Returns
        -------
        self: `autogl.train.GraphCLSemisupervisedTrainer`
            A reference of current trainer.

        """
        try:
            valid_loader = utils.graph_get_split(
                dataset, "val", batch_size=self.batch_size, num_workers=self.num_workers
            )
        except ValueError:
            valid_loader = None
        self._train_only(dataset)
        if keep_valid_result and valid_loader:
            # save the validation result after training
            pred = self._predict_only(valid_loader)
            self.valid_result = pred.max(1)[1]
            self.valid_result_prob = pred
            self.valid_score = self.evaluate(dataset, mask="val", feval=self.feval)
    
    def predict(self, dataset, mask="test"):
        """
        The function of predicting on the given dataset.

        Parameters
        ----------
        dataset: The graph classification dataset used to be predicted.

        mask: `str`
        "train", "val" or "test"
        The dataset mask.

        Returns
        -------
        The prediction result of `predict_proba`.
        """
        loader = utils.graph_get_split(
            dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return self._predict_proba(loader, in_log_format=True).max(1)[1]
    
    def predict_proba(self, dataset, mask="test", in_log_format=False):
        """
        The function of predicting the probability on the given dataset.

        Parameters
        ----------
        dataset: The graph dataset used to be predicted.

        mask: `str`
        "train", "val" or "test"
        The dataset mask.

        in_log_format: `bool`
        If True(False), the probability will (not) be log format.

        Returns
        -------
        The prediction result.
        """
        loader = utils.graph_get_split(
            dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return self._predict_proba(loader, in_log_format)

    def _predict_proba(self, loader, in_log_format=False, return_label=False):
        if return_label:
            ret, label = self._predict_only(loader, return_label=True)
        else:
            ret = self._predict_only(loader, return_label=False)
        if in_log_format is False:
            ret = torch.exp(ret)
        if return_label:
            return ret, label
        else:
            return ret

    def _predict_only(self, loader, return_label=False):
        model = self._compose_model()
        model.eval()
        pred = []
        label = []
        for data in loader:
            data = data.to(self.device)
            out = model(data)
            pred.append(out)
            label.append(data.y)
        ret = torch.cat(pred, 0)
        label = torch.cat(label, 0)
        if return_label:
            return ret, label
        else:
            return ret

    def evaluate(self, dataset, mask="val", feval=None):
        """
        The function of evaluating the model on the given dataset and keeping valid result.

        Parameters
        ----------
        dataset: The graph dataset used to be evaluated.

        mask: `str`
        "Train", "val" or "test"
        The dataset mask

        feval: `str`
        The evaluation method used in this function

        Returns
        -------
        res: The evaluation result on the given dataset.
        """
        loader = utils.graph_get_split(
            dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return self._evaluate(loader, feval)
    
    def _evaluate(self, loader, feval=None):
        if feval is None:
            feval = self.feval
        else:
            feval = get_feval(feval)

        y_pred_prob, y_true = self._predict_proba(loader=loader, return_label=True)
        y_pred = y_pred_prob.max(1)[1]

        if not isinstance(feval, list):
            feval = [feval]
            return_signle = True
        else:
            return_signle = False

        res = []
        for f in feval:
            flag = False
            try:
                res.append(f.evaluate(y_pred_prob, y_true))
                flag = False
            except:
                flag = True
            if flag:
                try:
                    res.append(
                        f.evaluate(y_pred_prob.cpu().numpy(), y_true.cpu().numpy())
                    )
                    flag = False
                except:
                    flag = True
            if flag:
                try:
                    res.append(
                        f.evaluate(
                            y_pred_prob.detach().numpy(), y_true.detach().numpy()
                        )
                    )
                    flag = False
                except:
                    flag = True
            if flag:
                try:
                    res.append(
                        f.evaluate(
                            y_pred_prob.cpu().detach().numpy(),
                            y_true.cpu().detach().numpy(),
                        )
                    )
                    flag = False
                except:
                    flag = True
            if flag:
                assert False

        if return_signle:
            return res[0]
        return res

    def duplicate_from_hyper_parameter(self, hp, encoder="same", decoder="same", prediction_head="same", restricted=True):
        """
        The function of duplicating a new instance from the given hyperparameter.

        Parameters
        ----------
        hp: `dict`
            The hyperparameter used in the new instance. Should contain 4 keys "trainer", "encoder"
            "decoder" and "phead", with corresponding hyperparameters as values.
        
        encoder: The new encoder
            Encoder can be `str` or `autogl.module.model.encoders.BaseEncoderMaintainer`
        
        decoder: The new decoder
            Decoder can be `str` or `autogl.module.model.decoders.BaseDecoderMaintainer`

        prediction_head: The new prediction_head
            Prediction head can be `str` or `autogl.module.model.decoders.BaseDecoderMaintainer`

        restricted: `bool`
            If False(True), the hyperparameter should (not) be updated from origin hyperparameter.

        Returns
        -------
        self: `autogl.train.ssl.GraphCLSemisupervisedTrainer`
            A new instance of trainer.

        """
        hp_trainer = hp.get("trainer", {})
        hp_encoder = hp.get("encoder", {})
        hp_decoder = hp.get("decoder", {})
        hp_phead = hp.get("prediction_head", {})
        if not restricted:
            origin_hp = deepcopy(self.hyper_parameters)
            origin_hp.update(hp_trainer)
            hp = origin_hp
        else:
            hp = hp_trainer
        encoder = encoder if encoder != "same" else self.encoder
        decoder = decoder if decoder != "same" else self.decoder
        prediction_head = prediction_head if prediction_head != "same" else self.prediction_model_head
        encoder = encoder.from_hyper_parameter(hp_encoder)
        if isinstance(encoder, BaseEncoderMaintainer) and isinstance(decoder, BaseDecoderMaintainer):
            decoder = decoder.from_hyper_parameter_and_encoder(hp_decoder, encoder)
        if isinstance(encoder, BaseEncoderMaintainer) and isinstance(prediction_head, BaseDecoderMaintainer):
            prediction_head = prediction_head.from_hyper_parameter_and_encoder(hp_phead, encoder)
        ret = self.__class__(
            model=(encoder, decoder),
            prediction_model_head=prediction_head,
            num_features=self.num_features,
            num_classes=self.num_classes,
            num_graph_features=self.num_graph_features,
            device=self.device,
            feval=self.feval,
            loss=self.loss,
            f_loss=self.f_loss,
            views_fn=self.views_fn_opt,
            aug_ratio=self.aug_ratio,
            z_dim=self.last_dim,
            neg_by_crpt=self.neg_by_crpt,
            tau=self.tau,
            model_path=self.model_path,
            num_workers=self.num_workers,
            batch_size=hp["batch_size"],
            p_optim=self.p_opt_received,
            p_lr=hp["p_lr"],
            p_lr_scheduler_type=self.p_lr_scheduler_type,
            p_epoch=hp["p_epoch"],
            p_early_stopping_round=hp["p_early_stopping_round"],
            p_weight_decay=hp["p_weight_decay"],
            f_optim=self.f_opt_received,
            f_lr=hp["f_lr"],
            f_lr_scheduler_type=self.f_lr_scheduler_type,
            f_epoch=hp["f_epoch"],
            f_early_stopping_round=hp["f_early_stopping_round"],
            f_weight_decay=hp["f_weight_decay"],
            init=True,
            *self.args,
            **self.kwargs
        )

        return ret
