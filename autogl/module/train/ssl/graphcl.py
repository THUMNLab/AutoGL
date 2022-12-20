import os
import torch
import logging

import torch.nn.functional as F
import torch.utils.data

import numpy as np
import typing as _typing
import torch.multiprocessing as mp

from typing import Union, Tuple, Sequence, Type, Callable

from tqdm import trange
from copy import deepcopy

from .base import BaseContrastiveTrainer

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

from torch_geometric.nn.glob import (
    global_add_pool, global_max_pool, global_mean_pool
)

LOGGER = get_logger("graphcl trainer")

@register_trainer("GraphCLSemisupervisedTrainer")
class GraphCLSemisupervisedTrainer(BaseContrastiveTrainer):
    def __init__(
        self, 
        model: Tuple[BaseEncoderMaintainer, BaseDecoderMaintainer] = None,
        prediction_head: Union[BaseDecoderMaintainer, str, None] = None,
        num_features: Union[int, None] = None,
        num_classes: Union[int,None] = None,
        num_graph_features: Union[int, None] = 0,
        device: Union[torch.device, str] = "auto",
        feval: Union[
            Sequence[str], Sequence[Type[Evaluation]]
        ] = (Acc,),
        views_fn: Union[
            Sequence[str], Sequence[Callable], None
        ] = None,
        aug_ratio: Union[float, Sequence[float]] = 0.2,
        z_dim: Union[int, None] = 128,
        tau: int = 0.5,
        model_path: Union[str, None] = "./models",
        num_workers: int = 0,
        batch_size: int = 128,
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
        tau: `int`
            The temperature parameter in NT_Xent loss. Only used when `loss` = "NT_Xent"
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
        # init contrastive learning
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            decoder_node=None,
            num_features=num_features,
            num_graph_features=num_graph_features,
            views_fn=views_fn,
            aug_ratio=aug_ratio,
            graph_level=True,
            node_level=False,
            device=device,
            feval=feval,
            z_dim=z_dim,
            z_node_dim=None,
            tau=tau,
            model_path=model_path,
            *args,
            **kwargs
        )
        self.views_fn = views_fn
        self.aug_ratio = aug_ratio
        self._prediction_head = None
        self.num_classes = num_classes
        self.prediction_head = prediction_head
        # recording the information about dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers > 0:
            mp.set_start_method("fork", force=True)

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

    def _get_views_fn(self, views_fn, aug_ratio):
        # GraphCL only need two kinds of augmentation methods
        assert (views_fn is None) or (len(views_fn) == 2)
        return super()._get_views_fn(views_fn, aug_ratio)

    def _initialize(self):
        self.encoder.initialize()
        if self.decoder is not None:
            self.decoder.initialize(self.encoder)
        if self.prediction_head is not None:
            self.prediction_head.initialize(self.encoder)

    @classmethod
    def get_task_name(cls):
        return "GraphCLSemisupervisedTrainer"

    def __repr__(self) -> str:
        import yaml

        return yaml.dump(
            {
                "trainer_name": self.__class__.__name__,
                "p_optimizer": self.p_optimizer,
                "p_learning_rate": self.p_lr,
                "p_max_epoch": self.p_epoch,
                "p_early_stopping_round": self.p_early_stopping_round,
                "f_optimizer": self.f_optimizer,
                "f_learning_rate": self.f_lr,
                "f_max_epoch": self.f_epoch,
                "f_early_stopping_round": self.f_early_stopping_round,
                "encoder": repr(self.encoder),
                "decoder": repr(self.decoder),
                "prediction_head": repr(self.prediction_head)
            }
        )
    
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
        except AttributeError:
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

    def _train_pretraining_only(self, dataset, per_epoch=False):
        for epoch in enumerate(super()._train_pretraining_only(dataset, per_epoch=per_epoch)):
            pass

    def _train_only(self, dataset):
        """
        Training the model on the given dataset.
        """
        self.encoder.encoder.to(self.device)
        self.decoder.decoder.to(self.device)
        self._train_pretraining_only(dataset)
        self._train_finetuning_only(dataset)

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

    def duplicate_from_hyper_parameter(self, hp, encoder="same", decoder="same", prediction_head="same", restricted=True):
        """
        The function of duplicating a new instance from the given hyperparameter.

        Parameters
        ----------
        hp: `dict`
            The hyperparameter used in the new instance. Should contain 4 keys "trainer", "encoder"
            "decoder" and "phead", with corresponding hyperparameters as values.
        
        encoder: `str` or `autogl.module.model.encoders.BaseEncoderMaintainer`
            The new encoder
        
        decoder: `str` or `autogl.module.model.decoders.BaseDecoderMaintainer`
            The new decoder

        prediction_head: `str` or `autogl.module.model.decoders.BaseDecoderMaintainer`
            The new predictor

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
        prediction_head = prediction_head if prediction_head != "same" else self.prediction_head
        encoder = encoder.from_hyper_parameter(hp_encoder)
        decoder.output_dimension = tuple(encoder.get_output_dimensions())[-1]
        if isinstance(encoder, BaseEncoderMaintainer) and isinstance(decoder, BaseDecoderMaintainer):
            decoder = decoder.from_hyper_parameter_and_encoder(hp_decoder, encoder)
        if isinstance(encoder, BaseEncoderMaintainer) and isinstance(prediction_head, BaseDecoderMaintainer):
            prediction_head = prediction_head.from_hyper_parameter_and_encoder(hp_phead, encoder)
        ret = self.__class__(
            model=(encoder, decoder),
            prediction_head=prediction_head,
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


@register_trainer("GraphCLUnsupervisedTrainer")
class GraphCLUnsupervisedTrainer(BaseContrastiveTrainer):
    def __init__(
        self, 
        model: Tuple[BaseEncoderMaintainer, BaseDecoderMaintainer] = None,
        prediction_head: Union[BaseDecoderMaintainer, str, None] = None,
        num_features: Union[int, None] = None,
        num_classes: Union[int,None] = None,
        num_graph_features: Union[int, None] = 0,
        device: Union[torch.device, str] = "auto",
        feval: Union[
            Sequence[str], Sequence[Type[Evaluation]]
        ] = (Acc,),
        views_fn: Union[
            Sequence[str], Sequence[Callable], None
        ] = None,
        aug_ratio: Union[float, Sequence[float]] = 0.2,
        z_dim: Union[int, None] = 128,
        num_workers: int = 0,
        batch_size: int = 128,
        eval_interval: int = 10,
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
        tau: `int`
            The temperature parameter in NT_Xent loss. Only used when `loss` = "NT_Xent"
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
        self.eval_interval = eval_interval
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
            z_dim=z_dim,
            z_node_dim=None,
            *args,
            **kwargs
        )
        self.views_fn = views_fn
        self.aug_ratio = aug_ratio
        self._prediction_head = None
        self.num_classes = num_classes
        self.prediction_head = prediction_head
        # recording the information about dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers > 0:
            mp.set_start_method("fork", force=True)

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

    def _get_views_fn(self, views_fn, aug_ratio):
        # GraphCL only need two kinds of augmentation methods
        assert (views_fn is None) or (len(views_fn) == 2)
        return super()._get_views_fn(views_fn, aug_ratio)

    def _initialize(self):
        self.encoder.initialize()
        if self.decoder is not None:
            self.decoder.initialize(self.encoder)
        if self.prediction_head is not None and isinstance(self.prediction_head, BaseDecoderMaintainer):
            self.prediction_head.initialize(self.encoder)

    @classmethod
    def get_task_name(cls):
        return "GraphCLUnsupervisedTrainer"

    def __repr__(self) -> str:
        import yaml

        return yaml.dump(
            {
                "trainer_name": self.__class__.__name__,
                "p_optimizer": self.p_optimizer,
                "p_learning_rate": self.p_lr,
                "p_max_epoch": self.p_epoch,
                "p_early_stopping_round": self.p_early_stopping_round,
                "f_optimizer": self.f_optimizer,
                "f_learning_rate": self.f_lr,
                "f_max_epoch": self.f_epoch,
                "f_early_stopping_round": self.f_early_stopping_round,
                "encoder": repr(self.encoder),
                "decoder": repr(self.decoder),
                "prediction_head": repr(self.prediction_head)
            }
        )

    def _get_eval_embed(self, loader):
        self.encoder.encoder.eval()
        num_layers = self.encoder.hyper_parameters['num_layers']
        ret, y = [], []
        for _ in range(num_layers):
            ret.append([])
        with torch.no_grad():
            for data in loader:
                y.append(data.y.numpy())
                data.to(self.device)
                embed = self.encoder.encoder(data)
                for i in range(num_layers):
                    embed[i] = global_add_pool(embed[i], data.batch)
                    ret[i].append(embed[i].cpu().numpy())
        for i in range(num_layers):
            ret[i] = np.concatenate(ret[i], 0)
        y = np.concatenate(y, 0)
        return ret, y

    def _train_only(self, dataset):
        """
        Training the model on the given dataset.
        """
        self.encoder.encoder.to(self.device)
        self.decoder.decoder.to(self.device)
        test_scores = []
        for i, epoch in enumerate(super()._train_pretraining_only(dataset, per_epoch=True)):
            if (i + 1) % self.eval_interval == 0:
                train_loader = utils.graph_get_split(dataset, "train", batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
                test_loader = utils.graph_get_split(dataset, "test", batch_size=self.batch_size, num_workers=self.num_workers)
                # test_embed, test_lbls = self._get_eval_embed(test_loader)
                if not isinstance(self.prediction_head, BaseDecoderMaintainer):
                    train_embed, train_lbls = self._get_eval_embed(train_loader)
                    self.prediction_head.initialize()
                    self.prediction_head.fit(train_embed, train_lbls)
                    acc = self._evaluate(test_loader)[0]
                    test_scores.append(acc)
                    self.prediction_head.save_checkpoint(acc)
                    self.p_early_stopping(-acc, self.encoder.encoder)
                else:
                    self.encoder.encoder.eval()
                    self.tmp_f_early_stopping = EarlyStopping(
                        patience=self.f_early_stopping_round, verbose=False
                    )
                    self.prediction_head.initialize(self.encoder)
                    model = self.prediction_head.decoder
                    optimizer = self.f_optimizer(model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                    scheduler = self._get_scheduler("finetune", optimizer)
                    with trange(self.f_epoch) as t:
                        for epoch in t:
                            model.train()
                            for data in train_loader:
                                optimizer.zero_grad()
                                data = data.to(self.device)
                                embeds = self.encoder.encoder(data)
                                out = model(embeds, data)
                                loss = self.f_loss(out, data.y)
                                loss.backward()
                                optimizer.step()
                                if self.f_lr_scheduler_type:
                                    scheduler.step()
                            eval_func = (
                                self.feval if not isinstance(self.feval, list) else self.feval[0]
                            )
                            val_loss = self._evaluate(test_loader, eval_func)

                            if eval_func.is_higher_better():
                                val_loss = -val_loss
                            self.tmp_f_early_stopping(val_loss, model)
                            if self.f_early_stopping.early_stop:
                                LOGGER.debug("Early stopping at", epoch)
                                break
                                
                    self.tmp_f_early_stopping.load_checkpoint(model)
                    acc = self._evaluate(test_loader)
                    test_scores.append(acc[0])
                    self.f_early_stopping(self.tmp_f_early_stopping.val_loss_min, self.prediction_head.decoder)

        idx = np.argmax(test_scores)
        acc = test_scores[idx]
        if isinstance(self.prediction_head, BaseDecoderMaintainer):
            self.f_early_stopping.load_checkpoint(self.prediction_head.decoder)
        else:
            self.prediction_head.load_checkpoint()
            self.p_early_stopping.load_checkpoint(self.encoder.encoder)
        print("Best epoch %d: acc %.4f" % ((idx + 1) * self.eval_interval, acc))

    def _predict_only(self, loader, return_label=False):
        if isinstance(self.prediction_head, BaseDecoderMaintainer):
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
        else:
            embeds, label = self._get_eval_embed(loader)
            model = self.prediction_head
            tmpret = model.predict(embeds)
            ret = torch.zeros(tmpret.shape[0], self.num_classes)
            for i in range(tmpret.shape[0]):
                ret[i, tmpret[i]] = 1.
        if return_label:
            return ret, label
        else:
            return ret

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
        prediction_head = prediction_head if prediction_head != "same" else self.prediction_head
        encoder = encoder.from_hyper_parameter(hp_encoder)
        decoder.output_dimension = tuple(encoder.get_output_dimensions())[-1]
        if isinstance(encoder, BaseEncoderMaintainer) and isinstance(decoder, BaseDecoderMaintainer):
            decoder = decoder.from_hyper_parameter_and_encoder(hp_decoder, encoder)
        if isinstance(encoder, BaseEncoderMaintainer) and isinstance(prediction_head, BaseDecoderMaintainer):
            prediction_head = prediction_head.from_hyper_parameter_and_encoder(hp_phead, encoder)
        ret = self.__class__(
            model=(encoder, decoder),
            prediction_head=prediction_head,
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
            tau=self.tau,
            model_path=self.model_path,
            num_workers=self.num_workers,
            batch_size=hp["batch_size"],
            eval_interval=self.eval_interval,
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