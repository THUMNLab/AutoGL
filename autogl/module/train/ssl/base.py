import os
import torch
import torch_geometric
import logging

import numpy as np
import typing as _typing
from tqdm import trange
from typing import Union, Tuple, Sequence, Type, Callable

import torch.nn.functional as F
import torch.utils.data

from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)

from .losses import NTXent_loss
from .utils import get_view_by_name

from autogl.module.model.encoders.base_encoder import AutoHomogeneousEncoderMaintainer

from ..base import BaseTrainer, EarlyStopping, _DummyModel
from ..evaluation import Evaluation, get_feval, Acc
from ...model import (
    BaseAutoModel,
    BaseEncoderMaintainer, 
    BaseDecoderMaintainer,
    EncoderUniversalRegistry,
    DecoderUniversalRegistry
)
from ....utils import get_logger
from ....datasets import utils

LOGGER = get_logger("contrastive trainer")

class BaseContrastiveTrainer(BaseTrainer):
    def __init__(
        self, 
        encoder: _typing.Union[BaseEncoderMaintainer, str, None], 
        decoder: _typing.Union[BaseDecoderMaintainer, str, None],
        decoder_node: _typing.Union[BaseDecoderMaintainer, None],
        num_features: _typing.Union[int, None] = None,
        num_graph_features: _typing.Union[int, None] = None,
        device: _typing.Union[torch.device, str] = "auto",
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: Union[str, Callable] = "NT_Xent",
        f_loss: Union[str, Callable] = "nll_loss",
        views_fn: _typing.Union[
            _typing.Sequence[_typing.Callable], None
        ] = None,
        aug_ratio: Union[float, Sequence[float]] = 0.2,
        graph_level: bool = True,
        node_level: bool = False,
        z_dim: _typing.Union[int, None] = None,
        z_node_dim: _typing.Union[int, None] = None,
        tau: int = 0.5,
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
        model_path: _typing.Union[str, None] = "./models"
    ):
        """
        The basic trainer for self-supervised learning with contrastive method.
        Used to automatically train the self-supervised problems.
        Parameters
        ----------
        encoder: `BaseEncoderMaintainer`, `str` or None
            A graph encoder shared by all views.
        decoder: `BaseDecoderMaintainer`, `str` or None
            A decoder which can be understood as a projection head for graph-level representations.
            Only required if `graph_level` = True.
        decoder_node: `BaseDecoderMaintainer`, `str` or None
            A decoder which can be understood as a projection head for node-level representations.
            Only required if `node_level` = True.
        num_features: `int` or None, Optional
            The number of features in dataset.
        num_graph_features: `int` or None, Optional
            The number of graph level features in dataset.
        device: `torch.device` or `str`, Optional
            The device this trainer will use.
            When `device` = "auto", if GPU exists in the device and dependency is installed, 
            the trainer will give priority to GPU, otherwise CPU will be used
        feval: a sequence of `str` or a sequence of `Evaluation`, Optional
            The evaluation methods.
        loss: `str` or `Callable`, Optional
            The loss function or the learning objective of contrastive model.
        views_fn: a list of `Callable` or None, Optional
            List of functions or augmentation methods to generate views from give graphs.
        graph_level: `bool`, Optional
            Whether to include graph-level representations
        node_level: `bool`, Optional
            Whether to include node-level representations
        z_dim: `int`, Optional
            The dimension of graph-level representations
        z_node_dim: `int`, Optional
            The dimension of node-level representations
        tau: `int`, Optional
            The temperature parameter in NT_Xent loss. Only used when `loss` = "NT_Xent"
        model_path: `str` or None, Optional
            The directory to restore the saved model.
            If `model_path` = None, the model will not be saved.
        """
        assert (node_level or graph_level) is True
        assert isinstance(encoder, BaseEncoderMaintainer) or isinstance(encoder, str) or encoder is None
        self.loss = self._get_loss(loss)
        self.node_level = node_level
        self.graph_level = graph_level
        self.z_dim = z_dim
        self.z_node_dim = z_node_dim
        self._encoder = None
        # parameters that record the results
        self.valid_result = None
        self.valid_result_prob = None
        self.valid_score = None
        # TODO
        # do not support method with both node-level representation and graph-level representation
        # so the decoder will be either _decoder or _decoder_node, one of them
        self._decoder = None
        self.views_fn_opt = views_fn
        self._views_fn = None
        self._aug_ratio = aug_ratio
        self.last_dim = z_dim if graph_level else z_node_dim
        self.num_features = num_features
        self.num_graph_features = num_graph_features
        self.tau = tau
        self.model_path = model_path
        if isinstance(device, str):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        if isinstance(f_loss, str) and hasattr(F, f_loss):
            self.f_loss = getattr(F, f_loss)
        elif callable(f_loss):
            self.f_loss = f_loss
        else:
            raise NotImplementedError(f"The loss {f_loss} is not supported yet.")
        super().__init__(
            encoder=encoder,
            decoder=decoder if graph_level else decoder_node,
            device=self.device,
            feval=feval,
            loss=self.loss
        )

    def _get_loss(self, loss):
        if callable(loss):
            return loss
        elif isinstance(loss, str):
            assert loss in ['NT_Xent']
            return {'NT_Xent': NTXent_loss}[loss]
        else:
            raise NotImplementedError("The argument `loss` should be str or callable which returns a loss tensor")

    # override encoder and decoder to depend on contrastive learning
    @property
    def encoder(self):
        return self._encoder
    
    @encoder.setter
    def encoder(self, enc: _typing.Union[BaseEncoderMaintainer, str, None]):
        if isinstance(enc, str):
            if enc in EncoderUniversalRegistry:
                if self.node_level:
                    self._encoder = EncoderUniversalRegistry.get_encoder(enc)(
                        self.num_features, final_dimension=self.last_dim, device=self.device, init=self.initialized
                    )
                elif self.graph_level:
                    self._encoder = EncoderUniversalRegistry.get_encoder(enc)(
                        self.num_features,
                        final_dimension=self.last_dim,
                        num_graph_featues=self.num_graph_features,
                        device=self.device,
                        init=self.initialized
                    )
            else:
                raise NotImplementedError(f"Sorry. Encoder {enc} is not supported yet.")
        elif isinstance(enc, BaseEncoderMaintainer):
            self._encoder = enc
        elif enc is None:
            self._encoder = None
        else:
            raise NotImplementedError(f"Sorry. Encoder {enc} is not supported yet.")
        self.num_features = self.num_features
        self.last_dim = self.last_dim
        self.num_graph_features = self.num_graph_features
    
    @property
    def decoder(self):
        if isinstance(self.encoder, BaseAutoModel): return None
        return self._decoder

    @decoder.setter
    def decoder(self, dec: _typing.Union[BaseDecoderMaintainer, str, None]):
        if isinstance(self.encoder, BaseAutoModel):
            logging.warn("Ignore passed dec since enc is a whole model")
            self._decoder = None
            return
        if isinstance(dec, str):
            if self.node_level:
                self._decoder = DecoderUniversalRegistry.get_decoder(dec)(
                    self.last_dim,
                    input_dimension=self.last_dim,
                    device=self.device,
                    init=self.initialized
                )
            elif self.graph_level:
                self._decoder = DecoderUniversalRegistry.get_decoder(dec)(
                    self.last_dim,
                    input_dimension=self.last_dim,
                    num_graph_features=self.num_graph_features,
                    device=self.device,
                    init=self.initialized
                )
        elif isinstance(dec, BaseDecoderMaintainer) or dec is None:
            self._decoder = dec
        else:
            raise NotImplementedError(f"Sorry. The decoder {dec} is not supported yet.")
        self.num_features = self.num_features
        self.last_dim = self.last_dim
        self.num_graph_features = self.num_graph_features

    @property
    def num_graph_features(self):
        return self._num_graph_features

    @num_graph_features.setter
    def num_graph_features(self, num_graph_featues):
        self._num_graph_features = num_graph_featues
        if self.graph_level:
            if self.encoder is not None: self.encoder.num_graph_features = self._num_graph_features
            if self.decoder is not None: self.decoder.num_graph_features = self._num_graph_features

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, num_features):
        self._num_features = num_features
        if self.encoder is not None:
            self.encoder.input_dimension = num_features

    @property
    def num_classes(self):
        return self._num_classes
    
    @num_classes.setter
    def num_classes(self, num_classes):
        self._num_classes = num_classes
        if self.prediction_head is not None:
            self.prediction_head.output_dimension = num_classes

    @property
    def views_fn(self):
        return self._views_fn
    
    @views_fn.setter
    def views_fn(self, views_fn):
        self.views_fn_opt = views_fn
        # set augmentation methods
        if isinstance(views_fn, list):
            if isinstance(self.aug_ratio, float):
                self.aug_ratio = [self.aug_ratio] * len(views_fn)
            if isinstance(self.aug_ratio, list) and len(self.aug_ratio) != len(views_fn):
                self.aug_ratio = [self.aug_ratio[0]] * len(views_fn)
        self._views_fn = self._get_views_fn(views_fn, self.aug_ratio)

    @property
    def aug_ratio(self):
        return self._aug_ratio
    
    @aug_ratio.setter
    def aug_ratio(self, aug_ratio):
        # set augmentation methods
        if isinstance(self.views_fn, list):
            if isinstance(aug_ratio, float):
                aug_ratio = [aug_ratio] * len(self.views_fn)
            assert len(aug_ratio) >= len(self.views_fn)
        self._aug_ratio = aug_ratio
        self._views_fn = self._get_views_fn(self.views_fn_opt, self.aug_ratio)

    @property
    def prediction_head(self):
        return self._prediction_head
    
    @prediction_head.setter
    def prediction_head(self, head: _typing.Union[BaseDecoderMaintainer, str, None]):
        if isinstance(self.encoder, BaseAutoModel):
            raise ValueError("Encoder shouldn't be a `BaseAutoModel` in GraphCLSemisupervisedTrainer.")
        if isinstance(head, str):
            self._prediction_head = DecoderUniversalRegistry.get_decoder(head)(
                self.num_classes,
                input_dim=self.last_dim,
                num_graph_features=self.num_graph_features,
                device=self.device,
                init=self.initialized
            )
        elif isinstance(head, BaseDecoderMaintainer) or head is None or (hasattr(head, 'fit') and hasattr(head, 'predict')):
            self._prediction_head = head
        else:
            raise NotImplementedError(f"Sorry. The head {head} is not supported yet.")
        self.num_features = self.num_features
        self.last_dim = self.last_dim
        self.num_graph_features = self.num_graph_features
        self.num_classes = self.num_classes

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
            if self.graph_level:
                valid_loader = utils.graph_get_split(
                    dataset, "val", batch_size=self.batch_size, num_workers=self.num_workers
                )
            else:
                # TODO node_level
                valid_loader = None
        except ValueError:
            valid_loader = None
        except AttributeError:
            valid_loader = None
        self._train_only(dataset)
        if keep_valid_result and valid_loader:
            # save the validation result after training
            pred = self._predict_only(valid_loader)
            self.valid_result = pred.max(1)[1]
            self.valid_result_prob = pred
            self.valid_score = self.evaluate(dataset, mask="val", feval=self.feval)

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
        if views_fn is None:
            return None
        final_views_fn = []
        for i, view in enumerate(views_fn):
            if isinstance(view, str):
                assert view in ["dropN", "permE", "subgraph", "maskN", "random2", "random3", "random4"]
                final_views_fn.append(get_view_by_name(view, aug_ratio[i]))
            else:
                final_views_fn.append(view)
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
                loss = self.loss(zs, tau=self.tau)
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
                loss = self.loss(zs, tau=self.tau)
                epoch_loss += loss.item()
                last_loss = loss.item()
        return epoch_loss, last_loss

    def _train_pretraining_only(self, dataset, per_epoch=False):
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
        if self.graph_level:
            pre_train_loader = DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
            )
        else:
            # TODO node level
            pre_train_loader = None
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
                try:
                    epoch_loss, last_loss = self._get_loader_loss(pre_train_loader, optimizer, scheduler, "train")
                except ValueError:
                    epoch_loss = 1e10
                    pass
                t.set_postfix(loss="{:.6f}".format(float(last_loss)))
                if per_epoch:
                    yield epoch
                else:
                    self.p_early_stopping(epoch_loss, self.encoder.encoder)
                    if self.p_early_stopping.early_stop:
                        LOGGER.debug("Early stopping at", epoch)
                        break
        
        if not per_epoch:
            self.p_early_stopping.load_checkpoint(self.encoder.encoder)
            yield self.p_epoch

    def _compose_model(self, pretrain=False):
        if pretrain:
            return _DummyModel(self.encoder, self.decoder).to(self.device)
        elif self.prediction_head is not None and isinstance(self.prediction_head, BaseDecoderMaintainer):
            return _DummyModel(self.encoder, self.prediction_head).to(self.device)
        else:
            return self.encoder.encoder.to(self.device)

    def _get_embed(self, view):
        z = self.encoder.encoder(view)
        return z

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
        if self.graph_level:
            loader = utils.graph_get_split(
                dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
            )
        else:
            # TODO node level
            loader = None
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
        if self.graph_level:
            loader = utils.graph_get_split(
                dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
            )
        else:
            # TODO node level
            loader = None
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
        raise NotImplementedError()

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
        if self.graph_level:
            loader = utils.graph_get_split(
                dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
            )
        else:
            # TODO node level
            loader = None
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

    def combined_hyper_parameter_space(self):
        return {
            "trainer": self.hyper_parameter_space,
            "encoder": self.encoder.hyper_parameter_space,
            "decoder": [] if self.decoder is None else self.decoder.hyper_parameter_space,
            "prediction_head": [] if self.prediction_head is None else self.prediction_head.hyper_parameter_space
        }

    def get_valid_predict(self):
    # """Get the valid result."""
        return self.valid_result

    def get_valid_predict_proba(self):
        # """Get the valid result (prediction probability)."""
        return self.valid_result_prob

    def get_valid_score(self, return_major=True):
        """
        The function of getting the valid score.

        Parameters
        ----------
        return_major: ``bool``.
            If True, then return only consists of the major result.
            If False, then return consists of the all results.

        Returns
        -------
        result: The valid score in training stage.
        """
        if isinstance(self.feval, list):
            if return_major:
                return self.valid_score[0], self.feval[0].is_higher_better()
            else:
                return self.valid_score, [f.is_higher_better() for f in self.feval]
        else:
            return self.valid_score, self.feval.is_higher_better()

    def __repr__(self) -> str:
        import yaml

        return yaml.dump(
            {
                "trainer_name": self.__class__.__name__,
                "p_optimizer": self.p_optimizer,
                "p_learning_rate": self.p_lr,
                "p_max_epoch": self.p_epoch,
                "p_early_stopping_round": self.p_early_stopping_round,
                "encoder": repr(self.encoder),
                "decoder": repr(self.decoder)
            }
        )