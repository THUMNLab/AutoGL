import numpy as np
import typing as _typing

import torch
import pickle

from autogl.module.model.encoders.base_encoder import AutoHomogeneousEncoderMaintainer

from ..model import (
    EncoderUniversalRegistry,
    DecoderUniversalRegistry,
    BaseEncoderMaintainer,
    BaseDecoderMaintainer,
    BaseAutoModel,
    ModelUniversalRegistry
)
from ..hpo import AutoModule
import logging
from .evaluation import Evaluation, get_feval, Acc
from ...utils import get_logger

LOGGER_ES = get_logger("early-stopping")

class _DummyModel(torch.nn.Module):
    def __init__(self, encoder: _typing.Union[BaseEncoderMaintainer, BaseAutoModel], decoder: _typing.Optional[BaseDecoderMaintainer]):
        super().__init__()
        if isinstance(encoder, BaseAutoModel):
            self.encoder = encoder.model
            self.decoder = None
        else:
            self.encoder = encoder.encoder
            self.decoder = None if decoder is None else decoder.decoder

    def __str__(self, ):
        return "DummyModel(encoder={}, decoder={})".format(self.encoder, self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        if self.decoder is None: return args[0]
        return self.decoder(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        res = self.encode(*args, **kwargs)
        return self.decode(res, *args, **kwargs)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=LOGGER_ES.info,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = 100 if patience is None else patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose is True:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.best_param = pickle.dumps(model.state_dict())
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        """Load models"""
        if hasattr(self, "best_param"):
            model.load_state_dict(pickle.loads(self.best_param))
        else:
            LOGGER_ES.warn("try to load checkpoint while no checkpoint is saved")


class BaseTrainer(AutoModule):
    def __init__(
        self,
        encoder: _typing.Union[BaseAutoModel, BaseEncoderMaintainer, None],
        decoder: _typing.Union[BaseDecoderMaintainer, None],
        device: _typing.Union[torch.device, str],
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        """
        The basic trainer.
        Used to automatically train the problems, e.g., node classification, graph classification, etc.
        Parameters
        ----------
        model: `BaseModel` or `str`
            The (name of) model used to train and predict.
        init: `bool`
            If True(False), the model will (not) be initialized.
        """
        super().__init__(device)
        self.encoder = encoder
        self.decoder = None if isinstance(encoder, BaseAutoModel) else decoder
        self.feval = feval
        self.loss = loss

    def _compose_model(self):
        return _DummyModel(self.encoder, self.decoder).to(self.device)

    def _initialize(self):
        self.encoder.initialize()
        if self.decoder is not None:
            self.decoder.initialize(self.encoder)

    @property
    def feval(self) -> _typing.Sequence[_typing.Type[Evaluation]]:
        return self.__feval

    @feval.setter
    def feval(
        self,
        _feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ],
    ):
        self.__feval: _typing.Sequence[_typing.Type[Evaluation]] = get_feval(_feval)

    @property
    def model(self):
        # compatible with v0.2
        return self.encoder
    
    @model.setter
    def model(self, model):
        # compatible with v0.2
        self.encoder = model

    def to(self, device: _typing.Union[str, torch.device]):
        """
        Transfer the trainer to another device
        Parameters
        ----------
        device: `str` or `torch.device`
            The device this trainer will use
        """
        self.device = device
        if self.encoder is not None:
            self.encoder.to_device(self.device)
        if self.decoder is not None:
            self.decoder.to_device(self.device)

    def get_feval(
        self, return_major: bool = False
    ) -> _typing.Union[
        _typing.Type[Evaluation], _typing.Sequence[_typing.Type[Evaluation]]
    ]:
        """
        Parameters
        ----------
        return_major: ``bool``
            Wether to return the major ``feval``. Default ``False``.
        Returns
        -------
        ``evaluation`` or list of ``evaluation``:
            If ``return_major=True``, will return the major ``evaluation`` method.
            Otherwise, will return the ``evaluation`` element passed when constructing.
        """
        if return_major:
            if isinstance(self.feval, _typing.Sequence):
                return self.feval[0]
            else:
                return self.feval
        return self.feval

    @classmethod
    def save(cls, instance, path):
        with open(path, "wb") as output:
            pickle.dump(instance, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as inputs:
            instance = pickle.load(inputs)
            return instance

    def duplicate_from_hyper_parameter(self, *args, **kwargs) -> "BaseTrainer":
        """Create a new trainer with the given hyper parameter."""
        raise NotImplementedError()

    def train(self, dataset, keep_valid_result):
        """
        Train on the given dataset.
        Parameters
        ----------
        dataset: The dataset used in training.
        keep_valid_result: `bool`
            If True(False), save the validation result after training.
        Returns
        -------
        """
        raise NotImplementedError()

    def predict(self, dataset, mask=None):
        """
        Predict on the given dataset.
        Parameters
        ----------
        dataset: The dataset used in predicting.
        mask: `train`, `val`, or `test`.
            The dataset mask.
        Returns
        -------
        prediction result
        """
        raise NotImplementedError()

    def predict_proba(self, dataset, mask=None, in_log_format=False):
        """
        Predict the probability on the given dataset.
        Parameters
        ----------
        dataset: The dataset used in predicting.
        mask: `train`, `val`, or `test`.
            The dataset mask.
        in_log_format: `bool`.
            If True(False), the probability will (not) be log format.
        Returns
        -------
        The prediction result.
        """
        raise NotImplementedError()

    def get_valid_predict_proba(self):
        """Get the valid result (prediction probability)."""
        raise NotImplementedError()

    def get_valid_predict(self):
        """Get the valid result."""
        raise NotImplementedError()

    def get_valid_score(self, return_major=True):
        """Get the validation score."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError

    def evaluate(self, dataset, mask=None, feval=None):
        """
        Parameters
        ----------
        dataset: The dataset used in evaluation.
        mask: `train`, `val`, or `test`.
            The dataset mask.
        feval: The evaluation methods.
        Returns
        -------
        The evaluation result.
        """
        raise NotImplementedError

    def update_parameters(self, **kwargs):
        """
        Update parameters of this trainer
        """
        for k, v in kwargs.items():
            if k == "feval":
                self.feval = get_feval(v)
            elif k == "device":
                self.to(v)
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                raise KeyError("Cannot set parameter", k, "for trainer", self.__class__)

    def combined_hyper_parameter_space(self):
        return {
            "trainer": self.hyper_parameter_space,
            "encoder": self.encoder.hyper_parameter_space,
            "decoder": [] if self.decoder is None else self.decoder.hyper_parameter_space
        }


class _BaseClassificationTrainer(BaseTrainer):
    """ Base class of trainer for classification tasks """

    def __init__(
        self,
        encoder: _typing.Union[BaseAutoModel, BaseEncoderMaintainer, str, None],
        decoder: _typing.Union[BaseDecoderMaintainer, str, None],
        num_features: int,
        num_classes: int,
        last_dim: _typing.Union[int, str] = "auto",
        device: _typing.Union[torch.device, str, None] = "auto",
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        self._encoder = None
        self._decoder = None
        self.num_features = num_features
        self.num_classes = num_classes
        self.last_dim: _typing.Union[int, str] = last_dim
        super(_BaseClassificationTrainer, self).__init__(
            encoder, decoder, device, feval, loss
        )
    
    @property
    def encoder(self):
        return self._encoder
    
    @encoder.setter
    def encoder(self, enc: _typing.Union[BaseAutoModel, BaseEncoderMaintainer, str, None]):
        if isinstance(enc, str):
            if enc in EncoderUniversalRegistry:
                self._encoder = EncoderUniversalRegistry.get_encoder(enc)(
                    self.num_features, final_dimension=self.last_dim, device=self.device, init=self.initialized
                )
            else:
                self._encoder = ModelUniversalRegistry.get_model(enc)(
                    self.num_features, final_dimension=self.last_dim, device=self.device
                )
                
        elif isinstance(enc, BaseEncoderMaintainer):
            self._encoder = enc
        elif isinstance(enc, BaseAutoModel):
            self._encoder = enc
            if self.decoder is not None:
                logging.warn("will disable decoder since a whole model is passed")
                self.decoder = None
        elif enc is None:
            self._encoder = None
        else:
            raise ValueError("Enc {} is not supported!".format(enc))
        self.num_features = self.num_features
        self.num_classes = self.num_classes
        self.last_dim = self.last_dim

    @property
    def decoder(self):
        return self._decoder
    
    @decoder.setter
    def decoder(self, dec: _typing.Union[BaseDecoderMaintainer, str, None]):
        if isinstance(self.encoder, BaseAutoModel):
            logging.warn("Ignore passed dec since enc is a whole model")
            self._decoder = None
            return
        if isinstance(dec, str):
            self._decoder = DecoderUniversalRegistry.get_decoder(dec)(
                self.num_classes, input_dimension=self.last_dim, device=self.device, init=self.initialized
            )
        elif isinstance(dec, BaseDecoderMaintainer):
            self._decoder = dec
        elif dec is None:
            self._decoder = None
        else:
            raise ValueError("Dec {} is not supported!".format(dec))
        self.num_features = self.num_features
        self.num_classes = self.num_classes
        self.last_dim = self.last_dim
    
    @property
    def num_classes(self):
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, num_classes):
        self.__num_classes = num_classes
        if isinstance(self.encoder, BaseAutoModel):
            self.encoder.output_dimension = num_classes
        elif isinstance(self.decoder, BaseDecoderMaintainer):
            self.decoder.output_dimension = num_classes

    @property
    def last_dim(self):
        return self._last_dim
    
    @last_dim.setter
    def last_dim(self, dim):
        self._last_dim = dim
        if isinstance(self.encoder, AutoHomogeneousEncoderMaintainer):
            self.encoder.final_dimension = self._last_dim

    @property
    def num_features(self):
        return self._num_features
    
    @num_features.setter
    def num_features(self, num_features):
        self._num_features = num_features
        if self.encoder is not None:
            self.encoder.input_dimension = num_features

class BaseNodeClassificationTrainer(_BaseClassificationTrainer):
    def __init__(
        self,
        encoder: _typing.Union[BaseAutoModel, BaseEncoderMaintainer, str, None],
        decoder: _typing.Union[BaseDecoderMaintainer, str, None],
        num_features: int,
        num_classes: int,
        device: _typing.Union[torch.device, str, None] = None,
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        super(BaseNodeClassificationTrainer, self).__init__(
            encoder, decoder, num_features, num_classes, num_classes, device, feval, loss
        )
    
    # override num_classes property to support last_dim setting
    @property
    def num_classes(self):
        return self.__num_classes
    
    @num_classes.setter
    def num_classes(self, num_classes):
        self.__num_classes = num_classes
        if isinstance(self.encoder, BaseAutoModel):
            self.encoder.output_dimension = num_classes
        elif isinstance(self.decoder, BaseDecoderMaintainer):
            self.decoder.output_dimension = num_classes
        self.last_dim = num_classes


class BaseGraphClassificationTrainer(_BaseClassificationTrainer):
    def __init__(
        self,
        encoder: _typing.Union[BaseAutoModel, BaseEncoderMaintainer, str, None] = None,
        decoder: _typing.Union[BaseDecoderMaintainer, str, None] = None,
        num_features: _typing.Optional[int] = None,
        num_classes: _typing.Optional[int] = None,
        num_graph_features: int = 0,
        last_dim: _typing.Union[int, str] = "auto",
        device: _typing.Union[torch.device, str, None] = None,
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        self._encoder = None
        self._decoder = None
        self.num_graph_features: int = num_graph_features
        super(BaseGraphClassificationTrainer, self).__init__(
            encoder, decoder, num_features, num_classes, last_dim, device, feval, loss
        )

    # override encoder and decoder to depend on graph level features
    @property
    def encoder(self):
        return self._encoder
    
    @encoder.setter
    def encoder(self, enc: _typing.Union[BaseAutoModel, BaseEncoderMaintainer, str, None]):
        if isinstance(enc, str):
            if enc in EncoderUniversalRegistry:
                self._encoder = EncoderUniversalRegistry.get_encoder(enc)(
                    self.num_features,
                    last_dim=self.last_dim,
                    num_graph_features=self.num_graph_features,
                    device=self.device,
                    init=self.initialized
                )
            else:
                self._encoder = ModelUniversalRegistry.get_model(enc)(
                    self.num_features,
                    self.last_dim,
                    device=self.device,
                    num_graph_features=self.num_graph_features,
                )

        elif isinstance(enc, (BaseAutoModel, BaseEncoderMaintainer)):
            self._encoder = enc
            if isinstance(enc, BaseAutoModel) and self.decoder is not None:
                logging.warn("will disable decoder since a whole model is passed")
                self.decoder = None
        elif enc is None:
            self._encoder = None
        else:
            raise ValueError("Enc {} is not supported!".format(enc))
        self.num_features = self.num_features
        self.num_classes = self.num_classes
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
            self._decoder = DecoderUniversalRegistry.get_decoder(dec)(
                self.num_classes,
                input_dim=self.last_dim,
                num_graph_features=self.num_graph_features,
                device=self.device,
                init=self.initialized
            )
        elif isinstance(dec, (BaseDecoderMaintainer, None)):
            self._decoder = dec
        else:
            raise ValueError("Invalid decoder setting")
        self.num_features = self.num_features
        self.num_classes = self.num_classes
        self.last_dim = self.last_dim

    # override num_classes property to support last_dim setting
    @property
    def num_classes(self):
        return self.__num_classes
    
    @num_classes.setter
    def num_classes(self, num_classes):
        self.__num_classes = num_classes
        if isinstance(self.encoder, BaseAutoModel):
            self.encoder.output_dimension = num_classes
        elif isinstance(self.decoder, BaseDecoderMaintainer):
            self.decoder.output_dimension = num_classes

    @property
    def num_graph_features(self):
        return self._num_graph_features
    
    @num_graph_features.setter
    def num_graph_features(self, num_graph_features):
        self._num_graph_features = num_graph_features
        if self.encoder is not None: self.encoder.num_graph_features = self._num_graph_features
        if self.decoder is not None: self.decoder.num_graph_features = self._num_graph_features


# TODO: according to discussion, link prediction may not belong to classification tasks
class BaseLinkPredictionTrainer(_BaseClassificationTrainer):
    def __init__(
        self,
        encoder: _typing.Union[BaseAutoModel, BaseEncoderMaintainer, str, None] = None,
        decoder: _typing.Union[BaseDecoderMaintainer, str, None] = None,
        num_features: _typing.Optional[int] = None,
        last_dim: _typing.Union[int, str] = "auto",
        device: _typing.Union[torch.device, str, None] = None,
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        super(BaseLinkPredictionTrainer, self).__init__(
            encoder, decoder, num_features, 2, last_dim, device, feval, loss
        )

    # override decoder since no num_classes is needed
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
            self._decoder = DecoderUniversalRegistry.get_decoder(dec)(
                input_dim=self.last_dim,
                device=self.device,
                init=self.initialized
            )
        elif isinstance(dec, BaseDecoderMaintainer):
            self._decoder = dec
        elif dec is None:
            self._decoder = None
        else:
            raise ValueError("Invalid decoder setting")
        self.num_features = self.num_features
        self.num_classes = self.num_classes
        self.last_dim = self.last_dim


# ============== Het =================
class BaseNodeClassificationHetTrainer(BaseNodeClassificationTrainer):
    """ Base class of trainer for classification tasks """

    def __init__(
        self,
        model: _typing.Union[BaseAutoModel, str],
        dataset: None,
        num_features: int,
        num_classes: int,
        device: _typing.Union[torch.device, str, None] = "auto",
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        self._dataset = dataset
        super(BaseNodeClassificationHetTrainer, self).__init__(
            model, None, num_features, num_classes, device, feval, loss
        )
        self.from_dataset(dataset)
    
    def from_dataset(self, dataset):
        self._dataset = dataset
        if self.encoder is not None:
            self.encoder.from_dataset(self._dataset)

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, enc: _typing.Union[BaseAutoModel, str, None]):
        if isinstance(enc, str):
            self._encoder = ModelUniversalRegistry.get_model(enc)(
                self.num_features, self.num_classes, device=self.device
            )
        elif isinstance(enc, BaseAutoModel):
            self._encoder = enc
            if self.decoder is not None:
                logging.warn("will disable decoder since a whole model is passed")
                self.decoder = None
        elif enc is None:
            self._encoder = None
        else:
            raise ValueError("Enc {} is not supported!".format(enc))
        self.num_features = self.num_features
        self.num_classes = self.num_classes
        if self._dataset is not None:
            self.from_dataset(self._dataset)
