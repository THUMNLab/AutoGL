import numpy as np
import typing as _typing

import torch
import pickle
from ..model import BaseModel, ModelUniversalRegistry
from .evaluation import Evaluation, get_feval, Acc
from ...utils import get_logger

LOGGER_ES = get_logger("early-stopping")


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
        elif score < self.best_score + self.delta:
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


class BaseTrainer:
    def __init__(
        self,
        model: BaseModel,
        device: _typing.Union[torch.device, str],
        init: bool = True,
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
        super().__init__()
        self.model: BaseModel = model
        if type(device) == torch.device or (
            type(device) == str and device.lower() != "auto"
        ):
            self.__device: torch.device = torch.device(device)
        else:
            self.__device: torch.device = torch.device(
                "cuda"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )
        self.init: bool = init
        self.__feval: _typing.Sequence[_typing.Type[Evaluation]] = get_feval(feval)
        self.loss: str = loss

    @property
    def device(self) -> torch.device:
        return self.__device

    @device.setter
    def device(self, __device: _typing.Union[torch.device, str]):
        if type(__device) == torch.device or (
            type(__device) == str and __device.lower() != "auto"
        ):
            self.__device: torch.device = torch.device(__device)
        else:
            self.__device: torch.device = torch.device(
                "cuda"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )

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

    def to(self, device: torch.device):
        """
        Transfer the trainer to another device

        Parameters
        ----------
        device: `str` or `torch.device`
            The device this trainer will use
        """
        self.device = torch.device(device)

    def initialize(self):
        """Initialize the auto model in trainer."""
        pass

    def get_model(self):
        """Get auto model used in trainer."""
        raise NotImplementedError()

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

    @property
    def hyper_parameter_space(self):
        """Get the space of hyperparameter."""
        raise NotImplementedError()

    @hyper_parameter_space.setter
    def hyper_parameter_space(self, space):
        """Set the space of hyperparameter."""
        pass

    def duplicate_from_hyper_parameter(
        self, hp, model: _typing.Optional[BaseModel] = ...
    ) -> "BaseTrainer":
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


class _BaseClassificationTrainer(BaseTrainer):
    """ Base class of trainer for classification tasks """

    def __init__(
        self,
        model: _typing.Union[BaseModel, str],
        num_features: int,
        num_classes: int,
        device: _typing.Union[torch.device, str, None] = "auto",
        init: bool = True,
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        self.num_features: int = num_features
        self.num_classes: int = num_classes
        if type(device) == torch.device or (
            type(device) == str and device.lower() != "auto"
        ):
            __device: torch.device = torch.device(device)
        else:
            __device: torch.device = torch.device(
                "cuda"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )
        if type(model) == str:
            _model: BaseModel = ModelUniversalRegistry.get_model(model)(
                num_features, num_classes, __device, init=init
            )
        elif isinstance(model, BaseModel):
            _model: BaseModel = model
        elif model is None:
            _model = None
        else:
            raise TypeError(
                f"Model argument only support str or BaseModel, got {model}."
            )
        super(_BaseClassificationTrainer, self).__init__(
            _model, __device, init, feval, loss
        )


class BaseNodeClassificationTrainer(_BaseClassificationTrainer):
    def __init__(
        self,
        model: _typing.Union[BaseModel, str],
        num_features: int,
        num_classes: int,
        device: _typing.Union[torch.device, str, None] = None,
        init: bool = True,
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        super(BaseNodeClassificationTrainer, self).__init__(
            model, num_features, num_classes, device, init, feval, loss
        )


class BaseGraphClassificationTrainer(_BaseClassificationTrainer):
    def __init__(
        self,
        model: _typing.Union[BaseModel, str],
        num_features: int,
        num_classes: int,
        num_graph_features: int = 0,
        device: _typing.Union[torch.device, str, None] = None,
        init: bool = True,
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        self.num_graph_features: int = num_graph_features
        super(BaseGraphClassificationTrainer, self).__init__(
            model, num_features, num_classes, device, init, feval, loss
        )


class BaseLinkPredictionTrainer(_BaseClassificationTrainer):
    def __init__(
        self,
        model: _typing.Union[BaseModel, str],
        num_features: int,
        device: _typing.Union[torch.device, str, None] = None,
        init: bool = True,
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: str = "nll_loss",
    ):
        super(BaseLinkPredictionTrainer, self).__init__(
            model, num_features, 2, device, init, feval, loss
        )
