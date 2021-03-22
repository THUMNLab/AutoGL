import numpy as np
from typing import Union, Iterable

import torch
from ..model import BaseModel, MODEL_DICT
import pickle
from ...utils import get_logger
from . import EVALUATE_DICT

LOGGER_ES = get_logger("early-stopping")


def get_feval(feval):
    if isinstance(feval, str):
        return EVALUATE_DICT[feval]
    if isinstance(feval, type) and issubclass(feval, Evaluation):
        return feval
    if isinstance(feval, list):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")


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
        device: Union[torch.device, str],
        init=True,
        feval=["acc"],
        loss="nll_loss",
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
        self.model = model
        self.to(device)
        self.init = init
        self.feval = get_feval(feval)
        self.loss = loss

    def to(self, device):
        """
        Migrate trainer to new device

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
    ) -> Union["Evaluation", Iterable["Evaluation"]]:
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
            if isinstance(self.feval, list):
                return self.feval[0]
            else:
                return self.feval
        return self.feval

    @classmethod
    def get_task_name(cls):
        """Get task name, e.g., `base`, `NodeClassification`, `GraphClassification`, etc."""
        return "base"

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
        self, hp, model: Union[BaseModel, str, None] = None
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

    def get_name_with_hp(self):
        """Get the name of hyperparameter."""
        raise NotImplementedError()

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
        raise NotImplementedError()

    def set_feval(self, feval):
        """Set the evaluation metrics."""
        self.feval = get_feval(feval)

    def update_parameters(self, **kwargs):
        """
        Update parameters of this trainer
        """
        for k, v in kwargs.items():
            if k == "feval":
                self.set_feval(v)
            elif k == "device":
                self.to(v)
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                raise KeyError("Cannot set parameter", k, "for trainer", self.__class__)


# a static class for evaluating results
class Evaluation:
    @staticmethod
    def get_eval_name():
        """
        Should return the name of this evaluation method
        """
        raise NotImplementedError()

    @staticmethod
    def is_higher_better():
        """
        Should return whether this evaluation method is higher better (bool)
        """
        return True

    @staticmethod
    def evaluate(predict, label):
        """
        Should return: the evaluation result (float)
        """
        raise NotImplementedError()


class BaseNodeClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        model: Union[BaseModel, str],
        num_features,
        num_classes,
        device="auto",
        init=True,
        feval=["acc"],
        loss="nll_loss",
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        if isinstance(model, str):
            assert model in MODEL_DICT, "Cannot parse model name " + model
            self.model = MODEL_DICT[model](num_features, num_classes, device, init=init)
        elif isinstance(model, BaseModel):
            self.model = model
        else:
            raise TypeError(
                "Model argument only support str or BaseModel, get",
                type(model),
                "instead.",
            )
        super().__init__(model, device=device, init=init, feval=feval, loss=loss)

    @classmethod
    def get_task_name(cls):
        return "GraphClassification"


class BaseGraphClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        model: Union[BaseModel, str],
        num_features,
        num_classes,
        num_graph_features=0,
        device=None,
        init=True,
        feval=["acc"],
        loss="nll_loss",
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_graph_features = num_graph_features
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        if isinstance(model, str):
            assert model in MODEL_DICT, "Cannot parse model name " + model
            self.model = MODEL_DICT[model](
                num_features,
                num_classes,
                device,
                init=init,
                num_graph_features=num_graph_features,
            )
        elif isinstance(model, BaseModel):
            self.model = model
        else:
            raise TypeError(
                "Model argument only support str or BaseModel, get",
                type(model),
                "instead.",
            )

        super().__init__(model, device=device, init=init, feval=feval, loss=loss)

    @classmethod
    def get_task_name(cls):
        return "NodeClassification"
