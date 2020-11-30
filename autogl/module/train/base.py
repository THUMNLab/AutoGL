import numpy as np
from typing import Union, Iterable
from ..model import BaseModel
import pickle
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
        model: Union[BaseModel, str],
        optimizer=None,
        lr=None,
        max_epoch=None,
        early_stopping_round=None,
        device=None,
        init=True,
        feval=["acc"],
        loss="nll_loss",
        *args,
        **kwargs,
    ):
        """
        The basic trainer.

        Used to automatically train the problems, e.g., node classification, graph classification, etc.

        Parameters
        ----------
        model: `BaseModel` or `str`
            The (name of) model used to train and predict.

        optimizer: `Optimizer` of `str`
            The (name of) optimizer used to train and predict.

        lr: `float`
            The learning rate.

        max_epoch: `int`
            The max number of epochs in training.

        early_stopping_round: `int`
            The round of early stop.

        device: `torch.device` or `str`
            The device where model will be running on.

        init: `bool`
            If True(False), the model will (not) be initialized.

        args: Other parameters.

        kwargs: Other parameters.
        """
        super().__init__()

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
        with open(path, "rb") as input:
            instance = pickle.load(input)
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
        raise NotImplementedError()


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
        raise True

    @staticmethod
    def evaluate(predict, label):
        """
        Should return: the evaluation result (float)
        """
        raise NotImplementedError()
