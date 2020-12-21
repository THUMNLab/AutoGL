from . import register_trainer, BaseTrainer, Evaluation, EVALUATE_DICT, EarlyStopping
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from ..model import MODEL_DICT, BaseModel
from .evaluate import Logloss
from typing import Union
from ...datasets import utils
from copy import deepcopy

from ...utils import get_logger

LOGGER = get_logger('graph classification solver')

def get_feval(feval):
    if isinstance(feval, str):
        return EVALUATE_DICT[feval]
    if isinstance(feval, type) and issubclass(feval, Evaluation):
        return feval
    if isinstance(feval, list):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")


@register_trainer("GraphClassification")
class GraphClassificationTrainer(BaseTrainer):
    """
    The graph classification trainer.

    Used to automatically train the graph classification problem.

    Parameters
    ----------
    model: ``BaseModel`` or ``str``
        The (name of) model used to train and predict.

    optimizer: ``Optimizer`` of ``str``
        The (name of) optimizer used to train and predict.

    lr: ``float``
        The learning rate of graph classification task.

    max_epoch: ``int``
        The max number of epochs in training.

    early_stopping_round: ``int``
        The round of early stop.

    device: ``torch.device`` or ``str``
        The device where model will be running on.

    init: ``bool``
        If True(False), the model will (not) be initialized.
    """

    space = None

    def __init__(
        self,
        model: Union[BaseModel, str],
        num_features,
        num_classes,
        num_graph_features=0,
        optimizer=None,
        lr=None,
        max_epoch=None,
        batch_size=None,
        early_stopping_round=7,
        weight_decay=1e-4,
        device=None,
        init=True,
        feval=[Logloss],
        loss="nll_loss",
        *args,
        **kwargs
    ):
        super(GraphClassificationTrainer, self).__init__(model)

        self.loss_type = loss

        # init model
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

        if type(optimizer) == str and optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam
        elif type(optimizer) == str and optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD
        else:
            self.optimizer = torch.optim.Adam

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_graph_features = num_graph_features
        self.lr = lr if lr is not None else 1e-4
        self.max_epoch = max_epoch if max_epoch is not None else 100
        self.batch_size = batch_size if batch_size is not None else 64
        self.early_stopping_round = (
            early_stopping_round if early_stopping_round is not None else 100
        )
        # GraphClassificationTrainer.space = self.model.hyper_parameter_space
        self.device = device
        self.args = args
        self.kwargs = kwargs

        self.feval = get_feval(feval)

        self.weight_decay = weight_decay

        self.early_stopping = EarlyStopping(
            patience=early_stopping_round, verbose=False
        )

        self.valid_result = None
        self.valid_result_prob = None
        self.valid_score = None

        self.initialized = False
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device

        self.space = [
            {
                "parameterName": "max_epoch",
                "type": "INTEGER",
                "maxValue": 300,
                "minValue": 10,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "batch_size",
                "type": "INTEGER",
                "maxValue": 128,
                "minValue": 32,
                "scalingType": "LOG",
            },
            {
                "parameterName": "early_stopping_round",
                "type": "INTEGER",
                "maxValue": 30,
                "minValue": 10,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "lr",
                "type": "DOUBLE",
                "maxValue": 1e-3,
                "minValue": 1e-4,
                "scalingType": "LOG",
            },
            {
                "parameterName": "weight_decay",
                "type": "DOUBLE",
                "maxValue": 5e-3,
                "minValue": 5e-4,
                "scalingType": "LOG",
            },
        ]
        self.space += self.model.space
        GraphClassificationTrainer.space = self.space

        self.hyperparams = {
            "max_epoch": self.max_epoch,
            "batch_size": self.batch_size,
            "early_stopping_round": self.early_stopping_round,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
        self.hyperparams = {**self.hyperparams, **self.model.get_hyper_parameter()}

        if init is True:
            self.initialize()

    def initialize(self):
        # """Initialize the auto model in trainer."""
        if self.initialized is True:
            return
        self.initialized = True
        self.model.initialize()

    def get_model(self):
        # """Get auto model used in trainer."""
        return self.model

    @classmethod
    def get_task_name(cls):
        # """Get task name, i.e., `GraphClassification`."""
        return "GraphClassification"

    def to(self, new_device):
        assert isinstance(new_device, torch.device)
        self.device = new_device
        if self.model is not None:
            self.model.to(self.device)

    def train_only(self, train_loader, valid_loader=None):
        """
        The function of training on the given dataset and mask.

        Parameters
        ----------
        data: The graph classification dataset used to be trained. It should consist of masks, including train_mask, and etc.
        train_mask: The mask used in training stage.

        Returns
        -------
        self: ``autogl.train.GraphClassificationTrainer``
            A reference of current trainer.

        """
        optimizer = self.optimizer(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(1, self.max_epoch):
            self.model.model.train()
            loss_all = 0
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.model.model(data)
                # loss = F.nll_loss(output, data.y)
                if hasattr(F, self.loss_type):
                    loss = getattr(F, self.loss_type)(output, data.y)
                else:
                    raise TypeError("PyTorch does not support loss type {}".format(self.loss_type))
                loss.backward()
                loss_all += data.num_graphs * loss.item()
                optimizer.step()
                scheduler.step()

            # loss = loss_all / len(train_loader.dataset)
            # train_loss = self.evaluate(train_loader)
            eval_func = (
                self.feval if not isinstance(self.feval, list) else self.feval[0]
            )
            val_loss = self._evaluate(valid_loader, eval_func) if valid_loader else 0.0

            if eval_func.is_higher_better():
                val_loss = -val_loss
            self.early_stopping(val_loss, self.model.model)
            if self.early_stopping.early_stop:
                LOGGER.debug("Early stopping at", epoch)
                self.early_stopping.load_checkpoint(self.model.model)
                break

    def predict_only(self, loader):
        """
        The function of predicting on the given dataset and mask.

        Parameters
        ----------
        data: The graph classification dataset used to be predicted.
        train_mask: The mask used in training stage.

        Returns
        -------
        res: The result of predicting on the given dataset.

        """
        self.model.model.eval()
        pred = []
        for data in loader:
            data = data.to(self.device)
            pred.append(self.model.model(data))
        ret = torch.cat(pred, 0)
        return ret

    def train(self, dataset, keep_valid_result=True):
        """
        The function of training on the given dataset and keeping valid result.

        Parameters
        ----------
        dataset: The graph classification dataset used to be trained.

        keep_valid_result: ``bool``
            If True(False), save the validation result after training.

        Returns
        -------
        self: ``autogl.train.GraphClassificationTrainer``
            A reference of current trainer.

        """
        train_loader = utils.graph_get_split(
            dataset, "train", batch_size=self.batch_size
        )  # DataLoader(dataset['train'], batch_size=self.batch_size)
        valid_loader = utils.graph_get_split(
            dataset, "val", batch_size=self.batch_size
        )  # DataLoader(dataset['val'], batch_size=self.batch_size)
        self.train_only(train_loader, valid_loader)
        if keep_valid_result and valid_loader:
            pred = self.predict_only(valid_loader)
            self.valid_result = pred.max(1)[1]
            self.valid_result_prob = pred
            self.valid_score = self.evaluate(dataset, mask="val", feval=self.feval)

    def predict(self, dataset, mask="test"):
        """
        The function of predicting on the given dataset.

        Parameters
        ----------
        dataset: The graph classification dataset used to be predicted.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        Returns
        -------
        The prediction result of ``predict_proba``.
        """
        loader = utils.graph_get_split(dataset, mask, batch_size=self.batch_size)
        return self._predict_proba(loader, in_log_format=True).max(1)[1]

    def predict_proba(self, dataset, mask="test", in_log_format=False):
        """
        The function of predicting the probability on the given dataset.

        Parameters
        ----------
        dataset: The graph classification dataset used to be predicted.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        in_log_format: ``bool``.
            If True(False), the probability will (not) be log format.

        Returns
        -------
        The prediction result.
        """
        loader = utils.graph_get_split(dataset, mask, batch_size=self.batch_size)
        return self._predict_proba(loader, in_log_format)

    def _predict_proba(self, loader, in_log_format=False):
        ret = self.predict_only(loader)
        if in_log_format is True:
            return ret
        else:
            return torch.exp(ret)

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
            If True, the return only consists of the major result.
            If False, the return consists of the all results.

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

    def get_name_with_hp(self):
        # """Get the name of hyperparameter."""
        name = "-".join(
            [
                str(self.optimizer),
                str(self.lr),
                str(self.max_epoch),
                str(self.early_stopping_round),
                str(self.model),
                str(self.device),
            ]
        )
        name = (
            name
            + "|"
            + "-".join(
                [
                    str(x[0]) + "-" + str(x[1])
                    for x in self.model.get_hyper_parameter().items()
                ]
            )
        )
        return name

    def evaluate(self, dataset, mask="val", feval=None):
        """
        The function of training on the given dataset and keeping valid result.

        Parameters
        ----------
        dataset: The graph classification dataset used to be evaluated.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        feval: ``str``.
            The evaluation method used in this function.

        Returns
        -------
        res: The evaluation result on the given dataset.

        """
        loader = utils.graph_get_split(dataset, mask, batch_size=self.batch_size)
        return self._evaluate(loader, feval)

    def _evaluate(self, loader, feval=None):
        if feval is None:
            feval = self.feval
        else:
            feval = get_feval(feval)
        y_pred_prob = self._predict_proba(loader=loader)
        y_pred = y_pred_prob.max(1)[1]

        y_true_tmp = []
        for data in loader:
            y_true_tmp.append(data.y)
        y_true = torch.cat(y_true_tmp, 0)

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

    def duplicate_from_hyper_parameter(self, hp, model=None, restricted=True):
        """
        The function of duplicating a new instance from the given hyperparameter.

        Parameters
        ----------
        hp: ``dict``.
            The hyperparameter used in the new instance.

        model: The model used in the new instance of trainer.

        restricted: ``bool``.
            If False(True), the hyperparameter should (not) be updated from origin hyperparameter.

        Returns
        -------
        self: ``autogl.train.GraphClassificationTrainer``
            A new instance of trainer.

        """
        if not restricted:
            origin_hp = deepcopy(self.hyperparams)
            origin_hp.update(hp)
            hp = origin_hp
        if model is None:
            model = self.model
        model = model.from_hyper_parameter(
            dict(
                [
                    x
                    for x in hp.items()
                    if x[0] in [y["parameterName"] for y in model.space]
                ]
            )
        )

        ret = self.__class__(
            model=model,
            num_features=self.num_features,
            num_classes=self.num_classes,
            num_graph_features=self.num_graph_features,
            optimizer=self.optimizer,
            lr=hp["lr"],
            max_epoch=hp["max_epoch"],
            batch_size=hp["batch_size"],
            early_stopping_round=hp["early_stopping_round"],
            weight_decay=hp["weight_decay"],
            device=self.device,
            feval=self.feval,
            init=True,
            *self.args,
            **self.kwargs
        )

        return ret

    def set_feval(self, feval):
        # """Get the space of hyperparameter."""
        self.feval = get_feval(feval)

    @property
    def hyper_parameter_space(self):
        # """Set the space of hyperparameter."""
        return self.space

    @hyper_parameter_space.setter
    def hyper_parameter_space(self, space):
        # """Set the space of hyperparameter."""
        self.space = space
        GraphClassificationTrainer.space = space

    def get_hyper_parameter(self):
        # """Get the hyperparameter in this trainer."""
        return self.hyperparams
