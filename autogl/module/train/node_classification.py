from . import register_trainer, BaseTrainer, Evaluation, EVALUATE_DICT, EarlyStopping
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import torch.nn.functional as F
from ..model import MODEL_DICT, BaseModel
from .evaluate import Logloss, Acc, Auc
from typing import Union
from copy import deepcopy

from ...utils import get_logger

LOGGER = get_logger("node classification trainer")


def get_feval(feval):
    if isinstance(feval, str):
        return EVALUATE_DICT[feval]
    if isinstance(feval, type) and issubclass(feval, Evaluation):
        return feval
    if isinstance(feval, list):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")


@register_trainer("NodeClassification")
class NodeClassificationTrainer(BaseTrainer):
    """
    The node classification trainer.

    Used to automatically train the node classification problem.

    Parameters
    ----------
    model: ``BaseModel`` or ``str``
        The (name of) model used to train and predict.

    optimizer: ``Optimizer`` of ``str``
        The (name of) optimizer used to train and predict.

    lr: ``float``
        The learning rate of node classification task.

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
        optimizer=None,
        lr=None,
        max_epoch=None,
        early_stopping_round=None,
        weight_decay=1e-4,
        device=None,
        init=True,
        feval=[Logloss],
        loss="nll_loss",
        lr_scheduler_type='steplr',
        *args,
        **kwargs
    ):
        super(NodeClassificationTrainer, self).__init__(model)

        self.loss_type = loss

        if device is None:
            device = "cpu"

        # init model
        if isinstance(model, str):
            assert model in MODEL_DICT, "Cannot parse model name " + model
            self.model = MODEL_DICT[model](num_features, num_classes, device, init=init)
        elif isinstance(model, BaseModel):
            self.model = model

        if type(optimizer) == str and optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam
        elif type(optimizer) == str and optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD
        else:
            self.optimizer = torch.optim.Adam

        self.lr_scheduler_type = lr_scheduler_type

        self.num_features = num_features
        self.num_classes = num_classes
        self.lr = lr if lr is not None else 1e-4
        self.max_epoch = max_epoch if max_epoch is not None else 100
        self.early_stopping_round = (
            early_stopping_round if early_stopping_round is not None else 100
        )
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
                "maxValue": 500,
                "minValue": 10,
                "scalingType": "LINEAR",
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
                "maxValue": 1e-1,
                "minValue": 1e-4,
                "scalingType": "LOG",
            },
            {
                "parameterName": "weight_decay",
                "type": "DOUBLE",
                "maxValue": 1e-2,
                "minValue": 1e-4,
                "scalingType": "LOG",
            },
        ]
        # self.space += self.model.space
        NodeClassificationTrainer.space = self.space

        self.hyperparams = {
            "max_epoch": self.max_epoch,
            "early_stopping_round": self.early_stopping_round,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
        self.hyperparams = {**self.hyperparams, **self.model.get_hyper_parameter()}

        if init is True:
            self.initialize()

    def initialize(self):
        #  Initialize the auto model in trainer.
        if self.initialized is True:
            return
        self.initialized = True
        self.model.initialize()

    def get_model(self):
        # Get auto model used in trainer.
        return self.model

    @classmethod
    def get_task_name(cls):
        # Get task name, i.e., `NodeClassification`.
        return "NodeClassification"

    def train_only(self, data, train_mask=None):
        """
        The function of training on the given dataset and mask.

        Parameters
        ----------
        data: The node classification dataset used to be trained. It should consist of masks, including train_mask, and etc.
        train_mask: The mask used in training stage.

        Returns
        -------
        self: ``autogl.train.NodeClassificationTrainer``
            A reference of current trainer.

        """
        data = data.to(self.device)
        mask = data.train_mask if train_mask is None else train_mask
        optimizer = self.optimizer(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        lr_scheduler_type = self.lr_scheduler_type
        if type(lr_scheduler_type) == str and lr_scheduler_type == 'steplr':
            scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        elif type(lr_scheduler_type) == str and lr_scheduler_type == 'multisteplr':
            scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        elif type(lr_scheduler_type) == str and lr_scheduler_type == 'exponentiallr':
            scheduler = ExponentialLR(optimizer, gamma=0.1)
        elif type(lr_scheduler_type) == str and lr_scheduler_type == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(optimizer, 'min')
        else:
            scheduler = None

        for epoch in range(1, self.max_epoch):
            self.model.model.train()
            optimizer.zero_grad()
            res = self.model.model.forward(data)
            if hasattr(F, self.loss_type):
                loss = getattr(F, self.loss_type)(res[mask], data.y[mask])
            else:
                raise TypeError(
                    "PyTorch does not support loss type {}".format(self.loss_type)
                )

            loss.backward()
            optimizer.step()
            if self.lr_scheduler_type:
                scheduler.step()

            if type(self.feval) is list:
                feval = self.feval[0]
            else:
                feval = self.feval
            val_loss = self.evaluate([data], mask=data.val_mask, feval=feval)
            if feval.is_higher_better() is True:
                val_loss = -val_loss
            self.early_stopping(val_loss, self.model.model)
            if self.early_stopping.early_stop:
                LOGGER.debug("Early stopping at %d", epoch)
                self.early_stopping.load_checkpoint(self.model.model)
                break

    def predict_only(self, data, test_mask=None):
        """
        The function of predicting on the given dataset and mask.

        Parameters
        ----------
        data: The node classification dataset used to be predicted.
        train_mask: The mask used in training stage.

        Returns
        -------
        res: The result of predicting on the given dataset.

        """
        # mask = data.test_mask if test_mask is None else test_mask
        data = data.to(self.device)
        self.model.model.eval()
        with torch.no_grad():
            res = self.model.model.forward(data)
        return res

    def train(self, dataset, keep_valid_result=True):
        """
        The function of training on the given dataset and keeping valid result.

        Parameters
        ----------
        dataset: The node classification dataset used to be trained.

        keep_valid_result: ``bool``
            If True(False), save the validation result after training.

        Returns
        -------
        self: ``autogl.train.NodeClassificationTrainer``
            A reference of current trainer.

        """
        data = dataset[0]
        self.train_only(data)
        if keep_valid_result:
            self.valid_result = self.predict_only(data)[data.val_mask].max(1)[1]
            self.valid_result_prob = self.predict_only(data)[data.val_mask]
            self.valid_score = self.evaluate(
                dataset, mask=data.val_mask, feval=self.feval
            )

    def predict(self, dataset, mask=None):
        """
        The function of predicting on the given dataset.

        Parameters
        ----------
        dataset: The node classification dataset used to be predicted.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        Returns
        -------
        The prediction result of ``predict_proba``.
        """
        return self.predict_proba(dataset, mask=mask, in_log_format=True).max(1)[1]

    def predict_proba(self, dataset, mask=None, in_log_format=False):
        """
        The function of predicting the probability on the given dataset.

        Parameters
        ----------
        dataset: The node classification dataset used to be predicted.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        in_log_format: ``bool``.
            If True(False), the probability will (not) be log format.

        Returns
        -------
        The prediction result.
        """
        data = dataset[0]
        data = data.to(self.device)
        if mask is not None:
            if mask == "val":
                mask = data.val_mask
            elif mask == "test":
                mask = data.test_mask
            elif mask == "train":
                mask = data.train_mask
        else:
            mask = data.test_mask
        ret = self.predict_only(data, mask)[mask]
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

    def evaluate(self, dataset, mask=None, feval=None):
        """
        The function of training on the given dataset and keeping valid result.

        Parameters
        ----------
        dataset: The node classification dataset used to be evaluated.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        feval: ``str``.
            The evaluation method used in this function.

        Returns
        -------
        res: The evaluation result on the given dataset.

        """
        data = dataset[0]
        data = data.to(self.device)
        test_mask = mask
        if feval is None:
            feval = self.feval
        else:
            feval = get_feval(feval)
        if test_mask is None:
            test_mask = data.test_mask
        elif test_mask == "test":
            test_mask = data.test_mask
        elif test_mask == "val":
            test_mask = data.val_mask
        elif test_mask == "train":
            test_mask = data.train_mask
        y_pred_prob = self.predict_proba(dataset, mask)
        y_pred = y_pred_prob.max(1)[1]
        y_true = data.y[test_mask]

        if not isinstance(feval, list):
            feval = [feval]
            return_signle = True
        else:
            return_signle = False

        res = []
        for f in feval:
            try:
                res.append(f.evaluate(y_pred_prob, y_true))
            except:
                res.append(f.evaluate(y_pred_prob.cpu().numpy(), y_true.cpu().numpy()))
        if return_signle:
            return res[0]
        return res

    def to(self, new_device):
        assert isinstance(new_device, torch.device)
        self.device = new_device
        if self.model is not None:
            self.model.to(self.device)

    def duplicate_from_hyper_parameter(self, hp: dict, model=None, restricted=True):
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
        self: ``autogl.train.NodeClassificationTrainer``
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
            optimizer=self.optimizer,
            lr=hp["lr"],
            max_epoch=hp["max_epoch"],
            early_stopping_round=hp["early_stopping_round"],
            device=self.device,
            weight_decay=hp["weight_decay"],
            feval=self.feval,
            init=True,
            *self.args,
            **self.kwargs
        )

        return ret

    def set_feval(self, feval):
        # """Set the evaluation metrics."""
        self.feval = get_feval(feval)

    @property
    def hyper_parameter_space(self):
        # """Get the space of hyperparameter."""
        return self.space

    @hyper_parameter_space.setter
    def hyper_parameter_space(self, space):
        # """Set the space of hyperparameter."""
        self.space = space
        NodeClassificationTrainer.space = space

    def get_hyper_parameter(self):
        # """Get the hyperparameter in this trainer."""
        return self.hyperparams
