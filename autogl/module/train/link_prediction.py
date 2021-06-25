from . import register_trainer, Evaluation
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from ..model import MODEL_DICT, BaseModel
from .evaluation import Auc, EVALUATE_DICT
from .base import EarlyStopping, BaseLinkPredictionTrainer
from typing import Union
from copy import deepcopy
from torch_geometric.utils import negative_sampling

from ...utils import get_logger

LOGGER = get_logger("link prediction trainer")


def get_feval(feval):
    if isinstance(feval, str):
        return EVALUATE_DICT[feval]
    if isinstance(feval, type) and issubclass(feval, Evaluation):
        return feval
    if isinstance(feval, list):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")


@register_trainer("LinkPredictionFull")
class LinkPredictionTrainer(BaseLinkPredictionTrainer):
    """
    The link prediction trainer.

    Used to automatically train the link prediction problem.

    Parameters
    ----------
    model: ``BaseModel`` or ``str``
        The (name of) model used to train and predict.

    optimizer: ``Optimizer`` of ``str``
        The (name of) optimizer used to train and predict.

    lr: ``float``
        The learning rate of link prediction task.

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
        model: Union[BaseModel, str] = None,
        num_features=None,
        optimizer=None,
        lr=1e-4,
        max_epoch=100,
        early_stopping_round=101,
        weight_decay=1e-4,
        device="auto",
        init=True,
        feval=[Auc],
        loss="binary_cross_entropy_with_logits",
        *args,
        **kwargs,
    ):
        super().__init__(model, num_features, device, init, feval, loss)

        if type(optimizer) == str and optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam
        elif type(optimizer) == str and optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD
        else:
            self.optimizer = torch.optim.Adam

        self.lr = lr
        self.max_epoch = max_epoch
        self.early_stopping_round = early_stopping_round
        self.device = device
        self.args = args
        self.kwargs = kwargs
        self.weight_decay = weight_decay

        self.early_stopping = EarlyStopping(
            patience=early_stopping_round, verbose=False
        )

        self.valid_result = None
        self.valid_result_prob = None
        self.valid_score = None

        self.initialized = False
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

        LinkPredictionTrainer.space = self.space

        self.hyperparams = {
            "max_epoch": self.max_epoch,
            "early_stopping_round": self.early_stopping_round,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

        if init is True:
            self.initialize()

    def initialize(self):
        #  Initialize the auto model in trainer.
        if self.initialized is True:
            return
        self.initialized = True
        self.model.set_num_classes(self.num_classes)
        self.model.set_num_features(self.num_features)
        self.model.initialize()

    def get_model(self):
        # Get auto model used in trainer.
        return self.model

    @classmethod
    def get_task_name(cls):
        # Get task name, i.e., `LinkPrediction`.
        return "LinkPrediction"

    def train_only(self, data, train_mask=None):
        """
        The function of training on the given dataset and mask.

        Parameters
        ----------
        data: The link prediction dataset used to be trained. It should consist of masks, including train_mask, and etc.
        train_mask: The mask used in training stage.

        Returns
        -------
        self: ``autogl.train.LinkPredictionTrainer``
            A reference of current trainer.

        """

        # data.train_mask = data.val_mask = data.test_mask = data.y = None
        # data = train_test_split_edges(data)
        data = data.to(self.device)
        # mask = data.train_mask if train_mask is None else train_mask
        optimizer = self.optimizer(
            self.model.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(1, self.max_epoch):
            self.model.model.train()

            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.train_pos_edge_index.size(1),
            )

            optimizer.zero_grad()
            # res = self.model.model.forward(data)
            z = self.model.model.lp_encode(data)
            link_logits = self.model.model.lp_decode(
                z, data.train_pos_edge_index, neg_edge_index
            )
            link_labels = self.get_link_labels(
                data.train_pos_edge_index, neg_edge_index
            )
            # loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
            if hasattr(F, self.loss):
                loss = getattr(F, self.loss)(link_logits, link_labels)
            else:
                raise TypeError(
                    "PyTorch does not support loss type {}".format(self.loss)
                )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if type(self.feval) is list:
                feval = self.feval[0]
            else:
                feval = self.feval
            val_loss = self.evaluate([data], mask="val", feval=feval)
            if feval.is_higher_better() is True:
                val_loss = -val_loss
            self.early_stopping(val_loss, self.model.model)
            if self.early_stopping.early_stop:
                LOGGER.debug("Early stopping at %d", epoch)
                break
        self.early_stopping.load_checkpoint(self.model.model)

    def predict_only(self, data, test_mask=None):
        """
        The function of predicting on the given dataset and mask.

        Parameters
        ----------
        data: The link prediction dataset used to be predicted.
        train_mask: The mask used in training stage.

        Returns
        -------
        res: The result of predicting on the given dataset.

        """
        data = data.to(self.device)
        self.model.model.eval()
        with torch.no_grad():
            z = self.model.model.lp_encode(data)
        return z

    def train(self, dataset, keep_valid_result=True):
        """
        The function of training on the given dataset and keeping valid result.

        Parameters
        ----------
        dataset: The link prediction dataset used to be trained.

        keep_valid_result: ``bool``
            If True(False), save the validation result after training.

        Returns
        -------
        self: ``autogl.train.LinkPredictionTrainer``
            A reference of current trainer.

        """
        data = dataset[0]
        data.edge_index = data.train_pos_edge_index
        self.train_only(data)
        if keep_valid_result:
            self.valid_result = self.predict_only(data)
            self.valid_result_prob = self.predict_proba(dataset, "val")
            self.valid_score = self.evaluate(dataset, mask="val", feval=self.feval)

    def predict(self, dataset, mask=None):
        """
        The function of predicting on the given dataset.

        Parameters
        ----------
        dataset: The link prediction dataset used to be predicted.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        Returns
        -------
        The prediction result of ``predict_proba``.
        """
        return self.predict_proba(dataset, mask=mask, in_log_format=False)

    def predict_proba(self, dataset, mask=None, in_log_format=False):
        """
        The function of predicting the probability on the given dataset.

        Parameters
        ----------
        dataset: The link prediction dataset used to be predicted.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        in_log_format: ``bool``.
            If True(False), the probability will (not) be log format.

        Returns
        -------
        The prediction result.
        """
        data = dataset[0]
        data.edge_index = data.train_pos_edge_index
        data = data.to(self.device)
        if mask in ["train", "val", "test"]:
            pos_edge_index = data[f"{mask}_pos_edge_index"]
            neg_edge_index = data[f"{mask}_neg_edge_index"]
        else:
            pos_edge_index = data[f"test_pos_edge_index"]
            neg_edge_index = data[f"test_neg_edge_index"]

        self.model.model.eval()
        with torch.no_grad():
            z = self.predict_only(data)
            link_logits = self.model.model.lp_decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()

        return link_probs

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
        dataset: The link prediction dataset used to be evaluated.

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

        if mask in ["train", "val", "test"]:
            pos_edge_index = data[f"{mask}_pos_edge_index"]
            neg_edge_index = data[f"{mask}_neg_edge_index"]
        else:
            pos_edge_index = data[f"test_pos_edge_index"]
            neg_edge_index = data[f"test_neg_edge_index"]

        self.model.model.eval()
        with torch.no_grad():
            link_probs = self.predict_proba(dataset, mask)
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)

        if not isinstance(feval, list):
            feval = [feval]
            return_signle = True
        else:
            return_signle = False

        res = []
        for f in feval:
            res.append(f.evaluate(link_probs.cpu().numpy(), link_labels.cpu().numpy()))
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
        self: ``autogl.train.LinkPredictionTrainer``
            A new instance of trainer.

        """
        if not restricted:
            origin_hp = deepcopy(self.hyperparams)
            origin_hp.update(hp)
            hp = origin_hp
        if model is None:
            model = self.model
        model.set_num_classes(self.num_classes)
        model.set_num_features(self.num_features)
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
            optimizer=self.optimizer,
            lr=hp["lr"],
            max_epoch=hp["max_epoch"],
            early_stopping_round=hp["early_stopping_round"],
            device=self.device,
            weight_decay=hp["weight_decay"],
            feval=self.feval,
            init=True,
            *self.args,
            **self.kwargs,
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
        LinkPredictionTrainer.space = space

    def get_hyper_parameter(self):
        # """Get the hyperparameter in this trainer."""
        return self.hyperparams

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=self.device)
        link_labels[: pos_edge_index.size(1)] = 1.0
        return link_labels
