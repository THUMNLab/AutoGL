"""
Node classification Het Trainer Implementation
"""

from . import register_trainer

from .base import BaseNodeClassificationHetTrainer, EarlyStopping
import torch
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
import torch.nn.functional as F
from ..model import BaseAutoModel
from .evaluation import get_feval, Logloss
from typing import Union
from copy import deepcopy
from sklearn.metrics import f1_score

from ...utils import get_logger

from ...backend import DependentBackend

LOGGER = get_logger("node classification het trainer")

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

@register_trainer("NodeClassificationHet")
class NodeClassificationHetTrainer(BaseNodeClassificationHetTrainer):
    """
    The heterogeneous node classification trainer.

    Parameters
    ----------
    model: ``autogl.module.model.BaseAutoModel``
        Currently Heterogeneous trainer doesn't support decoupled model setting.

    num_features: ``int`` (Optional)
        The number of features in dataset. default None
    
    num_classes: ``int`` (Optional)
        The number of classes. default None

    optimizer: ``Optimizer`` of ``str``
        The (name of) optimizer used to train and predict. default torch.optim.Adam

    lr: ``float``
        The learning rate of node classification task. default 1e-4

    max_epoch: ``int``
        The max number of epochs in training. default 100

    early_stopping_round: ``int``
        The round of early stop. default 100

    weight_decay: ``float``
        weight decay ratio, default 1e-4

    device: ``torch.device`` or ``str``
        The device where model will be running on.

    init: ``bool``
        If True(False), the model will (not) be initialized.

    feval: (Sequence of) ``Evaluation`` or ``str``
        The evaluation functions, default ``[LogLoss]``
    
    loss: ``str``
        The loss used. Default ``nll_loss``.

    lr_scheduler_type: ``str`` (Optional)
        The lr scheduler type used. Default None.

    """

    def __init__(
        self,
        model: Union[BaseAutoModel, str] = None,
        dataset = None,
        num_features=None,
        num_classes=None,
        optimizer=torch.optim.AdamW,
        lr=1e-4,
        max_epoch=100,
        early_stopping_round=100,
        weight_decay=1e-4,
        device="auto",
        init=False,
        feval=[Logloss],
        loss="nll_loss",
        lr_scheduler_type=None,
        *args,
        **kwargs
    ):
        super().__init__(
            model,
            dataset,
            num_features,
            num_classes,
            device=device,
            feval=feval,
            loss=loss,
        )
       
        self.opt_received = optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam": self.optimizer = torch.optim.Adam
            elif optimizer.lower() == "sgd": self.optimizer = torch.optim.SGD
            else: raise ValueError("Currently not support optimizer {}".format(optimizer))
        elif isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError("Currently not support optimizer {}".format(optimizer))

        self.lr_scheduler_type = lr_scheduler_type

        self.lr = lr
        self.max_epoch = max_epoch
        self.early_stopping_round = early_stopping_round
        self.args = args
        self.kwargs = kwargs
        self.weight_decay = weight_decay

        self.early_stopping = EarlyStopping(
            patience=early_stopping_round, verbose=False
        )

        self.valid_result = None
        self.valid_result_prob = None
        self.valid_score = None

        self.pyg_dgl = DependentBackend.get_backend_name()

        self.hyper_parameter_space = [
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

        self.hyper_parameters = {
            "max_epoch": self.max_epoch,
            "early_stopping_round": self.early_stopping_round,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

        if init is True:
            self.initialize()

    def _initialize(self):
        self.encoder.initialize()

    @classmethod
    def get_task_name(cls):
        """
        Get task name ("NodeClassificationHet")
        """
        return "NodeClassificationHet"

    def _train_only(self, dataset, train_mask="train"):
        G = dataset[0].to(self.device)
        field = dataset.schema["target_node_type"]
        labels = G.nodes[field].data['label'].to(self.device)
        train_mask = self._get_mask(dataset, train_mask).to(self.device)
        val_mask = self._get_mask(dataset, "val").to(self.device)
        model = self.encoder.model.to(self.device)
        optimizer = self.optimizer(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        lr_scheduler_type = self.lr_scheduler_type
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

        for epoch in range(1, self.max_epoch):
            model.train()
            optimizer.zero_grad()
            logits = model(G)

            if hasattr(F, self.loss):
                loss = getattr(F, self.loss)(logits[train_mask], labels[train_mask])
            else:
                raise TypeError(
                    "PyTorch does not support loss type {}".format(self.loss)
                )

            loss.backward()
            optimizer.step()
            if self.lr_scheduler_type:
                scheduler.step()

            if val_mask is not None:
                if type(self.feval) is list:
                    feval = self.feval[0]
                else:
                    feval = self.feval

                val_loss = self.evaluate(dataset, mask=val_mask, feval=feval)
                if feval.is_higher_better() is True:
                    val_loss = -val_loss

                self.early_stopping(val_loss, model)
                if self.early_stopping.early_stop:
                    LOGGER.debug("Early stopping at %d", epoch)
                    break
        if val_mask is not None:
            self.early_stopping.load_checkpoint(model)

    def _predict_only(self, dataset, mask=None):
        model = self.encoder.model.to(self.device)
        model.eval()
        G = dataset[0].to(self.device)
        with torch.no_grad():
            res = model(G)

        if mask is None:
            return res
        else:
            return res[mask]

    def train(self, dataset, keep_valid_result=True, train_mask="train"):
        """
        The function of training on the given dataset and keeping valid result.
        Parameters
        ----------
        dataset: The node classification dataset used to be trained.
        keep_valid_result: ``bool``
            If True(False), save the validation result after training.
        train_mask: The mask for training data
        Returns
        -------
        self: ``autogl.train.NodeClassificationTrainer``
            A reference of current trainer.
        """
        self._train_only(dataset, train_mask)
        G = dataset[0].to(self.device)
        if keep_valid_result:
            # generate labels
            val_mask = G.nodes[dataset.schema["target_node_type"]].data["val_mask"]
            self.valid_result = self._predict_only(dataset)[val_mask].max(1)[1]
            self.valid_result_prob = self._predict_only(dataset)[val_mask]
            self.valid_score = self.evaluate(
                dataset, mask=val_mask, feval=self.feval
            )
            # print(self.valid_score)

    def predict(self, dataset, mask="test"):
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

    def predict_proba(self, dataset, mask="test", in_log_format=False):
        """
        The function of predicting the probability on the given dataset.
        Parameters
        ----------
        dataset: The node classification dataset used to be predicted.
        mask: ``train``, ``val``, ``test``, or ``Tensor``.
            The dataset mask.
        in_log_format: ``bool``.
            If True(False), the probability will (not) be log format.
        Returns
        -------
        The prediction result.
        """
        G = dataset[0].to(self.device)
        if mask in ["train", "val", "test"]:
            mask = G.nodes[dataset.schema["target_node_type"]].data[f"{mask}_mask"]
        ret = self._predict_only(dataset, mask)
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

    def __repr__(self) -> str:
        import yaml

        return yaml.dump(
            {
                "trainer_name": self.__class__.__name__,
                "optimizer": self.optimizer,
                "learning_rate": self.lr,
                "max_epoch": self.max_epoch,
                "early_stopping_round": self.early_stopping_round,
                "model": repr(self.model.model),
            }
        )
    
    def _get_mask(self, dataset, mask):
        if mask in ["train", "val", "test"]:
            return dataset[0].nodes[dataset.schema["target_node_type"]].data[f"{mask}_mask"]
        return mask

    def evaluate(self, dataset, mask='val', feval = None):
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
        G = dataset[0].to(self.device)

        mask = self._get_mask(dataset, mask)
        label = G.nodes[dataset.schema["target_node_type"]].data['label'].to(self.device)

        if feval is None:
            feval = self.feval
        else:
            feval = get_feval(feval)

        y_pred_prob = self.predict_proba(dataset, mask)

        y_true = label[mask] if mask is not None else label

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
        self.device = new_device
        if self.model is not None:
            self.model.to(self.device)

    def duplicate_from_hyper_parameter(self, hp: dict, model=None, restricted=True):
        """
        The function of duplicating a new instance from the given hyperparameter.
        
        Parameters
        ----------
        hp: ``dict``.
            The hyperparameter used in the new instance. Should contain 2 keys "trainer", "encoder"
            with corresponding hyperparameters as values.
        model: ``autogl.module.model.BaseAutoModel``
            Currently Heterogeneous trainer doesn't support decoupled model setting.
            If only encoder is specified, decoder will be default to "logsoftmax"

        restricted: ``bool``.
            If False(True), the hyperparameter should (not) be updated from origin hyperparameter.
        
        Returns
        -------
        self: ``autogl.train.NodeClassificationTrainer``
            A new instance of trainer.
        
        """
        trainer_hp = hp["trainer"]
        model_hp = hp["encoder"]
        if not restricted:
            origin_hp = deepcopy(self.hyper_parameters)
            origin_hp.update(trainer_hp)
            trainer_hp = origin_hp
        if model is None:
            model = self.model
        model = model.from_hyper_parameter(model_hp)

        ret = self.__class__(
            model=model,
            dataset=self._dataset,
            num_features=self.num_features,
            num_classes=self.num_classes,
            optimizer=self.opt_received,
            lr=trainer_hp["lr"],
            max_epoch=trainer_hp["max_epoch"],
            early_stopping_round=trainer_hp["early_stopping_round"],
            device=self.device,
            weight_decay=trainer_hp["weight_decay"],
            feval=self.feval,
            loss=self.loss,
            lr_scheduler_type=self.lr_scheduler_type,
            init=True,
            *self.args,
            **self.kwargs
        )

        return ret
