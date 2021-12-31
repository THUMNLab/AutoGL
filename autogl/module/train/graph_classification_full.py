from . import register_trainer
from .base import BaseGraphClassificationTrainer, EarlyStopping
import torch
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
import torch.nn.functional as F
from ..model import BaseAutoModel, BaseDecoderMaintainer, BaseEncoderMaintainer
from .evaluation import get_feval, Logloss
from typing import Tuple, Type, Union
from ...datasets import utils
from copy import deepcopy
import torch.multiprocessing as mp

from ...utils import get_logger

from ...backend import DependentBackend

LOGGER = get_logger("graph classification solver")


@register_trainer("GraphClassificationFull")
class GraphClassificationFullTrainer(BaseGraphClassificationTrainer):
    """
    The graph classification trainer.

    Used to automatically train the graph classification problem.

    Parameters
    ----------
    model:
        Models can be ``str``, ``autogl.module.model.BaseAutoModel``, 
        ``autogl.module.model.encoders.BaseEncoderMaintainer`` or a tuple of (encoder, decoder) 
        if need to specify both encoder and decoder. Encoder can be ``str`` or
        ``autogl.module.model.encoders.BaseEncoderMaintainer``, and decoder can be ``str``
        or ``autogl.module.model.decoders.BaseDecoderMaintainer``.
        If only encoder is specified, decoder will be default to "logsoftmax"

    num_features: int (Optional)
        The number of features in dataset. default None
    
    num_classes: int (Optional)
        The number of classes. default None
    
    num_graph_features: int (Optional)
        The number of graph level features. default 0.

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

    space = None

    def __init__(
        self,
        model: Union[Tuple[BaseEncoderMaintainer, BaseDecoderMaintainer], BaseEncoderMaintainer, BaseAutoModel, str] = None,
        num_features: int = None,
        num_classes: int = None,
        num_graph_features: int = 0,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-4,
        max_epoch: int = 100,
        batch_size: int = 64,
        num_workers: int = 0,
        early_stopping_round: int = 7,
        weight_decay: float = 1e-4,
        device: Union[str, torch.device] = "auto",
        init: bool = False,
        feval=[Logloss],
        loss="nll_loss",
        lr_scheduler_type=None,
        criterion=None,
        *args,
        **kwargs
    ):
        if isinstance(model, Tuple):
            encoder, decoder = model
        elif isinstance(model, BaseAutoModel):
            encoder, decoder = model, None
        else:
            encoder, decoder = model, "sumpoolmlp"

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            num_features=num_features,
            num_classes=num_classes,
            num_graph_features=num_graph_features,
            last_dim="auto",
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
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.num_workers > 0:
            mp.set_start_method("fork", force=True)

        self.early_stopping_round = (
            early_stopping_round if early_stopping_round is not None else 100
        )

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
        self.criterion = criterion

        self.hyper_parameter_space = [
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

        self.hyper_parameters = {
            "max_epoch": self.max_epoch,
            "batch_size": self.batch_size,
            "early_stopping_round": self.early_stopping_round,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

        if init is True:
            self.initialize()

    @classmethod
    def get_task_name(cls):
        return "GraphClassification"

    def _train_only(self, train_loader, valid_loader=None):
        model = self._compose_model()

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

        for epoch in range(1, self.max_epoch + 1):
            model.train()
            loss_all = 0
            for data in train_loader:
                if self.pyg_dgl == 'pyg':
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    
                    if hasattr(F, self.loss):
                        loss = getattr(F, self.loss)(output, data.y)
                    else:
                        raise TypeError(
                            "PyTorch does not support loss type {}".format(self.loss)
                        )
                    loss.backward()
                    loss_all += data.num_graphs * loss.item()
                elif self.pyg_dgl == 'dgl':
                    data = [data[i].to(self.device) for i in range(len(data))]
                    data, labels = data
                    optimizer.zero_grad()
                    output = model(data)

                    if hasattr(F, self.loss):
                        loss = getattr(F, self.loss)(output, labels)
                    else:
                        raise TypeError(
                            "PyTorch does not support loss type {}".format(self.loss)
                        )

                    loss.backward()
                    loss_all += len(labels) * loss.item()

                optimizer.step()
                if self.lr_scheduler_type:
                    scheduler.step()

            if valid_loader is not None:
                eval_func = (
                    self.feval if not isinstance(self.feval, list) else self.feval[0]
                )
                val_loss = self._evaluate(valid_loader, eval_func)

                if eval_func.is_higher_better():
                    val_loss = -val_loss
                self.early_stopping(val_loss, model)

                if self.early_stopping.early_stop:
                    LOGGER.debug("Early stopping at", epoch)
                    break

        if valid_loader is not None:
            self.early_stopping.load_checkpoint(model)

    def _predict_only(self, loader, return_label=False):
        model = self._compose_model()
        model.eval()
        pred = []
        label = []
        for data in loader:
            if self.pyg_dgl == 'pyg':
                data = data.to(self.device)
                out = model(data)
                pred.append(out)
                label.append(data.y)
            elif self.pyg_dgl == 'dgl':
                data = [data[i].to(self.device) for i in range(len(data))]
                data, labels = data
                out = model(data)
                pred.append(out)
                label.append(labels)

        ret = torch.cat(pred, 0)
        label = torch.cat(label, 0)
        if return_label:
            return ret, label
        else:
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
            dataset, "train", batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        valid_loader = utils.graph_get_split(
            dataset, "val", batch_size=self.batch_size, num_workers=self.num_workers
        )
        self._train_only(train_loader, valid_loader)
        if keep_valid_result and valid_loader:
            pred = self._predict_only(valid_loader)
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

        loader = utils.graph_get_split(
            dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
        )
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
        loader = utils.graph_get_split(
            dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return self._predict_proba(loader, in_log_format)

    def _predict_proba(self, loader, in_log_format=False, return_label=False):
        if return_label:
            ret, label = self._predict_only(loader, return_label=True)
        else:
            ret = self._predict_only(loader, return_label=False)

        if self.pyg_dgl == 'dgl':
            ret = F.log_softmax(ret, dim=1)
        if in_log_format is False:
            ret = torch.exp(ret)

        if return_label:
            return ret, label
        else:
            return ret

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
                "encoder": repr(self.encoder),
                "decoder": repr(self.decoder)
            }
        )

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

        loader = utils.graph_get_split(
            dataset, mask, batch_size=self.batch_size, num_workers=self.num_workers
        )
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

    def duplicate_from_hyper_parameter(self, hp, encoder="same", decoder="same", restricted=True):
        """
        The function of duplicating a new instance from the given hyperparameter.

        Parameters
        ----------
        hp: ``dict``.
            The hyperparameter used in the new instance. Should contain 3 keys "trainer", "encoder"
            "decoder", with corresponding hyperparameters as values.

        model: The new model
            Models can be ``str``, ``autogl.module.model.BaseAutoModel``, 
            ``autogl.module.model.encoders.BaseEncoderMaintainer`` or a tuple of (encoder, decoder) 
            if need to specify both encoder and decoder. Encoder can be ``str`` or
            ``autogl.module.model.encoders.BaseEncoderMaintainer``, and decoder can be ``str``
            or ``autogl.module.model.decoders.BaseDecoderMaintainer``.

        restricted: ``bool``.
            If False(True), the hyperparameter should (not) be updated from origin hyperparameter.

        Returns
        -------
        self: ``autogl.train.GraphClassificationTrainer``
            A new instance of trainer.

        """
        hp_trainer = hp.get("trainer", {})
        hp_encoder = hp.get("encoder", {})
        hp_decoder = hp.get("decoder", {})
        if not restricted:
            origin_hp = deepcopy(self.hyper_parameters)
            origin_hp.update(hp_trainer)
            hp = origin_hp
        else:
            hp = hp_trainer
        encoder = encoder if encoder != "same" else self.encoder
        decoder = decoder if decoder != "same" else self.decoder
        encoder = encoder.from_hyper_parameter(hp_encoder)
        if isinstance(encoder, BaseEncoderMaintainer) and isinstance(decoder, BaseDecoderMaintainer):
            decoder = decoder.from_hyper_parameter_and_encoder(hp_decoder, encoder)

        ret = self.__class__(
            model=(encoder, decoder),
            num_features=self.num_features,
            num_classes=self.num_classes,
            num_graph_features=self.num_graph_features,
            optimizer=self.opt_received,
            lr=hp["lr"],
            max_epoch=hp["max_epoch"],
            batch_size=hp["batch_size"],
            early_stopping_round=hp["early_stopping_round"],
            weight_decay=hp["weight_decay"],
            device=self.device,
            feval=self.feval,
            loss=self.loss,
            lr_scheduler_type=self.lr_scheduler_type,
            init=True,
            *self.args,
            **self.kwargs
        )

        return ret
