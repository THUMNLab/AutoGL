"""
Node classification Full Trainer Implementation
"""

from . import register_trainer

from .base import BaseNodeClassificationTrainer, EarlyStopping
import torch
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
import torch.nn.functional as F
from ..model import BaseEncoderMaintainer, BaseDecoderMaintainer, BaseAutoModel
from .evaluation import Evaluation, get_feval, Logloss
from typing import Callable, Iterable, Optional, Tuple, Type, Union
from copy import deepcopy

from ...utils import get_logger

from ...backend import DependentBackend

LOGGER = get_logger("node classification trainer")


@register_trainer("NodeClassificationFull")
class NodeClassificationFullTrainer(BaseNodeClassificationTrainer):
    """
    The node classification trainer.

    Used to automatically train the node classification problem.

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
        model: Union[Tuple[BaseEncoderMaintainer, BaseDecoderMaintainer], BaseEncoderMaintainer, BaseAutoModel, str] = None,
        num_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = torch.optim.Adam,
        lr: float = 1e-4,
        max_epoch: int = 100,
        early_stopping_round: int = 100,
        weight_decay: float = 1e-4,
        device: Union[torch.device, str] = "auto",
        init: bool = False,
        feval: Iterable[Type[Evaluation]] =[Logloss],
        loss: Union[Callable, str] = "nll_loss",
        lr_scheduler_type: Optional[str] = None,
        **kwargs
    ):
        if isinstance(model, Tuple):
            encoder, decoder = model
        elif isinstance(model, BaseAutoModel):
            encoder, decoder = model, None
        else:
            encoder, decoder = model, "logsoftmax"

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            num_features=num_features,
            num_classes=num_classes,
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

    @classmethod
    def get_task_name(cls):
        """
        Derive the task name. (NodeClassification)
        """
        return "NodeClassification"

    def __train_only(self, data, train_mask=None):
        data = data.to(self.device)
        model = self._compose_model()
        if train_mask is None:
            if self.pyg_dgl == 'pyg':
                mask = data.train_mask
            elif self.pyg_dgl == 'dgl':
                mask = data.ndata['train_mask']
        else:
            mask = train_mask
                
        optimizer = self.optimizer(
            model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay
        )
        # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
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
            optimizer.zero_grad()
            res = model(data)
            if hasattr(F, self.loss):
                if self.pyg_dgl == 'pyg':
                    loss = getattr(F, self.loss)(res[mask], data.y[mask])
                elif self.pyg_dgl == 'dgl':
                    loss = getattr(F, self.loss)(res[mask], data.ndata['label'][mask])
            else:
                raise TypeError(
                    "PyTorch does not support loss type {}".format(self.loss)
                )

            loss.backward()
            optimizer.step()
            if self.lr_scheduler_type:
                scheduler.step()

            # TODO: move this to autogl.backend.utils
            if self.pyg_dgl == 'pyg' and hasattr(data, "val_mask") and data.val_mask is not None:
                val_mask = data.val_mask
            elif self.pyg_dgl == 'dgl' and data.ndata.get('val_mask', None) is not None:
                val_mask = data.ndata['val_mask']
            else:
                val_mask = None

            if val_mask is not None:
                if type(self.feval) is list:
                    feval = self.feval[0]
                else:
                    feval = self.feval
                val_loss = self.evaluate([data], mask=val_mask, feval=feval)
                if feval.is_higher_better() is True:
                    val_loss = -val_loss

                self.early_stopping(val_loss, model)
                if self.early_stopping.early_stop:
                    LOGGER.debug("Early stopping at %d", epoch)
                    break

        if self.pyg_dgl == "pyg" and hasattr(data, "val_mask") and data.val_mask is not None:
            self.early_stopping.load_checkpoint(model)
        elif self.pyg_dgl == 'dgl' and data.ndata.get('val_mask', None) is not None:
            self.early_stopping.load_checkpoint(model)

    @torch.no_grad()
    def __predict_only(self, data, mask=None):
        if isinstance(mask, str):
            if self.pyg_dgl == 'pyg':
                mask = getattr(data, f'{mask}_mask')
            elif self.pyg_dgl == 'dgl':
                mask = data.ndata[f'{mask}_mask']
        
        model = self._compose_model()
        model.to(self.device)

        data = data.to(self.device)
        model.eval()
        res = model(data)
            
        if mask is None:
            return res
        else:
            return res[mask]

    def train(self, dataset, keep_valid_result=True, train_mask=None):
        """
        Train on the given dataset.

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
        data = dataset[0]
        self.__train_only(data, train_mask)
        if keep_valid_result:
            if self.pyg_dgl == 'pyg':
                val_mask = data.val_mask
            elif self.pyg_dgl == 'dgl':
                val_mask = data.ndata['val_mask']
            else:
                assert False
            self.valid_result = self.__predict_only(data)[val_mask].max(1)[1]
            self.valid_result_prob = self.__predict_only(data)[val_mask]
            self.valid_score = self.evaluate(
                dataset, mask=val_mask, feval=self.feval
            )
        
    def predict(self, dataset, mask=None):
        """
        Predict on the given dataset using specified mask.

        Parameters
        ----------
        dataset: The node classification dataset used to be predicted.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        Returns
        -------
        The prediction result.
        """
        return self.predict_proba(dataset, mask=mask, in_log_format=True).max(1)[1]

    def predict_proba(self, dataset, mask=None, in_log_format=False):
        """
        Predict the probability on the given dataset using specified mask.

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
        data = dataset[0]
        data = data.to(self.device)
        ret = self.__predict_only(data, mask)
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
                "encoder": repr(self.encoder),
                "decoder": repr(self.decoder)
            }
        )

    def evaluate(self, dataset, mask=None, feval=None):
        """
        Evaluate on the given dataset.

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
        
        if isinstance(mask, str):
            if self.pyg_dgl == 'pyg':
                mask = getattr(data, f'{mask}_mask')
            elif self.pyg_dgl == 'dgl':
                mask = data.ndata[f'{mask}_mask']
        
        if self.pyg_dgl == 'pyg': label = data.y
        elif self.pyg_dgl == 'dgl': label = data.ndata['label']

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

    def duplicate_from_hyper_parameter(self, hp: dict, model=None, restricted=True):
        """
        The function of duplicating a new instance from the given hyperparameter.

        Parameters
        ----------
        hp: ``dict``.
            The hyperparameter used in the new instance. Should contain 3 keys "trainer", "encoder"
            "decoder", with corresponding hyperparameters as values.

        model:
            Models can be ``str``, ``autogl.module.model.BaseAutoModel``, 
            ``autogl.module.model.encoders.BaseEncoderMaintainer`` or a tuple of (encoder, decoder) 
            if need to specify both encoder and decoder. Encoder can be ``str`` or
            ``autogl.module.model.encoders.BaseEncoderMaintainer``, and decoder can be ``str``
            or ``autogl.module.model.decoders.BaseDecoderMaintainer``.

        restricted: ``bool``.
            If False(True), the hyperparameter should (not) be updated from origin hyperparameter.

        Returns
        -------
        self: ``autogl.train.NodeClassificationTrainer``
            A new instance of trainer.

        """
        if isinstance(model, Tuple):
            encoder, decoder = model
        elif isinstance(model, BaseAutoModel):
            encoder, decoder = model, None
        elif isinstance(model, BaseEncoderMaintainer):
            encoder, decoder = model, self.decoder
        elif model is None:
            encoder, decoder = self.encoder, self.decoder
        else:
            raise TypeError("Cannot parse model with type", type(model))
        
        hp_trainer = hp.get("trainer", {})
        hp_encoder = hp.get("encoder", {})
        hp_decoder = hp.get("decoder", {})
        if not restricted:
            origin_hp = deepcopy(self.hyper_parameters)
            origin_hp.update(hp_trainer)
            hp = origin_hp
        else:
            hp = hp_trainer
        encoder = encoder.from_hyper_parameter(hp_encoder)
        if isinstance(encoder, BaseEncoderMaintainer) and isinstance(decoder, BaseDecoderMaintainer):
            decoder = decoder.from_hyper_parameter_and_encoder(hp_decoder, encoder)

        ret = self.__class__(
            model=(encoder, decoder),
            num_features=self.num_features,
            num_classes=self.num_classes,
            optimizer=self.opt_received,
            lr=hp["lr"],
            max_epoch=hp["max_epoch"],
            early_stopping_round=hp["early_stopping_round"],
            device=self.device,
            weight_decay=hp["weight_decay"],
            feval=self.feval,
            loss=self.loss,
            lr_scheduler_type=self.lr_scheduler_type,
            init=True,
            **self.kwargs
        )

        return ret
