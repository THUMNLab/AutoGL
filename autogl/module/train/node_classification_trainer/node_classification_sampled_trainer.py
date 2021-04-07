import torch
import logging
import typing as _typing
from torch.nn import functional as F

from .. import register_trainer
from ..base import BaseNodeClassificationTrainer, EarlyStopping, Evaluation
from ..evaluation import get_feval, Logloss
from ..sampling.sampler.neighbor_sampler import NeighborSampler
from ...model import BaseModel

LOGGER: logging.Logger = logging.getLogger("Node classification sampling trainer")


@register_trainer("NodeClassificationNeighborSampling")
class NodeClassificationNeighborSamplingTrainer(BaseNodeClassificationTrainer):
    """
    The node classification trainer
    for automatically training the node classification tasks
    with neighbour sampling
    """

    def __init__(
        self,
        model: _typing.Union[BaseModel, str],
        num_features: int,
        num_classes: int,
        optimizer: _typing.Union[_typing.Type[torch.optim.Optimizer], str, None] = None,
        lr: float = 1e-4,
        max_epoch: int = 100,
        early_stopping_round: int = 100,
        weight_decay: float = 1e-4,
        device: _typing.Optional[torch.device] = None,
        init: bool = True,
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Logloss,),
        loss: str = "nll_loss",
        lr_scheduler_type: _typing.Optional[str] = None,
        **kwargs
    ) -> None:
        if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            self._optimizer_class: _typing.Type[torch.optim.Optimizer] = optimizer
        elif type(optimizer) == str:
            if optimizer.lower() == "adam":
                self._optimizer_class: _typing.Type[
                    torch.optim.Optimizer
                ] = torch.optim.Adam
            elif optimizer.lower() == "adam" + "w":
                self._optimizer_class: _typing.Type[
                    torch.optim.Optimizer
                ] = torch.optim.AdamW
            elif optimizer.lower() == "sgd":
                self._optimizer_class: _typing.Type[
                    torch.optim.Optimizer
                ] = torch.optim.SGD
            else:
                self._optimizer_class: _typing.Type[
                    torch.optim.Optimizer
                ] = torch.optim.Adam
        else:
            self._optimizer_class: _typing.Type[
                torch.optim.Optimizer
            ] = torch.optim.Adam

        self._learning_rate: float = lr if lr > 0 else 1e-4
        self._lr_scheduler_type: _typing.Optional[str] = lr_scheduler_type
        self._max_epoch: int = max_epoch if max_epoch > 0 else 1e2

        self.__sampling_sizes: _typing.Sequence[int] = kwargs.get("sampling_sizes")

        self._weight_decay: float = weight_decay if weight_decay > 0 else 1e-4
        early_stopping_round: int = (
            early_stopping_round if early_stopping_round > 0 else 1e2
        )
        self._early_stopping = EarlyStopping(
            patience=early_stopping_round, verbose=False
        )
        super(NodeClassificationNeighborSamplingTrainer, self).__init__(
            model, num_features, num_classes, device, init, feval, loss
        )

        self._valid_result: torch.Tensor = torch.zeros(0)
        self._valid_result_prob: torch.Tensor = torch.zeros(0)
        self._valid_score = None

        self._hyper_parameter_space: _typing.List[_typing.Dict[str, _typing.Any]] = [
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

        self._hyper_parameter: _typing.Dict[str, _typing.Any] = {
            "max_epoch": self._max_epoch,
            "early_stopping_round": self._early_stopping.patience,
            "lr": self._learning_rate,
            "weight_decay": self._weight_decay,
        }

        self.__initialized: bool = False
        if init:
            self.initialize()

    def initialize(self) -> "NodeClassificationNeighborSamplingTrainer":
        if self.__initialized:
            return self
        self._model.initialize()
        self.__initialized = True
        return self

    def get_model(self) -> BaseModel:
        return self._model

    def __train_only(self, data) -> "NodeClassificationNeighborSamplingTrainer":
        """
        The function of training on the given dataset and mask.
        :param data: data of a specific graph
        :return: self
        """
        data = data.to(self.device)
        optimizer: torch.optim.Optimizer = self._optimizer_class(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        if type(self._lr_scheduler_type) == str:
            if self._lr_scheduler_type.lower() == "step" + "lr":
                lr_scheduler: torch.optim.lr_scheduler.StepLR = (
                    torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
                )
            elif self._lr_scheduler_type.lower() == "multi" + "step" + "lr":
                lr_scheduler: torch.optim.lr_scheduler.MultiStepLR = (
                    torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones=[30, 80], gamma=0.1
                    )
                )
            elif self._lr_scheduler_type.lower() == "exponential" + "lr":
                lr_scheduler: torch.optim.lr_scheduler.ExponentialLR = (
                    torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
                )
            elif self._lr_scheduler_type.lower() == "ReduceLROnPlateau".lower():
                lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = (
                    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
                )
            else:
                lr_scheduler: torch.optim.lr_scheduler.LambdaLR = (
                    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                )
        else:
            lr_scheduler: torch.optim.lr_scheduler.LambdaLR = (
                torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            )

        train_sampler: NeighborSampler = NeighborSampler(
            data, self.__sampling_sizes, batch_size=20
        )

        for current_epoch in range(self._max_epoch):
            self._model.model.train()
            """ epoch start """
            for target_node_indexes, edge_indexes in train_sampler:
                optimizer.zero_grad()
                data.edge_indexes = edge_indexes
                prediction = self._model.model(data)
                if not hasattr(F, self.loss):
                    raise TypeError(
                        "PyTorch does not support loss type {}".format(self.loss)
                    )
                loss_function = getattr(F, self.loss)
                loss: torch.Tensor = loss_function(
                    prediction[target_node_indexes], data.y[target_node_indexes]
                )
                loss.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            """ Validate performance """
            if hasattr(data, "val_mask") and getattr(data, "val_mask") is not None:
                validation_results: _typing.Sequence[float] = self.evaluate(
                    (data,), "val", [self.feval[0]]
                )

                if self.feval[0].is_higher_better():
                    validation_loss: float = -validation_results[0]
                else:
                    validation_loss: float = validation_results[0]
                self._early_stopping(validation_loss, self._model.model)
                if self._early_stopping.early_stop:
                    LOGGER.debug("Early stopping at %d", current_epoch)
                    break
        if hasattr(data, "val_mask") and data.val_mask is not None:
            self._early_stopping.load_checkpoint(self._model.model)
        return self

    def __predict_only(self, data):
        """
        The function of predicting on the given data.
        :param data: data of a specific graph
        :return: the result of prediction on the given dataset
        """
        data = data.to(self.device)
        self._model.model.eval()
        with torch.no_grad():
            prediction = self._model.model(data)
        return prediction

    def train(self, dataset, keep_valid_result: bool = True):
        """
        The function of training on the given dataset and keeping valid result.
        :param dataset:
        :param keep_valid_result: Whether to save the validation result after training
        """
        data = dataset[0]
        self.__train_only(data)
        if keep_valid_result:
            prediction: torch.Tensor = self.__predict_only(data)
            self._valid_result: torch.Tensor = prediction[data.val_mask].max(1)[1]
            self._valid_result_prob: torch.Tensor = prediction[data.val_mask]
            self._valid_score = self.evaluate(dataset, "val")

    def predict_proba(
        self, dataset, mask: _typing.Optional[str] = None, in_log_format: bool = False
    ) -> torch.Tensor:
        """
        The function of predicting the probability on the given dataset.
        :param dataset: The node classification dataset used to be predicted.
        :param mask:
        :param in_log_format:
        :return:
        """
        data = dataset[0].to(self.device)
        if mask is not None and type(mask) == str:
            if mask.lower() == "train":
                _mask = data.train_mask
            elif mask.lower() == "test":
                _mask = data.test_mask
            elif mask.lower() == "val":
                _mask = data.val_mask
            else:
                _mask = data.test_mask
        else:
            _mask = data.test_mask
        result = self.__predict_only(data)[_mask]
        return result if in_log_format else torch.exp(result)

    def predict(self, dataset, mask: _typing.Optional[str] = None) -> torch.Tensor:
        return self.predict_proba(dataset, mask, in_log_format=True).max(1)[1]

    def get_valid_predict(self) -> torch.Tensor:
        return self._valid_result

    def get_valid_predict_proba(self) -> torch.Tensor:
        return self._valid_result_prob

    def get_valid_score(self, return_major: bool = True):
        if return_major:
            return (self._valid_score[0], self.feval[0].is_higher_better())
        else:
            return (self._valid_score, [f.is_higher_better() for f in self.feval])

    def get_name_with_hp(self) -> str:
        # """Get the name of hyperparameter."""
        name = "-".join(
            [
                str(self._optimizer_class),
                str(self._learning_rate),
                str(self._max_epoch),
                str(self._early_stopping.patience),
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

    def evaluate(
        self,
        dataset,
        mask: _typing.Optional[str] = None,
        feval: _typing.Union[
            None, _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = None,
    ) -> _typing.Sequence[float]:
        data = dataset[0]
        data = data.to(self.device)
        if feval is None:
            _feval: _typing.Sequence[_typing.Type[Evaluation]] = self.feval
        else:
            _feval: _typing.Sequence[_typing.Type[Evaluation]] = get_feval(list(feval))
        if mask.lower() == "train":
            _mask = data.train_mask
        elif mask.lower() == "test":
            _mask = data.test_mask
        elif mask.lower() == "val":
            _mask = data.val_mask
        else:
            _mask = data.test_mask
        prediction_probability: torch.Tensor = self.predict_proba(dataset, mask)
        y_ground_truth = data.y[_mask]

        results = []
        for f in _feval:
            try:
                results.append(f.evaluate(prediction_probability, y_ground_truth))
            except:
                results.append(
                    f.evaluate(
                        prediction_probability.cpu().numpy(),
                        y_ground_truth.cpu().numpy(),
                    )
                )
        return results

    def to(self, device: torch.device):
        self.device = device
        if self._model is not None:
            self._model.to(self.device)

    def duplicate_from_hyper_parameter(
        self,
        hp: _typing.Dict[str, _typing.Any],
        model: _typing.Union[BaseModel, str, None] = None,
    ) -> "NodeClassificationNeighborSamplingTrainer":

        if model is None or not isinstance(model, BaseModel):
            model = self._model
        model = model.from_hyper_parameter(
            dict(
                [
                    x
                    for x in hp.items()
                    if x[0] in [y["parameterName"] for y in model.hyper_parameter_space]
                ]
            )
        )

        return NodeClassificationNeighborSamplingTrainer(
            model,
            self.num_features,
            self.num_classes,
            self._optimizer_class,
            device=self.device,
            init=True,
            feval=self.feval,
            loss=self.loss,
            lr_scheduler_type=self._lr_scheduler_type,
            **hp
        )

    @property
    def hyper_parameter_space(self):
        return self._hyper_parameter_space

    @hyper_parameter_space.setter
    def hyper_parameter_space(self, hp_space):
        self._hyper_parameter_space = hp_space
