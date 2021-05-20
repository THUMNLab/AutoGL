import torch
import logging
import typing as _typing
import torch.nn.functional
import torch.utils.data

from .. import register_trainer
from ..base import BaseNodeClassificationTrainer, EarlyStopping, Evaluation
from ..evaluation import get_feval, Logloss
from ..sampling.sampler.neighbor_sampler import NeighborSampler
from ..sampling.sampler.graphsaint_sampler import *
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
        **kwargs,
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
        self._valid_score: _typing.Sequence[float] = []

        self._hyper_parameter_space: _typing.Sequence[
            _typing.Dict[str, _typing.Any]
        ] = []

        self.__initialized: bool = False
        if init:
            self.initialize()

    def initialize(self) -> "NodeClassificationNeighborSamplingTrainer":
        if self.__initialized:
            return self
        self.model.initialize()
        self.__initialized = True
        return self

    def get_model(self) -> BaseModel:
        return self.model

    def __train_only(self, data) -> "NodeClassificationNeighborSamplingTrainer":
        """
        The function of training on the given dataset and mask.
        :param data: data of a specific graph
        :return: self
        """
        data = data.to(self.device)
        optimizer: torch.optim.Optimizer = self._optimizer_class(
            self.model.model.parameters(),
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
            self.model.model.train()
            """ epoch start """
            for target_node_indexes, edge_indexes in train_sampler:
                optimizer.zero_grad()
                data.edge_indexes = edge_indexes
                prediction = self.model.model(data)
                if not hasattr(torch.nn.functional, self.loss):
                    raise TypeError(
                        "PyTorch does not support loss type {}".format(self.loss)
                    )
                loss_function = getattr(torch.nn.functional, self.loss)
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
                self._early_stopping(validation_loss, self.model.model)
                if self._early_stopping.early_stop:
                    LOGGER.debug("Early stopping at %d", current_epoch)
                    break
        if hasattr(data, "val_mask") and data.val_mask is not None:
            self._early_stopping.load_checkpoint(self.model.model)
        return self

    def __predict_only(self, data):
        """
        The function of predicting on the given data.
        :param data: data of a specific graph
        :return: the result of prediction on the given dataset
        """
        data = data.to(self.device)
        self.model.model.eval()
        with torch.no_grad():
            prediction = self.model.model(data)
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
        if self.model is not None:
            self.model.to(self.device)

    def duplicate_from_hyper_parameter(
        self,
        hp: _typing.Dict[str, _typing.Any],
        model: _typing.Union[BaseModel, str, None] = None,
    ) -> "NodeClassificationNeighborSamplingTrainer":

        if model is None or not isinstance(model, BaseModel):
            model = self.model
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
            **hp,
        )

    @property
    def hyper_parameter_space(self):
        return self._hyper_parameter_space

    @hyper_parameter_space.setter
    def hyper_parameter_space(self, hp_space):
        self._hyper_parameter_space = hp_space


@register_trainer("NodeClassificationGraphSAINTTrainer")
class NodeClassificationGraphSAINTTrainer(BaseNodeClassificationTrainer):
    def __init__(
        self,
        model: _typing.Union[BaseModel],
        num_features: int,
        num_classes: int,
        optimizer: _typing.Union[_typing.Type[torch.optim.Optimizer], str, None],
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
        **kwargs,
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
        self._weight_decay: float = weight_decay if weight_decay > 0 else 1e-4
        early_stopping_round: int = (
            early_stopping_round if early_stopping_round > 0 else 1e2
        )
        self._early_stopping = EarlyStopping(
            patience=early_stopping_round, verbose=False
        )

        # Assign an empty initial hyper parameter space
        self._hyper_parameter_space: _typing.Sequence[
            _typing.Dict[str, _typing.Any]
        ] = []

        self._valid_result: torch.Tensor = torch.zeros(0)
        self._valid_result_prob: torch.Tensor = torch.zeros(0)
        self._valid_score: _typing.Sequence[float] = ()

        super(NodeClassificationGraphSAINTTrainer, self).__init__(
            model, num_features, num_classes, device, init, feval, loss
        )

        """ Set hyper parameters """
        if "num_subgraphs" not in kwargs:
            raise KeyError
        elif type(kwargs.get("num_subgraphs")) != int:
            raise TypeError
        elif not kwargs.get("num_subgraphs") > 0:
            raise ValueError
        else:
            self.__num_subgraphs: int = kwargs.get("num_subgraphs")
        if "sampling_budget" not in kwargs:
            raise KeyError
        elif type(kwargs.get("sampling_budget")) != int:
            raise TypeError
        elif not kwargs.get("sampling_budget") > 0:
            raise ValueError
        else:
            self.__sampling_budget: int = kwargs.get("sampling_budget")
        if "sampling_method" not in kwargs:
            self.__sampling_method_identifier: str = "node"
        elif type(kwargs.get("sampling_method")) != str:
            self.__sampling_method_identifier: str = "node"
        else:
            self.__sampling_method_identifier: str = kwargs.get("sampling_method")
            if self.__sampling_method_identifier.lower() not in ("node", "edge"):
                self.__sampling_method_identifier: str = "node"

        self.__is_initialized: bool = False
        if init:
            self.initialize()

    def initialize(self):
        if self.__is_initialized:
            return self
        self.model.initialize()
        self.__is_initialized = True
        return self

    def to(self, device: torch.device):
        self.device = device
        if self.model is not None:
            self.model.to(self.device)

    def get_model(self):
        return self.model

    def __train_only(self, data):
        """
        The function of training on the given dataset and mask.
        :param data: data of a specific graph
        :return: self
        """
        data = data.to(self.device)
        optimizer: torch.optim.Optimizer = self._optimizer_class(
            self.model.parameters(),
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

        if self.__sampling_method_identifier.lower() == "edge":
            sub_graph_sampler = GraphSAINTRandomEdgeSampler(
                self.__sampling_budget, self.__num_subgraphs
            )
        else:
            sub_graph_sampler = GraphSAINTRandomNodeSampler(
                self.__sampling_budget, self.__num_subgraphs
            )

        for current_epoch in range(self._max_epoch):
            self.model.model.train()
            """ epoch start """
            """ Sample sub-graphs """
            sub_graph_set = sub_graph_sampler.sample(data)
            sub_graphs_loader: torch.utils.data.DataLoader = (
                torch.utils.data.DataLoader(sub_graph_set)
            )
            integral_alpha: torch.Tensor = getattr(sub_graph_set, "alpha")
            integral_lambda: torch.Tensor = getattr(sub_graph_set, "lambda")
            """ iterate sub-graphs """
            for sub_graph_data in sub_graphs_loader:
                optimizer.zero_grad()
                sampled_edge_indexes: torch.Tensor = sub_graph_data.sampled_edge_indexes
                sampled_node_indexes: torch.Tensor = sub_graph_data.sampled_node_indexes
                sampled_train_mask: torch.Tensor = sub_graph_data.train_mask

                sampled_alpha = integral_alpha[sampled_edge_indexes]
                sub_graph_data.edge_weight = 1 / sampled_alpha

                prediction: torch.Tensor = self.model.model(sub_graph_data)

                if not hasattr(torch.nn.functional, self.loss):
                    raise TypeError(f"PyTorch does not support loss type {self.loss}")
                loss_func = getattr(torch.nn.functional, self.loss)
                unreduced_loss: torch.Tensor = loss_func(
                    prediction[sampled_train_mask],
                    data.y[sampled_train_mask],
                    reduction="none",
                )

                sampled_lambda: torch.Tensor = integral_lambda[sampled_node_indexes]
                sampled_train_lambda: torch.Tensor = sampled_lambda[sampled_train_mask]
                assert unreduced_loss.size() == sampled_train_lambda.size()
                loss_weighted_sum: torch.Tensor = torch.sum(
                    unreduced_loss / sampled_train_lambda
                )
                loss_weighted_sum.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            """ Validate performance """
            if (
                hasattr(data, "val_mask")
                and type(getattr(data, "val_mask")) == torch.Tensor
            ):
                validation_results: _typing.Sequence[float] = self.evaluate(
                    (data,), "val", [self.feval[0]]
                )
                if self.feval[0].is_higher_better():
                    validation_loss: float = -validation_results[0]
                else:
                    validation_loss: float = validation_results[0]
                self._early_stopping(validation_loss, self.model.model)
                if self._early_stopping.early_stop:
                    LOGGER.debug("Early stopping at %d", current_epoch)
                    break
        if hasattr(data, "val_mask") and data.val_mask is not None:
            self._early_stopping.load_checkpoint(self.model.model)
        return self

    def __predict_only(self, data):
        """
        The function of predicting on the given data.
        :param data: data of a specific graph
        :return: the result of prediction on the given dataset
        """
        data = data.to(self.device)
        self.model.model.eval()
        with torch.no_grad():
            predicted_x: torch.Tensor = self.model.model(data)
        return predicted_x

    def predict_proba(
        self, dataset, mask: _typing.Optional[str] = None, in_log_format=False
    ):
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
                _mask: torch.Tensor = data.train_mask
            elif mask.lower() == "test":
                _mask: torch.Tensor = data.test_mask
            elif mask.lower() == "val":
                _mask: torch.Tensor = data.val_mask
            else:
                _mask: torch.Tensor = data.test_mask
        else:
            _mask: torch.Tensor = data.test_mask
        result = self.__predict_only(data)[_mask]
        return result if in_log_format else torch.exp(result)

    def predict(self, dataset, mask: _typing.Optional[str] = None) -> torch.Tensor:
        return self.predict_proba(dataset, mask, in_log_format=True).max(1)[1]

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
        if mask is not None and type(mask) == str:
            if mask.lower() == "train":
                _mask: torch.Tensor = data.train_mask
            elif mask.lower() == "test":
                _mask: torch.Tensor = data.test_mask
            elif mask.lower() == "val":
                _mask: torch.Tensor = data.val_mask
            else:
                _mask: torch.Tensor = data.test_mask
        else:
            _mask: torch.Tensor = data.test_mask
        prediction_probability: torch.Tensor = self.predict_proba(dataset, mask)
        y_ground_truth: torch.Tensor = data.y[_mask]

        eval_results = []
        for f in _feval:
            try:
                eval_results.append(f.evaluate(prediction_probability, y_ground_truth))
            except:
                eval_results.append(
                    f.evaluate(
                        prediction_probability.cpu().numpy(),
                        y_ground_truth.cpu().numpy(),
                    )
                )
        return eval_results

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
            self._valid_score: _typing.Sequence[float] = self.evaluate(dataset, "val")

    def get_valid_predict(self) -> torch.Tensor:
        return self._valid_result

    def get_valid_predict_proba(self) -> torch.Tensor:
        return self._valid_result_prob

    def get_valid_score(
        self, return_major: bool = True
    ) -> _typing.Tuple[
        _typing.Union[float, _typing.Sequence[float]],
        _typing.Union[bool, _typing.Sequence[bool]],
    ]:
        if return_major:
            return self._valid_score[0], self.feval[0].is_higher_better()
        else:
            return (self._valid_score, [f.is_higher_better() for f in self.feval])

    @property
    def hyper_parameter_space(self) -> _typing.Sequence[_typing.Dict[str, _typing.Any]]:
        return self._hyper_parameter_space

    @hyper_parameter_space.setter
    def hyper_parameter_space(
        self, hp_space: _typing.Sequence[_typing.Dict[str, _typing.Any]]
    ) -> None:
        if not isinstance(hp_space, _typing.Sequence):
            raise TypeError
        self._hyper_parameter_space = hp_space

    def get_name_with_hp(self) -> str:
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

    def duplicate_from_hyper_parameter(
        self,
        hp: _typing.Dict[str, _typing.Any],
        model: _typing.Optional[BaseModel] = None,
    ) -> "NodeClassificationGraphSAINTTrainer":
        if model is None or not isinstance(model, BaseModel):
            model: BaseModel = self.model
        model = model.from_hyper_parameter(
            dict(
                [
                    x
                    for x in hp.items()
                    if x[0] in [y["parameterName"] for y in model.hyper_parameter_space]
                ]
            )
        )
        return NodeClassificationGraphSAINTTrainer(
            model,
            self.num_features,
            self.num_classes,
            self._optimizer_class,
            device=self.device,
            init=True,
            feval=self.feval,
            loss=self.loss,
            lr_scheduler_type=self._lr_scheduler_type,
            **hp,
        )
