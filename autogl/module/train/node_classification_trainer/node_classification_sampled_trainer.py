import os
import torch
import logging
import typing as _typing
import torch.nn.functional
import torch.utils.data

import autogl.data
from .. import register_trainer
from ..base import BaseNodeClassificationTrainer, EarlyStopping, Evaluation
from ..evaluation import get_feval, EvaluatorUtility, Logloss, MicroF1
from ..sampling.sampler.target_dependant_sampler import TargetDependantSampledData
from ..sampling.sampler.neighbor_sampler import NeighborSampler
from ..sampling.sampler.graphsaint_sampler import *
from ..sampling.sampler.layer_dependent_importance_sampler import (
    LayerDependentImportanceSampler
)
from ...model import BaseModel
from ...model.base import SequentialGraphNeuralNetwork

LOGGER: logging.Logger = logging.getLogger("Node classification sampling trainer")


@register_trainer("NodeClassificationGraphSAINTTrainer")
class NodeClassificationGraphSAINTTrainer(BaseNodeClassificationTrainer):
    def __init__(
        self,
        model: _typing.Union[BaseModel],
        num_features: int,
        num_classes: int,
        optimizer: _typing.Union[_typing.Type[torch.optim.Optimizer], str, None] = ...,
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
        self.__num_subgraphs: int = kwargs.get("num_subgraphs")
        self.__sampling_budget: int = kwargs.get("sampling_budget")
        if (
                kwargs.get("sampling_method") is not None
                and type(kwargs.get("sampling_method")) == str
                and kwargs.get("sampling_method") in ("node", "edge")
        ):
            self.__sampling_method_identifier: str = kwargs.get("sampling_method")
        else:
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
            return (
                self._valid_score, [f.is_higher_better() for f in self.feval]
            )

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

    def __repr__(self) -> dict:
        return {
            "trainer_name": self.__class__.__name__,
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
            "max_epoch": self.max_epoch,
            "early_stopping_round": self.early_stopping_round,
            "model": repr(self.model)
        }

    def __str__(self) -> str:
        import yaml
        return yaml.dump(repr(self))

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


class _DeterministicNeighborSamplerStore:
    def __init__(self):
        self.__neighbor_sampler_mapping: _typing.List[
            _typing.Tuple[torch.LongTensor, NeighborSampler]
        ] = []

    @classmethod
    def __is_target_node_indexes_equal(cls, a: torch.LongTensor, b: torch.LongTensor) -> bool:
        if not a.dtype == b.dtype == torch.int64:
            return False
        if a.size() != b.size():
            return False
        return torch.where(a != b)[0].size(0) == 0

    def __setitem__(self, target_nodes: torch.Tensor, neighbor_sampler: NeighborSampler):
        target_nodes: _typing.Any = target_nodes.cpu()
        if type(target_nodes) != torch.Tensor or target_nodes.dtype != torch.int64:
            raise TypeError
        if type(neighbor_sampler) != NeighborSampler:
            raise TypeError
        for i in range(len(self.__neighbor_sampler_mapping)):
            if self.__is_target_node_indexes_equal(
                    target_nodes, self.__neighbor_sampler_mapping[i][0]
            ):
                self.__neighbor_sampler_mapping[i] = (target_nodes, neighbor_sampler)
                return
        self.__neighbor_sampler_mapping.append((target_nodes, neighbor_sampler))

    def __getitem__(self, target_nodes: torch.Tensor) -> _typing.Optional[NeighborSampler]:
        target_nodes: _typing.Any = target_nodes.cpu()
        if type(target_nodes) != torch.Tensor or target_nodes.dtype != torch.int64:
            raise TypeError
        for __current_target_nodes, __neighbor_sampler in self.__neighbor_sampler_mapping:
            if self.__is_target_node_indexes_equal(target_nodes, __current_target_nodes):
                return __neighbor_sampler
        return None


@register_trainer("NodeClassificationLayerDependentImportanceSamplingTrainer")
class NodeClassificationLayerDependentImportanceSamplingTrainer(BaseNodeClassificationTrainer):
    def __init__(
            self,
            model: _typing.Union[BaseModel, str],
            num_features: int,
            num_classes: int,
            optimizer: _typing.Union[_typing.Type[torch.optim.Optimizer], str, None] = ...,
            lr: float = 1e-4,
            max_epoch: int = 100,
            early_stopping_round: int = 100,
            weight_decay: float = 1e-4,
            device: _typing.Optional[torch.device] = None,
            init: bool = True,
            feval: _typing.Union[
                _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
            ] = (MicroF1,),
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
        self._early_stopping = EarlyStopping(
            patience=early_stopping_round if early_stopping_round > 0 else 1e2,
            verbose=False
        )
        """ Assign an empty initial hyper parameter space """
        self._hyper_parameter_space: _typing.Sequence[_typing.Dict[str, _typing.Any]] = []

        self._valid_result: torch.Tensor = torch.zeros(0)
        self._valid_result_prob: torch.Tensor = torch.zeros(0)
        self._valid_score: _typing.Sequence[float] = ()

        """ Set hyper parameters """
        self.__sampled_node_sizes: _typing.Sequence[int] = kwargs.get("sampled_node_sizes")

        self.__training_batch_size: int = kwargs.get("training_batch_size", 1024)
        if not self.__training_batch_size > 0:
            self.__training_batch_size: int = 1024
        self.__predicting_batch_size: int = kwargs.get("predicting_batch_size", 1024)
        if not self.__predicting_batch_size > 0:
            self.__predicting_batch_size: int = 1024

        cpu_count: int = os.cpu_count() if os.cpu_count() is not None else 0
        self.__training_sampler_num_workers: int = kwargs.get(
            "training_sampler_num_workers", cpu_count
        )
        if self.__training_sampler_num_workers > cpu_count:
            self.__training_sampler_num_workers = cpu_count
        self.__predicting_sampler_num_workers: int = kwargs.get(
            "predicting_sampler_num_workers", cpu_count
        )
        if self.__predicting_sampler_num_workers > cpu_count:
            self.__predicting_sampler_num_workers = cpu_count

        super(NodeClassificationLayerDependentImportanceSamplingTrainer, self).__init__(
            model, num_features, num_classes, device, init, feval, loss
        )

        self.__neighbor_sampler_store: _DeterministicNeighborSamplerStore = (
            _DeterministicNeighborSamplerStore()
        )

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

    def __train_only(self, integral_data):
        """
        The function of training on the given dataset and mask.
        :param integral_data: data of a specific graph
        :return: self
        """
        optimizer: torch.optim.Optimizer = self._optimizer_class(
            self.model.model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay
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

        __layer_dependent_importance_sampler: LayerDependentImportanceSampler = (
            LayerDependentImportanceSampler(
                integral_data.edge_index, torch.where(integral_data.train_mask)[0].unique(),
                self.__sampled_node_sizes, batch_size=self.__training_batch_size,
                num_workers=self.__training_sampler_num_workers
            )
        )
        for current_epoch in range(self._max_epoch):
            self.model.model.train()
            optimizer.zero_grad()
            """ epoch start """
            " sample graphs "
            for sampled_data in __layer_dependent_importance_sampler:
                optimizer.zero_grad()
                sampled_data: TargetDependantSampledData = sampled_data
                # 由于现在的Model设计是接受Data的，所以只能组装一个采样的Data作为参数
                sampled_graph: autogl.data.Data = autogl.data.Data(
                    x=integral_data.x[sampled_data.all_sampled_nodes_indexes],
                    y=integral_data.y[sampled_data.all_sampled_nodes_indexes]
                )
                sampled_graph.to(self.device)
                sampled_graph.edge_indexes: _typing.Sequence[torch.LongTensor] = [
                    current_layer.edge_index_for_sampled_graph.to(self.device)
                    for current_layer in sampled_data.sampled_edges_for_layers
                ]
                sampled_graph.edge_weights: _typing.Sequence[torch.Tensor] = [
                    current_layer.edge_weight.to(self.device)
                    for current_layer in sampled_data.sampled_edges_for_layers
                ]
                prediction: torch.Tensor = self.model.model(sampled_graph)
                if not hasattr(torch.nn.functional, self.loss):
                    raise TypeError(
                        f"PyTorch does not support loss type {self.loss}"
                    )
                loss_function = getattr(torch.nn.functional, self.loss)
                loss_value: torch.Tensor = loss_function(
                    prediction[sampled_data.target_nodes_indexes.indexes_in_sampled_graph],
                    sampled_graph.y[sampled_data.target_nodes_indexes.indexes_in_sampled_graph]
                )
                loss_value.backward()
                optimizer.step()

            if self._lr_scheduler_type:
                lr_scheduler.step()

            if (
                    hasattr(integral_data, "val_mask") and
                    getattr(integral_data, "val_mask") is not None and
                    type(getattr(integral_data, "val_mask")) == torch.Tensor
            ):
                validation_results: _typing.Sequence[float] = self.evaluate(
                    (integral_data,), "val", [self.feval[0]]
                )
                if self.feval[0].is_higher_better():
                    validation_loss: float = -validation_results[0]
                else:
                    validation_loss: float = validation_results[0]
                self._early_stopping(validation_loss, self.model.model)
                if self._early_stopping.early_stop:
                    LOGGER.debug("Early stopping at %d", current_epoch)
                    break
        if (
                hasattr(integral_data, "val_mask") and
                getattr(integral_data, "val_mask") is not None and
                type(getattr(integral_data, "val_mask")) == torch.Tensor
        ):
            self._early_stopping.load_checkpoint(self.model.model)

    def __predict_only(
            self, integral_data,
            mask_or_target_nodes_indexes: _typing.Union[
                torch.BoolTensor, torch.LongTensor
            ]
    ) -> torch.Tensor:
        """
        The function of predicting on the given data.
        :param integral_data: data of a specific graph
        :param mask_or_target_nodes_indexes: ...
        :return: the result of prediction on the given dataset
        """
        self.model.model.eval()
        integral_data = integral_data.to(torch.device("cpu"))
        mask_or_target_nodes_indexes = mask_or_target_nodes_indexes.to(torch.device("cpu"))
        if isinstance(self.model.model, SequentialGraphNeuralNetwork):
            sequential_gnn_model: SequentialGraphNeuralNetwork = self.model.model
            __num_layers: int = len(self.__sampled_node_sizes)

            x: torch.Tensor = getattr(integral_data, "x")
            for _current_layer_index in range(__num_layers - 1):
                __next_x: _typing.Optional[torch.Tensor] = None

                _optional_neighbor_sampler: _typing.Optional[NeighborSampler] = (
                    self.__neighbor_sampler_store[torch.arange(x.size(0))]
                )
                if (
                        _optional_neighbor_sampler is not None and
                        type(_optional_neighbor_sampler) == NeighborSampler
                ):
                    current_neighbor_sampler: NeighborSampler = _optional_neighbor_sampler
                else:
                    current_neighbor_sampler: NeighborSampler = NeighborSampler(
                        integral_data.edge_index, torch.arange(x.size(0)).unique(),
                        [-1], batch_size=self.__predicting_batch_size,
                        num_workers=self.__predicting_sampler_num_workers, shuffle=False
                    )
                    self.__neighbor_sampler_store[torch.arange(x.size(0))] = current_neighbor_sampler

                for _target_dependant_sampled_data in current_neighbor_sampler:
                    _target_dependant_sampled_data: TargetDependantSampledData = (
                        _target_dependant_sampled_data
                    )
                    _sampled_graph: autogl.data.Data = autogl.data.Data(
                        x=x[_target_dependant_sampled_data.all_sampled_nodes_indexes],
                        edge_index=(
                            _target_dependant_sampled_data.sampled_edges_for_layers[0].edge_index_for_sampled_graph
                        )
                    )
                    _sampled_graph.edge_weight: torch.Tensor = (
                        _target_dependant_sampled_data.sampled_edges_for_layers[0].edge_weight
                    )
                    _sampled_graph: autogl.data.Data = _sampled_graph.to(self.device)

                    with torch.no_grad():
                        __sampled_graph_inferences: torch.Tensor = (
                            sequential_gnn_model.encoder_sequential_modules[_current_layer_index](_sampled_graph)
                        )
                        _sampled_target_nodes_inferences: torch.Tensor = __sampled_graph_inferences[
                            _target_dependant_sampled_data.target_nodes_indexes.indexes_in_sampled_graph
                        ].cpu()
                    if __next_x is None:
                        __next_x: torch.Tensor = torch.zeros(x.size(0), __sampled_graph_inferences.size(1))
                    __next_x[_target_dependant_sampled_data.target_nodes_indexes.indexes_in_integral_graph] = (
                        _sampled_target_nodes_inferences
                    )
                x: torch.Tensor = __next_x
            " The following procedures are for the top layer "
            if mask_or_target_nodes_indexes.dtype == torch.bool:
                target_nodes_indexes: _typing.Any = (
                    torch.where(mask_or_target_nodes_indexes)[0]
                )
            else:
                target_nodes_indexes: _typing.Any = mask_or_target_nodes_indexes.long()

            _optional_neighbor_sampler: _typing.Optional[NeighborSampler] = (
                self.__neighbor_sampler_store[target_nodes_indexes]
            )
            if (
                    _optional_neighbor_sampler is not None and
                    type(_optional_neighbor_sampler) == NeighborSampler
            ):
                current_neighbor_sampler: NeighborSampler = _optional_neighbor_sampler
            else:
                current_neighbor_sampler: NeighborSampler = NeighborSampler(
                    integral_data.edge_index, target_nodes_indexes,
                    [-1], batch_size=self.__predicting_batch_size,
                    num_workers=self.__predicting_sampler_num_workers, shuffle=False
                )
                self.__neighbor_sampler_store[target_nodes_indexes] = current_neighbor_sampler

            prediction_batch_cumulative_builder = (
                EvaluatorUtility.PredictionBatchCumulativeBuilder()
            )
            for _target_dependant_sampled_data in current_neighbor_sampler:
                _sampled_graph: autogl.data.Data = autogl.data.Data(
                    x[_target_dependant_sampled_data.all_sampled_nodes_indexes],
                    _target_dependant_sampled_data.sampled_edges_for_layers[0].edge_index_for_sampled_graph
                )
                _sampled_graph.edge_weight: torch.Tensor = (
                    _target_dependant_sampled_data.sampled_edges_for_layers[0].edge_weight
                )
                _sampled_graph: autogl.data.Data = _sampled_graph.to(self.device)
                with torch.no_grad():
                    prediction_batch_cumulative_builder.add_batch(
                        _target_dependant_sampled_data.target_nodes_indexes.indexes_in_integral_graph.cpu().numpy(),
                        sequential_gnn_model.decode(
                            sequential_gnn_model.encoder_sequential_modules[-1](_sampled_graph)
                        )[_target_dependant_sampled_data.target_nodes_indexes.indexes_in_sampled_graph].cpu().numpy()
                    )
            return torch.from_numpy(prediction_batch_cumulative_builder.compose()[1])
        else:
            if mask_or_target_nodes_indexes.dtype == torch.bool:
                target_nodes_indexes: _typing.Any = (
                    torch.where(mask_or_target_nodes_indexes)[0]
                )
            else:
                target_nodes_indexes: _typing.Any = mask_or_target_nodes_indexes.long()
            neighbor_sampler: NeighborSampler = NeighborSampler(
                torch_geometric.utils.add_remaining_self_loops(integral_data.edge_index)[0],
                target_nodes_indexes, [-1 for _ in self.__sampled_node_sizes],
                batch_size=self.__predicting_batch_size,
                num_workers=self.__predicting_sampler_num_workers,
                shuffle=False
            )
            prediction_batch_cumulative_builder = (
                EvaluatorUtility.PredictionBatchCumulativeBuilder()
            )
            self.model.model.eval()
            for sampled_data in neighbor_sampler:
                sampled_data: TargetDependantSampledData = sampled_data
                sampled_graph: autogl.data.Data = autogl.data.Data(
                    integral_data.x[sampled_data.all_sampled_nodes_indexes],
                    integral_data.y[sampled_data.all_sampled_nodes_indexes]
                )
                sampled_graph.to(self.device)
                sampled_graph.edge_indexes: _typing.Sequence[torch.LongTensor] = [
                    current_layer.edge_index_for_sampled_graph.to(self.device)
                    for current_layer in sampled_data.sampled_edges_for_layers
                ]
                sampled_graph.edge_weights: _typing.Sequence[torch.FloatTensor] = [
                    current_layer.edge_weight.to(self.device)
                    for current_layer in sampled_data.sampled_edges_for_layers
                ]
                with torch.no_grad():
                    prediction_batch_cumulative_builder.add_batch(
                        sampled_data.target_nodes_indexes.indexes_in_integral_graph.cpu().numpy(),
                        self.model.model(sampled_graph)[
                            sampled_data.target_nodes_indexes.indexes_in_sampled_graph
                        ].cpu().numpy()
                    )
            return torch.from_numpy(prediction_batch_cumulative_builder.compose()[1])

    def predict_proba(
            self, dataset, mask: _typing.Optional[str] = None,
            in_log_format: bool = False
    ):
        """
        The function of predicting the probability on the given dataset.
        :param dataset: The node classification dataset used to be predicted.
        :param mask:
        :param in_log_format:
        :return:
        """
        data = dataset[0].to(torch.device("cpu"))
        if mask is not None and type(mask) == str:
            if mask.lower() == "train":
                _mask: torch.BoolTensor = data.train_mask
            elif mask.lower() == "test":
                _mask: torch.BoolTensor = data.test_mask
            elif mask.lower() == "val":
                _mask: torch.BoolTensor = data.val_mask
            else:
                _mask: torch.BoolTensor = data.test_mask
        else:
            _mask: torch.BoolTensor = data.test_mask
        result = self.__predict_only(data, _mask)
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

        return [
            f.evaluate(
                prediction_probability.cpu().numpy(),
                y_ground_truth.cpu().numpy(),
            ) for f in _feval
        ]

    def train(self, dataset, keep_valid_result: bool = True):
        """
        The function of training on the given dataset and keeping valid result.
        :param dataset:
        :param keep_valid_result: Whether to save the validation result after training
        """
        data = dataset[0]
        self.__train_only(data)
        if keep_valid_result:
            data = data.to(torch.device("cpu"))
            prediction: torch.Tensor = self.__predict_only(data, data.val_mask)
            self._valid_result: torch.Tensor = prediction.max(1)[1]
            self._valid_result_prob: torch.Tensor = prediction
            self._valid_score: _typing.Sequence[float] = self.evaluate(dataset, "val")

    def get_valid_predict(self) -> torch.Tensor:
        return self._valid_result

    def get_valid_predict_proba(self) -> torch.Tensor:
        return self._valid_result_prob

    def get_valid_score(
            self, return_major: bool = True
    ) -> _typing.Union[
        _typing.Tuple[float, bool],
        _typing.Tuple[_typing.Sequence[float], _typing.Sequence[bool]]
    ]:
        if return_major:
            return self._valid_score[0], self.feval[0].is_higher_better()
        else:
            return self._valid_score, [f.is_higher_better() for f in self.feval]

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

    def __repr__(self) -> str:
        import yaml
        return yaml.dump(
            {
                "trainer_name": self.__class__.__name__,
                "optimizer": self._optimizer_class,
                "learning_rate": self._learning_rate,
                "max_epoch": self._max_epoch,
                "early_stopping_round": self._early_stopping.patience,
                "sampling_sizes": self.__sampled_node_sizes,
                "model": repr(self.model)
            }
        )

    def duplicate_from_hyper_parameter(
            self,
            hp: _typing.Dict[str, _typing.Any],
            model: _typing.Optional[BaseModel] = None,
    ) -> "NodeClassificationLayerDependentImportanceSamplingTrainer":
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
        return NodeClassificationLayerDependentImportanceSamplingTrainer(
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


@register_trainer("NodeClassificationNeighborSampling")
class NodeClassificationNeighborSamplingTrainer(BaseNodeClassificationTrainer):
    def __init__(
            self,
            model: _typing.Union[BaseModel, str],
            num_features: int,
            num_classes: int,
            optimizer: _typing.Union[_typing.Type[torch.optim.Optimizer], str, None] = ...,
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
    ):
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
        self._early_stopping = EarlyStopping(
            patience=early_stopping_round if early_stopping_round > 0 else 1e2,
            verbose=False
        )
        """ Assign an empty initial hyper parameter space """
        self._hyper_parameter_space: _typing.Sequence[_typing.Dict[str, _typing.Any]] = []

        self._valid_result: torch.Tensor = torch.zeros(0)
        self._valid_result_prob: torch.Tensor = torch.zeros(0)
        self._valid_score: _typing.Sequence[float] = ()

        """ Set hyper-parameter """
        self.__sampling_sizes: _typing.Sequence[int] = kwargs.get("sampling_sizes")

        self.__training_batch_size: int = kwargs.get("training_batch_size", 1024)
        if not self.__training_batch_size > 0:
            self.__training_batch_size: int = 1024
        self.__predicting_batch_size: int = kwargs.get("predicting_batch_size", 1024)
        if not self.__predicting_batch_size > 0:
            self.__predicting_batch_size: int = 1024

        cpu_count: int = os.cpu_count() if os.cpu_count() is not None else 0
        self.__training_sampler_num_workers: int = kwargs.get(
            "training_sampler_num_workers", cpu_count
        )
        if self.__training_sampler_num_workers > cpu_count:
            self.__training_sampler_num_workers = cpu_count
        self.__predicting_sampler_num_workers: int = kwargs.get(
            "predicting_sampler_num_workers", cpu_count
        )
        if self.__predicting_sampler_num_workers > cpu_count:
            self.__predicting_sampler_num_workers = cpu_count

        super(NodeClassificationNeighborSamplingTrainer, self).__init__(
            model, num_features, num_classes, device, init, feval, loss
        )

        self.__neighbor_sampler_store: _DeterministicNeighborSamplerStore = (
            _DeterministicNeighborSamplerStore()
        )

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

    def __train_only(self, integral_data):
        """
        The function of training on the given dataset and mask.
        :param integral_data: data of the integral graph
        :return: self
        """
        optimizer: torch.optim.Optimizer = self._optimizer_class(
            self.model.model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay
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

        neighbor_sampler: NeighborSampler = NeighborSampler(
            integral_data.edge_index, torch.where(integral_data.train_mask)[0].unique(),
            self.__sampling_sizes, batch_size=self.__training_batch_size,
            num_workers=self.__training_sampler_num_workers
        )
        for current_epoch in range(self._max_epoch):
            self.model.model.train()
            optimizer.zero_grad()
            """ epoch start """
            " sample graphs "
            for sampled_data in neighbor_sampler:
                optimizer.zero_grad()
                sampled_data: TargetDependantSampledData = sampled_data
                sampled_graph: autogl.data.Data = autogl.data.Data(
                    x=integral_data.x[sampled_data.all_sampled_nodes_indexes],
                    y=integral_data.y[sampled_data.all_sampled_nodes_indexes]
                )
                sampled_graph.to(self.device)
                sampled_graph.edge_indexes: _typing.Sequence[torch.LongTensor] = [
                    current_layer.edge_index_for_sampled_graph.to(self.device)
                    for current_layer in sampled_data.sampled_edges_for_layers
                ]
                prediction: torch.Tensor = self.model.model(sampled_graph)
                if not hasattr(torch.nn.functional, self.loss):
                    raise TypeError(
                        f"PyTorch does not support loss type {self.loss}"
                    )
                loss_function = getattr(torch.nn.functional, self.loss)
                loss_value: torch.Tensor = loss_function(
                    prediction[sampled_data.target_nodes_indexes.indexes_in_sampled_graph],
                    sampled_graph.y[sampled_data.target_nodes_indexes.indexes_in_sampled_graph]
                )
                loss_value.backward()
                optimizer.step()

            if self._lr_scheduler_type:
                lr_scheduler.step()

            if (
                    hasattr(integral_data, "val_mask") and
                    getattr(integral_data, "val_mask") is not None and
                    type(getattr(integral_data, "val_mask")) == torch.Tensor
            ):
                validation_results: _typing.Sequence[float] = self.evaluate(
                    (integral_data,), "val", [self.feval[0]]
                )
                if self.feval[0].is_higher_better():
                    validation_loss: float = -validation_results[0]
                else:
                    validation_loss: float = validation_results[0]
                self._early_stopping(validation_loss, self.model.model)
                if self._early_stopping.early_stop:
                    LOGGER.debug("Early stopping at %d", current_epoch)
                    break
        if (
                hasattr(integral_data, "val_mask") and
                getattr(integral_data, "val_mask") is not None and
                type(getattr(integral_data, "val_mask")) == torch.Tensor
        ):
            self._early_stopping.load_checkpoint(self.model.model)

    def __predict_only(
            self, integral_data,
            mask_or_target_nodes_indexes: _typing.Union[
                torch.BoolTensor, torch.LongTensor
            ]
    ) -> torch.Tensor:
        """
        The function of predicting on the given data.
        :param integral_data: data of a specific graph
        :param mask_or_target_nodes_indexes: ...
        :return: the result of prediction on the given dataset
        """
        self.model.model.eval()
        integral_data = integral_data.to(torch.device("cpu"))
        mask_or_target_nodes_indexes = mask_or_target_nodes_indexes.to(torch.device("cpu"))
        if isinstance(self.model.model, SequentialGraphNeuralNetwork):
            sequential_gnn_model: SequentialGraphNeuralNetwork = self.model.model
            __num_layers: int = len(self.__sampling_sizes)

            x: torch.Tensor = getattr(integral_data, "x")
            for _current_layer_index in range(__num_layers - 1):
                __next_x: _typing.Optional[torch.Tensor] = None

                _optional_neighbor_sampler: _typing.Optional[NeighborSampler] = (
                    self.__neighbor_sampler_store[torch.arange(x.size(0)).unique()]
                )
                if (
                        _optional_neighbor_sampler is not None and
                        type(_optional_neighbor_sampler) == NeighborSampler
                ):
                    current_neighbor_sampler: NeighborSampler = _optional_neighbor_sampler
                else:
                    current_neighbor_sampler: NeighborSampler = NeighborSampler(
                        integral_data.edge_index, torch.arange(x.size(0)).unique(),
                        [-1], batch_size=self.__predicting_batch_size,
                        num_workers=self.__predicting_sampler_num_workers, shuffle=False
                    )
                    __temp: _typing.Any = torch.arange(x.size(0))
                    self.__neighbor_sampler_store[__temp] = current_neighbor_sampler

                for _target_dependant_sampled_data in current_neighbor_sampler:
                    _target_dependant_sampled_data: TargetDependantSampledData = (
                        _target_dependant_sampled_data
                    )
                    _sampled_graph: autogl.data.Data = autogl.data.Data(
                        x=x[_target_dependant_sampled_data.all_sampled_nodes_indexes],
                        edge_index=(
                            _target_dependant_sampled_data.sampled_edges_for_layers[0].edge_index_for_sampled_graph
                        )
                    )
                    _sampled_graph: autogl.data.Data = _sampled_graph.to(self.device)

                    with torch.no_grad():
                        __sampled_graph_inferences: torch.Tensor = (
                            sequential_gnn_model.encoder_sequential_modules[_current_layer_index](_sampled_graph)
                        )
                        _sampled_target_nodes_inferences: torch.Tensor = __sampled_graph_inferences[
                            _target_dependant_sampled_data.target_nodes_indexes.indexes_in_sampled_graph
                        ].cpu()
                    if __next_x is None:
                        __next_x: torch.Tensor = torch.zeros(x.size(0), __sampled_graph_inferences.size(1))
                    __next_x[_target_dependant_sampled_data.target_nodes_indexes.indexes_in_integral_graph] = (
                        _sampled_target_nodes_inferences
                    )
                x: torch.Tensor = __next_x
            # The following procedures are for the top layer
            if mask_or_target_nodes_indexes.dtype == torch.bool:
                target_nodes_indexes: _typing.Any = (
                    torch.where(mask_or_target_nodes_indexes)[0]
                )
            else:
                target_nodes_indexes: _typing.Any = mask_or_target_nodes_indexes.long()

            _optional_neighbor_sampler: _typing.Optional[NeighborSampler] = (
                self.__neighbor_sampler_store[target_nodes_indexes]
            )
            if (
                    _optional_neighbor_sampler is not None and
                    type(_optional_neighbor_sampler) == NeighborSampler
            ):
                current_neighbor_sampler: NeighborSampler = _optional_neighbor_sampler
            else:
                current_neighbor_sampler: NeighborSampler = NeighborSampler(
                    integral_data.edge_index, target_nodes_indexes,
                    [-1], batch_size=self.__predicting_batch_size,
                    num_workers=self.__predicting_sampler_num_workers, shuffle=False
                )
                self.__neighbor_sampler_store[target_nodes_indexes] = current_neighbor_sampler

            prediction_batch_cumulative_builder = (
                EvaluatorUtility.PredictionBatchCumulativeBuilder()
            )
            for _target_dependant_sampled_data in current_neighbor_sampler:
                _sampled_graph: autogl.data.Data = autogl.data.Data(
                    x[_target_dependant_sampled_data.all_sampled_nodes_indexes],
                    _target_dependant_sampled_data.sampled_edges_for_layers[0].edge_index_for_sampled_graph
                )
                _sampled_graph: autogl.data.Data = _sampled_graph.to(self.device)
                with torch.no_grad():
                    prediction_batch_cumulative_builder.add_batch(
                        _target_dependant_sampled_data.target_nodes_indexes.indexes_in_integral_graph.cpu().numpy(),
                        sequential_gnn_model.decode(
                            sequential_gnn_model.encoder_sequential_modules[-1](_sampled_graph)
                        )[_target_dependant_sampled_data.target_nodes_indexes.indexes_in_sampled_graph].cpu().numpy()
                    )
            return torch.from_numpy(prediction_batch_cumulative_builder.compose()[1])
        else:
            if mask_or_target_nodes_indexes.dtype == torch.bool:
                target_nodes_indexes: _typing.Any = (
                    torch.where(mask_or_target_nodes_indexes)[0]
                )
            else:
                target_nodes_indexes: _typing.Any = mask_or_target_nodes_indexes.long()

            neighbor_sampler: NeighborSampler = NeighborSampler(
                integral_data.edge_index, target_nodes_indexes, [-1 for _ in self.__sampling_sizes],
                batch_size=self.__predicting_batch_size,
                num_workers=self.__predicting_sampler_num_workers, shuffle=False
            )

            prediction_batch_cumulative_builder = (
                EvaluatorUtility.PredictionBatchCumulativeBuilder()
            )
            self.model.model.eval()
            for _target_dependant_sampled_data in neighbor_sampler:
                _sampled_graph: autogl.data.Data = autogl.data.Data(
                    x=integral_data.x[
                        _target_dependant_sampled_data.all_sampled_nodes_indexes
                    ]
                )
                _sampled_graph = _sampled_graph.to(self.device)
                _sampled_graph.edge_indexes: _typing.Sequence[torch.LongTensor] = [
                    current_layer.edge_index_for_sampled_graph.to(self.device)
                    for current_layer in _target_dependant_sampled_data.sampled_edges_for_layers
                ]
                with torch.no_grad():
                    prediction_batch_cumulative_builder.add_batch(
                        _target_dependant_sampled_data.target_nodes_indexes.indexes_in_integral_graph.cpu().numpy(),
                        self.model.model(_sampled_graph)[
                            _target_dependant_sampled_data.target_nodes_indexes.indexes_in_sampled_graph
                        ].cpu().numpy()
                    )
            return torch.from_numpy(prediction_batch_cumulative_builder.compose()[1])

    def predict_proba(
            self, dataset, mask: _typing.Optional[str] = None,
            in_log_format: bool = False
    ):
        """
        The function of predicting the probability on the given dataset.
        :param dataset: The node classification dataset used to be predicted.
        :param mask:
        :param in_log_format:
        :return:
        """
        data = dataset[0].to(torch.device("cpu"))
        if mask is not None and type(mask) == str:
            if mask.lower() == "train":
                _mask: torch.BoolTensor = data.train_mask
            elif mask.lower() == "test":
                _mask: torch.BoolTensor = data.test_mask
            elif mask.lower() == "val":
                _mask: torch.BoolTensor = data.val_mask
            else:
                _mask: torch.BoolTensor = data.test_mask
        else:
            _mask: torch.BoolTensor = data.test_mask
        result = self.__predict_only(data, _mask)
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
            data = data.to(torch.device("cpu"))
            prediction: torch.Tensor = self.__predict_only(data, data.val_mask)
            self._valid_result: torch.Tensor = prediction.max(1)[1]
            self._valid_result_prob: torch.Tensor = prediction
            self._valid_score: _typing.Sequence[float] = self.evaluate(dataset, "val")

    def get_valid_predict(self) -> torch.Tensor:
        return self._valid_result

    def get_valid_predict_proba(self) -> torch.Tensor:
        return self._valid_result_prob

    def get_valid_score(
            self, return_major: bool = True
    ) -> _typing.Union[
        _typing.Tuple[float, bool],
        _typing.Tuple[_typing.Sequence[float], _typing.Sequence[bool]]
    ]:
        if return_major:
            return self._valid_score[0], self.feval[0].is_higher_better()
        else:
            return self._valid_score, [f.is_higher_better() for f in self.feval]

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

    def __repr__(self) -> str:
        import yaml
        return yaml.dump(
            {
                "trainer_name": self.__class__.__name__,
                "optimizer": self._optimizer_class,
                "learning_rate": self._learning_rate,
                "max_epoch": self._max_epoch,
                "early_stopping_round": self._early_stopping.patience,
                "sampling_sizes": self.__sampling_sizes,
                "model": repr(self.model)
            }
        )

    def duplicate_from_hyper_parameter(
            self,
            hp: _typing.Dict[str, _typing.Any],
            model: _typing.Optional[BaseModel] = None,
    ) -> "NodeClassificationNeighborSamplingTrainer":
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
