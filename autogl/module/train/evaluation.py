import numpy as np
import typing as _typing
from sklearn.metrics import (
    f1_score,
    log_loss,
    accuracy_score,
    roc_auc_score,
    label_ranking_average_precision_score,
)


class Evaluation:
    @staticmethod
    def get_eval_name() -> str:
        """ Expected to return the name of this evaluation method """
        raise NotImplementedError

    @staticmethod
    def is_higher_better() -> bool:
        """ Expected to return whether this evaluation method is higher better (bool) """
        raise NotImplementedError

    @staticmethod
    def evaluate(predict, label) -> float:
        """ Expected to return the evaluation result (float) """
        raise NotImplementedError


class EvaluatorUtility:
    """ Auxiliary utilities for evaluation """

    class PredictionBatchCumulativeBuilder:
        """
        Batch-cumulative builder for prediction
        For large graph, as it is infeasible to predict all the nodes
        in validation set and test set in single batch,
        and layer-wise prediction mechanism is a practical evaluation approach,
        a batch-cumulative prediction collector `PredictionBatchCumulativeBuilder`
        is implemented for prediction in mini-batch manner.
        """

        def __init__(self):
            self.__indexes_in_integral_data: _typing.Optional[np.ndarray] = None
            self.__prediction: _typing.Optional[np.ndarray] = None

        def clear_batches(
            self, *__args, **__kwargs
        ) -> "EvaluatorUtility.PredictionBatchCumulativeBuilder":
            self.__indexes_in_integral_data = None
            self.__prediction = None
            return self

        def add_batch(
            self, indexes_in_integral_data: np.ndarray, batch_prediction: np.ndarray
        ) -> "EvaluatorUtility.PredictionBatchCumulativeBuilder":
            if not (
                isinstance(indexes_in_integral_data, np.ndarray)
                and isinstance(batch_prediction, np.ndarray)
                and len(indexes_in_integral_data.shape) == 1
            ):
                raise TypeError
            elif indexes_in_integral_data.shape[0] != batch_prediction.shape[0]:
                raise ValueError

            if self.__indexes_in_integral_data is None:
                if (
                    indexes_in_integral_data.shape
                    != np.unique(indexes_in_integral_data).shape
                ):
                    raise ValueError(
                        f"There exists duplicate index "
                        f"in the argument indexes_in_integral_data {indexes_in_integral_data}"
                    )
                else:
                    self.__indexes_in_integral_data: np.ndarray = np.unique(
                        indexes_in_integral_data
                    )
            else:
                __indexes_in_integral_data = np.concatenate(
                    (self.__indexes_in_integral_data, indexes_in_integral_data)
                )
                if (
                    __indexes_in_integral_data.shape
                    != np.unique(__indexes_in_integral_data).shape
                ):
                    raise ValueError
                else:
                    self.__indexes_in_integral_data: np.ndarray = (
                        __indexes_in_integral_data
                    )

            if self.__prediction is None:
                self.__prediction: np.ndarray = batch_prediction
            else:
                self.__prediction: np.ndarray = np.concatenate(
                    (self.__prediction, batch_prediction)
                )

            return self

        def compose(
            self, __sorted: bool = True, **__kwargs
        ) -> _typing.Tuple[np.ndarray, np.ndarray]:
            if __sorted:
                sorted_index = np.argsort(self.__indexes_in_integral_data)
                return (
                    self.__indexes_in_integral_data[sorted_index],
                    self.__prediction[sorted_index],
                )
            else:
                return self.__indexes_in_integral_data, self.__prediction


EVALUATE_DICT: _typing.Dict[str, _typing.Type[Evaluation]] = {}


def register_evaluate(*name):
    def register_evaluate_cls(cls):
        for n in name:
            if n in EVALUATE_DICT:
                raise ValueError("Cannot register duplicate evaluator ({})".format(n))
            if not issubclass(cls, Evaluation):
                raise ValueError(
                    "Evaluator ({}: {}) must extend Evaluation".format(n, cls.__name__)
                )
            EVALUATE_DICT[n] = cls
        return cls

    return register_evaluate_cls


def get_feval(feval):
    if isinstance(feval, str):
        return EVALUATE_DICT[feval]
    if isinstance(feval, type) and issubclass(feval, Evaluation):
        return feval
    if isinstance(feval, _typing.Sequence):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")


class EvaluationUniversalRegistry:
    @classmethod
    def register_evaluation(
        cls, *names
    ) -> _typing.Callable[[_typing.Type[Evaluation]], _typing.Type[Evaluation]]:
        def _register_evaluation(
            _class: _typing.Type[Evaluation],
        ) -> _typing.Type[Evaluation]:
            for n in names:
                if n in EVALUATE_DICT:
                    raise ValueError(
                        "Cannot register duplicate evaluator ({})".format(n)
                    )
                if not issubclass(_class, Evaluation):
                    raise ValueError(
                        "Evaluator ({}: {}) must extend Evaluation".format(
                            n, cls.__name__
                        )
                    )
                EVALUATE_DICT[n] = _class
            return _class

        return _register_evaluation


@register_evaluate("logloss")
class Logloss(Evaluation):
    @staticmethod
    def get_eval_name():
        return "logloss"

    @staticmethod
    def is_higher_better():
        """
        Should return whether this evaluation method is higher better (bool)
        """
        return False

    @staticmethod
    def evaluate(predict, label):
        """
        Should return: the evaluation result (float)
        """
        return log_loss(label, predict)


@register_evaluate("auc", "ROC-AUC")
class Auc(Evaluation):
    @staticmethod
    def get_eval_name():
        return "auc"

    @staticmethod
    def is_higher_better():
        """
        Should return whether this evaluation method is higher better (bool)
        """
        return True

    @staticmethod
    def evaluate(predict, label):
        """
        Should return: the evaluation result (float)
        """
        if len(predict.shape) == 1:
            pos_predict = predict
        else:
            assert (
                predict.shape[1] == 2
            ), "Cannot use auc on given data with %d classes!" % (predict.shape[1])
            pos_predict = predict[:, 1]
        return roc_auc_score(label, pos_predict)


@register_evaluate("acc", "Accuracy")
class Acc(Evaluation):
    @staticmethod
    def get_eval_name():
        return "acc"

    @staticmethod
    def is_higher_better():
        """
        Should return whether this evaluation method is higher better (bool)
        """
        return True

    @staticmethod
    def evaluate(predict, label):
        """
        Should return: the evaluation result (float)
        """
        if len(predict.shape) == 2:
            predict = np.argmax(predict, axis=1)
        else:
            predict = [1 if p > 0.5 else 0 for p in predict]
        return accuracy_score(label, predict)


@register_evaluate("mrr")
class Mrr(Evaluation):
    @staticmethod
    def get_eval_name():
        return "mrr"

    @staticmethod
    def is_higher_better():
        """
        Should return whether this evaluation method is higher better (bool)
        """
        return True

    @staticmethod
    def evaluate(predict, label):
        """
        Should return: the evaluation result (float)
        """
        if len(predict.shape) == 2:
            assert (
                predict.shape[1] == 2
            ), "Cannot use mrr on given data with %d classes!" % (predict.shape[1])
            pos_predict = predict[:, 1]
        else:
            pos_predict = predict
        return label_ranking_average_precision_score(label, pos_predict)


@register_evaluate("MicroF1")
class MicroF1(Evaluation):
    @staticmethod
    def get_eval_name() -> str:
        return "MicroF1"

    @staticmethod
    def is_higher_better() -> bool:
        return True

    @staticmethod
    def evaluate(predict, label) -> float:
        return f1_score(label, np.argmax(predict, axis=1), average="micro")
