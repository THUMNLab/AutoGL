import numpy as np
import typing as _typing
from sklearn.metrics import (
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
        return True

    @staticmethod
    def evaluate(predict, label) -> float:
        """ Expected to return the evaluation result (float) """
        raise NotImplementedError


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
        return accuracy_score(label, np.argmax(predict, axis=1))


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
        pos_predict = predict[:, 1]
        return label_ranking_average_precision_score(label, pos_predict)
