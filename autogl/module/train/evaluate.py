import numpy as np
from . import register_evaluate, Evaluation
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    roc_auc_score,
    label_ranking_average_precision_score,
)


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
