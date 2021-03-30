import importlib
import os

TRAINER_DICT = {}
EVALUATE_DICT = {}
from .base import (
    BaseTrainer,
    Evaluation,
    BaseNodeClassificationTrainer,
    BaseGraphClassificationTrainer,
)


def register_trainer(name):
    def register_trainer_cls(cls):
        if name in TRAINER_DICT:
            raise ValueError("Cannot register duplicate trainer ({})".format(name))
        if not issubclass(cls, BaseTrainer):
            raise ValueError(
                "Trainer ({}: {}) must extend BaseTrainer".format(name, cls.__name__)
            )
        TRAINER_DICT[name] = cls
        return cls

    return register_trainer_cls


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
    if isinstance(feval, list):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")


from .graph_classification_full import GraphClassificationFullTrainer
from .node_classification_full import NodeClassificationFullTrainer
from .node_classification_trainer import *
from .evaluate import Acc, Auc, Logloss

__all__ = [
    "BaseTrainer",
    "BaseNodeClassificationTrainer",
    "BaseGraphClassificationTrainer",
    "GraphClassificationFullTrainer",
    "NodeClassificationFullTrainer",
    "Evaluation",
    "Acc",
    "Auc",
    "Logloss",
]
