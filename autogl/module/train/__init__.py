import importlib
import os
from .base import BaseTrainer, Evaluation, EarlyStopping

TRAINER_DICT = {}


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


EVALUATE_DICT = {}


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


"""
# automatically import any Python files in this directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        module = importlib.import_module("autograph.module.train." + file_name)
"""


def get_feval(feval):
    if isinstance(feval, str):
        return EVALUATE_DICT[feval]
    if isinstance(feval, type) and issubclass(feval, Evaluation):
        return feval
    if isinstance(feval, list):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")


from .graph_classification import GraphClassificationTrainer
from .node_classification import NodeClassificationTrainer
from .evaluate import Acc, Auc, Logloss

__all__ = [
    "BaseTrainer",
    "GraphClassificationTrainer",
    "NodeClassificationTrainer",
    "Evaluation",
    "Acc",
    "Auc",
    "Logloss",
]
