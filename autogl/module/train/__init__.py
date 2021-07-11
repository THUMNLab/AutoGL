TRAINER_DICT = {}
from .base import (
    BaseTrainer,
    Evaluation,
    BaseNodeClassificationTrainer,
    BaseGraphClassificationTrainer,
    BaseLinkPredictionTrainer,
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


from .graph_classification_full import GraphClassificationFullTrainer
from .node_classification_full import NodeClassificationFullTrainer
from .link_prediction import LinkPredictionTrainer
from .node_classification_trainer import *
from .evaluation import get_feval, Acc, Auc, Logloss, Mrr, MicroF1

__all__ = [
    "BaseTrainer",
    "Evaluation",
    "BaseGraphClassificationTrainer",
    "BaseNodeClassificationTrainer",
    "BaseLinkPredictionTrainer",
    "GraphClassificationFullTrainer",
    "NodeClassificationFullTrainer",
    "NodeClassificationGraphSAINTTrainer",
    "NodeClassificationLayerDependentImportanceSamplingTrainer",
    "NodeClassificationNeighborSamplingTrainer",
    "LinkPredictionTrainer",
    "Acc",
    "Auc",
    "Logloss",
    "Mrr",
    "MicroF1",
    "get_feval",
]
