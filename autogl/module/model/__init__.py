import importlib
import os

MODEL_DICT = {}


def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_DICT:
            raise ValueError("Cannot register duplicate trainer ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError(
                "Trainer ({}: {}) must extend BaseModel".format(name, cls.__name__)
            )
        MODEL_DICT[name] = cls
        return cls

    return register_model_cls


# automatically import any Python files in this directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith(".py") and not file.startswith("_"):
#         file_name = file[: file.find(".py")]
#         module = importlib.import_module("autograph.module.model." + file_name)

from .base import BaseModel
from .topkpool import AutoTopkpool
from .graphsage import AutoSAGE
from .gcn import AutoGCN
from .gat import AutoGAT
from .gin import AutoGIN


__all__ = [
    "BaseModel",
    "AutoTopkpool",
    "AutoSAGE",
    "AutoGCN",
    "AutoGAT",
    "AutoGIN",
]
