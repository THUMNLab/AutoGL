import importlib
import os
from .base import BaseFeatureAtom
from .base import BaseFeatureEngineer

FEATURE_DICT = {}


def register_feature(name):
    def register_feature_cls(cls):
        if name in FEATURE_DICT:
            raise ValueError(
                "Cannot register duplicate feature engineer ({})".format(name)
            )
        # if not issubclass(cls, BaseFeatureEngineer):
        if not issubclass(cls, BaseFeatureAtom):
            raise ValueError(
                "Trainer ({}: {}) must extend BaseFeatureEngineer".format(
                    name, cls.__name__
                )
            )
        FEATURE_DICT[name] = cls
        return cls

    return register_feature_cls

from .auto_feature import AutoFeatureEngineer
from .base import BaseFeatureEngineer

from .generators import BaseGenerator
from .selectors import BaseSelector

from .subgraph import BaseSubgraph

__all__ = [
    "BaseFeatureEngineer",
    "AutoFeatureEngineer",
    "BaseFeatureAtom",
]
