"""
NAS algorithms
"""

import importlib
import os
from .base import BaseNAS

NAS_ALGO_DICT = {}


def register_nas_algo(name):
    def register_nas_algo_cls(cls):
        if name in NAS_ALGO_DICT:
            raise ValueError(
                "Cannot register duplicate NAS algorithm ({})".format(name)
            )
        if not issubclass(cls, BaseNAS):
            raise ValueError(
                "Model ({}: {}) must extend NAS algorithm".format(name, cls.__name__)
            )
        NAS_ALGO_DICT[name] = cls
        return cls

    return register_nas_algo_cls


from .darts import Darts
from .enas import Enas
from .random_search import RandomSearch
from .rl import RL, GraphNasRL
from ..backend import *
if not is_dgl():
    from .gasso import Gasso
    from .grna import GRNA
from .spos import Spos

def build_nas_algo_from_name(name: str) -> BaseNAS:
    """
    Parameters
    ----------
    name: ``str``
        the name of nas algorithm.

    Returns
    -------
    BaseNAS:
        the NAS algorithm built using default parameters

    Raises
    ------
    AssertionError
        If an invalid name is passed in
    """
    assert name in NAS_ALGO_DICT, "HPO module do not have name " + name
    return NAS_ALGO_DICT[name]()


__all__ = ["BaseNAS", "Darts", "Enas", "RandomSearch", "RL", "GraphNasRL","Spos"]
if not is_dgl():
    __all__.append("Gasso")
    __all__.append("GRNA")
