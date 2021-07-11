import importlib
import os
from .base import BaseSpace

NAS_SPACE_DICT = {}


def register_nas_space(name):
    def register_nas_space_cls(cls):
        if name in NAS_SPACE_DICT:
            raise ValueError("Cannot register duplicate NAS space ({})".format(name))
        if not issubclass(cls, BaseSpace):
            raise ValueError(
                "Model ({}: {}) must extend NAS space".format(name, cls.__name__)
            )
        NAS_SPACE_DICT[name] = cls
        return cls

    return register_nas_space_cls


from .graph_nas_macro import GraphNasMacroNodeClassificationSpace
from .graph_nas import GraphNasNodeClassificationSpace
from .single_path import SinglePathNodeClassificationSpace


def build_nas_space_from_name(name: str) -> BaseSpace:
    """
    Parameters
    ----------
    name: ``str``
        the name of nas space.

    Returns
    -------
    BaseSpace:
        the NAS space built using default parameters

    Raises
    ------
    AssertionError
        If an invalid name is passed in
    """
    assert name in NAS_SPACE_DICT, "HPO module do not have name " + name
    return NAS_SPACE_DICT[name]()


__all__ = [
    "BaseSpace",
    "GraphNasMacroNodeClassificationSpace",
    "GraphNasNodeClassificationSpace",
    "SinglePathNodeClassificationSpace",
]
