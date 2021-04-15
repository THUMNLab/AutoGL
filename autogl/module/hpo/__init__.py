import importlib
import os
from .base import BaseHPOptimizer

HPO_DICT = {}


def register_hpo(name):
    def register_hpo_cls(cls):
        if name in HPO_DICT:
            raise ValueError("Cannot register duplicate HP Optimizer ({})".format(name))
        if not issubclass(cls, BaseHPOptimizer):
            raise ValueError(
                "Model ({}: {}) must extend HPOptimizer".format(name, cls.__name__)
            )
        HPO_DICT[name] = cls
        return cls

    return register_hpo_cls


from .anneal_advisorhpo import AnnealAdvisorHPO
from .autone import AutoNE
from .bayes_advisor import BayesAdvisor
from .cmaes_advisorchoco import CmaesAdvisorChoco
from .grid_advisor import GridAdvisor
from .mocmaes_advisorchoco import MocmaesAdvisorChoco
from .quasi_advisorchoco import QuasiAdvisorChoco
from .rand_advisor import RandAdvisor
from .tpe_advisorhpo import TpeAdvisorHPO


def build_hpo_from_name(name: str) -> BaseHPOptimizer:
    """
    Parameters
    ----------
    name: ``str``
        the name of hpo module.

    Returns
    -------
    BaseHPOptimizer:
        the HPO built using default parameters

    Raises
    ------
    AssertionError
        If an invalid name is passed in
    """
    assert name in HPO_DICT, "HPO module do not have name " + name
    return HPO_DICT[name]()


__all__ = [
    "BaseHPOptimizer",
    "AnnealAdvisorHPO",
    "AutoNE",
    "BayesAdvisor",
    "CmaesAdvisorChoco",
    "GridAdvisor",
    "MocmaesAdvisorChoco",
    "QuasiAdvisorChoco",
    "RandAdvisor",
    "TpeAdvisorHPO",
    "build_hpo_from_name",
]
