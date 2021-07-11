import importlib
import os
from .base import BaseEstimator

NAS_ESTIMATOR_DICT = {}


def register_nas_estimator(name):
    def register_nas_estimator_cls(cls):
        if name in NAS_ESTIMATOR_DICT:
            raise ValueError(
                "Cannot register duplicate NAS estimator ({})".format(name)
            )
        if not issubclass(cls, BaseEstimator):
            raise ValueError(
                "Model ({}: {}) must extend NAS estimator".format(name, cls.__name__)
            )
        NAS_ESTIMATOR_DICT[name] = cls
        return cls

    return register_nas_estimator_cls


from .one_shot import OneShotEstimator
from .train_scratch import TrainEstimator


def build_nas_estimator_from_name(name: str) -> BaseEstimator:
    """
    Parameters
    ----------
    name: ``str``
        the name of nas estimator.

    Returns
    -------
    BaseEstimator:
        the NAS estimator built using default parameters

    Raises
    ------
    AssertionError
        If an invalid name is passed in
    """
    assert name in NAS_ESTIMATOR_DICT, "HPO module do not have name " + name
    return NAS_ESTIMATOR_DICT[name]()


__all__ = ["BaseEstimator", "OneShotEstimator", "TrainEstimator"]
