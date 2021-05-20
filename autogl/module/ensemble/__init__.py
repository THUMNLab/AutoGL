from .base import BaseEnsembler

ENSEMBLE_DICT = {}


def register_ensembler(name):
    def register_ensembler_cls(cls):
        if name in ENSEMBLE_DICT:
            raise ValueError("Cannot register duplicate ensembler ({})".format(name))
        if not issubclass(cls, BaseEnsembler):
            raise ValueError(
                "Model ({}: {}) must extend BaseEnsembler".format(name, cls.__name__)
            )
        ENSEMBLE_DICT[name] = cls
        return cls

    return register_ensembler_cls


from .voting import Voting
from .stacking import Stacking


def build_ensembler_from_name(name: str) -> BaseEnsembler:
    """
    Parameters
    ----------
    name: ``str``
        the name of ensemble module.

    Returns
    -------
    BaseEnsembler:
        the ensembler built using default parameters

    Raises
    ------
    AssertionError
        If an invalid name is passed in
    """
    assert name in ENSEMBLE_DICT, "ensemble module do not have name " + name
    return ENSEMBLE_DICT[name]()


__all__ = ["BaseEnsembler", "Voting", "Stacking", "build_ensembler_from_name"]
