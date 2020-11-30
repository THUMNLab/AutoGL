"""
Base solver for classification problems
"""

from typing import Any
from ..base import BaseSolver
from ...module.ensemble import ENSEMBLE_DICT
from ...module import BaseEnsembler


class BaseClassifier(BaseSolver):
    """
    Base solver for classification problems
    """

    def predict_proba(self, *args, **kwargs) -> Any:
        """
        Predict the node probability.

        Returns
        -------
        result: Any
            The predicted probability
        """
        raise NotImplementedError()

    def set_ensemble_module(self, ensemble_module, *args, **kwargs) -> "BaseClassifier":
        """
        Set the ensemble module used in current solver.

        Parameters
        ----------
        ensemble_module: autogl.module.ensemble.BaseEnsembler or str or None
            The (name of) ensemble module used to ensemble the multi-models found.
            Disable ensemble by setting it to ``None``.

        Returns
        -------
        self: autogl.solver.BaseSolver
            A reference of current solver.
        """
        # load ensemble module
        if ensemble_module is None:
            self.ensemble_module = None
        elif isinstance(ensemble_module, BaseEnsembler):
            self.ensemble_module = ensemble_module
        elif isinstance(ensemble_module, str):
            if ensemble_module in ENSEMBLE_DICT:
                self.ensemble_module = ENSEMBLE_DICT[ensemble_module](*args, **kwargs)
            else:
                raise KeyError("cannot find ensemble module %s." % (ensemble_module))
        else:
            ValueError(
                "need ensemble module to be str or a BaseEnsembler instance, get",
                type(ensemble_module),
                "instead.",
            )
