"""
Ensemble module.
"""

from collections import Counter
import numpy as np

from .base import BaseEnsembler
from . import register_ensembler
from ...utils import get_logger

VOTE_LOGGER = get_logger("voting")


@register_ensembler("voting")
class Voting(BaseEnsembler):
    """
    An ensembler using the voting method.

    Parameters
    ----------
    ensemble_size : int
        The number of base models selected by the voter. These selected models can be redundant. Default as 10.
    """

    def __init__(self, ensemble_size=10, *args, **kwargs):
        super().__init__()
        self.ensemble_size = ensemble_size

    def fit(self, predictions, label, identifiers, feval, *args, **kwargs):
        """
        Fit the ensembler to the given data using Rich Caruana's ensemble selection method.

        Parameters
        ----------
        predictions : a list of np.ndarray
            Predictions of base learners (corresponding to the elements in identifiers).
        labels : a list of int
            Class labels of instances.
        identifiers : a list of str
            The names of base models.
        feval : (a list of) instances in autogl.module.train.evaluate
            Performance evaluation metrices.

        Returns
        -------
        (a list of) ``float``
            The validation performance of the final voter.
        """

        self._re_initialize(identifiers, len(predictions))

        if not isinstance(feval, list):
            feval = [feval]

        weights = self._specify_weights(predictions, label, feval)
        self.model_to_weight = dict(zip(self.identifiers, weights))

        VOTE_LOGGER.debug(self.identifiers, weights)

        training_score = self._eval(predictions, label, feval)

        return training_score

    def ensemble(self, predictions, identifiers, *args, **kwargs):
        """
        Ensemble the predictions of base models.

        Parameters
        ----------
        predictions : a list of np.ndarray
            Predictions of base learners (corresponding to the elements in identifiers).
        identifiers : a list of str
            The names of base models.

        Returns
        -------
        np.ndarray
            The ensembled predictions.
        """

        weights = np.zeros([len(predictions)])
        for idx, model in enumerate(identifiers):
            weights[idx] = self.model_to_weight[model]
        weights = weights / np.sum(weights)

        return np.average(predictions, axis=0, weights=weights)

    def _specify_weights(self, predictions, label, feval):
        ensemble_prediction = []
        combinations = []
        history = []

        for i in range(self.ensemble_size):
            eval_score_full = []
            eval_score = np.zeros([self.n_models])

            for j, pred in enumerate(predictions):
                ensemble_prediction.append(pred)
                pred_mean = np.mean(ensemble_prediction, axis=0)
                eval_score_full.append(
                    [
                        fx.evaluate(pred_mean, label)
                        * (1 if fx.is_higher_better else -1)
                        for fx in feval
                    ]
                )
                eval_score[j] = eval_score_full[-1][0]
                ensemble_prediction.pop()

            best_model = np.argmax(eval_score)
            ensemble_prediction.append(predictions[best_model])
            history.append(eval_score_full[best_model])
            combinations.append(best_model)

        frequency = Counter(combinations).most_common()
        weights = np.zeros([self.n_models])
        for model, freq in frequency:
            weights[model] = float(freq)

        weights = weights / np.sum(weights)

        return weights

    def _re_initialize(self, identifiers, n_models):
        self.identifiers = identifiers
        self.n_models = n_models

    def _eval(self, predictions, label, feval):
        pred_ensemble = self.ensemble(predictions, self.identifiers)
        return [fx.evaluate(pred_ensemble, label) for fx in feval]
