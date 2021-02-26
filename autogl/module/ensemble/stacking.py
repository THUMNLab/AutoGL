"""
Ensemble module.
"""

import numpy as np

import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .base import BaseEnsembler
from . import register_ensembler
from ...utils import get_logger

STACKING_LOGGER = get_logger("stacking")


@register_ensembler("stacking")
class Stacking(BaseEnsembler):
    """
    A stacking ensembler. Currently we support gradient boosting as the meta-algorithm.

    Parameters
    ----------
    meta_model : 'gbm' or 'glm' (Optional)
        Type of the stacker:
            'gbm' : Gradient boosting model. This is the default.
            'glm' : Generalized linear model.

    meta_params : a ``dict``  (Optional)
        When ``meta_model`` is specified, you can customize the parameters of the stacker.
        If this argument is not provided, the stacker will be configurated with default parameters.
        Default ``{}``.
    """

    def __init__(self, meta_model="gbm", meta_params={}, *args, **kwargs):
        super().__init__()
        self.model_name = meta_model.lower()
        assert self.model_name in [
            "gbm",
            "glm",
        ], "Only support gbm and glm when ensemble!"
        self.meta_params = meta_params

    def fit(
        self, predictions, label, identifiers, feval, n_classes="auto", *args, **kwargs
    ):
        """
        Fit the ensembler to the given data using Stacking method.

        Parameters
        ----------
        predictions : a list of np.ndarray
            Predictions of base learners (corresponding to the elements in identifiers).

        label : a list of int
            Class labels of instances.

        identifiers : a list of str
            The names of base models.

        feval : (a list of) autogl.module.train.evaluate
            Performance evaluation metrices.

        n_classes : int or str (Optional)
            The number of classes. Default as ``'auto'``, which will use maximum label.

        Returns
        -------
        (a list of) float
            The validation performance of the final stacker.
        """

        n_classes = n_classes if not n_classes == "auto" else max(label) + 1
        assert n_classes > max(
            label
        ), "Detect max label passed (%d) exceeeds" " n_classes given (%d)" % (
            max(label),
            n_classes,
        )

        assert len(identifiers) == len(
            set(identifiers)
        ), "Duplicate name" " in identifiers {} !".format(identifiers)

        self.fit_identifiers = identifiers

        if not isinstance(feval, list):
            feval = [feval]

        self._re_initialize(identifiers, len(predictions))

        config = self.meta_params

        STACKING_LOGGER.debug("meta-model name %s", self.model_name)

        if self.model_name == "gbm":
            meta_X = (
                torch.tensor(predictions).transpose(0, 1).flatten(start_dim=1).numpy()
            )
            meta_Y = np.array(label)
            config = {}
            model = GradientBoostingClassifier(**config)
            model.fit(meta_X, meta_Y)

            self.model = model
            ensemble_prediction = model.predict_proba(meta_X)

        elif self.model_name == "glm":
            meta_X = (
                torch.tensor(predictions).transpose(0, 1).flatten(start_dim=1).numpy()
            )
            meta_Y = np.array(label)

            config["multi_class"] = "auto"
            config["solver"] = "lbfgs"
            model = LogisticRegression(**config)
            model.fit(meta_X, meta_Y)

            self.model = model
            ensemble_prediction = model.predict_proba(meta_X)

        elif self.model_name == "nn":
            meta_X = torch.tensor(predictions).transpose(0, 1).flatten(start_dim=1)
            meta_Y = F.one_hot(
                torch.tensor(label, dtype=torch.int64), n_classes
            ).double()
            # print(meta_Y.type())

            n_instance, n_input = meta_X.size()
            n_learners = len(identifiers)

            fc = torch.nn.Linear(n_input, n_input // n_learners).double()

            config["lr"] = 1e-1
            # config['weight_decay'] = 1e-2
            optimizer = torch.optim.SGD(fc.parameters(), **config)

            max_epoch = 100
            for epoch in range(max_epoch):
                optimizer.zero_grad()
                ensemble_prediction = F.normalize(fc.forward(meta_X), dim=0)
                loss = F.mse_loss(ensemble_prediction, meta_Y)
                loss.backward()
                optimizer.step()

            self.model = fc
            ensemble_prediction = (
                F.normalize(fc.forward(meta_X), dim=0).detach().numpy()
            )

        else:
            STACKING_LOGGER.error(
                "Cannot parse stacking ensemble model name %s", self.model_name
            )

        return [fx.evaluate(ensemble_prediction, label) for fx in feval]

    def ensemble(self, predictions, identifiers, *args, **kwargs):
        """
        Ensemble the predictions of base models.

        Parameters
        ----------
        predictions : a list of ``np.ndarray``
            Predictions of base learners (corresponding to the elements in identifiers).
        identifiers : a list of ``str``
            The names of base models.

        Returns
        -------
        ``np.ndarray``
            The ensembled predictions.
        """

        assert len(identifiers) == len(
            set(identifiers)
        ), "Duplicate name in" " identifiers {} !".format(identifiers)

        assert set(self.fit_identifiers) == set(
            identifiers
        ), "Different identifiers" " passed in fit {} and ensemble {} !".format(
            self.fit_identifiers, identifiers
        )

        # re-order predictions if needed
        if not self.fit_identifiers == identifiers:
            re_id = [
                identifiers.index(identifier) for identifier in self.fit_identifiers
            ]
            predictions = [predictions[i] for i in re_id]

        if self.model_name in ["gbm", "glm"]:
            pred_packed = (
                torch.tensor(predictions).transpose(0, 1).flatten(start_dim=1).numpy()
            )
            return self.model.predict_proba(pred_packed)

        elif self.model_name in ["nn"]:
            pred_packed = torch.tensor(predictions).transpose(0, 1).flatten(start_dim=1)
            return F.normalize(self.model.forward(pred_packed), dim=0).detach().numpy()

    def _re_initialize(self, identifiers, n_models):
        self.identifiers = identifiers
        self.model = None
