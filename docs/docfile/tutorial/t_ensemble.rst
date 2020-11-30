.. _ensemble:

Ensemble
========

We currently support voting and stacking methods. 

Voting
------

A voter essentially constructs a weighted sum of the predictions of base learners. Given an evaluation metric, the weights of base learners are specified in some way to maximize the validation score. 

We adopt Rich Caruana's method for weight specification. This method first finds a collection of (possibly redundant) base learners with equal weights via a greedy search, then specifies the weights in the voter by the number of occurrence in the collection. 

You can customize your own weight specification method by overwriting the ``_specify_weights`` method. 

.. code-block :: python

    # An example : use equal weights for all base learners.
    class EqualWeightVoting(Voting):
        def _specify_weights(self, predictions, label, feval):
            return np.ones(self.n_models)/self.n_models
            # just allocate the same weight for each base learner

Stacking
--------

A stacker trains a meta-model with the predictions of base learners as input to find an optimal combination of these base learners. 

Currently we support generalized linear model (GLM) and gradient boosting model (GBM) as the meta-model. 

Create a New Ensembler
----------------------

You can create your own ensembler by inheriting the base ensembler, and overloading methods ``fit`` and ``ensemble``.

.. code-block :: python

    # An example : use the currently available best model.
    from autogl.module.ensemble.base import BaseEnsembler
    import numpy as np
    class BestModel(BaseEnsembler):
        def fit(self, predictions, label, identifiers, feval):
            if not isinstance(feval, list):
                feval = [feval]
            scores = np.array([feval[0].evaluate(pred, label) for pred in predictions]) * (1 if feval[0].is_higher_better else -1)
            self.scores = dict(zip(identifiers, scores)) # record validation score of base learners
            ensemble_pred = predictions[np.argmax(scores)]
            return [fx.evaluate(ensemble_pred, label) for fx in feval]

        def ensemble(self, predictions, identifiers):
            best_idx = np.argmax([self.scores[model_name] for model_name in identifiers]) # choose the currently best model in the identifiers
            return predictions[best_idx]

