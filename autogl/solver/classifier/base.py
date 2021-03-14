"""
Base solver for classification problems
"""

from typing import Any
from ..base import BaseSolver
from ...module.ensemble import ENSEMBLE_DICT
from ...module.train import TRAINER_DICT
from ...module.model import MODEL_DICT
from ...module import BaseEnsembler, BaseModel, BaseTrainer


class BaseClassifier(BaseSolver):
    """
    Base solver for classification problems
    """

    def _init_graph_module(
        self,
        graph_models,
        num_classes,
        num_features,
        *args,
        **kwargs,
    ) -> "BaseClassifier":
        # load graph network module
        self.graph_model_list = []
        if isinstance(graph_models, list):
            for model in graph_models:
                if isinstance(model, str):
                    if model in MODEL_DICT:
                        self.graph_model_list.append(
                            MODEL_DICT[model](
                                num_classes=num_classes,
                                num_features=num_features,
                                *args,
                                **kwargs,
                                init=False,
                            )
                        )
                    else:
                        raise KeyError("cannot find model %s" % (model))
                elif isinstance(model, type) and issubclass(model, BaseModel):
                    self.graph_model_list.append(
                        model(
                            num_classes=num_classes,
                            num_features=num_features,
                            *args,
                            **kwargs,
                            init=False,
                        )
                    )
                elif isinstance(model, BaseModel):
                    # setup the hp of num_classes and num_features
                    model.set_num_classes(num_classes)
                    model.set_num_features(num_features)
                    self.graph_model_list.append(model.to(self.runtime_device))
                elif isinstance(model, BaseTrainer):
                    # receive a trainer list, put trainer to list
                    self.graph_model_list.append(model)
                else:
                    raise KeyError("cannot find graph network %s." % (model))
        else:
            raise ValueError(
                "need graph network to be (list of) str or a BaseModel class/instance, get",
                graph_models,
                "instead.",
            )

        # wrap all model_cls with specified trainer
        for i, model in enumerate(self.graph_model_list):
            # set model hp space
            if self._model_hp_spaces is not None:
                if self._model_hp_spaces[i] is not None:
                    if isinstance(model, BaseTrainer):
                        model.model.hyper_parameter_space = self._model_hp_spaces[i]
                    else:
                        model.hyper_parameter_space = self._model_hp_spaces[i]
            # initialize trainer if needed
            if isinstance(model, BaseModel):
                name = (
                    self._default_trainer
                    if isinstance(self._default_trainer, str)
                    else self._default_trainer[i]
                )
                model = TRAINER_DICT[name](
                    model=model,
                    num_features=num_features,
                    num_classes=num_classes,
                    *args,
                    **kwargs,
                    init=False,
                )
            # set trainer hp space
            if self._trainer_hp_space is not None:
                if isinstance(self._trainer_hp_space[0], list):
                    current_hp_for_trainer = self._trainer_hp_space[i]
                else:
                    current_hp_for_trainer = self._trainer_hp_space
                model.hyper_parameter_space = current_hp_for_trainer
            self.graph_model_list[i] = model

        return self

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
