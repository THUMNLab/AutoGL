"""
Solver base class

Provide some standard solver interface.
"""

from typing import Any, Tuple
from copy import deepcopy

import torch

from ..module.feature import FEATURE_DICT
from ..module.hpo import HPO_DICT
from ..module.model import MODEL_DICT
from ..module.nas.algorithm import NAS_ALGO_DICT
from ..module.nas.estimator import NAS_ESTIMATOR_DICT
from ..module.nas.space import NAS_SPACE_DICT
from ..module import BaseFeature, BaseHPOptimizer, BaseTrainer
from .utils import LeaderBoard
from ..utils import get_logger

LOGGER = get_logger("BaseSolver")


def _initialize_single_model(model_name, parameters=None):
    if parameters:
        return MODEL_DICT[model_name](**parameters)
    return MODEL_DICT[model_name]()


def _parse_hp_space(spaces):
    if spaces is None:
        return None
    for space in spaces:
        if "cutFunc" in space and isinstance(space["cutFunc"], str):
            space["cutFunc"] = eval(space["cutFunc"])
    return spaces


class BaseSolver:
    r"""
    Base solver class, define some standard solver interfaces.

    Parameters
    ----------
    feature_module: autogl.module.feature.BaseFeatureEngineer or str or None
        The (name of) auto feature engineer used to process the given dataset.
        Disable feature engineer by setting it to ``None``.

    graph_models: list of autogl.module.model.BaseModel or str
        The (name of) models to be optimized as backbone.

    hpo_module: autogl.module.hpo.BaseHPOptimizer or str or None
        The (name of) hpo module used to search for best hyper parameters.
        Disable hpo by setting it to ``None``.

    ensemble_module: autogl.module.ensemble.BaseEnsembler or str or None
        The (name of) ensemble module used to ensemble the multi-models found.
        Disable ensemble by setting it to ``None``.

    max_evals: int (Optional)
        If given, will set the number eval times the hpo module will use.
        Only be effective when hpo_module is  of type ``str``. Default ``50``.

    default_trainer: str or list of str (Optional)
        Default trainer class to be used.
        If a single trainer class is given, will set all trainer to default trainer.
        If a list of trainer class is given, will set every model with corresponding trainer
        cls. Default ``None``.

    trainer_hp_space: list of dict (Optional)
        trainer hp space or list of trainer hp spaces configuration.
        If a single trainer hp is given, will specify the hp space of trainer for every model.
        If a list of trainer hp is given, will specify every model with corrsponding
        trainer hp space. Default ``None``

    model_hp_spaces: list of list of dict (Optional)
        model hp space configuration.
        If given, will specify every hp space of every passed model. Default ``None``.

    size: int (Optional)
        The max models ensemble module will use. Default ``None``.

    device: torch.device or str
        The device where model will be running on. If set to ``auto``, will use gpu when available.
        You can also specify the device by directly giving ``gpu`` or ``cuda:0``, etc.
        Default ``auto``.
    """

    # pylint: disable=W0102

    def __init__(
        self,
        feature_module,
        graph_models,
        nas_spaces,
        nas_algorithms,
        nas_estimators,
        hpo_module,
        ensemble_module,
        max_evals=50,
        default_trainer=None,
        trainer_hp_space=None,
        model_hp_spaces=None,
        size=4,
        device="auto",
    ):

        # set default device
        if device == "auto":
            self.runtime_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, torch.device):
            self.runtime_device = device
        elif isinstance(device, str) and (device == "cpu" or device.startswith("cuda")):
            self.runtime_device = torch.device(device)
        else:
            LOGGER.error("Cannot parse device %s", str(device))
            raise ValueError("Cannot parse device {}".format(device))

        # initialize modules
        self.graph_model_list = []
        self.set_graph_models(
            graph_models, default_trainer, trainer_hp_space, model_hp_spaces
        )
        self.set_feature_module(feature_module)
        self.set_hpo_module(hpo_module, max_evals=max_evals)
        self.set_ensemble_module(ensemble_module, size=size)
        self.set_nas_module(nas_algorithms, nas_spaces, nas_estimators)

        # initialize leaderboard
        self.leaderboard = None

        # trained model is saved here
        self.trained_models = {}

    def set_feature_module(
        self,
        feature_module,
        *args,
        **kwargs,
    ) -> "BaseSolver":
        r"""
        Set the feature module of current solver.

        Parameters
        ----------
        feature_module: autogl.module.feature.BaseFeature or str or None
            The (name of) auto feature engineer used to process the given dataset.
            Disable feature engineer by setting it to ``None``.

        Returns
        -------
        self: autogl.solver.BaseSolver
            A reference of current solver.
        """
        # load feature engineer module

        def get_feature(feature_engineer):
            if isinstance(feature_engineer, BaseFeature):
                return feature_engineer
            if isinstance(feature_engineer, str):
                if feature_engineer in FEATURE_DICT:
                    return FEATURE_DICT[feature_engineer](*args, **kwargs)
                raise ValueError(
                    "cannot find feature engineer %s." % (feature_engineer)
                )
            raise TypeError(
                f"Cannot parse feature argument {str(feature_engineer)} of"
                " type {str(type(feature_engineer))}"
            )

        if feature_module is None:
            self.feature_module = None
        elif isinstance(feature_module, (BaseFeature, str)):
            self.feature_module = get_feature(feature_module)
        elif isinstance(feature_module, list):
            self.feature_module = get_feature(feature_module[0])
            for feature_engineer in feature_module[1:]:
                self.feature_module &= get_feature(feature_engineer)
        else:
            raise ValueError(
                "need feature module to be str or a BaseFeatureEngineer instance, get",
                type(feature_module),
                "instead.",
            )

        return self

    def set_graph_models(
        self,
        graph_models,
        default_trainer=None,
        trainer_hp_space=None,
        model_hp_spaces=None,
    ) -> "BaseSolver":
        r"""
        Set the graph models used in current solver.

        Parameters
        ----------
        graph_models: list of autogl.module.model.BaseModel or list of str
            The (name of) models to be optimized as backbone.

        default_trainer: str or list of str (Optional)
            Default trainer class to be used.
            If a single trainer class is given, will set all trainer to default trainer.
            If a list of trainer class is given, will set every model with corresponding trainer
            cls. Default ``None``.

        trainer_hp_space: list of dict (Optional)
            trainer hp space or list of trainer hp spaces configuration.
            If a single trainer hp is given, will specify the hp space of trainer for every model.
            If a list of trainer hp is given, will specify every model with corrsponding
            trainer hp space.
            Default ``None``.

        model_hp_spaces: list of list of dict (Optional)
            model hp space configuration.
            If given, will specify every hp space of every passed model. Default ``None``.

        Returns
        -------
        self: autogl.solver.BaseSolver
            A reference of current solver.
        """
        self.gml = graph_models
        self._default_trainer = default_trainer
        self._trainer_hp_space = trainer_hp_space
        self._model_hp_spaces = model_hp_spaces
        return self

    def set_hpo_module(self, hpo_module, *args, **kwargs) -> "BaseSolver":
        r"""
        Set the hpo module used in current solver.

        Parameters
        ----------
        hpo_module: autogl.module.hpo.BaseHPOptimizer or str or None
            The (name of) hpo module used to search for best hyper parameters.
            Disable hpo by setting it to ``None``.

        Returns
        -------
        self: autogl.solver.BaseSolver
            A reference of current solver.
        """

        # load hpo module
        if hpo_module is None:
            self.hpo_module = None
        elif isinstance(hpo_module, BaseHPOptimizer):
            self.hpo_module = hpo_module
        elif isinstance(hpo_module, str):
            if hpo_module in HPO_DICT:
                self.hpo_module = HPO_DICT[hpo_module](*args, **kwargs)
            else:
                raise KeyError("cannot find hpo module %s." % (hpo_module))
        else:
            raise ValueError(
                "need hpo module to be str or a BaseHPOtimizer instance, get",
                type(hpo_module),
                "instead.",
            )
        return self

    def set_nas_module(
        self, nas_algorithms=None, nas_spaces=None, nas_estimators=None
    ) -> "BaseSolver":
        """
        Set the neural architecture search module in current solver.

        Parameters
        ----------
        nas_spaces: (list of) `autogl.module.hpo.nas.GraphSpace`
            The search space of nas. You can pass a list of space to enable
            multiple space search. If list passed, the length of `nas_spaces`,
            `nas_algorithms` and `nas_estimators` should be the same. If set
            to `None`, will disable the whole nas module.

        nas_algorithms: (list of) `autogl.module.hpo.nas.BaseNAS`
            The search algorithm of nas. You can pass a list of algorithms
            to enable multiple algorithms search. If list passed, the length of
            `nas_spaces`, `nas_algorithms` and `nas_estimators` should be the same.
            Default `None`.

        nas_estimators: (list of) `autogl.module.hpo.nas.BaseEstimators`
            The nas estimators. You can pass a list of estimators to enable multiple
            estimators search. If list passed, the length of `nas_spaces`, `nas_algorithms`
            and `nas_estimators` should be the same. Default `None`.

        Returns
        -------
        self: autogl.solver.BaseSolver
            A reference of current solver.
        """
        if nas_algorithms is None and nas_estimators is None and nas_spaces is None:
            self.nas_algorithms = self.nas_estimators = self.nas_spaces = None
            return
        assert None not in [
            nas_algorithms,
            nas_estimators,
            nas_spaces,
        ], "The algorithms, estimators and spaces should all be set"

        nas_algorithms = (
            nas_algorithms
            if isinstance(nas_algorithms, (list, tuple))
            else [nas_algorithms]
        )
        nas_spaces = (
            nas_spaces if isinstance(nas_spaces, (list, tuple)) else [nas_spaces]
        )
        nas_estimators = (
            nas_estimators
            if isinstance(nas_estimators, (list, tuple))
            else [nas_estimators]
        )

        # parse all str elements
        nas_algorithms = [
            algo if not isinstance(algo, str) else NAS_ALGO_DICT[algo]()
            for algo in nas_algorithms
        ]
        nas_spaces = [
            space if not isinstance(space, str) else NAS_SPACE_DICT[space]()
            for space in nas_spaces
        ]
        nas_estimators = [
            estimator
            if not isinstance(estimator, str)
            else NAS_ESTIMATOR_DICT[estimator]()
            for estimator in nas_estimators
        ]

        max_number = max([len(x) for x in [nas_algorithms, nas_spaces, nas_estimators]])
        assert all(
            [
                len(x) in [1, max_number]
                for x in [nas_algorithms, nas_spaces, nas_estimators]
            ]
        ), "lengths of algorithms/spaces/estimators do not match!"

        self.nas_algorithms = (
            [deepcopy(nas_algorithms) for _ in range(max_number)]
            if len(nas_algorithms) == 1 and max_number > 1
            else nas_algorithms
        )
        self.nas_spaces = (
            [deepcopy(nas_spaces) for _ in range(max_number)]
            if len(nas_spaces) == 1 and max_number > 1
            else nas_spaces
        )
        self.nas_estimators = (
            [deepcopy(nas_estimators) for _ in range(max_number)]
            if len(nas_estimators) == 1 and max_number > 1
            else nas_estimators
        )

        return self

    def set_ensemble_module(self, ensemble_module, *args, **kwargs) -> "BaseSolver":
        r"""
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

        raise NotImplementedError()

    def fit(self, *args, **kwargs) -> "BaseSolver":
        r"""
        Fit current solver on given dataset.

        Returns
        -------
        self: autogl.solver.BaseSolver
            A reference of current solver.
        """
        raise NotImplementedError()

    def fit_predict(self, *args, **kwargs) -> Any:
        r"""
        Fit current solver on given dataset and return the predicted value.

        Returns
        -------
        result: Any
            The predicted result
        """
        raise NotImplementedError()

    def predict(self, *args, **kwargs) -> Any:
        r"""
        Predict the node class number.

        Returns
        -------
        result: Any
            The predicted result
        """
        raise NotImplementedError()

    def get_leaderboard(self) -> LeaderBoard:
        r"""
        Get the current leaderboard of this solver.

        Returns
        -------
        lb: autogl.solver.leaderboard
            A leaderboard instance.
        """
        return self.leaderboard

    def get_model_by_name(self, name) -> BaseTrainer:
        r"""
        Find and get the model instance by name.

        Parameters
        ----------
        name: str
            The name of model

        Returns
        -------
        trainer: autogl.module.train.BaseTrainer
            A trainer instance containing the trained models and training status.
        """
        assert name in self.trained_models, "cannot find model by name" + name
        return self.trained_models[name]

    def get_model_by_performance(self, index) -> Tuple[BaseTrainer, str]:
        r"""
        Find and get the model instance by performance.

        Parameters
        ----------
        index: int
            The performance index of model (from good to bad). Index from 0.

        Returns
        -------
        trainer: autogl.module.train.BaseTrainer
            A trainer instance containing the trained models and training status.
        name: str
            The name of current trainer.
        """
        name = self.leaderboard.get_best_model(index=index)
        return self.trained_models[name], name

    @classmethod
    def from_config(cls, path_or_dict, filetype="auto") -> "BaseSolver":
        r"""
        Load solver from config file.

        You can use this function to directly load a solver from predefined config dict
        or config file path.

        Parameters
        ----------
        path_or_dict: str or dict
            The path to the config file or the config dictionary object

        filetype: str
            The filetype the given file if the path is specified.

        Returns
        -------
        solver: autogl.solver.AutoGraphClassifier
            The solver that is created from given file or dictionary.
        """
        raise NotImplementedError()
