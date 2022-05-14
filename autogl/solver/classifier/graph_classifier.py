"""
Auto Classfier for Graph Node Classification
"""
import time
import json

from copy import deepcopy
from typing import Sequence

import torch
import numpy as np
import yaml

from .base import BaseClassifier
from ...module.feature import FEATURE_DICT
from ...module.train import TRAINER_DICT, get_feval, BaseGraphClassificationTrainer
from ..base import _initialize_single_model, _parse_hp_space, _parse_model_hp
from ...module.nas.space import NAS_SPACE_DICT
from ...module.nas.algorithm import NAS_ALGO_DICT
from ...module.nas.estimator import NAS_ESTIMATOR_DICT, BaseEstimator
from ..utils import LeaderBoard, get_dataset_labels, set_seed, get_graph_from_dataset, get_graph_node_features, convert_dataset
from ...datasets import utils
from ..utils import get_logger
from ...backend import DependentBackend

LOGGER = get_logger("GraphClassifier")
BACKEND = DependentBackend.get_backend_name()

class AutoGraphClassifier(BaseClassifier):
    """
    Auto Multi-class Graph Classifier.

    Used to automatically solve the graph classification problems.

    Parameters
    ----------
    feature_module: autogl.module.feature.BaseFeatureEngineer or str or None
        The (name of) auto feature engineer used to process the given dataset.
        Disable feature engineer by setting it to ``None``. Default ``deepgl``.

    graph_models: list of autogl.module.model.BaseModel or list of str
        The (name of) models to be optimized as backbone. Default ``['gat', 'gcn']``.

    hpo_module: autogl.module.hpo.BaseHPOptimizer or str or None
        The (name of) hpo module used to search for best hyper parameters.
        Disable hpo by setting it to ``None``. Default ``anneal``.

    ensemble_module: autogl.module.ensemble.BaseEnsembler or str or None
        The (name of) ensemble module used to ensemble the multi-models found.
        Disable ensemble by setting it to ``None``. Default ``voting``.

    max_evals: int (Optional)
        If given, will set the number eval times the hpo module will use.
        Only be effective when hpo_module is ``str``. Default ``None``.

    default_trainer: str (Optional)
        The (name of) the trainer used in this solver. Default to ``NodeClassificationFull``.

    trainer_hp_space: Iterable[dict] (Optional)
        trainer hp space or list of trainer hp spaces configuration.
        If a single trainer hp is given, will specify the hp space of trainer for
        every model. If a list of trainer hp is given, will specify every model
        with corrsponding trainer hp space. Default ``None``.

    model_hp_spaces: list of list of dict (Optional)
        model hp space configuration.
        If given, will specify every hp space of every passed model. Default ``None``.
        If the encoder(-decoder) is passed, the space should be a dict containing keys "encoder"
        and "decoder", specifying the detailed encoder decoder hp spaces.

    size: int (Optional)
        The max models ensemble module will use. Default ``None``.

    device: torch.device or str
        The device where model will be running on. If set to ``auto``, will use gpu
        when available. You can also specify the device by directly giving ``gpu`` or
        ``cuda:0``, etc. Default ``auto``.
    """

    # pylint: disable=W0102

    def __init__(
        self,
        feature_module=None,
        graph_models=("gin", "topkpool"),
        nas_algorithms=None, #
        nas_spaces=None, #
        nas_estimators=None, #
        hpo_module="anneal",
        ensemble_module="voting",
        max_evals=50,
        default_trainer="GraphClassificationFull",
        trainer_hp_space=None,
        model_hp_spaces=None,
        size=4,
        device="auto",
    ):

        super().__init__(
            feature_module=feature_module,
            graph_models=graph_models,
            nas_algorithms=nas_algorithms,#
            nas_spaces=nas_spaces,#
            nas_estimators=nas_estimators,#
            hpo_module=hpo_module,
            ensemble_module=ensemble_module,
            max_evals=max_evals,
            default_trainer=default_trainer,
            trainer_hp_space=trainer_hp_space,
            model_hp_spaces=model_hp_spaces,
            size=size,
            device=device,
        )

        self.dataset = None

    def _init_graph_module(
        self,
        graph_models,
        num_classes,
        num_features,
        feval,
        device,
        loss,
        num_graph_features,
    ) -> "AutoGraphClassifier":
        # load graph network module
        self.graph_model_list = []

        for i, model in enumerate(graph_models):
            # init the trainer
            if not isinstance(model, BaseGraphClassificationTrainer):
                trainer = (
                    self._default_trainer if not isinstance(self._default_trainer, (tuple, list))
                    else self._default_trainer[i]
                )
                if isinstance(trainer, str):
                    if trainer not in TRAINER_DICT:
                        raise KeyError(f"Does not support trainer {trainer}")
                    trainer = TRAINER_DICT[trainer]()
                if isinstance(model, (tuple, list)):
                    trainer.encoder = model[0]
                    trainer.decoder = model[1]
                else:
                    trainer.encoder = model
            else:
                trainer = model

            # set model hp space
            if self._model_hp_spaces is not None:
                if self._model_hp_spaces[i] is not None:
                    if isinstance(self._model_hp_spaces[i], dict):
                        encoder_hp_space = self._model_hp_spaces[i].get('encoder', None)
                        decoder_hp_space = self._model_hp_spaces[i].get('decoder', None)
                    else:
                        encoder_hp_space = self._model_hp_spaces[i]
                        decoder_hp_space = None
                    if encoder_hp_space is not None:
                        trainer.encoder.hyper_parameter_space = encoder_hp_space
                    if decoder_hp_space is not None:
                        trainer.decoder.hyper_parameter_space = decoder_hp_space
            
            # set trainer hp space
            if self._trainer_hp_space is not None:
                if isinstance(self._trainer_hp_space[0], list):
                    current_hp_for_trainer = self._trainer_hp_space[i]
                else:
                    current_hp_for_trainer = self._trainer_hp_space
                trainer.hyper_parameter_space = current_hp_for_trainer

            trainer.num_features = num_features
            trainer.num_classes = num_classes
            trainer.num_graph_features = num_graph_features
            trainer.loss = loss
            trainer.feval = feval
            trainer.to(device)
            self.graph_model_list.append(trainer)

        return self

    def _init_nas_module(self, num_features, num_classes, feval, device, loss):
        for algo, space, estimator in zip(
            self.nas_algorithms, self.nas_spaces, self.nas_estimators
        ):
            estimator: BaseEstimator
            algo.to(device)
            space.instantiate(input_dim=num_features, output_dim=num_classes)
            estimator.setEvaluation(feval)
            estimator.setLossFunction(loss)

    # pylint: disable=arguments-differ
    def fit(
        self,
        dataset,
        time_limit=-1,
        inplace=False,
        train_split=None,
        val_split=None,
        evaluation_method="infer",
        seed=None,
    ) -> "AutoGraphClassifier":
        """
        Fit current solver on given dataset.

        Parameters
        ----------
        dataset: autogl.data.dataset
            The multi-graph dataset needed to fit on.

        time_limit: int
            The time limit of the whole fit process (in seconds). If set below 0, will ignore
            time limit. Default ``-1``.

        inplace: bool
            Whether we process the given dataset in inplace manner. Default ``False``.
            Set it to True if you want to save memory by modifying the given dataset directly.

        train_split: float or int (Optional)
            The train ratio (in ``float``) or number (in ``int``) of dataset. If you want to use
            default train/val/test split in dataset, please set this to ``None``.
            Default ``None``.

        val_split: float or int (Optional)
            The validation ratio (in ``float``) or number (in ``int``) of dataset. If you want to
            use default train/val/test split in dataset, please set this to ``None``.
            Default ``None``.

        evaluation_method: (list of) str autogl.module.train.evaluation
            A (list of) evaluation method for current solver. If ``infer``, will automatically
            determine. Default ``infer``.

        seed: int (Optional)
            The random seed. If set to ``None``, will run everything at random.
            Default ``None``.

        Returns
        -------
        self: autogl.solver.AutoGraphClassifier
            A reference of current solver.
        """

        set_seed(seed)

        num_classes = get_dataset_labels(dataset).max().item() + 1

        if time_limit < 0:
            time_limit = 3600 * 24
        time_begin = time.time()

        # initialize leaderboard
        if evaluation_method == "infer":
            if hasattr(dataset, "metric"):
                evaluation_method = [dataset.metric]
            else:
                if num_classes == 2:
                    evaluation_method = ["auc"]
                else:
                    evaluation_method = ["acc"]
        assert isinstance(evaluation_method, list)
        evaluator_list = get_feval(evaluation_method)

        self.leaderboard = LeaderBoard(
            [e.get_eval_name() for e in evaluator_list],
            {e.get_eval_name(): e.is_higher_better() for e in evaluator_list},
        )

        # set up the dataset
        if train_split is None and val_split is None:
            assert hasattr(dataset, "train_split") and hasattr(dataset, "val_split"), (
                "The dataset has no default train/val split! "
                "Please manually pass train and val ratio."
            )
            LOGGER.info("Use the default train/val/test ratio in given dataset")
            # if hasattr(dataset.train_split, "n_splits"):
            #    cross_validation = True

        elif train_split is not None and val_split is not None:
            utils.graph_random_splits(dataset, train_split, val_split, seed=seed)
        else:
            LOGGER.error(
                "Please set both train_split and val_split explicitly. Detect %s is None.",
                "train_split" if train_split is None else "val_split",
            )
            raise ValueError(
                "In consistent setting of train/val split. Detect {} is None.".format(
                    "train_split" if train_split is None else "val_split"
                )
            )

        # feature engineering
        if self.feature_module is not None:
            self.feature_module.fit(dataset.train_split)
            dataset = self.feature_module.transform(dataset, inplace=inplace)

        self.dataset = dataset
        
        # check whether the dataset has features.
        # currently we only support graph classification with features.
        
        feat = get_graph_node_features(get_graph_from_dataset(dataset))
        assert feat is not None, (
            "Does not support fit on non node-feature dataset!"
            " Please add node features to dataset or specify feature engineers that generate"
            " node features."
        )
        num_features = feat.size(-1)

        # initialize graph networks
        self._init_graph_module(
            self.gml,
            num_features=num_features,
            num_classes=num_classes,
            feval=evaluator_list,
            device=self.runtime_device,
            loss="cross_entropy" if not hasattr(dataset, "loss") else dataset.loss,
            num_graph_features=(0
            if not hasattr(dataset[0], "gf")
            else dataset[0].gf.size(1)) if BACKEND == 'pyg' else 
            (0 if 'gf' not in dataset[0].data else dataset[0].data['gf'].size(1)),
        )

        if self.nas_algorithms is not None:
            # perform neural architecture search
            self._init_nas_module(
                num_features=num_features,
                num_classes=num_classes,
                feval=evaluator_list,
                device=self.runtime_device,
                loss="nll_loss" if not hasattr(dataset, "loss") else dataset.loss,
            )

            assert not isinstance(self._default_trainer, list) or len(
                self.nas_algorithms
            ) == len(self._default_trainer) - len(
                self.graph_model_list
            ), "length of default trainer should match total graph models and nas models passed"

            # perform nas and add them to model list
            idx_trainer = len(self.graph_model_list)
            for algo, space, estimator in zip(
                self.nas_algorithms, self.nas_spaces, self.nas_estimators
            ):
                model = algo.search(space, convert_dataset(self.dataset), estimator)
                # insert model into default trainer
                if isinstance(self._default_trainer, list):
                    train_name = self._default_trainer[idx_trainer]
                    idx_trainer += 1
                else:
                    train_name = self._default_trainer
                if isinstance(train_name, str):
                    trainer = TRAINER_DICT[train_name](
                        model=model,
                        num_features=num_features,
                        num_classes=num_classes,
                        loss="nll_loss"
                        if not hasattr(dataset, "loss")
                        else dataset.loss,
                        feval=evaluator_list,
                        device=self.runtime_device,
                        init=False,
                    )
                elif isinstance(train_name, BaseGraphClassificationTrainer):
                    trainer = train_name
                    trainer.encoder = model
                    trainer.num_features = num_features
                    trainer.num_classes = num_classes
                    trainer.loss = "nll_loss" if not hasattr(dataset, "loss") else dataset.loss
                    trainer.feval = evaluator_list
                    trainer.to(self.runtime_device)
                else:
                    raise ValueError()
                self.graph_model_list.append(trainer)

        # train the models and tune hpo
        result_valid = []
        names = []
        for idx, model in enumerate(self.graph_model_list):
            if time_limit < 0:
                time_for_each_model = None
            else:
                time_for_each_model = (time_limit - time.time() + time_begin) / (
                    len(self.graph_model_list) - idx
                )
            if self.hpo_module is None:
                model.initialize()
                model.train(convert_dataset(dataset), True)
                optimized = model
            else:
                optimized, _ = self.hpo_module.optimize(
                    trainer=model, dataset=convert_dataset(dataset), time_limit=time_for_each_model
                )
            # to save memory, all the trainer derived will be mapped to cpu
            optimized.to(torch.device("cpu"))
            name = str(optimized) + "_idx%d" % (idx)
            names.append(name)
            performance_on_valid, _ = optimized.get_valid_score(return_major=False)
            result_valid.append(
                optimized.get_valid_predict_proba().detach().cpu().numpy()
            )
            self.leaderboard.insert_model_performance(
                name,
                dict(
                    zip(
                        [e.get_eval_name() for e in evaluator_list],
                        performance_on_valid,
                    )
                ),
            )
            self.trained_models[name] = optimized

        # fit the ensemble model
        if self.ensemble_module is not None:
            performance = self.ensemble_module.fit(
                result_valid,
                get_dataset_labels(dataset)[dataset.val_index].cpu().numpy(),
                names,
                evaluator_list,
                n_classes=num_classes,
            )
            self.leaderboard.insert_model_performance(
                "ensemble",
                dict(zip([e.get_eval_name() for e in evaluator_list], performance)),
            )

        return self

    def fit_predict(
        self,
        dataset,
        time_limit=-1,
        inplace=False,
        train_split=None,
        val_split=None,
        evaluation_method="infer",
        seed=None,
        use_ensemble=True,
        use_best=True,
        name=None,
    ) -> np.ndarray:
        """
        Fit current solver on given dataset and return the predicted value.

        Parameters
        ----------
        dataset: torch_geometric.data.dataset.Dataset
            The dataset needed to fit on. This dataset must have only one graph.

        time_limit: int
            The time limit of the whole fit process (in seconds). If set below 0, will
            ignore time limit. Default ``-1``.

        inplace: bool
            Whether we process the given dataset in inplace manner. Default ``False``.
            Set it to True if you want to save memory by modifying the given dataset directly.

        train_split: float or int (Optional)
            The train ratio (in ``float``) or number (in ``int``) of dataset. If you want to
            use default train/val/test split in dataset, please set this to ``None``.
            Default ``None``.

        val_split: float or int (Optional)
            The validation ratio (in ``float``) or number (in ``int``) of dataset. If you want
            to use default train/val/test split in dataset, please set this to ``None``.
            Default ``None``.

        evaluation_method: (list of) str or autogl.module.train.evaluation
            A (list of) evaluation method for current solver. If ``infer``, will automatically
            determine. Default ``infer``.

        seed: int (Optional)
            The random seed. If set to ``None``, will run everything at random.
            Default ``None``.

        use_ensemble: bool
            Whether to use ensemble to do the predict. Default ``True``.

        use_best: bool
            Whether to use the best single model to do the predict. Will only be effective when
            ``use_ensemble`` is ``False``. Default ``True``.

        name: str or None
            The name of model used to predict. Will only be effective when ``use_ensemble`` and
            ``use_best`` both are ``False``. Default ``None``.

        Returns
        -------
        result: np.ndarray
            An array of shape ``(N,)``, where ``N`` is the number of test nodes. The prediction
            on given dataset.
        """
        self.fit(
            dataset=dataset,
            time_limit=time_limit,
            inplace=inplace,
            train_split=train_split,
            val_split=val_split,
            evaluation_method=evaluation_method,
            seed=seed,
        )
        return self.predict(
            dataset=dataset,
            inplaced=inplace,
            inplace=inplace,
            use_ensemble=use_ensemble,
            use_best=use_best,
            name=name,
        )

    def predict_proba(
        self,
        dataset=None,
        inplaced=False,
        inplace=False,
        use_ensemble=True,
        use_best=True,
        name=None,
        mask="test",
    ) -> np.ndarray:
        """
        Predict the node probability.

        Parameters
        ----------
        dataset: autogl.data.Dataset or None
            The dataset needed to predict. If ``None``, will use the processed dataset
            passed to ``fit()`` instead. Default ``None``.

        inplaced: bool
            Whether the given dataset is processed. Only be effective when ``dataset``
            is not ``None``. If you pass the dataset to ``fit()`` with ``inplace=True``,
            and you pass the dataset again to this method, you should set this argument
            to ``True``. Otherwise ``False``. Default ``False``.

        inplace: bool
            Whether we process the given dataset in inplace manner. Default ``False``.
            Set it to True if you want to save memory by modifying the given dataset directly.

        use_ensemble: bool
            Whether to use ensemble to do the predict. Default ``True``.

        use_best: bool
            Whether to use the best single model to do the predict. Will only be effective when
            ``use_ensemble`` is ``False``. Default ``True``.

        name: str or None
            The name of model used to predict. Will only be effective when ``use_ensemble`` and
            ``use_best`` both are ``False``. Default ``None``.

        mask: str
            The data split to give prediction on. Default ``test``.

        Returns
        -------
        result: np.ndarray
            An array of shape ``(N,C,)``, where ``N`` is the number of test nodes and ``C`` is
            the number of classes. The prediction on given dataset.
        """
        if dataset is None:
            dataset = self.dataset
        elif not inplaced:
            if self.feature_module is not None:
                dataset = self.feature_module.transform(dataset, inplace=inplace)

        if use_ensemble:
            LOGGER.info("Ensemble argument on, will try using ensemble model.")

        if not use_ensemble and use_best:
            LOGGER.info(
                "Ensemble argument off and best argument on, will try using best model."
            )

        if (use_ensemble and self.ensemble_module is not None) or (
            not use_best and name == "ensemble"
        ):
            # we need to get all the prediction of every model trained
            predict_result = []
            names = []
            for model_name in self.trained_models:
                predict_result.append(
                    self._predict_proba_by_name(dataset, model_name, mask)
                )
                names.append(model_name)
            return self.ensemble_module.ensemble(predict_result, names)

        if use_ensemble and self.ensemble_module is None:
            LOGGER.warning(
                "Cannot use ensemble because no ensebmle module is given. "
                "Will use best model instead."
            )

        if use_best or (use_ensemble and self.ensemble_module is None):
            # just return the best model we have found
            best_model_name = self.leaderboard.get_best_model()
            return self._predict_proba_by_name(dataset, best_model_name, mask)

        if name is not None:
            # return model performance by name
            return self._predict_proba_by_name(dataset, name, mask)

        LOGGER.error(
            "No model name is given while ensemble and best arguments are off."
        )
        raise ValueError(
            "You need to specify a model name if you do not want use ensemble and best model."
        )

    def _predict_proba_by_name(self, dataset, name, mask):
        self.trained_models[name].to(self.runtime_device)
        predicted = (
            self.trained_models[name]
            .predict_proba(convert_dataset(dataset), mask=mask)
            .detach()
            .cpu()
            .numpy()
        )
        self.trained_models[name].to(torch.device("cpu"))
        return predicted

    def predict(
        self,
        dataset=None,
        inplaced=False,
        inplace=False,
        use_ensemble=True,
        use_best=True,
        name=None,
        mask="test",
    ) -> np.ndarray:
        """
        Predict the node class number.

        Parameters
        ----------
        dataset: autogl.data.Dataset or None
            The dataset needed to predict. If ``None``, will use the processed dataset passed
            to ``fit()`` instead. Default ``None``.

        inplaced: bool
            Whether the given dataset is processed. Only be effective when ``dataset``
            is not ``None``. If you pass the dataset to ``fit()`` with ``inplace=True``, and
            you pass the dataset again to this method, you should set this argument to ``True``.
            Otherwise ``False``. Default ``False``.

        inplace: bool
            Whether we process the given dataset in inplace manner. Default ``False``.
            Set it to True if you want to save memory by modifying the given dataset directly.

        use_ensemble: bool
            Whether to use ensemble to do the predict. Default ``True``.

        use_best: bool
            Whether to use the best single model to do the predict. Will only be effective
            when ``use_ensemble`` is ``False``. Default ``True``.

        name: str or None
            The name of model used to predict. Will only be effective when ``use_ensemble``
            and ``use_best`` both are ``False``. Default ``None``.

        Returns
        -------
        result: np.ndarray
            An array of shape ``(N,)``, where ``N`` is the number of test nodes.
            The prediction on given dataset.
        """
        proba = self.predict_proba(
            dataset, inplaced, inplace, use_ensemble, use_best, name, mask
        )
        return np.argmax(proba, axis=1)

    def evaluate(self, dataset=None,
        inplaced=False,
        inplace=False,
        use_ensemble=True,
        use_best=True,
        name=None,
        mask="test",
        label=None,
        metric="acc"
    ):
        """
        Evaluate the given dataset.


        Parameters
        ----------
        dataset: torch_geometric.data.dataset.Dataset or None
            The dataset needed to predict. If ``None``, will use the processed dataset passed
            to ``fit()`` instead. Default ``None``.

        inplaced: bool
            Whether the given dataset is processed. Only be effective when ``dataset``
            is not ``None``. If you pass the dataset to ``fit()`` with ``inplace=True``, and
            you pass the dataset again to this method, you should set this argument to ``True``.
            Otherwise ``False``. Default ``False``.

        inplace: bool
            Whether we process the given dataset in inplace manner. Default ``False``. Set it to
            True if you want to save memory by modifying the given dataset directly.

        use_ensemble: bool
            Whether to use ensemble to do the predict. Default ``True``.

        use_best: bool
            Whether to use the best single model to do the predict. Will only be effective when
            ``use_ensemble`` is ``False``. Default ``True``.

        name: str or None
            The name of model used to predict. Will only be effective when ``use_ensemble`` and
            ``use_best`` both are ``False``. Default ``None``.

        mask: str
            The data split to give prediction on. Default ``test``.

        label: torch.Tensor (Optional)
            The groud truth label of the given predicted dataset split. If not passed, will extract
            labels from the input dataset.
        
        metric: str
            The metric to be used for evaluating the model. Default ``acc``.

        Returns
        -------
        score(s): (list of) evaluation scores
            the evaluation results according to the evaluator passed.

        """
        predicted = self.predict_proba(dataset, inplaced, inplace, use_ensemble, use_best, name, mask)
        if dataset is None:
            dataset = self.dataset
        if label is None:
            if mask == "all":
                masked_dataset = dataset
            else:
                masked_dataset = utils.graph_get_split(dataset, mask, False)
            label = get_dataset_labels(masked_dataset)
        evaluator = get_feval(metric)
        if isinstance(evaluator, Sequence):
            return [evals.evaluate(predicted, label) for evals in evaluator]
        return evaluator.evaluate(predicted, label)

    @classmethod
    def from_config(cls, path_or_dict, filetype="auto") -> "AutoGraphClassifier":
        """
        Load solver from config file.

        You can use this function to directly load a solver from predefined config dict
        or config file path. Currently, only support file type of ``json`` or ``yaml``,
        if you pass a path.

        Parameters
        ----------
        path_or_dict: str or dict
            The path to the config file or the config dictionary object

        filetype: str
            The filetype the given file if the path is specified. Currently only support
            ``json`` or ``yaml``. You can set to ``auto`` to automatically detect the file
            type (from file name). Default ``auto``.

        Returns
        -------
        solver: autogl.solver.AutoGraphClassifier
            The solver that is created from given file or dictionary.
        """
        assert filetype in ["auto", "yaml", "json"], (
            "currently only support yaml file or json file type, but get type "
            + filetype
        )
        if isinstance(path_or_dict, str):
            if filetype == "auto":
                if path_or_dict.endswith(".yaml") or path_or_dict.endswith(".yml"):
                    filetype = "yaml"
                elif path_or_dict.endswith(".json"):
                    filetype = "json"
                else:
                    LOGGER.error(
                        "cannot parse the type of the given file name, "
                        "please manually set the file type"
                    )
                    raise ValueError(
                        "cannot parse the type of the given file name, "
                        "please manually set the file type"
                    )
            if filetype == "yaml":
                path_or_dict = yaml.load(
                    open(path_or_dict, "r").read(), Loader=yaml.FullLoader
                )
            else:
                path_or_dict = json.load(open(path_or_dict, "r"))

        # load the dictionary
        path_or_dict = deepcopy(path_or_dict)
        solver = cls(None, [], None, None)
        fe_list = path_or_dict.pop("feature", None)
        if fe_list is not None:
            fe_list_ele = []
            for feature_engineer in fe_list:
                name = feature_engineer.pop("name")
                if name is not None:
                    fe_list_ele.append(FEATURE_DICT[name](**feature_engineer))
            if fe_list_ele != []:
                solver.set_feature_module(fe_list_ele)

        models = path_or_dict.pop("models", [{"name": "gcn"}, {"name": "gat"}, {"name": "sage"}, {"name": "gin"}])
        # models should be a list of model
        # with each element in two cases
        # * a dict describing a certain model
        # * a dict containing {"encoder": encoder, "decoder": decoder}
        model_hp_space = [
            _parse_model_hp(model) for model in models
        ]
        model_list = [
            _initialize_single_model(model) for model in models
        ]

        trainer = path_or_dict.pop("trainer", None)
        default_trainer = "GraphClassificationFull"
        trainer_space = None
        if isinstance(trainer, dict):
            # global default
            default_trainer = trainer.pop("name", "GraphClassificationFull")
            trainer_space = _parse_hp_space(trainer.pop("hp_space", None))
            default_kwargs = {"num_features": None, "num_classes": None}
            default_kwargs.update(trainer)
            default_kwargs["init"] = False
            for i in range(len(model_list)):
                model = model_list[i]
                trainer_wrapper = TRAINER_DICT[default_trainer](
                    model=model, **default_kwargs
                )
                model_list[i] = trainer_wrapper
        elif isinstance(trainer, list):
            # sequential trainer definition
            assert len(trainer) == len(
                model_list
            ), "The number of trainer and model does not match"
            trainer_space = []
            for i in range(len(model_list)):
                train, model = trainer[i], model_list[i]
                default_trainer = train.pop("name", "GraphClassificationFull")
                trainer_space.append(_parse_hp_space(train.pop("hp_space", None)))
                default_kwargs = {"num_features": None, "num_classes": None}
                default_kwargs.update(train)
                default_kwargs["init"] = False
                trainer_wrap = TRAINER_DICT[default_trainer](
                    model=model, **default_kwargs
                )
                model_list[i] = trainer_wrap

        solver.set_graph_models(
            model_list, default_trainer, trainer_space, model_hp_space
        )

        hpo_dict = path_or_dict.pop("hpo", {"name": "anneal"})
        if hpo_dict is not None:
            name = hpo_dict.pop("name")
            solver.set_hpo_module(name, **hpo_dict)

        ensemble_dict = path_or_dict.pop("ensemble", {"name": "voting"})
        if ensemble_dict is not None:
            name = ensemble_dict.pop("name")
            solver.set_ensemble_module(name, **ensemble_dict)

        return solver
