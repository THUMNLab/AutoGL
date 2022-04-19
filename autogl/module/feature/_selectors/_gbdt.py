import numpy as np
import pandas as pd
import torch
import typing as _typing
import autogl
from autogl.data.graph import GeneralStaticGraph
from .. import _feature_engineer_registry
import lightgbm
from sklearn.model_selection import train_test_split
from ._basic import BaseFeatureSelector


def _gbdt_generator(
        data: autogl.data.Data, fixlen: int = 1000,
        params: _typing.Mapping[str, _typing.Any] = ...,
        is_val: bool = True, train_val_ratio: float = 0.2,
        **optimizer_parameters
) -> _typing.Optional[np.ndarray]:
    parameters: _typing.Dict[str, _typing.Any] = (
        dict(params)
        if (
                params not in (Ellipsis, None) and
                isinstance(params, _typing.Mapping)
        )
        else {
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 47,
            "objective": "multiclass",
            "metric": ["multi_logloss"],
            "max_bin": 63,
            "save_binary": True,
            "num_threads": 20,
            "num_leaves": 16,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            # 'is_training_metric': True,
            # 'metric_freq': 1,
        }
    )

    num_classes: int = torch.max(data.y).item() + 1
    parameters["num_class"] = num_classes
    __optimizer_parameters = {
        "num_boost_round": 100,
        "early_stopping_rounds": 5,
        "verbose_eval": False
    }
    __optimizer_parameters.update(optimizer_parameters)
    if hasattr(data, "train_mask") and data.train_mask is not None and (
            isinstance(data.train_mask, np.ndarray) or
            isinstance(data.train_mask, torch.Tensor)
    ):
        x: np.ndarray = data.x[data.train_mask].numpy()
        label: np.ndarray = data.y[data.train_mask].numpy()
    else:
        x: np.ndarray = data.x.numpy()
        label: np.ndarray = data.y.numpy()
        is_val: bool = False
    _, num_features = x.shape
    if num_features < fixlen:
        return None

    feature_index: np.ndarray = np.array(
        [f"f{i}" for i in range(num_features)]
    )
    if is_val:
        x_train, x_val, y_train, y_val = train_test_split(
            x, label, test_size=train_val_ratio, stratify=label, random_state=47
        )
        dtrain = lightgbm.Dataset(x_train, label=y_train)
        dval = lightgbm.Dataset(x_val, label=y_val)
        clf = lightgbm.train(
            train_set=dtrain, params=parameters, valid_sets=dval,
            **__optimizer_parameters
        )
    else:
        train_x = pd.DataFrame(x, columns=feature_index, index=None)
        dtrain = lightgbm.Dataset(train_x, label=label)
        clf = lightgbm.train(
            train_set=dtrain, params=parameters,
            **__optimizer_parameters
        )

    imp = np.array(list(clf.feature_importance()))
    return np.argsort(imp)[-fixlen:]


@_feature_engineer_registry.FeatureEngineerUniversalRegistry.register_feature_engineer("gbdt")
class GBDTFeatureSelector(BaseFeatureSelector):
    r"""simple wrapper of lightgbm , using importance ranking to select top-k features.

    Parameters
    ----------
    fixlen : int
        K for top-K important features.
    """

    def __init__(self, fixlen: int = 10, *args, **kwargs):
        super(GBDTFeatureSelector, self).__init__()
        self.__fixlen = fixlen
        self.__args = args
        self.__kwargs = kwargs

    def _fit(self, homogeneous_static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        if not isinstance(homogeneous_static_graph, GeneralStaticGraph):
            raise TypeError
        elif not (
            homogeneous_static_graph.nodes.is_homogeneous and
            homogeneous_static_graph.edges.is_homogeneous
        ):
            raise ValueError
        if 'x' in homogeneous_static_graph.nodes.data:
            features: torch.Tensor = homogeneous_static_graph.nodes.data['x']
        elif 'feat' in homogeneous_static_graph.nodes.data:
            features: torch.Tensor = homogeneous_static_graph.nodes.data['feat']
        else:
            raise ValueError("Node features not exists")
        if 'y' in homogeneous_static_graph.nodes.data:
            label: torch.Tensor = homogeneous_static_graph.nodes.data['y']
        elif 'label' in homogeneous_static_graph.nodes.data:
            label: torch.Tensor = homogeneous_static_graph.nodes.data['label']
        else:
            raise ValueError("Node label not exists")
        if 'train_mask' in homogeneous_static_graph.nodes.data:
            train_mask: _typing.Optional[torch.Tensor] = (
                homogeneous_static_graph.nodes.data['train_mask']
            )
        else:
            train_mask: _typing.Optional[torch.Tensor] = None
        data = autogl.data.Data(
            edge_index=homogeneous_static_graph.edges.connections,
            x=features, y=label
        )
        data.train_mask = train_mask
        self._selection = _gbdt_generator(
            data, self.__fixlen, *self.__args, **self.__kwargs
        )
        return homogeneous_static_graph
