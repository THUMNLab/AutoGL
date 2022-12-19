import time
import numpy as np
import torch
import typing as _typing
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import tabulate
import autogl.data.graph
from ._feature_engineer import FeatureEngineer
from .._data_preprocessor_registry import DataPreprocessorUniversalRegistry
from ._selectors import GBDTFeatureSelector
from ....utils import get_logger

LOGGER = get_logger("Feature")


@DataPreprocessorUniversalRegistry.register_data_preprocessor("identity")
class IdentityFeature(FeatureEngineer):
    ...


@DataPreprocessorUniversalRegistry.register_data_preprocessor("OnlyConst".lower())
class OnlyConstFeature(FeatureEngineer):
    def _transform(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        if isinstance(data, autogl.data.graph.GeneralStaticGraph):
            for node_t in data.nodes:
                for candidate_feature_key in ('feat', 'x'):
                    if candidate_feature_key in data.nodes[node_t].data:
                        data.nodes[node_t].data[candidate_feature_key] = torch.ones(
                            (data.nodes[node_t].data[candidate_feature_key].size(0), 1)
                        ).to(data.nodes[node_t].data[candidate_feature_key])
                    elif len(data.nodes[node_t].data) > 0:
                        _ref = data.nodes[node_t].data[list(data.nodes[node_t].data)[0]]
                        data.nodes[node_t].data[candidate_feature_key] = (
                            torch.ones((_ref.size(0), 1)).to(_ref)
                        )
                    else:
                        data.nodes[node_t].data[candidate_feature_key] = torch.ones(
                            (torch.unique(data.edges.connections).size(0), 1)
                        )
        elif hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
            data.x = torch.ones((data.x.shape[0], 1)).to(data.x)
        elif hasattr(data, 'edge_index') and isinstance(data.edge_index, torch.Tensor):
            data.x = torch.ones((torch.unique(data.edge_index).size(0), 1)).to(data.edge_index)
        else:
            raise ValueError("Unsupported provided data")
        return data


def op_sum(x, nbs):
    res = np.zeros_like(x)
    for u in range(len(nbs)):
        nb = nbs[u]
        if len(nb != 0):
            res[u] = np.sum(x[nb], axis=0)
    return res


def op_mean(x, nbs):
    res = np.zeros_like(x)
    for u in range(len(nbs)):
        nb = nbs[u]
        if len(nb != 0):
            res[u] = np.mean(x[nb], axis=0)
    return res


def op_max(x, nbs):
    res = np.zeros_like(x)
    for u in range(len(nbs)):
        nb = nbs[u]
        if len(nb != 0):
            res[u] = np.max(x[nb], axis=0)
    return res


def op_min(x, nbs):
    res = np.zeros_like(x)
    for u in range(len(nbs)):
        nb = nbs[u]
        if len(nb != 0):
            res[u] = np.min(x[nb], axis=0)
    return res


def op_prod(x, nbs):
    res = np.zeros_like(x)
    for u in range(len(nbs)):
        nb = nbs[u]
        if len(nb != 0):
            res[u] = np.prod(x[nb], axis=0)
    return res


mms = preprocessing.MinMaxScaler()
ss = preprocessing.StandardScaler()


def scale(x):
    return ss.fit_transform(x)


class Timer:
    def __init__(self, timebudget=None):
        self._timebudget = timebudget
        self._esti_time = 0
        self._g_start = time.time()

    def start(self):
        self._start = time.time()

    def end(self):
        time_use = time.time() - self._start
        self._esti_time = (self._esti_time + time_use) / 2

    def is_timeout(self):
        timebudget = self._timebudget
        if timebudget:
            timebudget = self._timebudget - (time.time() - self._g_start)
            if timebudget < self._esti_time:
                return True
        return False


@DataPreprocessorUniversalRegistry.register_data_preprocessor('DeepGL'.lower())
class AutoFeatureEngineer(FeatureEngineer):
    r"""
    Notes
    -----
    An implementation of auto feature engineering method Deepgl [#]_ ,which iteratively generates features by aggregating neighbour features
    and select a fixed number of  features to automatically add important graph-aware features.
    References
    ----------
    .. [#] Rossi, R. A., Zhou, R., & Ahmed, N. K. (2020).
        Deep Inductive Graph Representation Learning.
        IEEE Transactions on Knowledge and Data Engineering, 32(3), 438–452.
        https://doi.org/10.1109/TKDE.2018.2878247
    Parameters
    ----------
    fix_length : int
        fixed number of features for every epoch. The final number of features added will be
        ``fixlen`` \times ``max_epoch``, 200 \times 5 in default.
    max_epoch : int
        number of epochs in total process.
    time_budget : int
        timebudget(seconds) for the feature engineering process, None for no time budget . Note that
        this time budget is a soft budget ,which is obtained by rough time estimation through previous iterations and
        may finally exceed the actual timebudget
    y_sel_func : Callable
        feature selector function object for selection at each iteration ,lightgbm in default. Note that in original paper,
        connected components of feature graph is used , and you may implement it by yourself if you want.
    verbosity : int
        hide any infomation except error and fatal if ``verbosity`` < 1
    """

    def __init__(
            self,
            fix_length: int = 200,
            max_epoch: int = 5,
            time_budget: _typing.Optional[int] = None,
            feature_selector=GBDTFeatureSelector,
            verbosity: int = 0,
            *args, **kwargs
    ):
        super(AutoFeatureEngineer, self).__init__()
        self._ops = [op_sum, op_mean, op_max, op_min]
        self._sim = cosine_similarity
        self._fixlen = fix_length
        self._max_epoch = max_epoch
        self._timebudget = time_budget
        self._feature_selector = feature_selector(
            fix_length, verbose_eval=verbosity >= 1, *args, **kwargs
        )
        self._verbosity = verbosity

    def _gen(self, x) -> np.ndarray:
        res = []
        for i, op in enumerate(self._ops):
            res.append(op(x, self.__neighbours))
        res = np.concatenate(res, axis=1)
        return res

    def _fit(self, homogeneous_static_graph: autogl.data.graph.GeneralStaticGraph):
        if not (
                homogeneous_static_graph.nodes.is_homogeneous and
                homogeneous_static_graph.edges.is_homogeneous
        ):
            raise ValueError
        if 'x' in homogeneous_static_graph.nodes.data:
            _feature_key = 'x'
            _original_features: torch.Tensor = (
                homogeneous_static_graph.nodes.data['x']
            )
        elif 'feat' in homogeneous_static_graph.nodes.data:
            _feature_key = 'feat'
            _original_features: torch.Tensor = (
                homogeneous_static_graph.nodes.data['feat']
            )
        else:
            raise ValueError

        num_nodes = _original_features.size(0)
        neighbours = [[] for _ in range(num_nodes)]
        for u, v in homogeneous_static_graph.edges.connections.t().numpy():
            neighbours[u].append(v)
        self.__neighbours: _typing.Sequence[np.ndarray] = tuple(
            [np.array(v) for v in neighbours]
        )

        x: np.ndarray = _original_features.numpy()
        gx: np.ndarray = x.copy()
        verbs = []
        soft_timer = Timer(self._timebudget)
        self._selection = []
        for epoch in tqdm.tqdm(range(self._max_epoch), disable=self._verbosity <= 0):
            soft_timer.start()
            verb = [epoch, gx.shape[1]]
            gx = self._gen(gx)
            gx = scale(gx)
            verb.append(gx.shape[1])

            homogeneous_static_graph.nodes.data[_feature_key] = torch.from_numpy(gx)
            self._feature_selector._fit(homogeneous_static_graph)
            self._selection.append(self._feature_selector._selection)
            homogeneous_static_graph = self._feature_selector._transform(
                homogeneous_static_graph
            )

            gx: np.ndarray = homogeneous_static_graph.nodes.data[_feature_key].numpy()
            verb.append(gx.shape[1])
            x = np.concatenate([x, gx], axis=1)
            verbs.append(verb)
            soft_timer.end()
            if soft_timer.is_timeout():
                break
        if self._verbosity >= 1:
            LOGGER.info(
                tabulate.tabulate(verbs, headers="epoch origin after-gen after-sel".split())
            )
        homogeneous_static_graph.nodes.data[_feature_key] = torch.from_numpy(x)
        return homogeneous_static_graph

    def _transform(self, homogeneous_static_graph: autogl.data.graph.GeneralStaticGraph):
        if not (
                homogeneous_static_graph.nodes.is_homogeneous and
                homogeneous_static_graph.edges.is_homogeneous
        ):
            raise ValueError
        if 'x' in homogeneous_static_graph.nodes.data:
            _feature_key = 'x'
            _original_features: torch.Tensor = (
                homogeneous_static_graph.nodes.data['x']
            )
        elif 'feat' in homogeneous_static_graph.nodes.data:
            _feature_key = 'feat'
            _original_features: torch.Tensor = (
                homogeneous_static_graph.nodes.data['feat']
            )
        else:
            raise ValueError

        x: np.ndarray = _original_features.numpy()
        gx: np.ndarray = x.copy()
        for selection in self._selection:
            gx = scale(self._gen(gx))[:, selection]
            x = np.concatenate([x, gx], axis=1)
        homogeneous_static_graph.nodes.data[_feature_key] = torch.from_numpy(x)
        return homogeneous_static_graph
