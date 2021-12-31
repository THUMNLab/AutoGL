from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import copy
from tqdm import tqdm
from tabulate import tabulate
import time

from .base import BaseFeature, BaseFeatureEngineer
from .selectors import SeGBDT
from . import register_feature

from ...utils import get_logger
import torch

LOGGER = get_logger("Feature")


@register_feature("identity")
class FeIdentity(BaseFeatureEngineer):
    r"""it is a dummy feature engineer , which directly returns identical data"""

    def __init__(self, *args, **kwargs):
        super(FeIdentity, self).__init__(multigraph=True, *args, **kwargs)


@register_feature("onlyconst")
class Onlyconst(BaseFeatureEngineer):
    r"""it is a dummy feature engineer , which directly returns identical data"""

    def __init__(self, *args, **kwargs):
        super(Onlyconst, self).__init__(
            data_t="tensor", multigraph=True, *args, **kwargs
        )

    def _transform(self, data):
        if "x" in data:
            data.x = torch.ones((data.x.shape[0], 1))
        else:
            data.x = torch.ones((torch.unique(data.edge_index).shape[0], 1))
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


@register_feature("deepgl")
class AutoFeatureEngineer(BaseFeatureEngineer):
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
    fixlen : int
        fixed number of features for every epoch. The final number of features added will be
        ``fixlen`` \times ``max_epoch``, 200 \times 5 in default.
    max_epoch : int
        number of epochs in total process.
    timebudget : int
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
        fixlen=200,
        max_epoch=5,
        timebudget=None,
        y_sel_func=SeGBDT,
        verbosity=-1,
        *args,
        **kwargs
    ):

        super(AutoFeatureEngineer, self).__init__(multigraph=False, *args, **kwargs)
        self._ops = [op_sum, op_mean, op_max, op_min]
        self._sim = cos_sim
        self._fixlen = fixlen
        self._max_epoch = max_epoch
        self._timebudget = timebudget
        self._y_sel_func = y_sel_func(
            fixlen, verbose_eval=verbosity >= 1, *args, **kwargs
        )
        self._verbosity = verbosity

    def _init(self, data):
        #         self._data=copy.deepcopy(data)
        self._data = data
        self._num_nodes = data.x.shape[0]
        self._x = data.x
        self._edges = data.edge_index
        self._neighbours = [[] for _ in range(self._num_nodes)]
        for u, v in self._edges.T:
            self._neighbours[u].append(v)
        self._neighbours = [np.array(v) for v in self._neighbours]

    def _gen(self, x):
        res = []
        for i, op in enumerate(self._ops):
            res.append(op(x, self._neighbours))
        res = np.concatenate(res, axis=1)
        return res

    def _fit(self, data):
        self._init(data)
        x = self._x.copy()
        gx = x.copy()
        verbs = []
        data = self._data
        max_epoch = self._max_epoch
        timebudget = self._timebudget
        y_sel_func = self._y_sel_func
        soft_timer = Timer(timebudget)
        self._sel = []
        for epoch in tqdm(range(max_epoch), disable=self._verbosity <= 0):
            soft_timer.start()
            verb = [epoch, gx.shape[1]]
            gx = self._gen(gx)
            gx = scale(gx)
            verb.append(gx.shape[1])
            data.x = gx
            #             data = feat_diffuse(data)
            y_sel_func._fit(data)
            self._sel.append(y_sel_func._sel)
            data = y_sel_func._transform(data)
            gx = data.x
            verb.append(gx.shape[1])
            x = np.concatenate([x, gx], axis=1)
            verbs.append(verb)
            soft_timer.end()
            if soft_timer.is_timeout():
                break
        if self._verbosity >= 1:
            LOGGER.info(
                tabulate(verbs, headers="epoch origin after-gen after-sel".split())
            )
        data.x = x
        return data

    def _transform(self, data):
        x = data.x
        gx = x.copy()
        for sel in self._sel:
            gx = self._gen(gx)
            gx = scale(gx)
            gx = gx[:, sel]
            x = np.concatenate([x, gx], axis=1)
        data.x = x
        return data
