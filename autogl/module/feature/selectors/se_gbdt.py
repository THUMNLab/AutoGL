from .. import register_feature
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import psutil
from .base import BaseSelector
import numpy as np
import copy
import pandas as pd


def gbdt_gen(
    data,
    fixlen=1000,
    params={
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
        #         'is_training_metric': True,
        #         'metric_freq': 1,
    },
    is_val=True,
    train_val_ratio=0.2,
    **param_o
):

    n_class = np.max(data.y) + 1
    params["num_class"] = n_class
    #     cpu_count = psutil.cpu_count()
    #     params['num_threads'] = cpu_count if data.train_ind.shape[0] > 10000 else min(32, cpu_count)

    param = {"num_boost_round": 100, "early_stopping_rounds": 5, "verbose_eval": False}
    param.update(param_o)
    if hasattr(data, "train_mask"):
        x = data.x[data.train_mask]
        label = data.y[data.train_mask]
    else:
        x = data.x
        label = data.y
        is_val = False
    _, num_feas = x.shape
    if num_feas < fixlen:
        return None

    fnames = np.array(["f{}".format(i) for i in range(num_feas)])

    if is_val:
        x_train, x_val, y_train, y_val = train_test_split(
            x, label, test_size=train_val_ratio, stratify=label, random_state=47
        )
        dtrain = lgb.Dataset(x_train, label=y_train)
        dval = lgb.Dataset(x_val, label=y_val)
        clf = lgb.train(train_set=dtrain, params=params, valid_sets=dval, **param)
    else:
        train_x = pd.DataFrame(x, columns=fnames, index=None)
        dtrain = lgb.Dataset(train_x, label=label)
        clf = lgb.train(train_set=dtrain, params=params, **param)

    imp = np.array(list(clf.feature_importance()))
    return np.argsort(imp)[-fixlen:]


@register_feature("gbdt")
class SeGBDT(BaseSelector):
    r"""simple wrapper of lightgbm , using importance ranking to select top-k features.

    Parameters
    ----------
    fixlen : int
        K for top-K important features.
    """

    def __init__(self, fixlen=10, *args, **kwargs):
        super(SeGBDT, self).__init__()
        self.fixlen = fixlen
        self.args = args
        self.kwargs = kwargs

    def _fit(self, data):
        self._sel = gbdt_gen(data, self.fixlen, *self.args, **self.kwargs)
