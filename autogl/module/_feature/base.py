from operator import xor
from .utils import data_is_tensor, data_tensor2np, data_np2tensor
import numpy as np
import copy
from torch_geometric.utils.convert import to_networkx
from torch.utils.data import Dataset
import torch
from ...utils import get_logger

LOGGER = get_logger("Feature")


class BaseFeature:
    r"""Any feature funcion object should inherit BaseFeature,
    which provides basic transformations and composing operation for feature
    engineering. Basic transformations include data type adjusting(tensor or numpy),
    complementing necessary attributes for future transform. Any subclass needs
    to overload methods ``_func`` and ``_transform`` to implement feature transformation.
    For specific needs, you may want to overload methods ``_preprocess`` and ``_postprocess``
    to enable specific processing before and after ``_transform`` .

    Parameters
    ----------
    pipe : list
        stores pipeline of ``BaseFeature``.
    data_t: str
        represents the data type needed for this transform, where 'tensor' accounts for ``torch.Tensor``,
        'np' for ``numpy.array`` and 'nx' for ``networkx``. When ``data_t`` values 'nx', then a ``networkx.DiGraph`` will
        be added to data as data.G .
    multigraph : bool
        determine whether it supports dataset with multiple graphs
    subgraph : bool
        determine whether it extracts subgraph features.
    """

    def __init__(self, pipe=None, data_t="tensor", multigraph=True, subgraph=False):
        r""""""
        if pipe is None:
            pipe = [self]
        self._pipe = pipe
        self._data_t = data_t
        self._multigraph = multigraph
        self._subgraph = subgraph

    def __and__(self, o):
        r"""enable and operation to support feature engineering pipeline syntax like
        SeFilterConstant()&GeEigen()&...
        """
        return BaseFeature(self._pipe + o._pipe)

    def _rebuild(self, dataset, datalist):
        dataset.__indices__ = None
        data, slices = dataset.collate(datalist)
        dataset.data.__dict__.update(data.__dict__)
        dataset.slices.update(slices)
        return dataset

    def _adjust_t(self, data):
        r"""adjust data type for current transform."""
        if self._data_t == "tensor":
            data_np2tensor(data)
        elif self._data_t == "np":
            data_tensor2np(data)
        elif self._data_t == "nx":
            if not hasattr(data, "G") or data.G is None:
                data.G = to_networkx(data, to_undirected=True)

    def _adjust_to_tensor(self, data):
        if self._data_t == "tensor":
            pass
        else:
            data_np2tensor(data)

    def _preprocess(self, data):
        pass

    def _postprocess(self, data):
        pass

    def _check_dataset(self, dataset):
        if len(dataset) > 1:
            for p in self._pipe:
                if not p._multigraph:
                    LOGGER.warn(p.__class__.__name__, " does not support multigraph")
                    return False
        return True

    def _fit(self, data):
        pass

    def _transform(self, data):
        return data

    def _fit_transform(self, data):
        self._fit(data)
        return self._transform(data)

    def fit(self, dataset):
        r"""fit dataset"""
        if not self._check_dataset(dataset):
            return
        dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for p in self._pipe:
                _dataset = [x for x in dataset]
                for i, datai in enumerate(_dataset):
                    p._adjust_t(datai)
                    p._preprocess(datai)
                    p._fit_transform(datai)
                    p._postprocess(datai)
                    p._adjust_to_tensor(datai)
                    _dataset[i] = datai
                dataset = self._rebuild(dataset, _dataset)

    def transform(self, dataset, inplace=True):
        r"""transform dataset inplace or not w.r.t bool argument ``inplace``"""
        if not self._check_dataset(dataset):
            return dataset
        if not inplace:
            dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for p in self._pipe:
                self._dataset = _dataset = [x for x in dataset]
                for i, datai in enumerate(_dataset):
                    p._adjust_t(datai)
                    p._preprocess(datai)
                    datai = p._transform(datai)
                    p._postprocess(datai)
                    p._adjust_to_tensor(datai)
                    _dataset[i] = datai
                dataset = self._rebuild(dataset, _dataset)
        dataset.data = data_np2tensor(dataset.data)
        return dataset

    def fit_transform(self, dataset, inplace=True):
        r"""fit and transform dataset inplace or not w.r.t bool argument ``inplace``"""
        self.fit(dataset)
        return self.transform(dataset, inplace=inplace)

    @staticmethod
    def compose(trans_list):
        r"""put a list of ``BaseFeature`` into feature engineering pipeline"""
        res = BaseFeature()
        for tran in trans_list:
            res = res & tran
        return res


class BaseFeatureEngineer(BaseFeature):
    def __init__(self, data_t="np", multigraph=False, *args, **kwargs):
        super(BaseFeatureEngineer, self).__init__(
            data_t=data_t, multigraph=multigraph, *args, **kwargs
        )
        self.args = args
        self.kwargs = kwargs


class TransformWrapper(BaseFeature):
    def __init__(self, cls, *args, **kwargs):
        super(TransformWrapper, self).__init__(data_t="tensor", *args, **kwargs)
        self._cls = cls
        self._func = None
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        if self._func is None:
            self._func = self._cls(*self._args, **self._kwargs)
            return self

    def _transform(self, data=None):
        return self._func(data)
