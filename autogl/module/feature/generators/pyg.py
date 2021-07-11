from torch_geometric.transforms.one_hot_degree import OneHotDegree
from torch_geometric.utils import degree
from torch_geometric.transforms.normalize_features import NormalizeFeatures
from torch_geometric.transforms.local_degree_profile import LocalDegreeProfile
from .. import register_feature
from .base import BaseGenerator
import numpy as np
from .. import register_feature
import torch
from functools import wraps

PYG_GENERATORS = []


def register_pyg(cls):
    PYG_GENERATORS.append(cls)
    register_feature(cls.__name__)(cls)
    return cls


class PYGGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super(PYGGenerator, self).__init__(data_t="tensor")
        self._args = args
        self._kwargs = kwargs

    def extract(self, data):
        return data.x

    def _transform(self, data):
        dsc = self.extract(data)
        # data.x = torch.cat([data.x, dsc], dim=1)
        data.x = dsc
        return data


def pygfunc(func):
    r"""A decorator for pyg transforms. You may want to use it to quickly wrap a feature transform function object.
    Examples
    --------
    @register_pyg
    @pygfunc(local_degree_profile)
    class PYGLocalDegreeProfile(local_degree_profile):pass
    """

    def decorator_func(cls):
        cls.extract = lambda s, data: (func(*s._args, **s._kwargs)(data)).x
        return cls

    return decorator_func


@register_pyg
@pygfunc(LocalDegreeProfile)
class PYGLocalDegreeProfile(PYGGenerator):
    pass


# def _preprocess(self,data):
#     num_nodes=data.num_nodes
#     self._num_nodes=num_nodes
#     data.num_nodes=np.sum(num_nodes)
# def _postprocess(self,data):
#     data.num_nodes=self._num_nodes


@register_pyg
@pygfunc(NormalizeFeatures)
class PYGNormalizeFeatures(PYGGenerator):
    pass


@register_pyg
@pygfunc(OneHotDegree)
class PYGOneHotDegree(PYGGenerator):
    def __init__(self, max_degree=1000):
        super(PYGOneHotDegree, self).__init__(max_degree=max_degree)

    """
    def _transform(self, data):
        #idx, x = data.edge_index[0], data.x
        #deg = degree(idx, data.num_nodes, dtype=torch.long)
        #self._kwargs["max_degree"] = np.min(
        #    [self._kwargs["max_degree"], torch.max(deg).numpy()]
        #)
        dsc = self.extract(data)
        data.x = torch.cat([data.x, dsc], dim=1)
        return data
    """
