import netlsd
from .base import BaseGraph
import numpy as np
import torch
from .. import register_feature


@register_feature("netlsd")
class SgNetLSD(BaseGraph):
    r"""
    Notes
    -----
    a graph feature generation method. This is a simple wrapper of NetLSD [#]_.
    References
    ----------
    .. [#] A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, and E. Müller, “NetLSD: Hearing the shape of a graph,”
     Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., pp. 2347–2356, 2018.
    """

    def __init__(self, *args, **kwargs):
        super(SgNetLSD, self).__init__(data_t="nx")
        self._args = args
        self._kwargs = kwargs

    def _transform(self, data):
        dsc = torch.FloatTensor([netlsd.heat(data.G, *self._args, **self._kwargs)])
        data.gf = torch.cat([data.gf, dsc], dim=1)
        return data
