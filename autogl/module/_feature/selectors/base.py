from ..base import BaseFeature
import numpy as np


class BaseSelector(BaseFeature):
    def __init__(self, data_t="np", multigraph=False, **kwargs):
        super(BaseSelector, self).__init__(
            data_t=data_t, multigraph=multigraph, **kwargs
        )
        self._sel = None

    def _transform(self, data):
        if self._sel is not None:
            data.x = data.x[:, self._sel]
        return data
