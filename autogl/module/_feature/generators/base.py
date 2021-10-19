import numpy as np
from .. import register_feature
from ..base import BaseFeature


class BaseGenerator(BaseFeature):
    def __init__(self, data_t="np", multigraph=True, **kwargs):
        super(BaseGenerator, self).__init__(
            data_t=data_t, multigraph=multigraph, **kwargs
        )


@register_feature("onehot")
class GeOnehot(BaseGenerator):
    def _transform(self, data):
        fe = np.eye(data.x.shape[0])
        data.x = np.concatenate([data.x, fe], axis=1)
        return data
