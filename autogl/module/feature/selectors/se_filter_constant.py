from .base import BaseSelector
import numpy as np
from .. import register_feature


@register_feature("SeFilterConstant")
class SeFilterConstant(BaseSelector):
    r"""drop constant features"""

    def _fit(self, data):
        d1, d2 = data.x.shape
        xx = data.x
        # if d2>=d1:
        #     if np.allclose(xx[:,:d1],np.eye(d1)):
        #         return np.empty((d1,0))
        self._sel = np.where(np.all(xx == xx[0, :], axis=0) == False)[0]
