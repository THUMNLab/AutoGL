from ..base import BaseFeature
import numpy as np
import torch
from .. import register_feature


@register_feature("graph")
class BaseGraph(BaseFeature):
    def __init__(self, data_t="np", multigraph=True, **kwargs):
        super(BaseGraph, self).__init__(
            data_t=data_t, multigraph=multigraph, subgraph=True, **kwargs
        )

    def _preprocess(self, data):
        if not hasattr(data, "gf") or data["gf"] is None:
            data.gf = torch.FloatTensor([[]])

    def _postprocess(self, data):
        pass
