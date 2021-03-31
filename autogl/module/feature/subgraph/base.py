from ..base import BaseFeatureAtom
import numpy as np
import torch
from .. import register_feature


@register_feature("subgraph")
class BaseSubgraph(BaseFeatureAtom):
    def __init__(self, data_t="np", multigraph=True, **kwargs):
        super(BaseSubgraph, self).__init__(
            data_t=data_t, multigraph=multigraph, subgraph=True, **kwargs
        )

    def _preprocess(self, data):
        if not hasattr(data, "gf") or data["gf"] is None:
            data.gf = torch.FloatTensor([[]])

    def _postprocess(self, data):
        pass
