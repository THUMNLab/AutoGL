# 1. Choose a dataset.
from autogl.datasets import build_dataset_from_name
data = build_dataset_from_name('cora')

# 2. Compose a feature engineering pipeline
from autogl.module.feature._base_feature_engineer._base_feature_engineer import _ComposedFeatureEngineer
from autogl.module.feature import EigenFeatureGenerator
from autogl.module.feature import NetLSD

# you may compose feature engineering bases through autogl.module.feature._base_feature_engineer
fe = _ComposedFeatureEngineer([
    EigenFeatureGenerator(size=32),
    NetLSD()
])

# 3. Fit and transform the data
fe.fit(data)
data1=fe.transform(data,inplace=False)

import autogl
import torch
from autogl.module.feature._generators._basic import BaseFeatureGenerator

class OneHotFeatureGenerator(BaseFeatureGenerator):
    # if overrider_features==False , concat the features with original features; otherwise override.
    def __init__(self, override_features: bool = False): 
        super(BaseFeatureGenerator, self).__init__(override_features)

    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        num_nodes: int = (
            data.x.size(0)
            if data.x is not None and isinstance(data.x, torch.Tensor)
            else (data.edge_index.max().item() + 1)
        )
        return torch.eye(num_nodes)