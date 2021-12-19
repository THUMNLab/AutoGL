import autogl

if autogl.backend.DependentBackend.is_dgl():
    from ._base_feature_engineer_dgl import BaseFeatureEngineer
else:
    from ._base_feature_engineer_pyg import BaseFeatureEngineer


class BaseFeature(BaseFeatureEngineer):
    ...
