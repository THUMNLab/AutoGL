from autogl.datasets import build_dataset_from_name
data = build_dataset_from_name('cora')

# 2. Compose a feature engineering pipeline
from autogl.module.preprocessing.feature_engineering import EigenFeatureGenerator

# you may compose feature engineering bases through autogl.module.feature._base_feature_engineer
from autogl.module.preprocessing.structure_engineering._structure_engineer import *
# fe = EigenFeatureGenerator(size=32)
fe=GCNJaccard()

# 3. Fit and transform the data
data1=fe.fit_transform(data,inplace=False)

# from autogl.data.graph import GeneralStaticGraph
# print(isinstance(data, GeneralStaticGraph))
# print(data.nodes)

