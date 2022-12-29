envs='dgl pyg'.split()
forbids=dict(zip(envs,"torch_geometric dgl".split()))
import os
from utils import *
import sys

def func(dev,env):
    sys.modules[forbids[env]]=None
    os.environ['AUTOGL_BACKEND']=env
    from autogl.backend import DependentBackend
    print('using backend :',DependentBackend.get_backend_name())
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')
    from autogl.module.preprocessing.feature_engineering import OneHotFeatureGenerator,EigenFeatureGenerator,GraphletGenerator,PageRankFeatureGenerator,LocalDegreeProfileGenerator,NormalizeFeatures,OneHotDegreeGenerator
    from autogl.module.preprocessing.feature_engineering import IdentityFeature, AutoFeatureEngineer
    from autogl.module.preprocessing.feature_engineering import FilterConstant, GBDTFeatureSelector
    from autogl.module.preprocessing.feature_engineering import NetLSD,NXLargeCliqueSize
    
    fes=[OneHotFeatureGenerator,EigenFeatureGenerator,GraphletGenerator,LocalDegreeProfileGenerator,NormalizeFeatures,OneHotDegreeGenerator]
    exceptions=[]
    for fe in fes:
        try:
            print(f'Doing {fe}')
            fe = fe()
            data=fe.fit_transform(data,inplace=False)
        except Exception as e:
            print(e)
            exceptions.append([fe,e])
    if len(exceptions)==0:
        return 'Test OK'
    return exceptions

# func(0,'dgl')
results=mp_exec([0,1],envs,func)
print(results)


