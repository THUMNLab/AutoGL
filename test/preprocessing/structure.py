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
    from autogl.module.preprocessing.structure_engineering import GCNJaccard,GCNSVD

    fes=[GCNJaccard,GCNSVD]
    for fe in fes:
        print(f'Doing {fe}')
        fe = fe()
        data=fe.fit_transform(data,inplace=False)

    return 'Test OK'

# func(0,'dgl')
results=mp_exec([0,1],envs,func)
print(results)


