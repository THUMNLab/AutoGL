from autogl.datasets import build_dataset_from_name
data = build_dataset_from_name('cora')
from autogl.module.preprocessing.structure_engineering._structure_engineer import *

fes=[GCNJaccard,GCNSVD]
for fe in fes:
    print(f'Doing {fe}')
    fe = fe()
    data=fe.fit_transform(data,inplace=False)


