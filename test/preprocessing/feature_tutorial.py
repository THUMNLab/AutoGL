import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-t',type=int)
args=parser.parse_args()
t=args.t

if t==0:
    # 1. Choose a dataset.
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')

    # 2. Compose a preprocessing pipeline
    from autogl.module.preprocessing import DataPreprocessor
    from autogl.module.preprocessing.feature_engineering import AutoFeatureEngineer
    from autogl.module.preprocessing.feature_engineering._generators import OneHotFeatureGenerator
    from autogl.module.preprocessing.feature_engineering._selectors import GBDTFeatureSelector
    from autogl.module.preprocessing.feature_engineering._graph import NXLargeCliqueSize
    # you may compose preprocessing bases through operator & 
    fe = OneHotFeatureGenerator() & GBDTFeatureSelector(fixlen=100) & NXLargeCliqueSize()

    # 3. Fit and transform the data
    fe.fit(data)
    data1=fe.transform(data,inplace=False)
elif t==1:
    import torch
    # 1. Choose a dataset.
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')
    
    from autogl.module.preprocessing.feature_engineering._generators._basic import BaseFeatureGenerator
    import numpy as np
    class GeOnehot(BaseFeatureGenerator):    
        def _extract_nodes_feature(self, data):
            num_nodes: int = (
                data.x.size(0)
                if data.x is not None and isinstance(data.x, torch.Tensor)
                else (data.edge_index.max().item() + 1)
            )
            return torch.eye(num_nodes)
    
    fe=GeOnehot()
    fe.fit(data)
    data1=fe.transform(data,inplace=False)
    
elif t==2:
    import torch
    # 1. Choose a dataset.
    from autogl.datasets import build_dataset_from_name
    data = build_dataset_from_name('cora')
    
    from autogl.module.preprocessing.structure_engineering import *
    from autogl.module.preprocessing.structure_engineering._structure_engineer import * 
    from torch_geometric.utils import add_self_loops
    class AddSelfLoop(StructureEngineer):
        def _transform(self,data):
            adj = get_edges(data) # edge list
            modified_adj=add_self_loops(adj)
            set_edges(data,modified_adj)
            return data
        
    fe=AddSelfLoop()
    fe.fit(data)
    data1=fe.transform(data,inplace=False)