from copy import deepcopy
import sys
from nni.nas.pytorch.fixed import apply_fixed_architecture
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
sys.path.append('../')
import torch
from autogl.solver import AutoNodeClassifier
from autogl.module.nas.nas import DartsNodeClfEstimator
from autogl.module.nas.space import GraphSpace
from autogl.datasets import build_dataset_from_name
from autogl.module.model import BaseModel
# from autogl.module.nas.darts import Darts
from autogl.utils import get_logger
from autogl.module.nas.enas import Enas
if __name__ == '__main__':
    dataset = build_dataset_from_name('cora')
    solver = AutoNodeClassifier(
        feature_module=None,
        graph_models=[],
        hpo_module="random",
        max_evals=10,
        ensemble_module=None,
        nas_algorithms=[Enas()],
        nas_spaces=[GraphSpace(hidden_dim=64, ops=[GATConv, GCNConv])],
        nas_estimators=[DartsNodeClfEstimator()]
    )
    solver.fit(dataset)
    out = solver.predict(dataset)