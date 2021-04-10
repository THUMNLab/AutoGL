import sys
sys.path.append('../')
from autogl.solver import AutoNodeClassifier
from autogl.module.hpo.nas import BaseNAS, BaseEstimator, GraphSpace, DartsNodeClfEstimator
from autogl.datasets import build_dataset_from_name
from autogl.module.model import AutoGAT
from autogl.module.train import NodeClassificationFullTrainer
from autogl.module.hpo.darts import Darts
from torch_geometric.nn import GATConv, GCNConv

class TestNASAlgorithm(BaseNAS):
    model = None
    def search(self, space, dset, trainer):
        num_classes = dset.num_classes
        num_features = dset.num_features
        return NodeClassificationFullTrainer(
            "gat",
            num_features=num_features,
            num_classes=num_classes,
            device="auto"
        )

if __name__ == '__main__':
    dataset = build_dataset_from_name('cora')
    solver = AutoNodeClassifier(
        feature_module=None,
        graph_models=[],
        hpo_module="random",
        max_evals=10,
        ensemble_module=None,
        nas_algorithms=[Darts()],
        nas_spaces=[GraphSpace()],
        #nas_spaces=[GraphSpace(dataset.num_features, 64, dataset.num_classes, [GATConv, GCNConv])],
        nas_estimators=[DartsNodeClfEstimator()]
    )
    solver.fit(dataset)
    out = solver.predict(dataset)