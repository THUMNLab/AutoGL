import sys
sys.path.append('../')
from autogl.solver import AutoGraphClassifier
from autogl.module.hpo.nas import BaseNAS, BaseEstimator, GraphSpace
from autogl.datasets import build_dataset_from_name
from autogl.module.train import GraphClassificationFullTrainer
from torch_geometric.nn import GATConv, GCNConv

class TestNASAlgorithm(BaseNAS):
    model = None
    def search(self, space, dset, trainer):
        num_classes = dset.num_classes
        num_features = dset.num_features
        return GraphClassificationFullTrainer(
            "gin",
            num_features=num_features,
            num_classes=num_classes,
            device="auto"
        )

if __name__ == '__main__':
    dataset = build_dataset_from_name('mutag')
    solver = AutoGraphClassifier(
        feature_module=None,
        graph_models=[],
        hpo_module="random",
        max_evals=10,
        ensemble_module=None,
        nas_algorithms=[TestNASAlgorithm()],
        nas_spaces=[GraphSpace(dataset.num_features, 64, dataset.num_classes, [GATConv, GCNConv])],
        nas_estimators=[BaseEstimator()]
    )
    solver.fit(dataset, train_split=0.8, val_split=0.1)
    out = solver.predict()